"""Rate-limited market reads plus live-position import for intraday alerts."""

from __future__ import annotations

import copy
import logging
import random
import threading
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, replace
from functools import partial
from typing import TypeVar

import requests

from .client import BitunixClient, BitunixError
from .intraday import (
    Candle,
    Decision,
    Market,
    Side,
    Tier,
    TradePlan,
    blank_checklist,
    closed_candles,
    evaluate_intraday,
    number,
    upsert_check,
    volatility,
)
from .signal_config import SignalsCfg, SignalSettings
from .signal_store import SignalStore, TrackedTrade, evaluate_exit
from .symbol_meta import (
    parse_symbol_meta,
    row_is_tradeable_usdt_perp,
    row_quote_volume_usdt,
    row_symbol,
)

log = logging.getLogger(__name__)
T = TypeVar("T")
MISSING_POSITION_KEYS = "Add Bitunix API keys to import live positions."


@dataclass(frozen=True)
class OpenPosition:
    position_id: str
    symbol: str
    side: Side
    quantity: float
    entry: float
    mark: float | None = None
    unrealized_pnl: float | None = None
    leverage: int | None = None
    opened_at: int | None = None


def parse_open_position(row: object) -> OpenPosition | None:
    if not isinstance(row, dict):
        return None
    symbol = str(row.get("symbol") or "").upper()
    if not symbol.endswith("USDT"):
        return None
    raw_side = str(row.get("side") or row.get("positionSide") or "").upper()
    if raw_side in ("LONG", "BUY"):
        side: Side = "long"
    elif raw_side in ("SHORT", "SELL"):
        side = "short"
    else:
        return None
    try:
        quantity = abs(
            number(row.get("qty") or row.get("size") or row.get("volume") or 0)
        )
        entry = number(
            row.get("avgOpenPrice") or row.get("entryPrice") or row.get("openPrice")
        )
    except (TypeError, ValueError):
        return None
    if quantity <= 0 or entry <= 0:
        return None
    position_id = str(row.get("positionId") or row.get("position_id") or "")
    if not position_id:
        position_id = f"{symbol}:{side}"
    mark: float | None = None
    for key in ("markPrice", "mark_price"):
        if row.get(key) not in (None, ""):
            try:
                mark = number(row[key])
            except (TypeError, ValueError):
                mark = None
            break
    unrealized: float | None = None
    for key in ("unrealizedPNL", "unrealizedPnl", "unrealized_pnl"):
        if row.get(key) not in (None, ""):
            try:
                unrealized = number(row[key])
            except (TypeError, ValueError):
                unrealized = None
            break
    leverage: int | None = None
    for key in ("leverage", "leverageLevel"):
        if row.get(key) not in (None, ""):
            try:
                parsed = int(number(row[key]))
            except (TypeError, ValueError):
                parsed = 0
            if parsed >= 1:
                leverage = parsed
            break
    opened_at: int | None = None
    raw_open = (
        row.get("ctime")
        or row.get("createdTime")
        or row.get("created_time")
        or row.get("openTime")
        or row.get("open_time")
    )
    if raw_open not in (None, ""):
        try:
            timestamp = number(raw_open)
            if timestamp > 10_000_000_000:
                timestamp /= 1000
            if timestamp > 0:
                opened_at = int(timestamp)
        except (TypeError, ValueError):
            opened_at = None
    return OpenPosition(
        position_id,
        symbol,
        side,
        quantity,
        entry,
        mark,
        unrealized,
        leverage,
        opened_at,
    )


READ_ERRORS = (
    BitunixError,
    requests.RequestException,
    ValueError,
    TypeError,
    KeyError,
    IndexError,
)


class SignalScanner:
    def __init__(self, client: BitunixClient, cfg: SignalsCfg, store: SignalStore):
        self.client, self.cfg, self.store = client, cfg, store
        self.lock = threading.RLock()
        self.refresh_lock = threading.Lock()
        self.decisions: dict[str, Decision] = {}
        self.frames: dict[str, dict[str, list[Candle]]] = {}
        self._cache: dict[str, tuple[float, object]] = {}
        self._failures: dict[str, tuple[int, float]] = {}
        self._consecutive_failures = 0
        self._circuit_until = 0.0
        self._last_request = 0.0
        self._last_scan = 0.0
        self.error: str | None = None
        self._featured_symbol: str | None = None
        self._handoff_to: str | None = None
        self._handoff_until: int = 0
        self._positions_error: str | None = None
        self._universe_cursor = 0
        self._universe: list[str] = []
        self._hot_symbols_last: list[str] = []
        self._evaluated_last: list[str] = []

    def _read(self, key: str, ttl: float, fetch: Callable[[], T]) -> T:
        now = time.time()
        cached = self._cache.get(key)
        if cached and now - cached[0] < ttl:
            return copy.deepcopy(cached[1])  # type: ignore[return-value]
        failures, retry_at = self._failures.get(key, (0, 0.0))
        if now < max(retry_at, self._circuit_until):
            raise ValueError("Market provider cooling down after failed reads")
        wait = 0.13 - (time.monotonic() - self._last_request)
        if wait > 0:
            time.sleep(wait)
        self._last_request = time.monotonic()
        try:
            value = fetch()
        except READ_ERRORS:
            self._failures[key] = (
                failures + 1,
                now + min(60, 2 ** min(failures + 1, 6)) + random.random(),
            )
            self._consecutive_failures += 1
            if self._consecutive_failures >= 3:
                self._circuit_until = time.time() + 30
            raise
        self._consecutive_failures = 0
        self._failures.pop(key, None)
        self._cache[key] = (time.time(), copy.deepcopy(value))
        return value

    def _frames(self, symbol: str, now: int) -> dict[str, list[Candle]]:
        frames = {}
        for interval, seconds in (("15m", 900), ("1h", 3600), ("4h", 14400)):
            # Cache only within this candle boundary; never retain the prior incomplete close.
            key = f"candles:{symbol}:{interval}"
            cached = self._cache.get(key)
            if cached and int(cached[0]) // seconds != now // seconds:
                self._cache.pop(key, None)
            rows = self._read(
                key,
                seconds,
                partial(self.client.klines, symbol, interval, limit=200),
            )
            try:
                frames[interval] = closed_candles(rows, interval, now)
            except ValueError:
                self._cache.pop(key, None)
                raise
        return frames

    def _market(
        self, symbol: str, ticker: dict[str, object], pair: dict[str, object]
    ) -> Market:
        funding = self._read(
            f"funding:{symbol}",
            self.cfg.refresh_seconds,
            lambda: self.client.funding_rate(symbol),
        )
        book = self._read(
            f"depth:{symbol}",
            self.cfg.refresh_seconds,
            lambda: self.client.depth(symbol),
        )
        tiers = self._read(
            f"tiers:{symbol}", 3600, lambda: self.client.position_tiers(symbol)
        )
        bids = [(number(p), number(q)) for p, q in book["bids"]]
        asks = [(number(p), number(q)) for p, q in book["asks"]]
        if not bids or not asks or any(min(p, q) <= 0 for p, q in bids + asks):
            raise ValueError("Missing or invalid order book")
        bids.sort(reverse=True)
        asks.sort()
        price, mark = number(funding["lastPrice"]), number(funding["markPrice"])
        base_precision = number(pair["basePrecision"])
        price_precision = number(pair["quotePrecision"])
        if (
            not 0 <= base_precision <= 18
            or not 0 <= price_precision <= 18
            or price_precision != int(price_precision)
        ):
            raise ValueError("Invalid exchange precision metadata")
        if base_precision >= 1 and base_precision != int(base_precision):
            raise ValueError("Invalid quantity precision")
        if number(pair["minTradeVolume"]) < 0:
            raise ValueError("Invalid minimum quantity")
        meta = parse_symbol_meta(pair)
        parsed_tiers = [
            Tier(
                number(t["startValue"]),
                number(t["endValue"]),
                number(t["maintenanceMarginRate"]),
                int(number(t["leverage"])),
            )
            for t in tiers
        ]
        if (
            price <= 0
            or mark <= 0
            or bids[0][0] > asks[0][0]
            or not parsed_tiers
            or any(
                t.minimum < 0
                or t.maximum <= t.minimum
                or not 0 < t.maintenance_rate < 1
                or t.max_leverage < 1
                for t in parsed_tiers
            )
        ):
            raise ValueError("Invalid prices or position tiers")
        next_funding = number(funding["nextFundingTime"])
        if next_funding > 10_000_000_000:
            next_funding /= 1000
        interval = number(funding["fundingInterval"])
        if not 0 < interval <= 24 or next_funding < time.time():
            raise ValueError("Missing or stale funding schedule")
        as_of = int(
            min(
                self._cache[f"funding:{symbol}"][0],
                self._cache[f"depth:{symbol}"][0],
                self._cache["tickers"][0],
            )
        )
        oi = ticker.get("openInterest")
        return Market(
            symbol,
            price,
            mark,
            bids[0][0],
            asks[0][0],
            sum(p * q for p, q in bids),
            sum(p * q for p, q in asks),
            row_quote_volume_usdt(ticker),
            number(funding["fundingRate"]),
            interval,
            int(next_funding),
            parsed_tiers,
            meta.base_precision,
            meta.min_qty,
            as_of,
            number(oi) if oi is not None else None,
        )

    def refresh(self, force: bool = False) -> None:
        if not force and time.time() - self._last_scan < self.cfg.refresh_seconds:
            return
        if not self.refresh_lock.acquire(blocking=False):
            return
        try:
            self._refresh()
        finally:
            self._last_scan = time.time()
            self.refresh_lock.release()

    def _has_account_keys(self) -> bool:
        key = getattr(self.client, "api_key", "")
        secret = getattr(self.client, "secret_key", "")
        return (
            isinstance(key, str)
            and isinstance(secret, str)
            and bool(key.strip() and secret.strip())
        )

    def _load_positions(self) -> tuple[list[OpenPosition], bool]:
        if not self._has_account_keys():
            self._positions_error = MISSING_POSITION_KEYS
            return [], False
        now = time.time()
        cached = self._cache.get("positions")
        if cached and now - cached[0] < self.cfg.refresh_seconds:
            rows = cached[1]
            if isinstance(rows, list):
                return (
                    [p for row in rows if (p := parse_open_position(row))],
                    True,
                )
        wait = 0.13 - (time.monotonic() - self._last_request)
        if wait > 0:
            time.sleep(wait)
        self._last_request = time.monotonic()
        try:
            rows = self.client.pending_positions()
            if not isinstance(rows, list):
                raise ValueError("Invalid positions payload")
            self._cache["positions"] = (time.time(), copy.deepcopy(rows))
            self._positions_error = None
            return [p for row in rows if (p := parse_open_position(row))], True
        except Exception as exc:
            log.warning("Live positions unavailable: %s", exc)
            self._positions_error = (
                "Could not read Bitunix positions; live tracking is paused"
            )
            if cached and isinstance(cached[1], list):
                return (
                    [p for row in cached[1] if (p := parse_open_position(row))],
                    False,
                )
            return [], False

    def _plan_for_position(
        self,
        position: OpenPosition,
        decision: Decision | None,
        settings: SignalSettings,
        now: int,
    ) -> TradePlan:
        side = position.side
        sign = 1 if side == "long" else -1
        entry = position.entry
        quantity = position.quantity
        leverage = max(1, min(position.leverage or settings.leverage, 125))
        mark = position.mark or (
            decision.price if decision and decision.price else entry
        )
        cost_pct = self.cfg.round_trip_fee_pct + self.cfg.slippage_pct
        target2: float | None = None
        funding_cost_pct = 0.0
        funding_payments = 0
        liquidation: float | None = None
        max_leverage = leverage
        if decision and decision.plan and decision.side == side:
            plan = decision.plan
            stop = plan.stop
            target = plan.target
            target2 = plan.target2
            cost_pct = plan.cost_pct
            funding_cost_pct = plan.funding_cost_pct
            funding_payments = plan.funding_payments
            liquidation = (
                plan.liquidation_estimate * entry / plan.entry
                if plan.liquidation_estimate is not None and plan.entry
                else None
            )
            max_leverage = plan.max_leverage
        else:
            bars = self.frames.get(position.symbol, {}).get("15m", [])
            atr_value = volatility(bars) if len(bars) >= 15 else 0.0
            stop_distance = max(
                entry * 0.015, atr_value * 1.5 if atr_value > 0 else entry * 0.015
            )
            stop = entry - sign * stop_distance
            target = entry + sign * 2 * stop_distance
            liquidation = entry * (1 - sign / leverage)
        if sign * (mark - stop) <= 0:
            stop = mark - sign * max(entry * 0.005, abs(entry - stop) * 0.15)
        notional = entry * quantity
        risk = quantity * (abs(entry - stop) + entry * cost_pct / 100)
        reward = quantity * (abs(target - entry) - entry * cost_pct / 100)
        return TradePlan(
            side,
            entry,
            min(entry, mark),
            max(entry, mark),
            stop,
            target,
            target2,
            quantity,
            notional,
            notional / leverage,
            risk,
            risk / settings.planning_equity * 100,
            reward / risk if risk > 0 else 0.0,
            abs(entry - stop) / entry * 100,
            cost_pct,
            funding_cost_pct,
            funding_payments,
            liquidation,
            max_leverage,
            leverage,
            settings.hold_hours,
            now + 1800,
        )

    def _sync_exchange_positions(
        self,
        positions: list[OpenPosition],
        decisions: dict[str, Decision],
        now: int,
        *,
        fetch_ok: bool,
    ) -> None:
        if not fetch_ok:
            return
        settings = self.store.settings()
        live_ids = {position.position_id for position in positions}
        matched: set[str] = set()
        seen: set[tuple[str, str]] = set()
        for position in positions:
            key = (position.symbol, position.side)
            if key in seen:
                continue
            seen.add(key)
            trade_id = f"exchange:{position.position_id}"
            trades = self.store.trades()
            existing = next((trade for trade in trades if trade.id == trade_id), None)
            if existing is None:
                existing = next(
                    (
                        trade
                        for trade in trades
                        if trade.closed_at is None
                        and trade.kind == "exchange"
                        and trade.symbol == position.symbol
                        and trade.plan.side == position.side
                    ),
                    None,
                )
            if existing is None:
                recorded = next(
                    (
                        trade
                        for trade in trades
                        if trade.closed_at is None
                        and trade.kind in ("paper", "manual")
                        and trade.symbol == position.symbol
                        and trade.plan.side == position.side
                    ),
                    None,
                )
                if recorded:
                    recorded.mark_price = position.mark
                    recorded.unrealized_pnl = position.unrealized_pnl
                    recorded.exchange_position_id = position.position_id
                    self.store.save_trade(recorded)
                    matched.add(recorded.id)
                    continue
                opened = position.opened_at or now
                if opened > now or opened <= 0:
                    opened = now
                plan = self._plan_for_position(
                    position, decisions.get(position.symbol), settings, now
                )
                trade = TrackedTrade(
                    trade_id,
                    position.symbol,
                    "exchange",
                    opened,
                    plan,
                    plan.stop,
                    position.mark or position.entry,
                    state=f"HOLD_{position.side.upper()}",
                    reason=(
                        "Imported live Bitunix position; alerts only, no exchange order"
                    ),
                    checked_at=opened,
                    mark_price=position.mark,
                    unrealized_pnl=position.unrealized_pnl,
                    exchange_position_id=position.position_id,
                )
                self.store.save_trade(trade)
                matched.add(trade.id)
                self.store.record_alert(
                    f"{trade_id}:imported:{opened}",
                    trade.state,
                    position.symbol,
                    trade.reason,
                    now,
                    side=position.side,
                )
                continue
            if existing.closed_at:
                existing.closed_at = None
                existing.exit_price = None
                existing.estimated_net_pnl = None
                existing.state = f"HOLD_{position.side.upper()}"
                existing.reason = "Live Bitunix position is still open"
            existing.exchange_position_id = position.position_id
            existing.mark_price = position.mark
            existing.unrealized_pnl = position.unrealized_pnl
            mark = position.mark or existing.best_price
            sign = 1 if existing.plan.side == "long" else -1
            existing.best_price = max(
                [existing.best_price, mark], key=lambda price: sign * price
            )
            if (
                abs(existing.plan.quantity - position.quantity) > 1e-12
                or abs(existing.plan.entry - position.entry) > 1e-8
            ):
                plan = self._plan_for_position(
                    position, decisions.get(position.symbol), settings, now
                )
                if sign * (existing.current_stop - plan.stop) <= 0:
                    existing.current_stop = plan.stop
                existing.plan = plan
            self.store.save_trade(existing)
            matched.add(existing.id)
        for trade in self.store.trades(active_only=True):
            if trade.kind == "exchange" and trade.id not in matched:
                mark = trade.mark_price or trade.plan.entry
                trade.closed_at = now
                trade.exit_price = mark
                trade.state = "CLOSED"
                sign = 1 if trade.plan.side == "long" else -1
                trade.estimated_net_pnl = (
                    sign * (mark - trade.plan.entry) * trade.plan.quantity
                    - trade.plan.notional * trade.plan.cost_pct / 100
                )
                trade.reason = (
                    "Bitunix position is no longer open; tracking closed automatically"
                )
                self.store.save_trade(trade)
                self.store.record_alert(
                    f"{trade.id}:closed:{now}",
                    trade.state,
                    trade.symbol,
                    trade.reason,
                    now,
                    side=trade.plan.side,
                )
            elif (
                trade.kind != "exchange"
                and trade.exchange_position_id
                and trade.exchange_position_id not in live_ids
            ):
                trade.exchange_position_id = ""
                trade.mark_price = None
                trade.unrealized_pnl = None
                self.store.save_trade(trade)

    def _rank_decisions(
        self, rows: dict[str, Decision] | None = None
    ) -> list[Decision]:
        return sorted(
            (rows if rows is not None else self.decisions).values(),
            key=lambda d: (
                d.state.startswith("ENTER_"),
                d.state.startswith("WATCH_"),
                sum(c.passed for c in d.checks),
                d.plan.net_reward_risk if d.plan else 0.0,
            ),
            reverse=True,
        )

    def _liquid_universe(self, by_ticker: dict[str, dict[str, object]]) -> list[str]:
        return sorted(
            (
                symbol
                for symbol in by_ticker
                if row_quote_volume_usdt(by_ticker[symbol]) >= self.cfg.min_quote_volume
            ),
            key=lambda symbol: row_quote_volume_usdt(by_ticker[symbol]),
            reverse=True,
        )[: self.cfg.universe_size]

    def _hot_symbols(self, liquid: list[str], pinned: list[str]) -> list[str]:
        hot: list[str] = []

        def add(symbol: str | None) -> None:
            if symbol and symbol not in hot:
                hot.append(symbol)

        add("BTCUSDT")
        for symbol in pinned:
            add(symbol)
        add(self._featured_symbol)
        for decision in self.decisions.values():
            if decision.state.startswith(("ENTER_", "WATCH_")):
                add(decision.symbol)
        for decision in self._rank_decisions()[: self.cfg.queue_size]:
            add(decision.symbol)
        for symbol in liquid:
            if len(hot) >= self.cfg.max_symbols:
                break
            add(symbol)
        return hot

    def _rotate_universe(self, liquid: list[str], already: set[str]) -> list[str]:
        if not liquid or self.cfg.evaluate_batch <= 0:
            return []
        picked: list[str] = []
        start = self._universe_cursor % len(liquid)
        for offset in range(len(liquid)):
            symbol = liquid[(start + offset) % len(liquid)]
            if symbol in already:
                continue
            picked.append(symbol)
            if len(picked) >= self.cfg.evaluate_batch:
                self._universe_cursor = (start + offset + 1) % len(liquid)
                return picked
        return picked

    def _evaluate_symbol(
        self,
        symbol: str,
        now: int,
        settings: SignalSettings,
        by_ticker: dict[str, dict[str, object]],
        by_pair: dict[str, dict[str, object]],
        frames: dict[str, dict[str, list[Candle]]],
    ) -> Decision:
        frames[symbol] = self._frames(symbol, now)
        market = self._market(symbol, by_ticker[symbol], by_pair[symbol])
        return evaluate_intraday(
            market,
            frames[symbol],
            frames.get("BTCUSDT", {}).get("1h"),
            settings,
            self.cfg,
            int(time.time()),
        )

    def _refresh(self) -> None:
        now = int(time.time())
        settings = self.store.settings()
        active = self.store.trades(active_only=True)
        positions, positions_ok = self._load_positions()
        try:
            pairs = self._read("pairs", 900, self.client.trading_pairs)
            tickers = self._read(
                "tickers", self.cfg.refresh_seconds, self.client.tickers
            )
            by_pair = {row_symbol(p): p for p in pairs if row_is_tradeable_usdt_perp(p)}
            by_ticker = {row_symbol(t): t for t in tickers if row_symbol(t) in by_pair}
            liquid = self._liquid_universe(by_ticker)
            pinned = [trade.symbol for trade in active] + [
                position.symbol for position in positions
            ]
            hot = self._hot_symbols(liquid, pinned)
            rotated = self._rotate_universe(liquid, set(hot))
            symbols = list(dict.fromkeys(hot + rotated))
            if not by_ticker:
                raise ValueError("Provider returned no tradable markets")
        except READ_ERRORS as exc:
            log.warning("Intraday universe unavailable: %s", exc)
            with self.lock:
                self.error = "Market provider unavailable; new entries are blocked"
                self.decisions = {}
                self._universe = []
                self._hot_symbols_last = []
                self._evaluated_last = []
                self._sync_exchange_positions(
                    positions, {}, now, fetch_ok=positions_ok
                )
                self._update_exits({}, now)
            return
        frames: dict[str, dict[str, list[Candle]]] = {}
        decisions: dict[str, Decision] = {}
        for symbol in symbols:
            try:
                decisions[symbol] = self._evaluate_symbol(
                    symbol, now, settings, by_ticker, by_pair, frames
                )
            except READ_ERRORS as exc:
                log.warning("Intraday data rejected for %s: %s", symbol, exc)
                decisions[symbol] = Decision(
                    symbol,
                    reasons=[f"Data unavailable: {exc}"],
                    checks=blank_checklist(
                        "Wait for valid candles, funding, depth and risk tiers"
                    ),
                )
        keep = set(liquid) | set(symbols) | set(pinned) | {"BTCUSDT"}
        with self.lock:
            # A settings update invalidates any scan that started with the old risk profile.
            if settings != self.store.settings():
                return
            self.error = None
            self._stamp_states(decisions, now)
            merged_frames = {
                symbol: bars
                for symbol, bars in self.frames.items()
                if symbol in keep
            }
            merged_frames.update(frames)
            merged = {
                symbol: decision
                for symbol, decision in self.decisions.items()
                if symbol in keep
            }
            merged.update(decisions)
            self.frames, self.decisions = merged_frames, merged
            self._universe = liquid
            self._hot_symbols_last = hot
            self._evaluated_last = symbols
            self._sync_exchange_positions(
                positions, merged, now, fetch_ok=positions_ok
            )
            self._update_exits(merged, int(time.time()))
            for decision in merged.values():
                self._portfolio_gate(decision)
            for decision in decisions.values():
                self._record_signal_alert(decision, now)

    def _portfolio_gate(self, decision: Decision) -> None:
        if not decision.state.startswith("ENTER_") or not decision.plan:
            return
        active = self.store.trades(active_only=True)
        settings = self.store.settings()
        blocked = (
            any(t.symbol == decision.symbol for t in active)
            or sum(t.plan.side == decision.side for t in active)
            >= self.cfg.max_same_direction
            or sum(t.plan.risk_usdt for t in active) + decision.plan.risk_usdt
            > settings.planning_equity * self.cfg.max_total_risk_pct / 100
        )
        if blocked:
            decision.state = f"WATCH_{decision.side.upper()}"
            upsert_check(
                decision.checks,
                "Tracked exposure",
                False,
                "Close or review existing tracked exposure before adding this trade",
            )
            decision.reasons = [
                "Tracked portfolio risk or same-direction limit reached"
            ]

    def _update_exits(self, decisions: dict[str, Decision], now: int) -> None:
        for trade in self.store.trades(active_only=True):
            previous = trade.state
            evaluate_exit(
                trade,
                decisions.get(trade.symbol),
                self.frames.get(trade.symbol, {}).get("15m", []),
                now,
                self.cfg,
            )
            self.store.save_trade(trade)
            if trade.state != previous and (
                trade.state.startswith("EXIT_") or trade.state == "REVIEW"
            ):
                self.store.record_alert(
                    f"{trade.id}:{trade.state}:{now if trade.state == 'REVIEW' else ''}",
                    trade.state,
                    trade.symbol,
                    trade.reason,
                    now,
                    side=trade.plan.side,
                )

    def _stamp_states(self, decisions: dict[str, Decision], now: int) -> None:
        previous = self.decisions
        for symbol, decision in decisions.items():
            prior = previous.get(symbol)
            identity = (
                decision.state,
                decision.side,
                decision.setup,
                decision.bar_time,
            )
            if (
                prior
                and (
                    prior.state,
                    prior.side,
                    prior.setup,
                    prior.bar_time,
                )
                == identity
                and prior.state_since
            ):
                decision.state_since = prior.state_since
            else:
                decision.state_since = now

    def _record_signal_alert(self, decision: Decision, now: int) -> None:
        if decision.state.startswith("ENTER_") and decision.signal_id:
            key = decision.signal_id
        elif decision.state.startswith("WATCH_") and decision.bar_time:
            key = (
                f"{decision.symbol}:{decision.state}:"
                f"{decision.setup or 'none'}:{decision.bar_time}"
            )
        else:
            return
        reason = decision.reasons[0] if decision.reasons else decision.state
        self.store.record_alert(
            key,
            decision.state,
            decision.symbol,
            reason,
            now,
            setup=decision.setup,
            side=decision.side,
        )

    def _queue_item(self, decision: Decision, now: int) -> dict[str, object]:
        expires = (
            decision.plan.expires_at
            if decision.plan and decision.state.startswith("ENTER_")
            else None
        )
        remaining = max(0, expires - now) if expires is not None else None
        return {
            "symbol": decision.symbol,
            "state": decision.state,
            "side": decision.side,
            "setup": decision.setup,
            "as_of": decision.as_of,
            "state_since": decision.state_since or decision.as_of or now,
            "expires_at": expires,
            "seconds_remaining": remaining,
            "checks_passed": sum(c.passed for c in decision.checks),
            "checks_total": len(decision.checks),
            "reason": decision.reasons[0] if decision.reasons else "",
            "price": decision.price,
        }

    def _featured_and_handoff(
        self, ranked: list[Decision], rows: dict[str, Decision], now: int
    ) -> tuple[str | None, dict[str, object] | None]:
        natural = ranked[0].symbol if ranked else None
        featured = self._featured_symbol
        if featured and featured not in rows:
            featured = None
        if not featured:
            featured = natural
            self._handoff_to = None
            self._handoff_until = 0
        elif natural != featured:
            if self._handoff_to != natural:
                self._handoff_to = natural
                self._handoff_until = now + self.cfg.handoff_seconds
            if now >= self._handoff_until:
                featured = natural
                self._handoff_to = None
                self._handoff_until = 0
        else:
            self._handoff_to = None
            self._handoff_until = 0
        self._featured_symbol = featured
        handoff: dict[str, object] | None = None
        if self._handoff_to and self._handoff_until > now:
            handoff = {
                "from_symbol": featured,
                "to_symbol": self._handoff_to,
                "reason": "A higher-ranked setup is ready",
                "expires_at": self._handoff_until,
                "seconds_remaining": max(0, self._handoff_until - now),
            }
        else:
            current = rows.get(featured) if featured else None
            if (
                current
                and current.plan
                and current.state.startswith("ENTER_")
            ):
                remaining = current.plan.expires_at - now
                if 0 < remaining <= self.cfg.expiry_warn_seconds:
                    nxt = next(
                        (d.symbol for d in ranked if d.symbol != featured), None
                    )
                    handoff = {
                        "from_symbol": featured,
                        "to_symbol": nxt,
                        "reason": "Entry window ending",
                        "expires_at": current.plan.expires_at,
                        "seconds_remaining": remaining,
                    }
        return featured, handoff

    def snapshot(self) -> dict[str, object]:
        now = int(time.time())
        with self.lock:
            rows = copy.deepcopy(self.decisions)
            self._update_exits(rows, now)
            for decision in rows.values():
                if (
                    not decision.as_of
                    or now - decision.as_of > self.cfg.max_data_age_seconds
                ):
                    decision.state = "WAIT"
                    if decision.as_of:
                        decision.reasons = [
                            "Market data is stale; wait for a fresh scan"
                        ]
                elif decision.plan and now >= decision.plan.expires_at:
                    decision.state = "WAIT"
                    decision.reasons = [
                        "Entry window expired; wait for the next completed candle"
                    ]
                self._portfolio_gate(decision)
            for decision in rows.values():
                if not decision.state_since:
                    decision.state_since = decision.as_of or now
            live_symbols = {
                trade.symbol
                for trade in self.store.trades(active_only=True)
            }
            published = {
                symbol: decision
                for symbol, decision in rows.items()
                if symbol in self._hot_symbols_last
                or symbol in self._evaluated_last
                or symbol in live_symbols
                or (
                    decision.as_of
                    and now - decision.as_of <= self.cfg.max_data_age_seconds
                )
            }
            ranked = self._rank_decisions(published)
            featured, handoff = self._featured_and_handoff(ranked, published, now)
            trades = self.store.trades()
            live = [
                t
                for t in trades
                if not t.closed_at and (t.kind == "exchange" or t.exchange_position_id)
            ]
            connected = self._has_account_keys()
            return {
                "version": 2,
                "strategy": "intraday",
                "mode": "alerts_only",
                "now": now,
                "settings": asdict(self.store.settings()),
                "data_max_age": self.cfg.max_data_age_seconds,
                "status": {
                    "ready": any(
                        d.as_of and now - d.as_of <= self.cfg.max_data_age_seconds
                        for d in rows.values()
                    ),
                    "error": self.error,
                },
                "positions": {
                    "connected": connected,
                    "imported": len(live),
                    "error": (
                        MISSING_POSITION_KEYS
                        if not connected
                        else self._positions_error
                    ),
                },
                "scan": {
                    "universe": len(self._universe),
                    "hot": len(self._hot_symbols_last),
                    "hot_symbols": list(self._hot_symbols_last),
                    "evaluated": list(self._evaluated_last),
                    "refresh_seconds": self.cfg.refresh_seconds,
                    "evaluate_batch": self.cfg.evaluate_batch,
                    "queue_size": self.cfg.queue_size,
                },
                "symbols": {d.symbol: asdict(d) for d in ranked},
                "queue": [
                    self._queue_item(d, now) for d in ranked[: self.cfg.queue_size]
                ],
                "best_symbol": featured,
                "handoff": handoff,
                "trades": [asdict(t) for t in trades if not t.closed_at],
                "closed_trades": [asdict(t) for t in trades if t.closed_at][:30],
                "history": self.store.history(),
            }

    def update_settings(self, values: dict[str, object]) -> None:
        settings = SignalSettings.from_dict(values)
        with self.lock:
            self.store.save_settings(settings)
            self.decisions = {}
            self._last_scan = 0
            self._universe_cursor = 0
            self._universe = []
            self._hot_symbols_last = []
            self._evaluated_last = []
            self._featured_symbol = None
            self._handoff_to = None
            self._handoff_until = 0

    def track(self, values: dict[str, object]) -> TrackedTrade:
        allowed = {"signal_id", "kind", "entry", "quantity"}
        extra = {"exchange_stop_confirmed"}
        if (
            not allowed <= set(values) <= allowed | extra
            or values["kind"] not in ("paper", "manual")
        ):
            raise ValueError(
                "Provide signal_id, kind (paper/manual), entry and quantity"
            )
        with self.lock:
            existing = next(
                (t for t in self.store.trades() if t.id == values["signal_id"]), None
            )
            if existing:
                return existing
            decision = next(
                (
                    d
                    for d in self.decisions.values()
                    if d.signal_id == values["signal_id"]
                ),
                None,
            )
            now = int(time.time())
            if (
                not decision
                or not decision.plan
                or decision.state != f"ENTER_{decision.side.upper()}"
            ):
                raise ValueError("A confirmed entry signal is required")
            self._portfolio_gate(decision)
            if (
                not decision.state.startswith("ENTER_")
                or now - decision.as_of > self.cfg.max_data_age_seconds
                or now >= decision.plan.expires_at
            ):
                raise ValueError(
                    "Signal expired, stale, or blocked by tracked exposure"
                )
            plan = decision.plan
            entry, quantity = number(values["entry"]), number(values["quantity"])
            if (
                not plan.entry_low <= entry <= plan.entry_high
                or not 0 < quantity <= plan.quantity
            ):
                raise ValueError(
                    "Recorded entry must be inside the entry zone and quantity no larger than the plan"
                )
            risk = quantity * (abs(entry - plan.stop) + entry * plan.cost_pct / 100)
            settings = self.store.settings()
            reward = quantity * (abs(plan.target - entry) - entry * plan.cost_pct / 100)
            sign = 1 if plan.side == "long" else -1
            liquidation = (
                (plan.liquidation_estimate * entry / plan.entry)
                if plan.liquidation_estimate is not None
                else None
            )
            atr_distance = (
                number(decision.metrics.get("atr_pct")) * number(decision.price) / 100
            )
            required_buffer = max(
                entry * self.cfg.liquidation_buffer_pct / 100, atr_distance * 0.5
            )
            if (
                liquidation is None
                or sign * (plan.stop - liquidation) + plan.adverse_mark_basis
                < required_buffer
            ):
                raise ValueError(
                    "Recorded fill no longer leaves the required liquidation buffer"
                )
            portfolio_risk = (
                sum(t.plan.risk_usdt for t in self.store.trades(active_only=True))
                + risk
            )
            if (
                risk > settings.planning_equity * settings.risk_pct / 100 + 1e-8
                or reward / risk < self.cfg.min_reward_risk
                or portfolio_risk
                > settings.planning_equity * self.cfg.max_total_risk_pct / 100
                or entry * quantity > plan.notional + 1e-8
            ):
                raise ValueError(
                    "Recorded fill no longer meets the risk budget or reward requirement"
                )
            updated = replace(
                plan,
                entry=entry,
                quantity=quantity,
                notional=entry * quantity,
                margin=entry * quantity / plan.leverage,
                risk_usdt=risk,
                risk_pct=risk / settings.planning_equity * 100,
                net_reward_risk=reward / risk,
                liquidation_estimate=liquidation,
                stop_pct=abs(entry - plan.stop) / entry * 100,
            )
            trade = TrackedTrade(
                decision.signal_id,
                decision.symbol,
                str(values["kind"]),
                now,
                updated,
                plan.stop,
                entry,
                state=f"HOLD_{decision.side.upper()}",
                checked_at=now,
                exchange_stop_confirmed=(
                    str(values["kind"]) == "paper"
                    or values.get("exchange_stop_confirmed") is True
                ),
            )
            self.store.save_trade(trade)
            return trade

    def confirm_stop(self, values: dict[str, object]) -> TrackedTrade:
        if set(values) != {"id"}:
            raise ValueError("Provide id")
        with self.lock:
            trade = next((t for t in self.store.trades() if t.id == values["id"]), None)
            if not trade:
                raise ValueError("Tracked trade not found")
            if trade.closed_at:
                return trade
            trade.exchange_stop_confirmed = True
            evaluate_exit(
                trade,
                self.decisions.get(trade.symbol),
                self.frames.get(trade.symbol, {}).get("15m", []),
                int(time.time()),
                self.cfg,
            )
            self.store.save_trade(trade)
            return trade

    def _matching_live_position(
        self, trade: TrackedTrade, positions: list[OpenPosition]
    ) -> OpenPosition | None:
        if trade.exchange_position_id:
            found = next(
                (
                    position
                    for position in positions
                    if position.position_id == trade.exchange_position_id
                ),
                None,
            )
            if found:
                return found
        return next(
            (
                position
                for position in positions
                if position.symbol == trade.symbol and position.side == trade.plan.side
            ),
            None,
        )

    def _pair_row(self, symbol: str) -> dict[str, object] | None:
        cached = self._cache.get("pairs")
        rows = cached[1] if cached and isinstance(cached[1], list) else None
        if not rows:
            try:
                fetched = self._read("pairs", 900, self.client.trading_pairs)
            except READ_ERRORS:
                fetched = []
            rows = fetched if isinstance(fetched, list) else []
        for row in rows:
            if isinstance(row, dict) and row_symbol(row) == symbol:
                return row
        return None

    def _price_digits(self, symbol: str) -> int:
        row = self._pair_row(symbol)
        if row is None:
            return 8
        digits = int(parse_symbol_meta(row).price_precision)
        return digits if 0 <= digits <= 18 else 8

    def _format_price(self, symbol: str, value: float) -> str:
        digits = self._price_digits(symbol)
        scale = 10**digits
        quantized = round(value * scale) / scale if scale else value
        text = f"{quantized:.{digits}f}"
        return text.rstrip("0").rstrip(".") if "." in text else text

    def _format_qty(self, symbol: str, value: float) -> str:
        row = self._pair_row(symbol)
        step = parse_symbol_meta(row).base_precision if row else 0.0
        if step <= 0:
            text = f"{value:.8f}"
            return text.rstrip("0").rstrip(".") if "." in text else text
        quantized = round(value / step) * step
        digits = max(0, min(18, len(f"{step:.16f}".rstrip("0").split(".")[-1])))
        if step >= 1:
            digits = 0
        text = f"{quantized:.{digits}f}"
        return text.rstrip("0").rstrip(".") if "." in text else text

    def _pending_stop_rows(
        self, symbol: str, position_id: str
    ) -> list[dict[str, object]]:
        matches: list[dict[str, object]] = []
        for row in self.client.pending_tpsl(symbol):
            if not isinstance(row, dict) or row.get("slPrice") in (None, ""):
                continue
            row_id = str(row.get("positionId") or row.get("position_id") or "")
            if row_id and row_id != str(position_id):
                continue
            if not row_id and row_symbol(row) not in ("", symbol):
                continue
            matches.append(row)
        return matches

    def _qty_stop_rows(
        self, rows: list[dict[str, object]]
    ) -> list[dict[str, object]]:
        return [row for row in rows if row.get("slQty") not in (None, "")]

    def _already_exists_stop(self, exc: BitunixError) -> bool:
        text = f"{exc.code} {exc.msg}".lower()
        return any(
            token in text
            for token in ("already", "exist", "duplicate", "only one", "has tpsl")
        )

    def _stop_matches(self, current: float | None, desired: float) -> bool:
        return current is not None and abs(current - desired) <= max(
            abs(desired) * 1e-8, 1e-10
        )

    def _is_tighter(self, trade: TrackedTrade, current: float | None, desired: float) -> bool:
        if current is None:
            return True
        return desired > current if trade.plan.side == "long" else desired < current

    def _modify_qty_stop(
        self, row: dict[str, object], stop_text: str, qty_text: str
    ) -> None:
        order_id = str(row.get("id") or row.get("orderId") or "")
        if not order_id:
            raise BitunixError(0, "Existing quantity stop is missing an order id", row)
        self.client.modify_tpsl_order(
            order_id,
            sl_price=stop_text,
            sl_qty=str(row.get("slQty") or qty_text),
            sl_stop_type=str(row.get("slStopType") or "LAST_PRICE"),
            sl_order_type=str(row.get("slOrderType") or "MARKET"),
        )

    def _write_exchange_stop(
        self,
        trade: TrackedTrade,
        position: OpenPosition,
        stop_text: str,
        qty_text: str,
    ) -> None:
        # The Bitunix ticket shows quantity TP/SL from /tpsl/place_order.
        # Position-level TPSL can succeed without filling that field.
        existing = self._pending_stop_rows(trade.symbol, position.position_id)
        qty_rows = self._qty_stop_rows(existing)
        reference = qty_rows[0] if qty_rows else existing[0] if existing else None
        current = None
        if reference is not None:
            try:
                current = number(reference.get("slPrice"))
            except (TypeError, ValueError):
                current = None
        desired = number(stop_text)
        if qty_rows and (
            self._stop_matches(current, desired)
            or not self._is_tighter(trade, current, desired)
        ):
            return
        if qty_rows and self._is_tighter(trade, current, desired):
            self._modify_qty_stop(qty_rows[0], stop_text, qty_text)
            return
        try:
            self.client.place_qty_tpsl(
                trade.symbol, position.position_id, stop_text, qty_text
            )
            return
        except BitunixError as exc:
            if not self._already_exists_stop(exc):
                raise
        existing = self._pending_stop_rows(trade.symbol, position.position_id)
        qty_rows = self._qty_stop_rows(existing)
        if qty_rows:
            self._modify_qty_stop(qty_rows[0], stop_text, qty_text)
            return
        if existing:
            self.client.modify_position_tpsl(
                trade.symbol, position.position_id, stop_text
            )
            return
        self.client.place_position_tpsl(
            trade.symbol, position.position_id, stop_text
        )

    def place_stop(self, values: dict[str, object]) -> TrackedTrade:
        if set(values) != {"id"}:
            raise ValueError("Provide id")
        with self.lock:
            trade = next((t for t in self.store.trades() if t.id == values["id"]), None)
            if not trade:
                raise ValueError("Tracked trade not found")
            if trade.closed_at:
                raise ValueError("Trade is already closed")
            if trade.kind == "paper":
                raise ValueError("Paper tracks do not place an exchange stop")
            if not self._has_account_keys():
                raise ValueError(
                    "Add Bitunix API keys with Trade permission to place the stop"
                )
            if trade.current_stop <= 0:
                raise ValueError("Working stop is missing")
            positions, ok = self._load_positions()
            if not ok:
                raise ValueError(
                    self._positions_error
                    or "Could not read live Bitunix positions"
                )
            position = self._matching_live_position(trade, positions)
            if position is None:
                raise ValueError(
                    "Open this position on Bitunix first, then click Set Bitunix stop"
                )
            decision = self.decisions.get(trade.symbol)
            live = (
                position.mark
                or trade.mark_price
                or (decision.price if decision and decision.price else None)
                or position.entry
            )
            sign = 1 if trade.plan.side == "long" else -1
            if sign * (live - trade.current_stop) <= 0:
                raise ValueError(
                    "Stop is already through the market; close on Bitunix instead of hoping"
                )
            stop_text = self._format_price(trade.symbol, trade.current_stop)
            qty_text = self._format_qty(trade.symbol, position.quantity)
            try:
                self._write_exchange_stop(trade, position, stop_text, qty_text)
            except BitunixError as exc:
                raise ValueError(f"Bitunix rejected the stop: {exc.msg}") from exc
            except READ_ERRORS as exc:
                raise ValueError(f"Could not place the Bitunix stop: {exc}") from exc
            trade.exchange_stop_confirmed = True
            trade.exchange_position_id = position.position_id
            trade.mark_price = position.mark
            trade.unrealized_pnl = position.unrealized_pnl
            evaluate_exit(
                trade,
                decision,
                self.frames.get(trade.symbol, {}).get("15m", []),
                int(time.time()),
                self.cfg,
            )
            self.store.save_trade(trade)
            return trade

    def close_track(self, values: dict[str, object]) -> TrackedTrade:
        if set(values) != {"id", "exit_price"}:
            raise ValueError("Provide id and exit_price")
        price = number(values["exit_price"])
        if price <= 0:
            raise ValueError("Exit price must be positive")
        with self.lock:
            trade = next((t for t in self.store.trades() if t.id == values["id"]), None)
            if not trade:
                raise ValueError("Tracked trade not found")
            if trade.closed_at:
                return trade
            trade.closed_at, trade.exit_price, trade.state = (
                int(time.time()),
                price,
                "CLOSED",
            )
            sign = 1 if trade.plan.side == "long" else -1
            trade.estimated_net_pnl = (
                sign * (price - trade.plan.entry) * trade.plan.quantity
                - trade.plan.notional * trade.plan.cost_pct / 100
            )
            trade.reason = (
                "Closure recorded by user; costs are estimates, not exchange settlement"
            )
            self.store.save_trade(trade)
            return trade
