"""Rate-limited market reads and durable manual/paper tracking for intraday alerts."""

from __future__ import annotations

import copy
import logging
import random
import threading
import time
from collections.abc import Callable
from dataclasses import asdict, replace
from functools import partial
from typing import TypeVar

import requests

from .client import BitunixClient, BitunixError
from .intraday import (
    Candle,
    Check,
    Decision,
    Market,
    Tier,
    closed_candles,
    evaluate_intraday,
    number,
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

    def _refresh(self) -> None:
        now = int(time.time())
        settings = self.store.settings()
        active = self.store.trades(active_only=True)
        try:
            pairs = self._read("pairs", 900, self.client.trading_pairs)
            tickers = self._read(
                "tickers", self.cfg.refresh_seconds, self.client.tickers
            )
            by_pair = {row_symbol(p): p for p in pairs if row_is_tradeable_usdt_perp(p)}
            by_ticker = {row_symbol(t): t for t in tickers if row_symbol(t) in by_pair}
            liquid = sorted(
                (
                    s
                    for s in by_ticker
                    if row_quote_volume_usdt(by_ticker[s]) >= self.cfg.min_quote_volume
                ),
                key=lambda s: row_quote_volume_usdt(by_ticker[s]),
                reverse=True,
            )[: self.cfg.max_symbols]
            symbols = list(
                dict.fromkeys(["BTCUSDT"] + liquid + [t.symbol for t in active])
            )
            if not by_ticker:
                raise ValueError("Provider returned no tradable markets")
        except READ_ERRORS as exc:
            log.warning("Intraday universe unavailable: %s", exc)
            with self.lock:
                self.error = "Market provider unavailable; new entries are blocked"
                self.decisions = {}
                self._update_exits({}, now)
            return
        frames: dict[str, dict[str, list[Candle]]] = {}
        decisions: dict[str, Decision] = {}
        for symbol in symbols:
            try:
                frames[symbol] = self._frames(symbol, now)
                market = self._market(symbol, by_ticker[symbol], by_pair[symbol])
                decisions[symbol] = evaluate_intraday(
                    market,
                    frames[symbol],
                    frames.get("BTCUSDT", {}).get("1h"),
                    settings,
                    self.cfg,
                    int(time.time()),
                )
            except READ_ERRORS as exc:
                log.warning("Intraday data rejected for %s: %s", symbol, exc)
                decisions[symbol] = Decision(
                    symbol,
                    reasons=[f"Data unavailable: {exc}"],
                    checks=[
                        Check(
                            "Complete market data",
                            False,
                            "Wait for valid candles, funding, depth and risk tiers",
                        )
                    ],
                )
        with self.lock:
            # A settings update invalidates any scan that started with the old risk profile.
            if settings != self.store.settings():
                return
            self.error = None
            self.frames, self.decisions = frames, decisions
            self._update_exits(decisions, int(time.time()))
            for decision in decisions.values():
                self._portfolio_gate(decision)
                if decision.state.startswith("ENTER_"):
                    self.store.record_alert(
                        decision.signal_id,
                        decision.state,
                        decision.symbol,
                        decision.reasons[0],
                        now,
                    )

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
            decision.checks.append(
                Check(
                    "Tracked exposure",
                    False,
                    "Close or review existing tracked exposure before adding this trade",
                )
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
                )

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
            ranked = sorted(
                rows.values(),
                key=lambda d: (
                    d.state.startswith("ENTER_"),
                    d.state.startswith("WATCH_"),
                    sum(c.passed for c in d.checks),
                ),
                reverse=True,
            )
            trades = self.store.trades()
            return {
                "version": 1,
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
                "symbols": {d.symbol: asdict(d) for d in ranked},
                "best_symbol": ranked[0].symbol if ranked else None,
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

    def track(self, values: dict[str, object]) -> TrackedTrade:
        if set(values) != {"signal_id", "kind", "entry", "quantity"} or values[
            "kind"
        ] not in ("paper", "manual"):
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
