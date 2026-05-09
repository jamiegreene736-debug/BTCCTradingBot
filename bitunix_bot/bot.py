"""Main trading loop — multi-symbol, cooldown-aware, bar-deduped.

Per tick:
  1. Pull all open positions and the account once.
  2. For each configured symbol:
     - Skip if already holding (max_positions_per_symbol reached).
     - Skip if cooldown window not elapsed.
     - Skip if global max_open_positions reached.
     - Skip if no fresh bar (last bar's timestamp unchanged since last eval).
     - Otherwise: evaluate signal, build risk-sized order, place with native SL/TP.
"""
from __future__ import annotations

import contextlib
import logging
import logging.handlers
import signal
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Deque

import numpy as np

from .client import BitunixClient, BitunixError
from .config import Config
from .indicators import adx as adx_fn
from .indicators import atr as atr_fn
from .journal import TradeJournal
from .orderbook import OrderBookFeed
from .risk import OrderPlan, adaptive_tp_r, build_order
from .state import get as get_state
from .order_executor import OrderExecutor
from .pnl import (
    as_float,
    closed_position_net_pnl,
    position_entry_price,
    position_exit_price,
    position_qty,
)
from .position_manager import PositionManager
from .strategy import Signal, compute_overlay_scores, evaluate
from .symbol_meta import DEFAULT_META as _DEFAULT_META
from .symbol_meta import SymbolMeta
from .symbol_meta import auto_symbol_risk_mult
from .symbol_meta import parse_symbol_meta
from .symbol_meta import row_max_leverage
from .symbol_meta import row_quote_volume_usdt
from .symbol_meta import row_symbol
from .symbol_meta import select_dynamic_symbols
from .tradetape import TradeFeed

log = logging.getLogger(__name__)


def configure_logging(cfg: Config) -> None:
    level = getattr(logging, cfg.logging.level.upper(), logging.INFO)
    Path(cfg.logging.file).parent.mkdir(parents=True, exist_ok=True)
    root = logging.getLogger()
    root.setLevel(level)
    fmt = logging.Formatter("%(asctime)s %(levelname)-7s %(name)s | %(message)s")
    fh = logging.handlers.RotatingFileHandler(cfg.logging.file, maxBytes=2_000_000, backupCount=3)
    fh.setFormatter(fmt)
    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    for h in list(root.handlers):
        root.removeHandler(h)
    root.addHandler(fh)
    root.addHandler(ch)


# SymbolMeta + _DEFAULT_META extracted to bitunix_bot/symbol_meta.py so
# position_manager / order_executor can reference them without importing
# from bot.py (which would create a circular import).


class BitunixBot:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.client = BitunixClient(
            cfg.creds.api_key,
            cfg.creds.secret_key,
            margin_coin=cfg.trading.margin_coin,
        )
        self.metas: dict[str, SymbolMeta] = {}
        self._configured_symbols = [s.upper() for s in cfg.trading.symbols]
        self._market_rows_by_symbol: dict[str, dict[str, Any]] = {}
        self._last_dynamic_symbol_refresh = 0.0
        # Per-symbol state for cooldown + bar-dedupe.
        self.last_action_at: dict[str, int] = {}    # unix sec
        self.last_bar_ts: dict[str, int] = {}       # unix sec/ms
        # HTF kline cache: symbol -> (fetched_unix_ts, closes list).
        # Refreshed at most once per HTF_CACHE_SECONDS to keep API calls down.
        self.htf_cache: dict[str, tuple[int, list[float]]] = {}
        # Funding rate cache: symbol -> (fetched_unix_ts, rate).
        self.funding_cache: dict[str, tuple[int, float]] = {}
        # Streak-protection state: symbol -> (paused_until_ts, reason).
        self.streak_pause_until: dict[str, int] = {}
        # Closed-positions watermark — used to detect newly-closed trades each
        # tick and update the consecutive-loss streak per symbol.
        self.last_seen_closed_mtime: int = 0
        self.consec_losses: dict[str, int] = {}   # symbol -> count
        # Recent-loss timestamps per symbol — used by the 2-loss mini-cooldown
        # circuit breaker (intercepts BEFORE the 3-loss streak pause).
        # If 2 losses happen within MINI_COOLDOWN_WINDOW seconds, that symbol
        # gets a 5-minute pause. Catches early "we're getting chopped" without
        # waiting for the 3rd loss.
        self.recent_losses: dict[str, list[float]] = {}
        self.mini_cooldown_until: dict[str, float] = {}
        # Adaptive self-defense: rolling R-multiple history of last N closed
        # trades. If the running tally drops too low (deep drawdown) the
        # fire_threshold ratchets UP (harder to fire) until equity recovers;
        # if it climbs high, threshold can ease slightly. Trade R is computed
        # in _update_streak_state when each closed position is observed.
        self.recent_trade_r: Deque[float] = deque(maxlen=20)
        # Daily drawdown circuit breaker state.
        self.session_start_equity: float | None = None
        self.session_start_day: int = 0       # UTC day-of-year, resets daily
        self.daily_dd_breached: bool = False
        # Liquidation-cascade circuit breaker. When BTC moves >X% in 3 min,
        # halt new entries for 5 min — every alt is in cascade and indicators
        # all read "BIG TREND" exactly when the move is exhausting.
        self._cascade_active: bool = False
        self._cascade_clear_at: float = 0.0
        self._cascade_check_at: float = 0.0
        # Per-position state (position_max_favor, partial_tp_done) lives in
        # PositionManager.* (Grok holistic review state migration). External
        # access via bot.position_manager.get_max_favor(pid) /
        # clear_position_state(pid). Init happens inside PositionManager.
        # Post-only entry tracking: symbol_u -> dict with order_id, place_ts, plan.
        # When a limit hits and a position appears, entry is removed. When the
        # timeout expires without a fill, the limit is cancelled and a market
        # fallback is placed.
        self.pending_limits: dict[str, dict[str, Any]] = {}
        # Live order-book feed (WebSocket). Started in start() / run_forever().
        self.ob_feed: OrderBookFeed | None = None
        # Live trade-tape feed (WebSocket). Companion to ob_feed — where the
        # OB shows what's RESTING, the tape shows what just EXECUTED and which
        # side aggressed. Source of real CVD / aggression / print-rate flow
        # signals. Started alongside ob_feed in run_forever().
        self.tape_feed: TradeFeed | None = None
        # Trade-quality journal — JSONL log of structured entry + exit events
        # for offline analysis. Prerequisite for data-driven tuning of factor
        # weights, threshold dynamics, and conviction calibration.
        self.journal = TradeJournal()
        # Position management subsystem (Grok holistic review — module split).
        # Wraps SL ratchet, BE/trailing, stale exit, tape exit, partial TP,
        # adaptive TP. State (position_max_favor, partial_tp_done) stays on
        # the bot for compat with other call sites; PositionManager just
        # operates on it via composition.
        self.position_manager = PositionManager(self)
        # Entry-execution subsystem (Grok holistic review — module split).
        # Wraps tape veto, maker-first post-only, market fallback, and
        # pending-limit timeout sweep. State (pending_limits) stays on the
        # bot for compat with the _tick max_open_positions counting logic.
        self.order_executor = OrderExecutor(self)
        self.stop_flag = False
        self.state = get_state()
        # Per-(symbol, horizon) rolling score history for the overlay's
        # 2-tick persistence filter. Shape: {sym: {horizon_key: [(score, side), ...]}}.
        # Keeps the last _PERSISTENCE_WINDOW entries; older ones get trimmed.
        self._overlay_score_history: dict[str, dict[str, list[tuple[float, str]]]] = {}
        # Per-symbol published one-hour decision memory. Short-term readings can
        # flip long/short on one aggressive tape burst; the API should publish
        # a stable call that every Chrome tab sees consistently.
        self._overlay_decision_memory: dict[str, dict[str, Any]] = {}

    # ------------------------------------------------------------------ setup

    def _fetch_market_rows(self) -> list[dict[str, Any]]:
        pairs = self.client.trading_pairs()
        by_symbol = {row_symbol(r): dict(r) for r in pairs if row_symbol(r)}
        try:
            ticker_rows = self.client.tickers()
            if not isinstance(ticker_rows, list):
                ticker_rows = []
        except Exception as e:
            log.debug("tickers() failed during symbol refresh: %s", e)
            ticker_rows = []
        for row in ticker_rows:
            sym = row_symbol(row)
            if not sym:
                continue
            merged = by_symbol.setdefault(sym, {})
            merged.update(row)
            merged.setdefault("symbol", sym)
        return list(by_symbol.values())

    def _apply_dynamic_symbol_universe(
        self,
        rows: list[dict[str, Any]],
        *,
        force: bool = False,
    ) -> None:
        trading = self.cfg.trading
        if not getattr(trading, "dynamic_symbols_enabled", False):
            return
        now = time.time()
        refresh_secs = max(60, int(getattr(trading, "dynamic_symbol_refresh_secs", 900)))
        if not force and (now - self._last_dynamic_symbol_refresh) < refresh_secs:
            return
        self._last_dynamic_symbol_refresh = now

        keep = self._configured_symbols if getattr(trading, "dynamic_symbol_keep_configured", True) else []
        selected = select_dynamic_symbols(
            rows,
            min_quote_volume_usdt=float(getattr(trading, "dynamic_symbol_min_quote_volume_usdt", 0.0)),
            min_open_interest_usdt=float(getattr(trading, "dynamic_symbol_min_open_interest_usdt", 0.0)),
            min_leverage=int(getattr(trading, "dynamic_symbol_min_leverage", 1)),
            max_symbols=int(getattr(trading, "dynamic_symbol_max_symbols", 30)),
            keep_symbols=keep,
        )
        if not selected:
            log.warning("Dynamic symbol scan found no eligible pairs; keeping %s", trading.symbols)
            return

        old_symbols = [s.upper() for s in trading.symbols]
        if selected != old_symbols:
            added = sorted(set(selected) - set(old_symbols))
            removed = sorted(set(old_symbols) - set(selected))
            trading.symbols = selected
            log.info("Dynamic symbol universe updated: %d symbols (+%s -%s)",
                     len(selected), added, removed)
            if self.ob_feed is not None:
                self.ob_feed.update_symbols(selected)
            if self.tape_feed is not None:
                self.tape_feed.update_symbols(selected)

        rows_by_symbol = {row_symbol(r): r for r in rows if row_symbol(r)}
        for sym in selected:
            row = rows_by_symbol.get(sym, {})
            if sym not in trading.symbol_risk_mult:
                trading.symbol_risk_mult[sym] = auto_symbol_risk_mult(
                    sym,
                    quote_volume_usdt=row_quote_volume_usdt(row),
                    max_leverage=row_max_leverage(row),
                )

    def _refresh_dynamic_symbols(self, *, force: bool = False) -> None:
        if not getattr(self.cfg.trading, "dynamic_symbols_enabled", False):
            return
        try:
            rows = self._fetch_market_rows()
        except Exception as e:
            log.warning("Dynamic symbol refresh failed: %s", e)
            return
        self._market_rows_by_symbol = {row_symbol(r): r for r in rows if row_symbol(r)}
        self._apply_dynamic_symbol_universe(rows, force=force)
        for sym in self.cfg.trading.symbols:
            row = self._market_rows_by_symbol.get(sym.upper())
            if row:
                self.metas[sym.upper()] = parse_symbol_meta(row)

    def _resolve_symbol_meta(self) -> None:
        try:
            pairs = self._fetch_market_rows()
        except Exception as e:
            log.warning("trading_pairs() failed: %s — using defaults for all", e)
            for s in self.cfg.trading.symbols:
                self.metas[s] = _DEFAULT_META
            return

        self._market_rows_by_symbol = {row_symbol(r): r for r in pairs if row_symbol(r)}
        self._apply_dynamic_symbol_universe(pairs, force=True)
        by_name = self._market_rows_by_symbol
        for sym in self.cfg.trading.symbols:
            row = by_name.get(sym.upper())
            if not row:
                log.warning("Symbol %s not in trading_pairs; using defaults", sym)
                self.metas[sym] = _DEFAULT_META
                continue
            meta = parse_symbol_meta(row)
            self.metas[sym] = meta
            log.info("Meta %s: step=%s priceDigits=%s minQty=%s maxLev=%s",
                     sym, meta.base_precision, meta.price_precision,
                     meta.min_qty, meta.max_leverage)

    def _configure_account(self) -> None:
        # Position mode is global; set once.
        try:
            self.client.set_position_mode("ONE_WAY")
            log.info("Set position_mode=ONE_WAY")
        except BitunixError as e:
            log.info("Skip position_mode: %s", e.msg or e.code)

        # Margin mode + leverage are per symbol. Cap leverage at the symbol's
        # max so Bitunix doesn't reject the call and silently leave us with
        # whatever was already set.
        for sym in self.cfg.trading.symbols:
            meta = self.metas.get(sym, _DEFAULT_META)
            eff_lev = self._target_leverage_for_symbol(sym, meta, self.cfg.trading.leverage)
            for fn, desc in [
                (lambda s=sym: self.client.set_margin_mode(s, self.cfg.trading.margin_mode),
                 f"{sym} margin_mode={self.cfg.trading.margin_mode}"),
                (lambda s=sym, lev=eff_lev: self.client.set_leverage(s, lev),
                 f"{sym} leverage={eff_lev}x (cap {meta.max_leverage}x)"),
            ]:
                try:
                    fn()
                    log.info("Set %s", desc)
                except BitunixError as e:
                    log.info("Skip %s: %s", desc, e.msg or e.code)
                except Exception as e:
                    log.warning("Error setting %s: %s", desc, e)

    # ------------------------------------------------------------------ loop

    def start(self) -> None:
        log.info("Starting Bitunix TraderBot in %s mode (symbols=%s)",
                 self.cfg.mode.upper(), self.cfg.trading.symbols)
        self._resolve_symbol_meta()
        if self.cfg.is_live:
            self._configure_account()

        signal.signal(signal.SIGINT, self._on_sig)
        with contextlib.suppress(AttributeError):
            signal.signal(signal.SIGTERM, self._on_sig)

        self.run_forever()

    def run_forever(self) -> None:
        """Loop without installing signal handlers (safe in a worker thread)."""
        # Start the order-book feed if not already running.
        if self.ob_feed is None:
            self.ob_feed = OrderBookFeed(
                symbols=self.cfg.trading.symbols,
                depth_levels=self.cfg.strategy.ob_depth_levels,
            )
            self.ob_feed.start()
        # Start the trade-tape feed alongside.
        if self.tape_feed is None:
            self.tape_feed = TradeFeed(symbols=self.cfg.trading.symbols)
            self.tape_feed.start()
        while not self.stop_flag:
            try:
                self._tick()
            except BitunixError as e:
                log.error("Bitunix API error: %s", e)
                self.state.record_error(f"{e.code}: {e.msg}")
            except Exception as e:
                log.exception("Tick failed")
                self.state.record_error(str(e))
            for _ in range(self.cfg.loop.tick_seconds):
                if self.stop_flag:
                    break
                time.sleep(1)
        if self.ob_feed:
            self.ob_feed.stop()
        if self.tape_feed:
            self.tape_feed.stop()
        log.info("Bot stopped")

    def _on_sig(self, *_: Any) -> None:
        self.stop_flag = True

    # ------------------------------------------------------------------ data fetchers

    _HTF_CACHE_SECONDS = 60          # HTF bars update slowly; 1 min is plenty
    _FUNDING_CACHE_SECONDS = 300     # funding rate updates every 8h on Bitunix
    _BTC_LEADER_CACHE_SECONDS = 30   # leader trend is checked frequently

    def _get_htf_closes(self, symbol: str) -> list[float] | None:
        sym_u = symbol.upper()
        now = int(time.time())
        cached = self.htf_cache.get(sym_u)
        if cached and (now - cached[0]) < self._HTF_CACHE_SECONDS:
            return cached[1]
        try:
            rows = self.client.klines(sym_u, self.cfg.strategy.htf_timeframe, limit=100)
            rows = sorted(rows, key=lambda r: int(r.get("time") or 0))
            closes = [float(r["close"]) for r in rows]
            self.htf_cache[sym_u] = (now, closes)
            return closes
        except Exception as e:
            log.debug("HTF kline fetch failed for %s: %s", sym_u, e)
            return None

    def _get_btc_trend(self) -> int | None:
        """Return +1 if BTC's recent 1m closes are above EMA(btc_leader_ema),
        -1 if below, None if can't tell. Cached briefly."""
        sym = self.cfg.strategy.btc_leader_symbol
        now = int(time.time())
        cached = self.htf_cache.get("__btc_trend__")
        if cached and (now - cached[0]) < self._BTC_LEADER_CACHE_SECONDS:
            return cached[1]
        try:
            rows = self.client.klines(sym, "1m", limit=80)
            rows = sorted(rows, key=lambda r: int(r.get("time") or 0))
            closes = np.array([float(r["close"]) for r in rows])
            if len(closes) < self.cfg.strategy.btc_leader_ema + 5:
                return None
            from .indicators import ema as ema_fn
            ema_arr = ema_fn(closes, self.cfg.strategy.btc_leader_ema)
            if np.isnan(ema_arr[-1]):
                return None
            trend = 1 if closes[-1] > ema_arr[-1] else (-1 if closes[-1] < ema_arr[-1] else 0)
            self.htf_cache["__btc_trend__"] = (now, trend)
            return trend
        except Exception as e:
            log.debug("BTC leader fetch failed: %s", e)
            return None

    @staticmethod
    def _session_weight() -> float:
        """Multiplier on combined score by UTC hour.

        Asia/EU overlap (04-07 UTC) and London/NY overlap (13-16 UTC) are
        the highest-edge windows for crypto per published exchange volume
        data. Weekends are lower-volume / wickier.
        """
        now = time.gmtime()
        hour = now.tm_hour
        wday = now.tm_wday  # Mon=0, Sun=6
        # Weekend dampener.
        weekend_mul = 0.85 if wday in (5, 6) else 1.0
        # High-edge overlap windows.
        if 4 <= hour < 7 or 13 <= hour < 16:
            return 1.20 * weekend_mul
        # Dead hours: 23-02 UTC (post-NY-close before Asia volume).
        if hour >= 23 or hour < 2:
            return 0.75 * weekend_mul
        return 1.0 * weekend_mul

    # Mini-cooldown tunables: 2 losses within this window → 5-minute pause.
    # Sits in front of the 3-loss / 2-hour streak pause as an early intercept.
    _MINI_COOLDOWN_WINDOW_SECS = 600     # 10 minutes
    _MINI_COOLDOWN_LOSS_LIMIT = 2
    _MINI_COOLDOWN_PAUSE_SECS = 300      # 5 minutes

    # Adaptive self-defense: threshold adjustments based on rolling R-tally.
    # Conservative bands — we tighten more aggressively than we loosen, and
    # require minimum sample size to avoid trippy noise on first few trades.
    _ADAPTIVE_MIN_SAMPLES = 5
    _ADAPTIVE_DD_R_THRESHOLD = -2.0      # last-20 sum < -2R → +0.04
    _ADAPTIVE_DD_BUMP = 0.04
    _ADAPTIVE_HOT_R_THRESHOLD = 3.0      # last-20 sum > +3R → -0.02
    _ADAPTIVE_HOT_RELAX = -0.02

    def _adaptive_threshold_adjustment(self) -> float:
        """Return the additive adjustment to fire_threshold based on the
        rolling sum of recent trade R-multiples.

        Logic:
          last-20 sum < -2R  → +0.04 (raise bar in drawdown)
          last-20 sum > +3R  → -0.02 (modest ease when on a streak)
          else               → 0.0

        Requires _ADAPTIVE_MIN_SAMPLES trades observed before kicking in,
        so we don't react to noise on the first few trades after startup.
        """
        if len(self.recent_trade_r) < self._ADAPTIVE_MIN_SAMPLES:
            return 0.0
        total_r = sum(self.recent_trade_r)
        if total_r < self._ADAPTIVE_DD_R_THRESHOLD:
            return self._ADAPTIVE_DD_BUMP
        if total_r > self._ADAPTIVE_HOT_R_THRESHOLD:
            return self._ADAPTIVE_HOT_RELAX
        return 0.0

    @staticmethod
    def _compute_trade_r(p: dict[str, Any], sl_pct_default: float) -> float:
        """Estimate trade R-multiple from a closed position record.

        R = net_pnl / risk_dollars where risk_dollars = qty * entry_price * sl_pct.
        Uses config sl_pct as the assumed entry SL distance (we don't track
        per-trade SL pct — would require keeping a per-position dict — but
        it's close enough for a rolling tally to detect drawdown). Returns
        0.0 on insufficient data.
        """
        open_px = position_entry_price(p)
        qty = position_qty(p)
        if open_px <= 0 or qty <= 0 or sl_pct_default <= 0:
            return 0.0
        sl_dist = open_px * sl_pct_default / 100.0
        risk_dollars = qty * sl_dist
        if risk_dollars <= 0:
            return 0.0
        return closed_position_net_pnl(p) / risk_dollars

    def _update_streak_state(self) -> None:
        """Pull recent closed positions and update consecutive-loss counts
        per symbol. Pause a symbol after streak_loss_limit consecutive losses."""
        try:
            hist = self.client.history_positions(limit=20)
            closed = hist.get("positionList", [])
        except Exception as e:
            log.debug("history_positions fetch for streak failed: %s", e)
            return
        # Filter to positions closed AFTER our watermark, in chronological order.
        new_closed = sorted(
            [p for p in closed
             if int(p.get("mtime") or 0) > self.last_seen_closed_mtime],
            key=lambda p: int(p.get("mtime") or 0),
        )
        for p in new_closed:
            sym = str(p.get("symbol") or "").upper()
            if not sym:
                continue
            realized = as_float(p.get("realizedPNL") or p.get("realizedPnl"))
            fee = as_float(p.get("fee"))
            funding = as_float(p.get("funding"))
            net = closed_position_net_pnl(p)

            # Surface non-zero funding for verification (Grok rescan):
            # Bitunix's docs aren't crystal clear on whether realizedPNL
            # already includes funding. If funding is consistently 0 in
            # responses, our `realized + fee + funding` math is fine; if
            # we see non-zero funding, log it once-per-position so we can
            # cross-check whether realizedPNL is double-counting.
            if funding != 0.0:
                log.info("Position close %s has non-zero funding: realized=%.6f "
                         "fee=%.6f funding=%.6f → net=%.6f",
                         p.get("positionId"), realized, fee, funding, net)

            # Compute trade-R first so we can use it for both flat detection
            # and the adaptive self-defense tally.
            trade_r = self._compute_trade_r(p, self.cfg.risk.stop_loss_pct)

            # Flat-trade detection. The BE ratchet at +1R favorable + price
            # reversal frequently produces "near-zero" exits where realized
            # loss ≈ fee/rebate offset → net within $0.001 of break-even.
            # These aren't real losses (the bot did its job — locked the
            # ratchet to BE — and the trade just didn't continue) and
            # shouldn't count toward the streak / mini-cooldown counters.
            # |trade_r| < FLAT_R_THRESHOLD = "essentially flat"
            FLAT_R_THRESHOLD = 0.10   # within ±10% of one R = flat
            is_flat = abs(trade_r) < FLAT_R_THRESHOLD

            # Adaptive self-defense: append trade R to the rolling tally —
            # but EXCLUDE flat trades (Grok holistic review). Flats dilute
            # the rolling tally toward zero and make the drawdown trigger
            # less responsive to actual losing streaks. The tally exists to
            # detect "we're bleeding" — flats are not bleeding.
            if not is_flat:
                self.recent_trade_r.append(trade_r)

            # Journal exit. exit_reason is best-effort; without per-trade
            # closeReason from Bitunix we tag generically. Future improvement:
            # match positionId against state-recorded events to reconstruct
            # whether SL/TP fired vs tape_exit / stale_exit / time_exit.
            pid_closed = str(p.get("positionId") or "")
            ctime_ms = int(p.get("ctime") or 0)
            mtime_ms = int(p.get("mtime") or 0)
            hold_sec = (mtime_ms - ctime_ms) / 1000.0 if (ctime_ms and mtime_ms) else 0.0
            entry_px = position_entry_price(p)
            exit_px = position_exit_price(p) or None
            exit_reason = "win" if net > 0 else ("loss" if net < 0 else "flat")
            self.journal.record_exit(
                symbol=sym,
                position_id=pid_closed,
                side=str(p.get("side") or ""),
                entry_price=entry_px,
                exit_price=exit_px,
                exit_reason=exit_reason,
                hold_time_sec=hold_sec,
                max_favor_r=self.position_manager.get_max_favor(pid_closed),
                net_pnl=net,
                realized_pnl=realized,
                fee=fee,
                funding=funding,
            )
            # Clear per-position state — the position is gone.
            self.position_manager.clear_position_state(pid_closed)
            if is_flat:
                # Flat (BE-ratchet-then-reversal pattern, or other near-zero
                # exits): don't touch streak counters in either direction.
                # Live data showed 6 of 14 "losses" were within $0.001 of
                # break-even — essentially flat trades that triggered streak
                # pauses inappropriately. Counting them defeats the
                # purpose of streak protection (which is to catch genuine
                # wrong-regime calls, not BE-ratchet-then-chop sequences).
                log.info("FLAT trade %s: r=%.3f (within ±%.2f), "
                         "skipping streak update", sym, trade_r,
                         FLAT_R_THRESHOLD)
            elif net > 0:
                # Win resets the streak.
                self.consec_losses[sym] = 0
            elif net < 0:
                self.consec_losses[sym] = self.consec_losses.get(sym, 0) + 1
                # 2-loss mini-cooldown: track timestamps of recent losses,
                # prune old, fire a 5-min pause when count crosses the limit.
                now_s = time.time()
                buf = self.recent_losses.setdefault(sym, [])
                buf.append(now_s)
                cutoff = now_s - self._MINI_COOLDOWN_WINDOW_SECS
                buf[:] = [t for t in buf if t > cutoff]
                if len(buf) >= self._MINI_COOLDOWN_LOSS_LIMIT:
                    until = now_s + self._MINI_COOLDOWN_PAUSE_SECS
                    # Only set if not already past the existing mini-cooldown.
                    self.mini_cooldown_until[sym] = max(
                        self.mini_cooldown_until.get(sym, 0.0), until
                    )
                    log.info("MINI-COOLDOWN %s: %d losses in %dm — pause %ds",
                             sym, len(buf),
                             self._MINI_COOLDOWN_WINDOW_SECS // 60,
                             self._MINI_COOLDOWN_PAUSE_SECS)
                    self.state.record_skip(
                        f"{sym}: 2-loss mini-cooldown — pause "
                        f"{self._MINI_COOLDOWN_PAUSE_SECS // 60}m"
                    )
                if self.consec_losses[sym] >= self.cfg.trading.streak_loss_limit:
                    until = int(time.time()) + self.cfg.trading.streak_loss_pause_seconds
                    self.streak_pause_until[sym] = until
                    log.warning("STREAK PAUSE %s after %d consecutive losses; "
                                "no entries until %ss",
                                sym, self.consec_losses[sym],
                                self.cfg.trading.streak_loss_pause_seconds)
                    self.state.record_skip(
                        f"{sym}: STREAK PAUSE — {self.consec_losses[sym]} losses, "
                        f"halt for {self.cfg.trading.streak_loss_pause_seconds // 60}min"
                    )
                    self.consec_losses[sym] = 0  # reset count after triggering
            self.last_seen_closed_mtime = max(self.last_seen_closed_mtime,
                                               int(p.get("mtime") or 0))

    # Cascade detector tunables. Threshold and halt duration are conservative:
    # 2% in 3 min on BTC is a severe move (1m ATR is typically 0.05–0.15%, so
    # a 3-min cumulative >2% means we're in the upper tail of the distribution).
    _CASCADE_PCT_3MIN = 2.0       # absolute % move on BTC over 3 min
    _CASCADE_HALT_SECS = 300      # halt new entries for 5 min after detection
    _CASCADE_CHECK_INTERVAL = 5   # re-evaluate at most every 5 sec

    def _check_liquidation_cascade(self) -> bool:
        """Detect rapid BTC moves indicative of a liquidation cascade.

        When BTC moves >_CASCADE_PCT_3MIN over the last 3 1m bars in either
        direction, halt all NEW entries for _CASCADE_HALT_SECS. Existing
        positions stay managed (SL ratchet, partial TP, time exit) — we
        only block new entries because indicators all read "BIG TREND"
        exactly when the move is exhausting and the next 1m bar is a
        violent reversal. Pro desks have these everywhere.

        Returns True if the cascade is active (block new entries).
        Cached: only re-evaluates every _CASCADE_CHECK_INTERVAL seconds
        and only when there's no active cascade timer running.
        """
        now = time.time()
        # Active cascade — let it expire on the timer; no need to recheck.
        if self._cascade_active:
            if now >= self._cascade_clear_at:
                self._cascade_active = False
                log.info("Cascade cleared, resuming entries")
            return self._cascade_active
        # Throttle the check itself.
        if now - self._cascade_check_at < self._CASCADE_CHECK_INTERVAL:
            return False
        self._cascade_check_at = now

        try:
            rows = self.client.klines("BTCUSDT", "1m", limit=5)
        except Exception as e:
            log.debug("Cascade check klines fetch failed: %s", e)
            return False
        if len(rows) < 4:
            return False
        rows = sorted(rows, key=lambda r: int(r.get("time") or 0))
        try:
            close_now = float(rows[-1]["close"])
            close_3m_ago = float(rows[-4]["close"])
        except (KeyError, ValueError, TypeError):
            return False
        if close_3m_ago <= 0:
            return False

        pct_3m = (close_now - close_3m_ago) / close_3m_ago * 100.0
        if abs(pct_3m) >= self._CASCADE_PCT_3MIN:
            self._cascade_active = True
            self._cascade_clear_at = now + self._CASCADE_HALT_SECS
            msg = (f"CASCADE DETECTED: BTC {pct_3m:+.2f}% in 3min "
                   f"(threshold {self._CASCADE_PCT_3MIN}%) — halt new entries "
                   f"for {self._CASCADE_HALT_SECS // 60}m")
            log.warning(msg)
            self.state.record_error(msg)
        return self._cascade_active

    def _daily_dd_risk_multiplier(self) -> float:
        """Return a risk multiplier in [0.0, 1.0] based on current DD.

        Gradual throttle (instead of binary halt) protects capital better:
          0%   to -2%  →  1.00× (normal)
          -2%  to -4%  →  0.75× (early caution — small overall loss)
          -4%  to -6%  →  0.50× (significant — half-size remaining trades)
          -6%  to -8%  →  0.25× (deep caution — quarter-size)
          ≥ -8%        →  0.00× (HALT — no new entries)

        Halt threshold is `max_daily_dd_pct`; intermediate steps scale
        from 25% → 75% → 100% of that. Resets at UTC midnight.
        """
        try:
            acct = self.client.account()
            avail = float(acct.get("available") or 0)
            margin = float(acct.get("margin") or 0)
            upnl = (float(acct.get("crossUnrealizedPNL") or 0)
                    + float(acct.get("isolationUnrealizedPNL") or 0))
            equity = avail + margin + upnl
        except Exception as e:
            log.debug("DD-check account fetch failed: %s", e)
            return 0.0 if self.daily_dd_breached else 1.0

        # Reset session at UTC midnight.
        today = time.gmtime().tm_yday
        if today != self.session_start_day:
            self.session_start_day = today
            self.session_start_equity = equity
            self.daily_dd_breached = False
            log.info("New session: start equity=$%.2f", equity)
            return 1.0

        if self.session_start_equity is None or self.session_start_equity <= 0:
            self.session_start_equity = equity
            return 1.0

        dd_pct = (self.session_start_equity - equity) / self.session_start_equity * 100.0
        threshold = self.cfg.trading.max_daily_dd_pct

        # Gradual ramp.
        if dd_pct >= threshold:
            if not self.daily_dd_breached:
                self.daily_dd_breached = True
                msg = (f"DAILY DD HALT: -{dd_pct:.2f}% from session start "
                       f"${self.session_start_equity:.2f} → ${equity:.2f} "
                       f"(threshold {threshold}%)")
                log.warning(msg)
                self.state.record_error(msg)
            return 0.0
        if dd_pct >= threshold * 0.75:        # e.g. -6% if threshold=8
            return 0.25
        if dd_pct >= threshold * 0.50:        # e.g. -4%
            return 0.50
        if dd_pct >= threshold * 0.25:        # e.g. -2%
            return 0.75
        return 1.0

    # Backwards-compat shim — older _tick path uses bool dd_halted check.
    def _check_daily_drawdown(self) -> bool:
        """Return True iff the DD circuit breaker is fully tripped (mult=0)."""
        return self._daily_dd_risk_multiplier() <= 0.0

    def _get_funding_rate(self, symbol: str) -> float | None:
        sym_u = symbol.upper()
        now = int(time.time())
        cached = self.funding_cache.get(sym_u)
        if cached and (now - cached[0]) < self._FUNDING_CACHE_SECONDS:
            return cached[1]
        try:
            data = self.client.funding_rate(sym_u)
            rate = float(data.get("fundingRate") or 0)
            self.funding_cache[sym_u] = (now, rate)
            return rate
        except Exception as e:
            log.debug("Funding rate fetch failed for %s: %s", sym_u, e)
            return None

    # ------------------------------------------------------------------ tick

    # Pump-fade overlay configuration. Each entry is:
    #   (key, klines_tf, cache_ttl_seconds, display_label)
    # The current strategy is single-purpose: find pump-fade shorts. The
    # decision only uses:
    #   h_15m -> 1m entry/rejection timing
    #   h_30m -> 5m pump structure
    #   h_1h  -> 15m trend/chop context
    # Older long-horizon rows are intentionally omitted here so the expanded
    # symbol scanner stays fresh instead of spending time computing unused
    # 1h/2h/4h overlays.
    _OVERLAY_HORIZONS: tuple[tuple[str, str, int, str], ...] = (
        ("h_15m", "1m",  5, "1m entry"),
        ("h_30m", "5m",  15, "5m pump"),
        ("h_1h",  "15m", 60, "trend context"),
    )

    # Persistence window — how many recent ticks of dominant scores per
    # (symbol, horizon) we keep. With the live 5s tick this is ~15s of
    # history: enough to require 2-tick confirmation without over-smoothing.
    _PERSISTENCE_WINDOW = 3
    # Score threshold that mirrors the frontend's ALARM_AT — kept here so
    # the backend "stable" flag uses the same gate.
    _ALARM_AT = 0.55
    # Same gap as the frontend's BIAS_GAP — directional bias requires
    # one side to lead by at least this much.
    _BIAS_GAP = 0.05

    # Dedicated one-hour decision model for the Chrome overlay. The shortest
    # horizons are too noisy to own the top card, so 1h is the anchor, 30m is
    # the near-term confirmation, and the 1m entry row is only timing / anti-chase
    # context.
    _NEXT_HOUR_WEIGHTS: tuple[tuple[str, float], ...] = (
        ("h_15m", 0.10),
        ("h_30m", 0.25),
        ("h_1h",  0.45),
        ("h_4h",  0.20),
    )
    _NEXT_HOUR_CORE_KEYS = ("h_30m", "h_1h")
    _NEXT_HOUR_MIN_ACTION_BIAS = 0.08
    _NEXT_HOUR_MIN_SIDE_GAP = 0.04
    _NEXT_HOUR_CORE_CONFLICT_GAP = 0.04
    _NEXT_HOUR_CONTEXT_CONFLICT_GAP = 0.08
    _NEXT_HOUR_CHOP_ADX = 25.0
    _NEXT_HOUR_NEWS_ATR_PCT = 0.75
    _NEXT_HOUR_CASCADE_10S_PCT = 1.00
    # Anti-chase guard for market entries: a high momentum score after a
    # vertical move can be the worst entry, because the next leg often
    # mean-reverts before any continuation becomes tradeable.
    _NEXT_HOUR_CHASE_3_ATR = 1.15
    _NEXT_HOUR_CHASE_5_ATR = 1.60
    _NEXT_HOUR_CHASE_RANGE_EDGE = 0.20
    _NEXT_HOUR_CHASE_RETEST_ATR = 0.35
    _FOCUS_ENTER_CONFIRM_TICKS = 3
    _FOCUS_FLIP_CONFIRM_TICKS = 12
    _FOCUS_MIN_HOLD_SECONDS = 300

    @classmethod
    def _next_hour_confidence_score(
        cls,
        *,
        action: str,
        abs_bias: float,
        core_agree: int,
        core_total: int,
        blockers: list[str],
        context_conflicts: list[dict[str, Any]],
    ) -> int:
        """Map the one-hour decision geometry to a human 0-100 confidence.

        This is not a calibrated win probability. It is a readability score:
        directional edge strength + core-horizon agreement, with explicit
        penalties for blockers and higher-timeframe context headwinds.
        """
        if action not in ("long", "short"):
            # Still show a useful number for WAIT: how close the setup is to
            # becoming actionable, capped below the LOW tier.
            edge = min(1.0, abs_bias / max(cls._NEXT_HOUR_MIN_ACTION_BIAS, 0.0001))
            agree = core_agree / max(1, core_total)
            score = 15 + (25 * edge) + (15 * agree) - (10 * len(blockers))
            return max(0, min(49, int(round(score))))

        # At the action threshold, start near 50. Stronger bias and broader
        # core alignment push toward 100.
        edge = min(1.0, abs_bias / 0.20)
        agree = core_agree / max(1, core_total)
        score = 35 + (40 * edge) + (25 * agree)
        if context_conflicts:
            score -= 12
        score -= 8 * len(blockers)
        return max(1, min(99, int(round(score))))

    @staticmethod
    def _gap_side(gap: float, neutral_gap: float = 0.0) -> str:
        if gap > neutral_gap:
            return "long"
        if gap < -neutral_gap:
            return "short"
        return "mixed"

    @staticmethod
    def _float_field(row: dict[str, Any], *keys: str, default: float = 0.0) -> float:
        for key in keys:
            try:
                value = row.get(key)
            except AttributeError:
                return default
            if value is None:
                continue
            try:
                out = float(value)
            except (TypeError, ValueError):
                continue
            if np.isfinite(out):
                return out
        return default

    @classmethod
    def _anti_chase_blocker(
        cls,
        lean: str,
        row: dict[str, Any],
        *,
        label: str = "move",
    ) -> str | None:
        if lean not in ("long", "short") or not row:
            return None

        move_3_atr = cls._float_field(row, "move_3_atr", "move3Atr")
        move_5_atr = cls._float_field(row, "move_5_atr", "move5Atr")
        range_pos = cls._float_field(
            row, "position_in_recent_range_15", "positionInRecentRange15", default=0.5
        )
        low_dist = cls._float_field(
            row, "distance_from_recent_low_atr", "distanceFromRecentLowAtr", default=999.0
        )
        high_dist = cls._float_field(
            row, "distance_from_recent_high_atr", "distanceFromRecentHighAtr", default=999.0
        )
        down_closes = int(cls._float_field(row, "down_closes_5", "downCloses5"))
        up_closes = int(cls._float_field(row, "up_closes_5", "upCloses5"))
        down_candles = int(cls._float_field(row, "down_candles_5", "downCandles5"))
        up_candles = int(cls._float_field(row, "up_candles_5", "upCandles5"))

        extended_down = (
            move_5_atr <= -cls._NEXT_HOUR_CHASE_5_ATR
            or move_3_atr <= -cls._NEXT_HOUR_CHASE_3_ATR
        )
        extended_up = (
            move_5_atr >= cls._NEXT_HOUR_CHASE_5_ATR
            or move_3_atr >= cls._NEXT_HOUR_CHASE_3_ATR
        )
        pinned_low = (
            range_pos <= cls._NEXT_HOUR_CHASE_RANGE_EDGE
            or low_dist <= cls._NEXT_HOUR_CHASE_RETEST_ATR
        )
        pinned_high = (
            range_pos >= 1.0 - cls._NEXT_HOUR_CHASE_RANGE_EDGE
            or high_dist <= cls._NEXT_HOUR_CHASE_RETEST_ATR
        )
        one_way_down = (
            down_closes >= 3
            or down_candles >= 3
            or move_3_atr <= -(cls._NEXT_HOUR_CHASE_3_ATR + 0.25)
        )
        one_way_up = (
            up_closes >= 3
            or up_candles >= 3
            or move_3_atr >= cls._NEXT_HOUR_CHASE_3_ATR + 0.25
        )

        if lean == "short" and extended_down and pinned_low and one_way_down:
            return f"anti-chase: {label} is already extended down near local lows; wait for bounce/retest before shorting"
        if lean == "long" and extended_up and pinned_high and one_way_up:
            return f"anti-chase: {label} is already extended up near local highs; wait for pullback/retest before longing"
        return None

    @classmethod
    def _parabolic_pump_short_setup(
        cls,
        h15: dict[str, Any],
        h30: dict[str, Any],
        h1: dict[str, Any],
    ) -> dict[str, Any] | None:
        """Detect the user's desired quick fade: a hard pump into exhaustion.

        This is deliberately a reversal override, not a trend-following long.
        It requires a large 30m impulse, price in the upper local range, and
        at least two short-term exhaustion tells. The bot used to fire on a
        single clue, which was too eager for clean momentum pumps.
        """
        status = cls._pump_fade_status(h15, h30, h1)
        if not status["usable"]:
            return None

        move_5_atr = float(status["move_5_atr"])
        range_pos = float(status["range_pos"])
        h15_cvd = float(status["h15_cvd"])
        h15_short_score = float(status["h15_short_score"])
        h30_short_score = float(status["h30_short_score"])
        h15_move_3_atr = float(status["h15_move_3_atr"])
        h15_down_closes = int(status["h15_down_closes"])
        big_green_candle = bool(status.get("big_green_candle"))
        rejection_candle = bool(status.get("one_min_rejection_candle"))
        if not (
            status["pump"]
            and status["not_far_from_high"]
            and status["entry_window"]
            and status["exhaustion"]
            and status["trend_not_too_clean"]
        ):
            return None

        confidence = 80
        confidence += min(8, max(0, int(round((move_5_atr - 0.75) * 4))))
        if status.get("micro_pump"):
            confidence += 4
        if status.get("micro_rejection"):
            confidence += 4
        if big_green_candle:
            confidence += 5
        if rejection_candle:
            confidence += 4
        if range_pos >= 0.92:
            confidence += 4
        elif range_pos >= 0.86:
            confidence += 2
        if status.get("not_far_from_high") and range_pos >= 0.82:
            confidence += 2
        if h15_cvd <= -5.0:
            confidence += 5
        if h15_short_score >= 0.28:
            confidence += 5
        elif h15_short_score >= 0.20:
            confidence += 3
        if h30_short_score >= 0.25:
            confidence += 2
        if h15_move_3_atr <= -0.25:
            confidence += 3
        if h15_down_closes >= 3:
            confidence += 2
        confidence = max(80, min(98, confidence))

        reasons = [
            f"micro pump: 30m move +{move_5_atr:.2f} ATR over 5 bars",
            f"price high in local range ({range_pos * 100:.0f}%)",
        ]
        if big_green_candle:
            reasons.append("5m pump candle closed strong on elevated volume")
        if rejection_candle:
            reasons.append("1m entry candle shows wick/body rejection")
        if h15_cvd <= -5.0:
            reasons.append("60s CVD flipped strongly negative into the pump")
        if h15_short_score >= 0.20:
            reasons.append("1m entry row has real bearish exhaustion votes")
        if h30_short_score >= 0.25:
            reasons.append("30m confirms short-side pressure")
        for reason in h15.get("short_reasons") or []:
            if reason not in reasons:
                reasons.append(str(reason))

        return {
            "action": "short",
            "confidence_score": confidence,
            "warning": "parabolic pump fade: quick short only; no long/trend-following trades",
            "reasons": reasons[:8],
            "move_5_atr": round(move_5_atr, 4),
            "range_pos": round(range_pos, 4),
            "fade_eta": status.get("fade_eta"),
            "fadeEta": status.get("fade_eta"),
            "checks": status["checks"],
        }

    @classmethod
    def _pump_fade_status(
        cls,
        h15: dict[str, Any],
        h30: dict[str, Any],
        h1: dict[str, Any],
    ) -> dict[str, Any]:
        """Explain how close the current chart is to the pump-fade short setup."""
        if not h15 or not h30:
            return {
                "usable": False,
                "score": 0,
                "move_5_atr": 0.0,
                "range_pos": 0.5,
                "h15_cvd": 0.0,
                "h15_short_score": 0.0,
                "pump": False,
                "not_far_from_high": False,
                "exhaustion": False,
                "trend_not_too_clean": False,
                "entry_window": False,
                "pump_watch": False,
                "pre_pump_building": False,
                "pre_pump_score": 0,
                "checks": [
                    {"key": "data", "label": "Data", "passed": False,
                     "detail": "waiting for 1m entry and 5m pump data"},
                ],
            }

        move_3_atr = cls._float_field(h30, "move_3_atr", "move3Atr")
        move_5_atr = cls._float_field(h30, "move_5_atr", "move5Atr")
        move_10_atr = cls._float_field(h30, "move_10_atr", "move10Atr")
        move_12_atr = cls._float_field(h30, "move_12_atr", "move12Atr")
        range_pos = cls._float_field(
            h30, "position_in_recent_range_15", "positionInRecentRange15", default=0.5
        )
        high_dist = cls._float_field(
            h30, "distance_from_recent_high_atr", "distanceFromRecentHighAtr", default=999.0
        )
        up_closes = int(cls._float_field(h30, "up_closes_5", "upCloses5"))
        up_candles = int(cls._float_field(h30, "up_candles_5", "upCandles5"))
        up_closes_10 = int(cls._float_field(h30, "up_closes_10", "upCloses10"))
        up_closes_12 = int(cls._float_field(h30, "up_closes_12", "upCloses12"))
        h1_adx = cls._float_field(h1, "adx", default=99.0)
        h15_long_score = cls._float_field(h15, "long_score", "longScore")
        h15_short_score = cls._float_field(h15, "short_score", "shortScore")
        h30_long_score = cls._float_field(h30, "long_score", "longScore")
        h30_short_score = cls._float_field(h30, "short_score", "shortScore")
        h15_cvd = cls._float_field(h15, "real_cvd", "realCvd")
        h15_aggression = cls._float_field(h15, "aggression_10s", "aggression10s")
        h15_move_3_atr = cls._float_field(h15, "move_3_atr", "move3Atr")
        h15_move_10_atr = cls._float_field(h15, "move_10_atr", "move10Atr")
        h15_move_15_atr = cls._float_field(h15, "move_15_atr", "move15Atr")
        h15_up_closes = int(cls._float_field(h15, "up_closes_5", "upCloses5"))
        h15_down_closes = int(cls._float_field(h15, "down_closes_5", "downCloses5"))
        h15_up_closes_10 = int(cls._float_field(h15, "up_closes_10", "upCloses10"))
        h15_up_closes_15 = int(cls._float_field(h15, "up_closes_15", "upCloses15"))
        h15_body_atr = cls._float_field(h15, "last_bar_body_atr", "lastBarBodyAtr")
        h15_upper_wick_atr = cls._float_field(h15, "last_bar_upper_wick_atr", "lastBarUpperWickAtr")
        h15_close_pos = cls._float_field(h15, "last_bar_close_position", "lastBarClosePosition", default=0.5)
        h15_volume_ratio = cls._float_field(h15, "volume_spike_ratio", "volumeSpikeRatio")
        h30_body_atr = cls._float_field(h30, "last_bar_body_atr", "lastBarBodyAtr")
        h30_close_pos = cls._float_field(h30, "last_bar_close_position", "lastBarClosePosition", default=0.5)
        h30_volume_ratio = cls._float_field(h30, "volume_spike_ratio", "volumeSpikeRatio")
        h15_short_reasons = [str(r).lower() for r in h15.get("short_reasons") or []]
        h15_long_reasons = [str(r).lower() for r in h15.get("long_reasons") or []]
        h30_long_reasons = [str(r).lower() for r in h30.get("long_reasons") or []]

        bearish_pattern = any(
            tag in reason
            for reason in h15_short_reasons
            for tag in (
                "tweezer_top",
                "bear",
                "rsi_bearish",
                "cvd_real-",
                "absorb(buyflow",
                "wick_pattern_short",
                "supertrend_down",
            )
        )
        strong_negative_tape = h15_cvd <= -5.0
        meaningful_short_votes = h15_short_score >= 0.20 and h30_short_score >= 0.18
        bearish_rejection = bearish_pattern and (
            h15_short_score >= 0.15
            or h30_short_score >= 0.18
            or strong_negative_tape
        )
        exhaustion_votes = sum((
            bool(strong_negative_tape),
            bool(meaningful_short_votes),
            bool(bearish_rejection),
        ))
        big_green_candle = (
            h30_body_atr >= 0.75
            and h30_close_pos >= 0.70
            and (h30_volume_ratio <= 0 or h30_volume_ratio >= 1.25)
        )
        one_min_rejection_candle = (
            h15_body_atr < 0
            or h15_upper_wick_atr >= 0.18
            or h15_close_pos <= 0.45
        )

        vertical_pump = (
            (move_5_atr >= 3.5 or move_3_atr >= 2.25 or big_green_candle)
            and (up_closes >= 3 or up_candles >= 3)
        )
        # High-leverage pump fades are not meant to wait for a giant move.
        # The target is a small, fast pop near the local high that starts
        # failing on the 1m tape. These thresholds intentionally key off ATR
        # rather than raw percent so BTC/ETH/DOGE/XRP normalize reasonably.
        micro_pump = (
            range_pos >= 0.72
            and (
                (move_3_atr >= 0.55 and up_closes >= 2)
                or (move_5_atr >= 0.75 and up_closes >= 2)
                or (h15_move_10_atr >= 0.85 and h15_up_closes_10 >= 5)
                or (h15_move_15_atr >= 1.10 and h15_up_closes_15 >= 7)
            )
        )
        micro_pump_watch = (
            range_pos >= 0.55
            and (
                (move_3_atr >= 0.35 and up_closes >= 2)
                or (move_5_atr >= 0.50 and up_closes >= 2)
                or (h15_move_10_atr >= 0.55 and h15_up_closes_10 >= 4)
                or (h15_move_15_atr >= 0.80 and h15_up_closes_15 >= 6)
            )
        )
        positive_tape = h15_cvd >= 2.0 or h15_aggression >= 0.35
        lift_tags = ("vol_spike", "squeeze_up", "supertrend_up", "cvd_real+", "agg+")
        momentum_tags = any(
            tag in reason
            for reason in (h15_long_reasons + h30_long_reasons)
            for tag in lift_tags
        )
        early_lift = (
            move_3_atr >= 0.25
            or move_5_atr >= 0.35
            or h15_move_3_atr >= 0.18
            or h15_move_10_atr >= 0.45
        )
        not_blown_off_yet = range_pos < 0.78 and high_dist > 0.45
        directional_lift = (
            h15_long_score >= h15_short_score + 0.10
            or h30_long_score >= h30_short_score + 0.10
        )
        early_pump_ignition = (
            bool(early_lift)
            and bool(positive_tape or momentum_tags)
            and bool(directional_lift)
            and (
                range_pos >= 0.35
                or move_3_atr >= 0.25
                or h15_move_10_atr >= 0.45
            )
        )
        pre_pump_votes = sum((
            range_pos >= 0.45,
            bool(early_lift),
            bool(positive_tape or momentum_tags),
            bool(directional_lift),
            bool(not_blown_off_yet),
        ))
        session_pump = (
            range_pos >= 0.78
            and (
                (move_12_atr >= 2.40 and up_closes_12 >= 6)
                or (move_10_atr >= 2.00 and up_closes_10 >= 5)
                or (h15_move_15_atr >= 2.20 and h15_up_closes_15 >= 8)
                or (h15_move_10_atr >= 1.65 and h15_up_closes_10 >= 6)
            )
        )
        pump_watch = (
            range_pos >= 0.55
            and (
                micro_pump_watch
                or session_pump
                or (
                    (move_5_atr >= 1.45 or move_3_atr >= 1.05)
                    and (up_closes >= 3 or up_candles >= 3)
                )
            )
        )
        post_pump_rejection = (
            move_5_atr >= 2.0
            and range_pos >= 0.88
            and h30_short_score >= 0.25
            and bearish_rejection
        )
        pump = (
            range_pos >= 0.78
            and (vertical_pump or post_pump_rejection or session_pump or micro_pump)
        )
        not_far_from_high = high_dist <= 1.25 or range_pos >= 0.82
        exhaustion = exhaustion_votes >= 2
        trend_not_too_clean = h1_adx < 30.0 or (
            strong_negative_tape and h30_short_score >= 0.18
        ) or (strong_negative_tape and h15_short_score >= 0.24)
        near_blowoff_high = high_dist <= 0.95 or range_pos >= 0.86
        lower_high_rejection = (
            high_dist <= 1.45
            and range_pos >= 0.78
            and h15_move_3_atr <= -0.18
            and h15_down_closes >= 2
            and h15_short_score >= 0.18
            and (strong_negative_tape or h15_short_score >= 0.28)
        )
        micro_rejection = (
            micro_pump
            and range_pos >= 0.76
            and h15_move_3_atr <= -0.12
            and h15_down_closes >= 2
            and (strong_negative_tape or h15_short_score >= 0.24)
        )
        cooled_off_recovery = (
            high_dist > 0.95
            and h15_move_3_atr >= 0.15
            and (
                h15_up_closes >= 3
                or h15_long_score >= h15_short_score + 0.10
                or h15_cvd > 0.0
            )
        )
        buyer_still_in_control = (
            h15_long_score >= h15_short_score + 0.20
            and h15_long_score >= 0.35
            and (
                h15_aggression >= 0.25
                or (h15_move_3_atr >= 0 and h15_up_closes >= 3)
            )
        )
        pre_pump_building = (
            (pre_pump_votes >= 4 or (pre_pump_votes >= 3 and early_pump_ignition))
            and not pump_watch
        )
        still_squeezing_up = (
            h15_move_3_atr >= 0.45
            and h15_up_closes >= 3
            and h15_long_score > h15_short_score
        )
        entry_window = (near_blowoff_high or lower_high_rejection or micro_rejection) and not (
            cooled_off_recovery or still_squeezing_up or buyer_still_in_control
        )
        if any(k in h15 for k in (
            "last_bar_body_atr",
            "lastBarBodyAtr",
            "last_bar_close_position",
            "lastBarClosePosition",
        )):
            entry_window = entry_window and (
                bool(one_min_rejection_candle)
                or bool(strong_negative_tape and h15_short_score >= 0.26)
            )

        checks = [
            {
                "key": "watch",
                "label": "Pump watch",
                "passed": bool(pump_watch),
                "detail": (
                    f"micro {'yes' if micro_pump_watch else 'no'} / "
                    f"5-bar +{move_5_atr:.2f} ATR, 12-bar +{move_12_atr:.2f} ATR"
                ),
            },
            {
                "key": "pump",
                "label": "Fade-ready pump",
                "passed": bool(pump),
                "detail": (
                    f"micro {'yes' if micro_pump else 'no'} / "
                    f"green {'yes' if big_green_candle else 'no'} / "
                    f"range {range_pos * 100:.0f}% / up {up_closes}/5, {up_closes_12}/12"
                ),
            },
            {
                "key": "high",
                "label": "Near blow-off high",
                "passed": bool(not_far_from_high),
                "detail": f"{high_dist:.2f} ATR below high" if high_dist < 900 else "high distance unavailable",
            },
            {
                "key": "entry_window",
                "label": "Fresh fade window",
                "passed": bool(entry_window),
                "detail": (
                    f"1m move {h15_move_3_atr:+.2f} ATR, "
                    f"up {h15_up_closes}/5, down {h15_down_closes}/5, "
                    f"wick {h15_upper_wick_atr:.2f} ATR, "
                    f"buyer control {'yes' if buyer_still_in_control else 'no'}"
                ),
            },
            {
                "key": "exhaustion",
                "label": "Confirmed rejection",
                "passed": bool(exhaustion),
                "detail": (
                    f"votes {exhaustion_votes}/3; 1m entry short {h15_short_score * 100:.0f}, "
                    f"5m pump short {h30_short_score * 100:.0f}, CVD {h15_cvd:.0f}"
                ),
            },
            {
                "key": "trend_risk",
                "label": "Clean-trend risk ok",
                "passed": bool(trend_not_too_clean),
                "detail": f"1h ADX {h1_adx:.1f}; needs tape rollover if >=30",
            },
        ]
        if entry_window and exhaustion:
            fade_eta = {
                "status": "live",
                "label": "NOW",
                "seconds_min": 0,
                "secondsMin": 0,
                "seconds_max": 10,
                "secondsMax": 10,
                "reason": "fade trigger is live: rejection window and exhaustion are both confirmed",
            }
        elif entry_window:
            fade_eta = {
                "status": "starting",
                "label": "5-20s",
                "seconds_min": 5,
                "secondsMin": 5,
                "seconds_max": 20,
                "secondsMax": 20,
                "reason": "1m rejection is starting; waiting for exhaustion/tape confirmation",
            }
        elif pump_watch:
            if still_squeezing_up or buyer_still_in_control:
                fade_eta = {
                    "status": "squeezing",
                    "label": "30-90s",
                    "seconds_min": 30,
                    "secondsMin": 30,
                    "seconds_max": 90,
                    "secondsMax": 90,
                    "reason": "pump is active but buyers still control the tape; wait for wick/CVD flip",
                }
            elif h15_move_3_atr < -0.05 or h15_down_closes >= 1 or h15_cvd < 0:
                fade_eta = {
                    "status": "starting",
                    "label": "10-30s",
                    "seconds_min": 10,
                    "secondsMin": 10,
                    "seconds_max": 30,
                    "secondsMax": 30,
                    "reason": "pump is near the fade window and the first selloff ticks are appearing",
                }
            elif high_dist <= 0.45 or range_pos >= 0.90:
                fade_eta = {
                    "status": "near",
                    "label": "15-45s",
                    "seconds_min": 15,
                    "secondsMin": 15,
                    "seconds_max": 45,
                    "secondsMax": 45,
                    "reason": "pump is close to the blow-off zone; watch for the first failed 1m candle",
                }
            else:
                fade_eta = {
                    "status": "watch",
                    "label": "30-90s",
                    "seconds_min": 30,
                    "secondsMin": 30,
                    "seconds_max": 90,
                    "secondsMax": 90,
                    "reason": "pump is active but has not reached the fade trigger zone yet",
                }
        elif pre_pump_building:
            fade_eta = {
                "status": "building",
                "label": "45-150s",
                "seconds_min": 45,
                "secondsMin": 45,
                "seconds_max": 150,
                "secondsMax": 150,
                "reason": "pump ignition is building; no short until price tags the upper range and rejects",
            }
        elif early_pump_ignition:
            fade_eta = {
                "status": "early",
                "label": "60-180s",
                "seconds_min": 60,
                "secondsMin": 60,
                "seconds_max": 180,
                "secondsMax": 180,
                "reason": "early lift detected but the pump is not mature enough for a fade setup",
            }
        else:
            fade_eta = {
                "status": "unknown",
                "label": "--",
                "seconds_min": None,
                "secondsMin": None,
                "seconds_max": None,
                "secondsMax": None,
                "reason": "no active pump timing edge",
            }
        score = sum(1 for row in checks if row["passed"]) * 25
        return {
            "usable": True,
            "score": score,
            "move_3_atr": round(move_3_atr, 4),
            "move_5_atr": round(move_5_atr, 4),
            "move_10_atr": round(move_10_atr, 4),
            "move_12_atr": round(move_12_atr, 4),
            "range_pos": round(range_pos, 4),
            "h15_cvd": round(h15_cvd, 4),
            "h15_short_score": round(h15_short_score, 4),
            "h30_short_score": round(h30_short_score, 4),
            "h15_long_score": round(h15_long_score, 4),
            "h30_long_score": round(h30_long_score, 4),
            "h15_aggression": round(h15_aggression, 4),
            "h15_move_3_atr": round(h15_move_3_atr, 4),
            "h15_move_10_atr": round(h15_move_10_atr, 4),
            "h15_move_15_atr": round(h15_move_15_atr, 4),
            "h15_up_closes": int(h15_up_closes),
            "h15_down_closes": int(h15_down_closes),
            "h15_up_closes_10": int(h15_up_closes_10),
            "h15_up_closes_15": int(h15_up_closes_15),
            "exhaustion_votes": int(exhaustion_votes),
            "big_green_candle": bool(big_green_candle),
            "one_min_rejection_candle": bool(one_min_rejection_candle),
            "h15_upper_wick_atr": round(h15_upper_wick_atr, 4),
            "h15_close_position": round(h15_close_pos, 4),
            "h30_body_atr": round(h30_body_atr, 4),
            "h30_volume_ratio": round(h30_volume_ratio, 4),
            "pump": bool(pump),
            "micro_pump": bool(micro_pump),
            "micro_pump_watch": bool(micro_pump_watch),
            "session_pump": bool(session_pump),
            "not_far_from_high": bool(not_far_from_high),
            "exhaustion": bool(exhaustion),
            "trend_not_too_clean": bool(trend_not_too_clean),
            "entry_window": bool(entry_window),
            "micro_rejection": bool(micro_rejection),
            "pump_watch": bool(pump_watch),
            "pre_pump_building": bool(pre_pump_building),
            "prePumpBuilding": bool(pre_pump_building),
            "pre_pump_score": int(min(49, max(35 if early_pump_ignition else 0, pre_pump_votes * 10))),
            "prePumpScore": int(min(49, max(35 if early_pump_ignition else 0, pre_pump_votes * 10))),
            "cooled_off_recovery": bool(cooled_off_recovery),
            "still_squeezing_up": bool(still_squeezing_up),
            "buyer_still_in_control": bool(buyer_still_in_control),
            "early_pump_ignition": bool(early_pump_ignition),
            "fade_eta": fade_eta,
            "fadeEta": fade_eta,
            "checks": checks,
        }

    @classmethod
    def _build_pump_fade_only_decision(cls, horizons: dict[str, dict[str, Any]]) -> dict[str, Any]:
        """Only publish the user's target setup: parabolic pump fade SHORT."""
        h15 = horizons.get("h_15m") or {}
        h30 = horizons.get("h_30m") or {}
        h1 = horizons.get("h_1h") or {}
        status = cls._pump_fade_status(h15, h30, h1)
        pump_fade = cls._parabolic_pump_short_setup(h15, h30, h1)

        details: list[dict[str, Any]] = []
        for key in ("h_15m", "h_30m", "h_1h"):
            row = horizons.get(key)
            if not row:
                continue
            long_score = cls._float_field(row, "long_score", "longScore")
            short_score = cls._float_field(row, "short_score", "shortScore")
            details.append({
                "key": key,
                "label": row.get("label", key),
                "long_score": round(long_score, 4),
                "short_score": round(short_score, 4),
                "gap": round(long_score - short_score, 4),
                "side": cls._gap_side(long_score - short_score, cls._NEXT_HOUR_MIN_SIDE_GAP),
            })

        if pump_fade:
            confidence_score = int(pump_fade["confidence_score"])
            primary_signal = {
                "key": "parabolic_pump_fade",
                "label": "Pump fade short",
                "timeframe": "3m30s",
                "side": "short",
                "score": round(confidence_score / 100.0, 4),
                "long_score": 0.0,
                "short_score": round(confidence_score / 100.0, 4),
                "gap": round(-(confidence_score / 100.0), 4),
                "stable": True,
                "firing": True,
                "reasons": pump_fade["reasons"],
            }
            return {
                "action": "short",
                "lean": "short",
                "confidence": "high",
                "confidence_score": confidence_score,
                "confidenceScore": confidence_score,
                "bias": round(-confidence_score / 100.0, 4),
                "weighted_long_score": 0.0,
                "weighted_short_score": round(confidence_score / 100.0, 4),
                "agreement": {"agree": 1, "total": 1, "ratio": 1.0},
                "warnings": [str(pump_fade["warning"])],
                "method": "parabolic_pump_fade_short_only",
                "mode": "pump_fade_only",
                "horizon": "3m30s",
                "setup": "parabolic_pump_fade",
                "setupLabel": "Parabolic pump fade short",
                "suggested_lev": 100,
                "suggestedLev": 100,
                "plan_horizon_key": "h_15m",
                "planHorizonKey": "h_15m",
                "primary_signal": primary_signal,
                "primarySignal": primary_signal,
                "fade_eta": pump_fade.get("fade_eta"),
                "fadeEta": pump_fade.get("fade_eta"),
                "pump_fade_checks": pump_fade["checks"],
                "pumpFadeChecks": pump_fade["checks"],
                "horizons": details,
            }

        checklist_score = min(49, int(status.get("score") or 0))
        pump_watch = bool(status.get("pump_watch"))
        pre_pump = bool(status.get("pre_pump_building"))
        warnings = ["waiting for parabolic pump + 1m entry rejection before shorting"]
        setup_stage = "hunting"
        if pump_watch:
            warnings = ["pump detected now; get ready for a fast fade-short entry"]
            setup_stage = "pump_watch"
        elif pre_pump:
            warnings = ["pump building; watch now before fade-ready rejection"]
            setup_stage = "pump_building"
            checklist_score = max(checklist_score, int(status.get("pre_pump_score") or 0))
        return {
            "action": "wait",
            "lean": "mixed",
            "confidence": "none",
            "confidence_score": 0,
            "confidenceScore": 0,
            "checklist_score": checklist_score,
            "checklistScore": checklist_score,
            "setup_stage": setup_stage,
            "setupStage": setup_stage,
            "pre_pump_building": pre_pump,
            "prePumpBuilding": pre_pump,
            "pre_pump_score": int(status.get("pre_pump_score") or 0),
            "prePumpScore": int(status.get("pre_pump_score") or 0),
            "early_pump_ignition": bool(status.get("early_pump_ignition")),
            "earlyPumpIgnition": bool(status.get("early_pump_ignition")),
            "fade_eta": status.get("fade_eta"),
            "fadeEta": status.get("fade_eta"),
            "bias": 0.0,
            "weighted_long_score": 0.0,
            "weighted_short_score": 0.0,
            "agreement": {"agree": 0, "total": 1 if status.get("usable") else 0, "ratio": 0.0},
            "warnings": warnings,
            "method": "parabolic_pump_fade_short_only",
            "mode": "pump_fade_only",
            "horizon": "3m30s",
            "setup": None,
            "setupLabel": "Parabolic pump fade short",
            "suggested_lev": 0,
            "suggestedLev": 0,
            "pump_fade_checks": status["checks"],
            "pumpFadeChecks": status["checks"],
            "horizons": details,
        }

    @classmethod
    def _build_next_hour_decision(cls, horizons: dict[str, dict[str, Any]]) -> dict[str, Any]:
        """Build a tradeable one-hour long/short/wait decision from horizons.

        This is intentionally stricter than raw overlay scores. A professional
        signal should be allowed to say "wait" when edge is too small, when
        30m and 1h disagree, or when the setup is too noisy for a one-hour
        trade.
        """
        details: list[dict[str, Any]] = []
        weighted_long = 0.0
        weighted_short = 0.0
        total_weight = 0.0

        for key, weight in cls._NEXT_HOUR_WEIGHTS:
            row = horizons.get(key)
            if not row:
                continue
            try:
                long_score = float(row.get("long_score") or 0.0)
                short_score = float(row.get("short_score") or 0.0)
            except (TypeError, ValueError):
                continue
            gap = long_score - short_score
            weighted_long += long_score * weight
            weighted_short += short_score * weight
            total_weight += weight
            details.append({
                "key": key,
                "label": row.get("label", key),
                "weight": weight,
                "long_score": round(long_score, 4),
                "short_score": round(short_score, 4),
                "gap": round(gap, 4),
                "side": cls._gap_side(gap, cls._NEXT_HOUR_MIN_SIDE_GAP),
            })

        if total_weight <= 0:
            return {
                "action": "wait",
                "lean": "mixed",
                "confidence": "none",
                "confidence_score": 0,
                "confidenceScore": 0,
                "bias": 0.0,
                "weighted_long_score": 0.0,
                "weighted_short_score": 0.0,
                "agreement": {"agree": 0, "total": 0, "ratio": 0.0},
                "warnings": ["no usable horizon data"],
                "method": "weighted_next_1h",
                "horizon": "1h",
                "horizons": details,
            }

        weighted_long /= total_weight
        weighted_short /= total_weight
        bias = weighted_long - weighted_short
        abs_bias = abs(bias)
        lean = cls._gap_side(bias)

        core_details = [d for d in details if d["key"] in cls._NEXT_HOUR_CORE_KEYS]
        core_total = len(core_details)
        core_agree = sum(
            1 for d in core_details
            if d["side"] == lean and abs(float(d["gap"])) >= cls._NEXT_HOUR_MIN_SIDE_GAP
        ) if lean != "mixed" else 0

        warnings: list[str] = []
        blockers: list[str] = []

        if lean == "mixed" or abs_bias < cls._NEXT_HOUR_MIN_ACTION_BIAS:
            blockers.append("weighted edge is too small")

        trigger = next((d for d in details if d["key"] == "h_1h"), None)
        if trigger and lean != "mixed":
            trigger_gap = float(trigger["gap"])
            trigger_side = cls._gap_side(trigger_gap, cls._NEXT_HOUR_CORE_CONFLICT_GAP)
            if trigger_side == "mixed":
                blockers.append("the one-hour anchor is not decisive")
            elif trigger_side != lean:
                blockers.append("the one-hour anchor disagrees")

        if core_total >= 2 and core_agree < 2:
            blockers.append("30m and 1h horizons are not aligned")

        h15 = horizons.get("h_15m") or {}
        h30 = horizons.get("h_30m") or {}
        h1 = horizons.get("h_1h") or {}
        pump_fade = cls._parabolic_pump_short_setup(h15, h30, h1)
        if pump_fade:
            confidence_score = int(pump_fade["confidence_score"])
            primary_signal = {
                "key": "parabolic_pump_fade",
                "label": "Pump fade",
                "timeframe": "3m30s",
                "side": "short",
                "score": round(confidence_score / 100.0, 4),
                "long_score": 0.0,
                "short_score": round(confidence_score / 100.0, 4),
                "gap": round(-(confidence_score / 100.0), 4),
                "stable": True,
                "firing": True,
                "reasons": pump_fade["reasons"],
            }
            return {
                "action": "short",
                "lean": "short",
                "confidence": "high" if confidence_score >= 80 else "medium",
                "confidence_score": confidence_score,
                "confidenceScore": confidence_score,
                "bias": round(-confidence_score / 100.0, 4),
                "weighted_long_score": round(weighted_long, 4),
                "weighted_short_score": round(weighted_short, 4),
                "agreement": {
                    "agree": 1,
                    "total": max(1, core_total),
                    "ratio": round(1 / max(1, core_total), 3),
                },
                "warnings": [str(pump_fade["warning"])],
                "method": "parabolic_pump_fade_short_only",
                "horizon": "3m30s",
                "setup": "parabolic_pump_fade",
                "setupLabel": "Parabolic pump fade",
                "suggested_lev": 100,
                "suggestedLev": 100,
                "plan_horizon_key": "h_15m",
                "planHorizonKey": "h_15m",
                "primary_signal": primary_signal,
                "primarySignal": primary_signal,
                "fade_eta": pump_fade.get("fade_eta"),
                "fadeEta": pump_fade.get("fade_eta"),
                "horizons": details,
            }
        if lean != "mixed":
            side_reasons = (
                list(h15.get(f"{lean}_reasons") or [])
                + list(h30.get(f"{lean}_reasons") or [])
            )
            flow_tags = ("agg", "cvd_real", "ob_imb", "absorb")
            has_flow = any(any(tag in str(reason) for tag in flow_tags) for reason in side_reasons)
            try:
                h30_adx = float(h30.get("adx") if h30.get("adx") is not None else "nan")
            except (TypeError, ValueError):
                h30_adx = float("nan")
            try:
                h1_adx = float(h1.get("adx") if h1.get("adx") is not None else "nan")
            except (TypeError, ValueError):
                h1_adx = float("nan")
            low_adx_pair = (
                not np.isnan(h30_adx)
                and not np.isnan(h1_adx)
                and h30_adx < cls._NEXT_HOUR_CHOP_ADX
                and h1_adx < cls._NEXT_HOUR_CHOP_ADX
            )
            if low_adx_pair and not has_flow:
                blockers.append("1h chop filter: ADX below 25 without flow confirmation")

            aggression = h15.get("aggression_10s")
            real_cvd = h15.get("real_cvd")
            try:
                agg_f = float(aggression) if aggression is not None else None
            except (TypeError, ValueError):
                agg_f = None
            try:
                cvd_f = float(real_cvd) if real_cvd is not None else None
            except (TypeError, ValueError):
                cvd_f = None
            if agg_f is not None and abs(agg_f) >= 0.60:
                if lean == "long" and agg_f < -0.30:
                    warnings.append("10s tape aggression opposes the long entry")
                elif lean == "short" and agg_f > 0.30:
                    warnings.append("10s tape aggression opposes the short entry")
            if cvd_f is not None and abs(cvd_f) >= 2.0:
                if lean == "long" and cvd_f < 0:
                    warnings.append("60s CVD opposes the long entry")
                elif lean == "short" and cvd_f > 0:
                    warnings.append("60s CVD opposes the short entry")

            try:
                atr_pct = float(h15.get("atr_pct") or 0.0)
            except (TypeError, ValueError):
                atr_pct = 0.0
            try:
                change_10s = abs(float(h15.get("price_change_10s_pct") or 0.0))
            except (TypeError, ValueError):
                change_10s = 0.0
            if atr_pct >= cls._NEXT_HOUR_NEWS_ATR_PCT:
                blockers.append("short-term volatility spike is too hot")
            if change_10s >= cls._NEXT_HOUR_CASCADE_10S_PCT:
                blockers.append("10s liquidation-cascade filter is active")
            chase_blockers = [
                cls._anti_chase_blocker(lean, h15, label="entry move"),
                cls._anti_chase_blocker(lean, h30, label="30m confirmation move"),
            ]
            for chase_blocker in chase_blockers:
                if chase_blocker:
                    blockers.append(chase_blocker)

        context_conflicts = [
            d for d in details
            if d["key"] in ("h_4h", "h_8h")
            and lean != "mixed"
            and d["side"] not in ("mixed", lean)
            and abs(float(d["gap"])) >= cls._NEXT_HOUR_CONTEXT_CONFLICT_GAP
        ]
        if context_conflicts:
            warnings.append("higher-timeframe context leans against the one-hour trade")

        action = "wait"
        confidence = "none"
        if not blockers and lean != "mixed":
            action = lean
            strong_core_agreement = core_agree >= min(2, core_total)
            if abs_bias >= 0.14 and strong_core_agreement and not context_conflicts:
                confidence = "high"
            elif abs_bias >= 0.09 and core_agree >= 2:
                confidence = "medium"
            else:
                confidence = "low"
            if context_conflicts and confidence == "high":
                confidence = "medium"
        confidence_score = cls._next_hour_confidence_score(
            action=action,
            abs_bias=abs_bias,
            core_agree=core_agree,
            core_total=core_total,
            blockers=blockers,
            context_conflicts=context_conflicts,
        )

        return {
            "action": action,
            "lean": lean,
            "confidence": confidence,
            "confidence_score": confidence_score,
            "confidenceScore": confidence_score,
            "bias": round(bias, 4),
            "weighted_long_score": round(weighted_long, 4),
            "weighted_short_score": round(weighted_short, 4),
            "agreement": {
                "agree": core_agree,
                "total": core_total,
                "ratio": round(core_agree / max(1, core_total), 3),
            },
            "warnings": warnings + blockers,
            "method": "weighted_next_1h",
            "horizon": "1h",
            "horizons": details,
        }

    def _smooth_next_hour_decision(
        self,
        symbol: str,
        raw_decision: dict[str, Any],
    ) -> dict[str, Any]:
        """Publish a stable one-hour decision instead of raw one-tick flips.

        A direct SHORT -> LONG publication is expensive for the human using the
        overlay: it invites whipsaw trades. Opposite actionable sides must
        persist across several bot ticks and the previous published side must
        be held for several minutes. While the flip is proving itself, publish
        WAIT with an explicit confirmation warning.
        """
        sym_u = symbol.upper()
        now = time.time()

        def _action(decision: dict[str, Any]) -> str:
            value = str(decision.get("action") or "wait").lower()
            return value if value in ("long", "short") else "wait"

        def _accept(decision: dict[str, Any], status: str = "stable") -> dict[str, Any]:
            out = {
                **decision,
                "smoothing": {
                    "status": status,
                    "published_action": _action(decision),
                    "pending_action": None,
                    "pending_count": 0,
                    "required_count": 0,
                    "min_hold_seconds": self._FOCUS_MIN_HOLD_SECONDS,
                },
            }
            self._overlay_decision_memory[sym_u] = {
                "shown": out,
                "changed_at": now,
                "pending_key": "",
                "pending_count": 0,
            }
            return out

        def _wait_for_confirmation(
            *,
            shown_action: str,
            raw_action: str,
            pending_count: int,
            required_count: int,
            held_seconds: int,
        ) -> dict[str, Any]:
            raw_score = int(raw_decision.get("confidence_score") or 0)
            warning = (
                f"{raw_action.upper()} confirming "
                f"({pending_count}/{required_count}); "
                f"holding WAIT until the 1h flip is stable"
            )
            if shown_action in ("long", "short") and held_seconds < self._FOCUS_MIN_HOLD_SECONDS:
                warning += f" and prior {shown_action.upper()} is at least 5m old"
            warnings = [warning] + list(raw_decision.get("warnings") or [])
            return {
                **raw_decision,
                "action": "wait",
                "confidence": "none",
                "confidence_score": max(0, min(49, raw_score)),
                "confidenceScore": max(0, min(49, raw_score)),
                "warnings": warnings,
                "smoothing": {
                    "status": "confirming",
                    "published_action": "wait",
                    "previous_action": shown_action,
                    "pending_action": raw_action,
                    "pending_count": pending_count,
                    "required_count": required_count,
                    "held_seconds": held_seconds,
                    "min_hold_seconds": self._FOCUS_MIN_HOLD_SECONDS,
                },
            }

        mem = self._overlay_decision_memory.get(sym_u)
        raw_action = _action(raw_decision)
        if not mem:
            if raw_action == "wait":
                return _accept(raw_decision)
            # First actionable tick after start/redeploy must prove itself once
            # more before we publish a tradeable long/short card.
            self._overlay_decision_memory[sym_u] = {
                "shown": {**raw_decision, "action": "wait", "confidence": "none",
                          "confidence_score": 0, "confidenceScore": 0},
                "changed_at": now,
                "pending_key": f"wait->{raw_action}",
                "pending_count": 1,
            }
            return _wait_for_confirmation(
                shown_action="wait",
                raw_action=raw_action,
                pending_count=1,
                required_count=self._FOCUS_ENTER_CONFIRM_TICKS,
                held_seconds=0,
            )

        shown = mem.get("shown") or {}
        shown_action = _action(shown)
        held_seconds = int(max(0, now - float(mem.get("changed_at") or now)))

        if raw_action == "wait":
            return _accept(raw_decision)

        if shown_action == raw_action:
            return _accept(raw_decision)

        pending_key = f"{shown_action}->{raw_action}"
        pending_count = int(mem.get("pending_count") or 0) + 1 if mem.get("pending_key") == pending_key else 1
        mem["pending_key"] = pending_key
        mem["pending_count"] = pending_count

        required_count = (
            self._FOCUS_ENTER_CONFIRM_TICKS
            if shown_action == "wait"
            else self._FOCUS_FLIP_CONFIRM_TICKS
        )
        hold_ok = shown_action == "wait" or held_seconds >= self._FOCUS_MIN_HOLD_SECONDS
        if pending_count >= required_count and hold_ok:
            return _accept(raw_decision, status="confirmed")

        return _wait_for_confirmation(
            shown_action=shown_action,
            raw_action=raw_action,
            pending_count=pending_count,
            required_count=required_count,
            held_seconds=held_seconds,
        )

    @classmethod
    def _build_sub_hour_payload(
        cls,
        horizons: dict[str, dict[str, Any]],
        decision: dict[str, Any],
    ) -> dict[str, Any]:
        """Compatibility payload for the Chrome overlay's sub-hour card.

        Older overlay builds look for a "sub-hour cache" rather than the newer
        `next_hour` decision. Expose the core confirmation horizons in that
        shape so the UI can render a real decision instead of sitting on
        WARMING UP.
        """
        signals: list[dict[str, Any]] = []
        for key in cls._NEXT_HOUR_CORE_KEYS:
            row = horizons.get(key)
            if not row:
                continue
            try:
                long_score = float(row.get("long_score") or 0.0)
                short_score = float(row.get("short_score") or 0.0)
            except (TypeError, ValueError):
                continue
            gap = long_score - short_score
            side = cls._gap_side(gap, cls._NEXT_HOUR_MIN_SIDE_GAP)
            if side == "long":
                reasons = list(row.get("long_reasons") or [])
                score = long_score
            elif side == "short":
                reasons = list(row.get("short_reasons") or [])
                score = short_score
            else:
                reasons = []
                score = max(long_score, short_score)
            signals.append({
                "key": key,
                "label": row.get("label", key),
                "timeframe": row.get("timeframe"),
                "side": side,
                "score": round(score, 4),
                "long_score": round(long_score, 4),
                "short_score": round(short_score, 4),
                "gap": round(gap, 4),
                "stable": bool(row.get("stable")),
                "firing": side != "mixed" and score >= cls._ALARM_AT,
                "reasons": reasons,
            })

        ready = bool(signals)
        actual_firing = [s for s in signals if s.get("firing")]
        primary = None
        if signals:
            directional = [s for s in signals if s.get("side") != "mixed"]
            pool = directional or signals
            primary = max(pool, key=lambda s: abs(float(s.get("gap") or 0.0)))
            if primary not in actual_firing:
                primary = {
                    **primary,
                    "candidate": True,
                    "reason": "strongest sub-hour lean below alarm threshold",
                    "threshold": cls._ALARM_AT,
                }
        decision_primary = decision.get("primary_signal") or decision.get("primarySignal")
        if isinstance(decision_primary, dict):
            primary = dict(decision_primary)
            if primary.get("firing") and primary not in actual_firing:
                actual_firing = [primary] + actual_firing
        # Compatibility: older overlay builds use the presence of
        # `firing_signals` as "cache filled". If nothing clears the alarm
        # threshold yet, expose the strongest candidate there too while keeping
        # `firing: false` on the object so consumers can label it as weak/wait.
        display_signals = actual_firing or ([primary] if primary else [])
        return {
            "ready": ready,
            "filled": ready,
            "warming_up": False,
            "warmingUp": False,
            "status": "ready" if ready else "warming_up",
            "action": decision.get("action", "wait"),
            "direction": decision.get("action", "wait"),
            "lean": decision.get("lean", "mixed"),
            "confidence": decision.get("confidence", "none"),
            "confidence_score": decision.get("confidence_score", 0),
            "confidenceScore": decision.get("confidence_score", 0),
            "trade_plan": decision.get("trade_plan"),
            "tradePlan": decision.get("trade_plan"),
            "bias": decision.get("bias", 0.0),
            "agreement": decision.get("agreement", {}),
            "warnings": list(decision.get("warnings") or []),
            "primary_signal": primary,
            "primarySignal": primary,
            "best_signal": primary,
            "bestSignal": primary,
            "signals": signals,
            "actual_firing_signals": actual_firing,
            "actualFiringSignals": actual_firing,
            "firing_signals": display_signals,
            "firingSignals": display_signals,
            "candidate_signals": signals,
            "candidateSignals": signals,
            "as_of": int(time.time()),
        }

    @staticmethod
    def _maker_limit_price(
        *,
        side: str,
        bid: float,
        ask: float,
        price_precision: int,
    ) -> float:
        """Return an aggressive maker limit that should not cross the spread."""
        tick_size = 10 ** -price_precision if price_precision >= 0 else 0.0
        spread = ask - bid
        is_buy = side == "BUY"
        if tick_size > 0 and spread > tick_size * 1.5:
            raw = bid + tick_size if is_buy else ask - tick_size
        else:
            raw = bid if is_buy else ask
        return round(raw, price_precision)

    @staticmethod
    def _first_float(*values: Any) -> float:
        for value in values:
            try:
                out = float(value)
            except (TypeError, ValueError):
                continue
            if out > 0:
                return out
        return 0.0

    def _target_leverage_for_symbol(
        self,
        symbol: str,
        meta: SymbolMeta,
        requested: int,
    ) -> int:
        """Cap leverage by symbol class.

        BTC/ETH can use the requested high leverage when Bitunix allows it.
        Dynamic alts are capped harder because their spreads, ATR and book
        depth are noisier even when they pass the liquidity scan.
        """
        sym = symbol.upper()
        cap = int(requested)
        if sym not in {"BTCUSDT", "ETHUSDT"}:
            cap = min(cap, 100)
            if sym not in self._configured_symbols:
                risk_mult = self.cfg.trading.symbol_risk_mult.get(sym, 0.35)
                if risk_mult <= 0.35:
                    cap = min(cap, 75)
        return max(1, min(cap, int(meta.max_leverage or 100)))

    def _max_entry_spread_pct_for_symbol(self, symbol: str, meta: SymbolMeta) -> float:
        base = float(self.cfg.trading.max_entry_spread_pct)
        sym = symbol.upper()
        if sym in {"BTCUSDT", "ETHUSDT"}:
            return base
        if sym not in self._configured_symbols:
            return min(0.25, max(base, base * 1.5))
        if meta.max_leverage < 100:
            return min(0.20, max(base, base * 1.25))
        return base

    def _build_suggested_trade_plan(
        self,
        symbol: str,
        decision: dict[str, Any],
        horizons: dict[str, dict[str, Any]],
        fallback_price: float | None,
    ) -> dict[str, Any]:
        """Build a display-only entry/exit plan for the Chrome overlay.

        This does not place an order. It mirrors the bot's entry preference:
        maker limit at top-of-book when available, market only for urgent,
        tight-spread momentum.
        """
        sym_u = symbol.upper()
        action = str(decision.get("action") or "wait").lower()
        confidence_score = int(decision.get("confidence_score") or 0)
        setup_stage = str(decision.get("setup_stage") or decision.get("setupStage") or "").lower()
        preview_only = False
        if action not in ("long", "short"):
            if setup_stage in ("pump_watch", "pump_building"):
                # Display-only short plan so the overlay can show the user
                # where the trade will likely be taken once rejection confirms.
                # This is NOT an executable-ready state.
                action = "short"
                preview_only = True
                confidence_score = max(
                    confidence_score,
                    int(decision.get("checklist_score") or decision.get("checklistScore") or 55),
                )
            else:
                return {
                    "status": "wait",
                    "order_type": "WAIT",
                    "orderType": "WAIT",
                    "reason": "no parabolic pump-fade short setup",
                }

        meta = self.metas.get(sym_u, _DEFAULT_META)
        side = "BUY" if action == "long" else "SELL"
        preferred_horizon = str(
            decision.get("plan_horizon_key") or decision.get("planHorizonKey") or ""
        )
        plan_horizon_key = (
            preferred_horizon if preferred_horizon in horizons
            else "h_15m" if preview_only and "h_15m" in horizons
            else next((key for key in ("h_1h", "h_30m", "h_15m") if key in horizons), None)
        )
        plan_horizon = horizons.get(plan_horizon_key or "", {})
        reference_price = self._first_float(
            horizons.get("h_15m", {}).get("price"),
            fallback_price,
            plan_horizon.get("price"),
        )
        if reference_price <= 0:
            return {
                "status": "wait",
                "order_type": "WAIT_FOR_PRICE",
                "orderType": "WAIT_FOR_PRICE",
                "reason": "price unavailable",
            }

        spread_pct = None
        bid = ask = maker_limit = taker_price = None
        order_type = "MARKET"
        entry_price = reference_price
        rationale = "market reference; order book unavailable"
        if self.ob_feed is not None:
            tob = self.ob_feed.get_top_of_book(sym_u)
            if tob:
                bid, ask = float(tob[0]), float(tob[1])
                maker_limit = self._maker_limit_price(
                    side=side,
                    bid=bid,
                    ask=ask,
                    price_precision=meta.price_precision,
                )
                taker_price = ask if side == "BUY" else bid
                spread_pct = self.ob_feed.get_spread_pct(sym_u)

                aggression = None
                if self.tape_feed is not None:
                    aggression = self.tape_feed.get_aggression_ratio(sym_u, window_secs=10)
                tape_aligned = (
                    aggression is None
                    or (side == "BUY" and aggression >= 0.30)
                    or (side == "SELL" and aggression <= -0.30)
                )
                tight_spread = (
                    spread_pct is not None
                    and spread_pct <= max(0.01, self.cfg.trading.max_entry_spread_pct * 0.5)
                )
                pump_fade_scalp = decision.get("setup") == "parabolic_pump_fade"
                urgent_market = (
                    pump_fade_scalp and not preview_only
                    or (confidence_score >= 85 and tight_spread and tape_aligned)
                )

                if urgent_market:
                    order_type = "MARKET"
                    entry_price = taker_price
                    rationale = (
                        "pump-fade scalp; market entry only if taking it immediately"
                        if pump_fade_scalp
                        else "high confidence with tight spread; market entry is acceptable"
                    )
                else:
                    order_type = "LIMIT_POST_ONLY" if self.cfg.trading.use_post_only_entries else "LIMIT"
                    entry_price = maker_limit
                    rationale = "maker limit at top-of-book; skip if it does not fill quickly"

        atr = self._first_float(
            plan_horizon.get("atr"),
            reference_price * self._first_float(plan_horizon.get("atr_pct")) / 100.0,
        )
        if preview_only:
            tick_size = 10 ** -meta.price_precision if meta.price_precision >= 0 else 0.0
            trigger_offset = max(
                tick_size,
                atr * (0.08 if setup_stage == "pump_watch" else 0.15),
            )
            if action == "short":
                entry_price = max(tick_size, entry_price - trigger_offset)
            else:
                entry_price = entry_price + trigger_offset
            entry_price = round(float(entry_price), meta.price_precision)
            order_type = "WAIT_FOR_REJECTION"
            rationale = (
                "preview only: enter short after 1m rejection/CVD flip confirms"
                if setup_stage == "pump_watch"
                else "preview only: pump is building; wait for high tag and rejection"
            )
        reasons = list(dict.fromkeys(
            list(plan_horizon.get(f"{action}_reasons") or [])
            + list(horizons.get("h_30m", {}).get(f"{action}_reasons") or [])
            + list(horizons.get("h_15m", {}).get(f"{action}_reasons") or [])
        ))
        signal = Signal(
            direction=action,  # type: ignore[arg-type]
            score=max(0.01, min(1.0, confidence_score / 100.0)),
            indicator_score=len(reasons),
            pattern_score=0.0,
            reasons=reasons,
            price=float(entry_price),
            atr=atr,
            fire_threshold_used=self.cfg.strategy.fire_threshold,
            last_bar_high=self._first_float(plan_horizon.get("last_bar_high")),
            last_bar_low=self._first_float(plan_horizon.get("last_bar_low")),
        )
        order_plan = build_order(
            signal,
            free_margin=100_000.0,
            trading=self.cfg.trading,
            risk=self.cfg.risk,
            min_volume=meta.min_qty,
            volume_step=meta.base_precision,
            digits=meta.price_precision,
            effective_leverage=self._target_leverage_for_symbol(sym_u, meta, self.cfg.trading.leverage),
            symbol=sym_u,
        )
        if order_plan is None:
            return {
                "status": "wait",
                "order_type": "WAIT",
                "orderType": "WAIT",
                "reason": "risk geometry rejected the setup",
            }

        risk_pct = abs(order_plan.price - order_plan.stop_loss) / order_plan.price * 100.0
        reward_pct = abs(order_plan.take_profit - order_plan.price) / order_plan.price * 100.0
        timeout_secs = self.cfg.trading.post_only_timeout_secs
        return {
            "status": "preview" if preview_only else "ready",
            "ready": not preview_only,
            "preview": preview_only,
            "side": side,
            "direction": action,
            "order_type": order_type,
            "orderType": order_type,
            "entry_price": order_plan.price,
            "entryPrice": order_plan.price,
            "limit_price": maker_limit,
            "limitPrice": maker_limit,
            "market_price": round(taker_price, meta.price_precision) if taker_price else None,
            "marketPrice": round(taker_price, meta.price_precision) if taker_price else None,
            "stop_loss": order_plan.stop_loss,
            "stopLoss": order_plan.stop_loss,
            "take_profit": order_plan.take_profit,
            "takeProfit": order_plan.take_profit,
            "target_exit_price": order_plan.take_profit,
            "targetExitPrice": order_plan.take_profit,
            "max_exit_price": order_plan.take_profit,
            "maxExitPrice": order_plan.take_profit,
            "risk_pct": round(risk_pct, 3),
            "riskPct": round(risk_pct, 3),
            "reward_pct": round(reward_pct, 3),
            "rewardPct": round(reward_pct, 3),
            "risk_reward_r": self.cfg.risk.take_profit_r,
            "riskRewardR": self.cfg.risk.take_profit_r,
            "valid_for_seconds": timeout_secs if order_type == "LIMIT_POST_ONLY" else None,
            "validForSeconds": timeout_secs if order_type == "LIMIT_POST_ONLY" else None,
            "horizon": plan_horizon.get("label") or plan_horizon_key,
            "spread_pct": round(float(spread_pct), 4) if spread_pct is not None else None,
            "spreadPct": round(float(spread_pct), 4) if spread_pct is not None else None,
            "bid": round(bid, meta.price_precision) if bid else None,
            "ask": round(ask, meta.price_precision) if ask else None,
            "rationale": rationale,
        }

    def _get_klines_cached(self, symbol: str, timeframe: str,
                           ttl_seconds: int, limit: int = 200) -> list | None:
        """Klines fetcher with per-(symbol, timeframe) TTL cache.

        Returns rows sorted ascending by time, or the last cached value if
        the network call fails (degrades gracefully — better to show
        slightly-stale data than to drop the row entirely).
        """
        key = (symbol.upper(), timeframe)
        now = int(time.time())
        cached = getattr(self, "_kline_cache", None)
        if cached is None:
            self._kline_cache = {}
            cached = self._kline_cache
        prev = cached.get(key)
        if prev and (now - prev[0]) < ttl_seconds:
            return prev[1]
        try:
            rows = self.client.klines(symbol, timeframe, limit=limit)
        except Exception as e:
            log.debug("klines fetch failed %s/%s: %s", symbol, timeframe, e)
            return prev[1] if prev else None
        if not rows:
            return prev[1] if prev else None
        rows = sorted(rows, key=lambda r: int(r.get("time") or 0))
        cached[key] = (now, rows)
        return rows

    def _compute_overlays(self) -> None:
        """Refresh per-symbol multi-horizon overlay scores.

        Runs once per tick for every configured symbol regardless of cooldown
        / streak-pause / position state — the overlay is for exit timing, so
        the user wants continuous momentum updates even when the trading
        loop wouldn't act on this symbol.

        For each symbol, runs compute_overlay_scores() at six different
        kline timeframes corresponding to forward-looking horizons of
        15m / 30m / 1h / 4h / 8h / 24h. Tape and order-book signals are
        only fed into the short-term horizons (≤5m bars) — they're noise
        on multi-hour bars.

        Errors per symbol/horizon are swallowed (logged at debug) so a
        single failing fetch doesn't take down the whole tick.
        """
        for sym in self.cfg.trading.symbols:
            sym_u = sym.upper()
            meta = self.metas.get(sym_u, _DEFAULT_META)

            # Live-tape inputs are only meaningful for the short-term horizons.
            real_cvd = aggression_10s = price_change_10s_pct = None
            if self.tape_feed is not None:
                real_cvd = self.tape_feed.get_cvd(sym, window_secs=60)
                aggression_10s = self.tape_feed.get_aggression_ratio(sym, window_secs=10)
                price_change_10s_pct = self.tape_feed.get_price_change_pct(sym, window_secs=10)
            ob_imb = self.ob_feed.get_imbalance(sym) if self.ob_feed else None
            htf_closes = self._get_htf_closes(sym)

            horizons: dict[str, dict[str, Any]] = {}
            latest_price = None

            for key, tf, ttl, label in self._OVERLAY_HORIZONS:
                rows = self._get_klines_cached(sym, tf, ttl,
                                                limit=self.cfg.loop.kline_lookback)
                if not rows or len(rows) < 30:
                    continue
                opens = [float(r["open"]) for r in rows]
                highs = [float(r["high"]) for r in rows]
                lows = [float(r["low"]) for r in rows]
                closes = [float(r["close"]) for r in rows]
                volumes = [float(r.get("baseVol") or r.get("quoteVol") or 0) for r in rows]

                # Only feed flow + HTF inputs to the shortest two timeframes.
                # On 15m+ bars, 10-second tape windows and 1h HTF context are
                # noise rather than signal.
                is_short_tf = tf in ("1m", "5m")
                tf_htf = htf_closes if is_short_tf else None
                tf_ob = ob_imb if is_short_tf else None
                tf_cvd = real_cvd if is_short_tf else None
                tf_agg = aggression_10s if is_short_tf else None
                tf_chg = price_change_10s_pct if is_short_tf else None

                try:
                    overlay = compute_overlay_scores(
                        opens, highs, lows, closes, self.cfg.strategy,
                        volumes=volumes, htf_closes=tf_htf,
                        ob_imbalance=tf_ob,
                        real_cvd=tf_cvd, aggression_10s=tf_agg,
                        price_change_10s_pct=tf_chg,
                    )
                except Exception as e:
                    log.debug("overlay failed %s/%s: %s", sym_u, tf, e)
                    continue
                if overlay is None:
                    continue
                if (
                    getattr(self.cfg.trading, "dynamic_symbols_enabled", False)
                    and sym_u not in self._configured_symbols
                ):
                    row = self._market_rows_by_symbol.get(sym_u, {})
                    self.cfg.trading.symbol_risk_mult[sym_u] = auto_symbol_risk_mult(
                        sym_u,
                        quote_volume_usdt=row_quote_volume_usdt(row),
                        max_leverage=row_max_leverage(row),
                        atr_pct=overlay.atr_pct,
                    )
                atr_abs = overlay.price * overlay.atr_pct / 100.0

                def _recent_move_pct(bars: int) -> float | None:
                    if len(closes) <= bars:
                        return None
                    start = closes[-(bars + 1)]
                    if start <= 0:
                        return None
                    return (closes[-1] - start) / start * 100.0

                def _recent_move_atr(bars: int) -> float | None:
                    if len(closes) <= bars or atr_abs <= 0:
                        return None
                    return (closes[-1] - closes[-(bars + 1)]) / atr_abs

                def _recent_close_counts(bars: int) -> tuple[int, int]:
                    if len(closes) <= bars:
                        return 0, 0
                    pairs = list(zip(closes[-(bars + 1):-1], closes[-bars:]))
                    down = sum(1 for prev, cur in pairs if cur < prev)
                    up = sum(1 for prev, cur in pairs if cur > prev)
                    return down, up

                def _round_metric(value: float | None, digits: int = 4) -> float | None:
                    if value is None or not np.isfinite(value):
                        return None
                    return round(value, digits)

                move_3_bars_pct = _round_metric(_recent_move_pct(3))
                move_5_bars_pct = _round_metric(_recent_move_pct(5))
                move_10_bars_pct = _round_metric(_recent_move_pct(10))
                move_12_bars_pct = _round_metric(_recent_move_pct(12))
                move_15_bars_pct = _round_metric(_recent_move_pct(15))
                move_3_atr = _round_metric(_recent_move_atr(3))
                move_5_atr = _round_metric(_recent_move_atr(5))
                move_10_atr = _round_metric(_recent_move_atr(10))
                move_12_atr = _round_metric(_recent_move_atr(12))
                move_15_atr = _round_metric(_recent_move_atr(15))
                recent_lows = lows[-15:]
                recent_highs = highs[-15:]
                range_low = min(recent_lows)
                range_high = max(recent_highs)
                range_span = range_high - range_low
                range_pos = (closes[-1] - range_low) / range_span if range_span > 0 else 0.5
                range_pos = max(0.0, min(1.0, range_pos))
                distance_from_low_atr = (closes[-1] - range_low) / atr_abs if atr_abs > 0 else None
                distance_from_high_atr = (range_high - closes[-1]) / atr_abs if atr_abs > 0 else None
                down_closes_5, up_closes_5 = _recent_close_counts(5)
                down_closes_10, up_closes_10 = _recent_close_counts(10)
                down_closes_12, up_closes_12 = _recent_close_counts(12)
                down_closes_15, up_closes_15 = _recent_close_counts(15)
                recent_candles = list(zip(opens[-5:], closes[-5:]))
                down_candles_5 = sum(1 for op, cl in recent_candles if cl < op)
                up_candles_5 = sum(1 for op, cl in recent_candles if cl > op)
                last_open = opens[-1]
                last_close = closes[-1]
                last_high = highs[-1]
                last_low = lows[-1]
                last_range = max(0.0, last_high - last_low)
                last_body = last_close - last_open
                last_upper_wick = max(0.0, last_high - max(last_open, last_close))
                last_lower_wick = max(0.0, min(last_open, last_close) - last_low)
                last_close_pos = ((last_close - last_low) / last_range) if last_range > 0 else 0.5
                vol_ma_values = volumes[-21:-1] if len(volumes) >= 21 else volumes[:-1]
                vol_ma = (sum(vol_ma_values) / len(vol_ma_values)) if vol_ma_values else 0.0
                volume_ratio = volumes[-1] / vol_ma if vol_ma > 0 else None
                range_pos_rounded = _round_metric(range_pos)
                distance_from_low_atr = _round_metric(distance_from_low_atr)
                distance_from_high_atr = _round_metric(distance_from_high_atr)

                horizons[key] = {
                    "label": label,
                    "timeframe": tf,
                    "price": round(overlay.price, meta.price_precision),
                    "atr": round(atr_abs, meta.price_precision),
                    "atr_pct": round(overlay.atr_pct, 4),
                    "last_bar_high": round(highs[-1], meta.price_precision),
                    "last_bar_low": round(lows[-1], meta.price_precision),
                    "last_bar_open": round(last_open, meta.price_precision),
                    "lastBarOpen": round(last_open, meta.price_precision),
                    "last_bar_close": round(last_close, meta.price_precision),
                    "lastBarClose": round(last_close, meta.price_precision),
                    "last_bar_body_atr": _round_metric(last_body / atr_abs if atr_abs > 0 else None),
                    "lastBarBodyAtr": _round_metric(last_body / atr_abs if atr_abs > 0 else None),
                    "last_bar_range_atr": _round_metric(last_range / atr_abs if atr_abs > 0 else None),
                    "lastBarRangeAtr": _round_metric(last_range / atr_abs if atr_abs > 0 else None),
                    "last_bar_upper_wick_atr": _round_metric(last_upper_wick / atr_abs if atr_abs > 0 else None),
                    "lastBarUpperWickAtr": _round_metric(last_upper_wick / atr_abs if atr_abs > 0 else None),
                    "last_bar_lower_wick_atr": _round_metric(last_lower_wick / atr_abs if atr_abs > 0 else None),
                    "lastBarLowerWickAtr": _round_metric(last_lower_wick / atr_abs if atr_abs > 0 else None),
                    "last_bar_close_position": _round_metric(last_close_pos),
                    "lastBarClosePosition": _round_metric(last_close_pos),
                    "volume_spike_ratio": _round_metric(volume_ratio),
                    "volumeSpikeRatio": _round_metric(volume_ratio),
                    "long_score": round(overlay.long_score, 4),
                    "short_score": round(overlay.short_score, 4),
                    "long_reasons": overlay.long_reasons,
                    "short_reasons": overlay.short_reasons,
                    "adx": round(overlay.adx, 2) if overlay.adx >= 0 else None,
                    "real_cvd": round(float(real_cvd), 4) if is_short_tf and real_cvd is not None else None,
                    "realCvd": round(float(real_cvd), 4) if is_short_tf and real_cvd is not None else None,
                    "aggression_10s": round(float(aggression_10s), 4) if is_short_tf and aggression_10s is not None else None,
                    "aggression10s": round(float(aggression_10s), 4) if is_short_tf and aggression_10s is not None else None,
                    "price_change_10s_pct": round(float(price_change_10s_pct), 4) if is_short_tf and price_change_10s_pct is not None else None,
                    "priceChange10sPct": round(float(price_change_10s_pct), 4) if is_short_tf and price_change_10s_pct is not None else None,
                    "order_book_imbalance": round(float(ob_imb), 4) if is_short_tf and ob_imb is not None else None,
                    "orderBookImbalance": round(float(ob_imb), 4) if is_short_tf and ob_imb is not None else None,
                    "move_3_bars_pct": move_3_bars_pct,
                    "move3BarsPct": move_3_bars_pct,
                    "move_5_bars_pct": move_5_bars_pct,
                    "move5BarsPct": move_5_bars_pct,
                    "move_10_bars_pct": move_10_bars_pct,
                    "move10BarsPct": move_10_bars_pct,
                    "move_12_bars_pct": move_12_bars_pct,
                    "move12BarsPct": move_12_bars_pct,
                    "move_15_bars_pct": move_15_bars_pct,
                    "move15BarsPct": move_15_bars_pct,
                    "move_3_atr": move_3_atr,
                    "move3Atr": move_3_atr,
                    "move_5_atr": move_5_atr,
                    "move5Atr": move_5_atr,
                    "move_10_atr": move_10_atr,
                    "move10Atr": move_10_atr,
                    "move_12_atr": move_12_atr,
                    "move12Atr": move_12_atr,
                    "move_15_atr": move_15_atr,
                    "move15Atr": move_15_atr,
                    "range_low_15": round(range_low, meta.price_precision),
                    "rangeLow15": round(range_low, meta.price_precision),
                    "range_high_15": round(range_high, meta.price_precision),
                    "rangeHigh15": round(range_high, meta.price_precision),
                    "position_in_recent_range_15": range_pos_rounded,
                    "positionInRecentRange15": range_pos_rounded,
                    "distance_from_recent_low_atr": distance_from_low_atr,
                    "distanceFromRecentLowAtr": distance_from_low_atr,
                    "distance_from_recent_high_atr": distance_from_high_atr,
                    "distanceFromRecentHighAtr": distance_from_high_atr,
                    "down_closes_5": down_closes_5,
                    "downCloses5": down_closes_5,
                    "up_closes_5": up_closes_5,
                    "upCloses5": up_closes_5,
                    "down_closes_10": down_closes_10,
                    "downCloses10": down_closes_10,
                    "up_closes_10": up_closes_10,
                    "upCloses10": up_closes_10,
                    "down_closes_12": down_closes_12,
                    "downCloses12": down_closes_12,
                    "up_closes_12": up_closes_12,
                    "upCloses12": up_closes_12,
                    "down_closes_15": down_closes_15,
                    "downCloses15": down_closes_15,
                    "up_closes_15": up_closes_15,
                    "upCloses15": up_closes_15,
                    "down_candles_5": down_candles_5,
                    "downCandles5": down_candles_5,
                    "up_candles_5": up_candles_5,
                    "upCandles5": up_candles_5,
                }
                if latest_price is None or key == "h_15m":
                    latest_price = overlay.price

            if not horizons:
                continue

            # ---- Score persistence ---------------------------------------
            # Reliability research consistently shows that single-tick
            # threshold crossings have far higher false-positive rates than
            # crossings sustained across 2+ samples. We track the last
            # _PERSISTENCE_WINDOW dominant scores per (symbol, horizon)
            # and mark a horizon "stable" only if it's been ≥ ALARM_AT for
            # at least 2 of the last 3 ticks AND on the same direction.
            #
            # This is the bot equivalent of "wait for 2-bar confirmation"
            # — a standard rule in pro setups for filtering chop spikes.
            hist = self._overlay_score_history.setdefault(sym_u, {})
            for key, h in horizons.items():
                ls = h["long_score"]
                ss = h["short_score"]
                dom = max(ls, ss)
                side = "long" if ls > ss else "short" if ss > ls else "tie"
                buf = hist.setdefault(key, [])
                buf.append((dom, side))
                if len(buf) > self._PERSISTENCE_WINDOW:
                    del buf[:-self._PERSISTENCE_WINDOW]
                # Stable iff the most recent reading is alarm-strength AND
                # at least one prior reading also was, on the same side.
                stable = False
                if buf and buf[-1][0] >= self._ALARM_AT and buf[-1][1] != "tie":
                    cur_side = buf[-1][1]
                    confirms = sum(1 for s, sd in buf
                                    if s >= self._ALARM_AT and sd == cur_side)
                    stable = confirms >= 2
                h["stable"] = stable

            # ---- Cross-horizon alignment ---------------------------------
            # Weighted-mean approach: sum the signed bias (long_score -
            # short_score) across all horizons and divide by N. This
            # captures direction AND magnitude — a small consistent lean
            # across all 6 horizons is far more meaningful than 3 mild
            # longs vs 3 mild shorts (which the old binary counting
            # called "mixed" even when the actual scores were nearly
            # identical to a clean bull case).
            #
            # Strength tiers are tuned to typical values observed in live
            # data:
            #   |mean_bias| ≥ 0.20  → ALL-IN  (strongest readings, e.g. all
            #                                   horizons solidly directional)
            #   |mean_bias| ≥ 0.13  → STRONG  (5+ horizons clearly leaning)
            #   |mean_bias| ≥ 0.07  → MODERATE (mild but consistent lean)
            #   |mean_bias| ≥ 0.03  → WEAK    (barely directional)
            #   else                → NO CONSENSUS (truly flat)
            signed = [h["long_score"] - h["short_score"] for h in horizons.values()]
            total_horizons = len(signed)
            mean_bias = sum(signed) / max(1, total_horizons) if signed else 0.0
            abs_bias = abs(mean_bias)

            if abs_bias < 0.03:
                dominant_side = "mixed"
                strength_tier = "none"
            else:
                dominant_side = "long" if mean_bias > 0 else "short"
                if abs_bias >= 0.20:   strength_tier = "all_in"
                elif abs_bias >= 0.13: strength_tier = "strong"
                elif abs_bias >= 0.07: strength_tier = "moderate"
                else:                   strength_tier = "weak"

            # Count horizons that lean the dominant direction (any positive
            # gap, no BIAS_GAP threshold — used for the secondary "5 of 6"
            # display detail).
            if dominant_side == "long":
                agree = sum(1 for sb in signed if sb > 0)
            elif dominant_side == "short":
                agree = sum(1 for sb in signed if sb < 0)
            else:
                agree = 0

            alignment = {
                "dominant": dominant_side,
                "strength": strength_tier,
                "mean_bias": round(mean_bias, 4),
                "agree": agree,
                "total": total_horizons,
                "ratio": round(agree / max(1, total_horizons), 3),
            }

            next_hour = self._build_pump_fade_only_decision(horizons)
            trade_plan = self._build_suggested_trade_plan(sym_u, next_hour, horizons, latest_price)
            next_hour = {
                **next_hour,
                "trade_plan": trade_plan,
                "tradePlan": trade_plan,
            }
            sub_hour = self._build_sub_hour_payload(horizons, next_hour)
            self.state.record_overlay(sym_u, {
                "symbol": sym_u,
                "price": latest_price,
                "horizons": horizons,
                "horizon_order": [k for k, _, _, _ in self._OVERLAY_HORIZONS if k in horizons],
                "alignment": alignment,
                "focus_horizon": "pump_fade",
                "focusHorizon": "pump_fade",
                "next_1h": next_hour,
                "next1h": next_hour,
                "next_15m": next_hour,
                "next15m": next_hour,
                "fifteen_minute": next_hour,
                "fifteenMinute": next_hour,
                "next_hour": next_hour,
                "nextHour": next_hour,
                "one_hour": next_hour,
                "oneHour": next_hour,
                "decision": next_hour,
                "recommendation": next_hour,
                "trade_plan": trade_plan,
                "tradePlan": trade_plan,
                "sub_hour": sub_hour,
                "subHour": sub_hour,
                "sub_hour_cache": sub_hour,
                "subHourCache": sub_hour,
                "sub_hour_signals": sub_hour["signals"],
                "subHourSignals": sub_hour["signals"],
                "sub_hour_firing_signals": sub_hour["firing_signals"],
                "subHourFiringSignals": sub_hour["firing_signals"],
                "sub_hour_primary_signal": sub_hour["primary_signal"],
                "subHourPrimarySignal": sub_hour["primary_signal"],
                "sub_hour_ready": sub_hour["ready"],
                "subHourReady": sub_hour["ready"],
                "sub_hour_warming_up": False,
                "subHourWarmingUp": False,
                "as_of": int(time.time()),
            })

    def _build_auto_pump_fade_plan(
        self,
        symbol: str,
        overlay: dict[str, Any],
        decision: dict[str, Any],
        *,
        free_margin: float,
        effective_leverage: int,
        dd_risk_mult: float,
    ) -> OrderPlan | None:
        """Build the actual market-short order plan for auto pump fades."""
        sym_u = symbol.upper()
        meta = self.metas.get(sym_u, _DEFAULT_META)
        horizons = overlay.get("horizons") or {}
        h15 = horizons.get("h_15m") or {}
        h30 = horizons.get("h_30m") or {}

        entry_price = 0.0
        if self.ob_feed is not None:
            tob = self.ob_feed.get_top_of_book(sym_u)
            if tob:
                entry_price = float(tob[0])  # SELL market should fill near bid.
        if entry_price <= 0:
            try:
                ticker = self.client.ticker(sym_u)
                if isinstance(ticker, dict):
                    entry_price = float(ticker.get("lastPrice") or 0)
            except Exception:
                entry_price = 0.0
        entry_price = self._first_float(entry_price, h15.get("price"), overlay.get("price"), h30.get("price"))
        if entry_price <= 0:
            return None

        atr = self._first_float(
            h15.get("atr"),
            h30.get("atr"),
            entry_price * self._first_float(h15.get("atr_pct"), h30.get("atr_pct")) / 100.0,
        )
        primary = decision.get("primary_signal") or decision.get("primarySignal") or {}
        reasons = list(primary.get("reasons") or [])
        signal = Signal(
            direction="short",
            score=max(0.01, min(1.0, float(decision.get("confidence_score") or 0) / 100.0)),
            indicator_score=len(reasons),
            pattern_score=0.0,
            reasons=reasons,
            price=float(entry_price),
            atr=atr,
            fire_threshold_used=float(getattr(self.cfg.trading, "pump_fade_auto_min_confidence", 95)) / 100.0,
            last_bar_high=self._first_float(h15.get("last_bar_high"), h30.get("last_bar_high")),
            last_bar_low=self._first_float(h15.get("last_bar_low"), h30.get("last_bar_low")),
        )
        return build_order(
            signal,
            free_margin=free_margin,
            trading=self.cfg.trading,
            risk=self.cfg.risk,
            min_volume=meta.min_qty,
            volume_step=meta.base_precision,
            digits=meta.price_precision,
            effective_leverage=effective_leverage,
            symbol=sym_u,
            dd_risk_mult=dd_risk_mult,
        )

    def _auto_execute_pump_fade_entries(
        self,
        *,
        n_open: int,
        per_sym_count: dict[str, int],
        short_count: int,
        cached_acct: dict[str, Any] | None,
        dd_risk_mult: float,
        now: int,
    ) -> tuple[dict[str, Any] | None, int, int]:
        """Market-short only the strongest parabolic pump-fade matches.

        This is intentionally separate from the legacy confluence strategy so
        enabling auto execution does not revive generic LONG/SHORT trading.
        """
        trading = self.cfg.trading
        if not getattr(trading, "auto_execute_pump_fade_shorts", False):
            return cached_acct, n_open, short_count

        min_conf = int(getattr(trading, "pump_fade_auto_min_confidence", 95))
        target_lev = int(getattr(trading, "pump_fade_auto_leverage", 100))
        overlays = self.state.overlay_snapshot()

        for sym in trading.symbols:
            if self.stop_flag or n_open >= trading.max_open_positions:
                break
            sym_u = sym.upper()
            if per_sym_count.get(sym_u, 0) >= trading.max_positions_per_symbol:
                continue
            if short_count >= trading.max_same_direction:
                self.state.record_skip(f"{sym_u}: same-direction cap ({short_count} shorts already)")
                continue

            mini_cd = self.mini_cooldown_until.get(sym_u, 0.0)
            if mini_cd and now < mini_cd:
                self.state.record_skip(
                    f"{sym_u}: 2-loss mini-cooldown — {int(mini_cd - now)}s left"
                )
                continue
            paused_until = self.streak_pause_until.get(sym_u, 0)
            if paused_until and now < paused_until:
                self.state.record_skip(
                    f"{sym_u}: streak-paused for {(paused_until - now) // 60}m more"
                )
                continue
            last = self.last_action_at.get(sym_u, 0)
            if now - last < trading.cooldown_seconds:
                continue

            overlay = overlays.get(sym_u) or {}
            decision = (
                overlay.get("decision")
                or overlay.get("recommendation")
                or overlay.get("next_1h")
                or overlay.get("next1h")
                or {}
            )
            action = str(decision.get("action") or "").lower()
            setup = str(decision.get("setup") or "")
            conf = int(decision.get("confidence_score") or decision.get("confidenceScore") or 0)
            if action != "short" or setup != "parabolic_pump_fade" or conf < min_conf:
                continue

            meta = self.metas.get(sym_u, _DEFAULT_META)
            max_spread_pct = self._max_entry_spread_pct_for_symbol(sym_u, meta)
            spread_pct = self.ob_feed.get_spread_pct(sym_u) if self.ob_feed else None
            if spread_pct is not None and spread_pct > max_spread_pct:
                self.state.record_skip(
                    f"{sym_u}: spread {spread_pct:.3f}% > "
                    f"{max_spread_pct:.3f}% threshold"
                )
                continue

            if cached_acct is None:
                try:
                    cached_acct = self.client.account()
                except Exception as e:
                    log.error("account fetch failed: %s", e)
                    self.state.record_error(f"account fetch failed: {e}")
                    return cached_acct, n_open, short_count
            free_margin = float(cached_acct.get("available") or 0)
            if free_margin <= 0:
                if not self.cfg.is_live:
                    free_margin = 1000.0
                else:
                    self.state.record_skip(f"{sym_u}: no available margin")
                    return cached_acct, n_open, short_count

            eff_lev = self._target_leverage_for_symbol(sym_u, meta, target_lev)
            plan = self._build_auto_pump_fade_plan(
                sym_u,
                overlay,
                decision,
                free_margin=free_margin,
                effective_leverage=eff_lev,
                dd_risk_mult=dd_risk_mult,
            )
            if plan is None:
                self.state.record_skip(f"{sym_u}: auto pump-fade risk manager rejected")
                continue

            min_notional = getattr(self.cfg.risk, "min_trade_notional", 0.0)
            trade_notional = plan.volume * plan.price
            if min_notional > 0 and trade_notional < min_notional:
                self.state.record_skip(
                    f"{sym_u}: notional ${trade_notional:.2f} < min ${min_notional:.2f} — fee drag too high"
                )
                continue

            if self.cfg.is_live:
                try:
                    self.client.set_leverage(sym_u, eff_lev)
                except BitunixError as e:
                    self.state.record_error(
                        f"{sym_u} auto pump-fade leverage set failed: {e.code} {e.msg}"
                    )
                    continue
                except Exception as e:
                    self.state.record_error(f"{sym_u} auto pump-fade leverage set failed: {e}")
                    continue

            self.state.record_signal(
                f"{sym_u} AUTO PUMP-FADE SHORT conf={conf}/100 "
                f"lev={eff_lev}x market @ {plan.price}"
            )
            if not self._execute(sym_u, plan, force_market=True):
                continue

            self.last_action_at[sym_u] = now
            per_sym_count[sym_u] = per_sym_count.get(sym_u, 0) + 1
            n_open += 1
            short_count += 1
            used_margin = (plan.volume * plan.price) / max(plan.leverage, 1)
            cached_acct["available"] = str(max(0.0, free_margin - used_margin))
            if n_open >= trading.max_open_positions:
                break

        return cached_acct, n_open, short_count

    def _tick(self) -> None:
        # 0. Update streak-loss state from newly-closed positions.
        self._update_streak_state()

        # 0a. Periodically broaden/narrow the scan universe to all liquid
        # Bitunix USDT perpetuals that pass the configured leverage/liquidity
        # filters. Existing manually configured symbols remain pinned.
        self._refresh_dynamic_symbols()

        # 0b. Refresh overlay scores for all symbols. Runs unconditionally so
        # the Chrome-extension overlay keeps updating even when the trading
        # loop is paused / capped / cooldowned for a symbol.
        self._compute_overlays()

        # 0c. Daily drawdown — gradual throttle on risk sizing as DD deepens,
        # full halt at the configured threshold. Existing positions still
        # managed normally.
        dd_risk_mult = self._daily_dd_risk_multiplier()
        dd_halted = dd_risk_mult <= 0.0

        # 0d. Liquidation-cascade check — halt new entries when BTC moves
        # rapidly enough that indicators are about to give a false "huge
        # trend" reading at the worst possible moment.
        cascade_halted = self._check_liquidation_cascade()

        # 1. Snapshot global state once per tick.
        all_open = [p for p in self.client.pending_positions() if float(p.get("qty") or 0) != 0]

        # 1b. Sweep post-only pending limits BEFORE position management:
        #   - if a position now exists for the symbol → entry filled, clear tracking
        #   - if timeout exceeded → cancel the limit and place market fallback
        # Runs only in live mode (paper mode never tracks pending limits).
        if self.cfg.is_live:
            self._check_pending_limits(all_open)

        # Optional time-based exit. Disabled when max_position_age_seconds is 0.
        # If enabled, time_exit_only_if_losing can restrict the sweep to losers.
        max_age = self.cfg.trading.max_position_age_seconds
        only_if_losing = self.cfg.trading.time_exit_only_if_losing
        if max_age > 0 and self.cfg.is_live:
            now_ms = int(time.time() * 1000)
            still_open = []
            for p in all_open:
                ctime = int(p.get("ctime") or 0)
                age_s = (now_ms - ctime) // 1000 if ctime else 0
                upnl = float(p.get("unrealizedPNL") or 0)
                if age_s >= max_age and (not only_if_losing or upnl < 0):
                    pid = str(p.get("positionId") or "")
                    sym = str(p.get("symbol") or "")
                    log.info("Force-close stale position %s (%s) age=%ss uPnL=%s",
                             pid, sym, age_s, upnl)
                    try:
                        self.client.flash_close_position(pid)
                        self.state.record_order(
                            f"{sym} TIME_EXIT positionId={pid} age={age_s}s uPnL={upnl:.4f}"
                        )
                    except BitunixError as e:
                        log.error("Force-close failed for %s: %s", pid, e)
                        self.state.record_error(f"{sym} time-exit failed: {e.code} {e.msg}")
                        still_open.append(p)
                else:
                    still_open.append(p)
            all_open = still_open

        # Dynamic SL management: ratchet stops forward once price moves favorably.
        # Skipped in paper mode (no real positions to manage).
        if all_open and self.cfg.is_live:
            self._manage_open_positions(all_open)

        n_open = len(all_open)
        per_sym_count: dict[str, int] = {}
        long_count = short_count = 0
        for p in all_open:
            s = str(p.get("symbol", "")).upper()
            per_sym_count[s] = per_sym_count.get(s, 0) + 1
            side = str(p.get("side", "")).upper()
            if side == "BUY" or side == "LONG":
                long_count += 1
            elif side == "SELL" or side == "SHORT":
                short_count += 1

        # Include pending post-only limits in cap math — they're "almost"
        # positions. Without this, a tick that just placed a maker limit would
        # think it has free slots and place ANOTHER order before the limit fills.
        for sym_pl, info in self.pending_limits.items():
            per_sym_count[sym_pl] = per_sym_count.get(sym_pl, 0) + 1
            side_pl = str(info["plan"].side).upper()
            if side_pl == "BUY":
                long_count += 1
            elif side_pl == "SELL":
                short_count += 1
        n_open += len(self.pending_limits)

        self.state.record_tick(None, len(all_open))

        if n_open >= self.cfg.trading.max_open_positions:
            log.debug("Global cap reached (%d/%d open); waiting", n_open, self.cfg.trading.max_open_positions)
            return

        # If daily drawdown OR liquidation cascade is active, no new entries.
        # Existing positions keep being managed (SL ratchet, partial TP, etc).
        if dd_halted or cascade_halted:
            return

        # Account fetched lazily — only when we're about to size an order.
        cached_acct: dict[str, Any] | None = None

        now = int(time.time())
        cached_acct, n_open, short_count = self._auto_execute_pump_fade_entries(
            n_open=n_open,
            per_sym_count=per_sym_count,
            short_count=short_count,
            cached_acct=cached_acct,
            dd_risk_mult=dd_risk_mult,
            now=now,
        )
        if n_open >= self.cfg.trading.max_open_positions:
            return
        if (
            getattr(self.cfg.trading, "auto_execute_pump_fade_shorts", False)
            and getattr(self.cfg.trading, "auto_execute_pump_fade_only", False)
        ):
            return

        for sym in self.cfg.trading.symbols:
            if self.stop_flag:
                return
            sym_u = sym.upper()

            # Per-symbol cap.
            if per_sym_count.get(sym_u, 0) >= self.cfg.trading.max_positions_per_symbol:
                continue

            # 2-loss mini-cooldown — early-intercept circuit breaker that
            # fires before the 3-loss streak pause. If 2 losses hit within
            # 10 minutes, pause that symbol for 5 minutes.
            mini_cd = self.mini_cooldown_until.get(sym_u, 0.0)
            if mini_cd and now < mini_cd:
                self.state.record_skip(
                    f"{sym}: 2-loss mini-cooldown — "
                    f"{int(mini_cd - now)}s left"
                )
                continue

            # Streak-loss circuit breaker — pause this symbol if it's hit
            # streak_loss_limit consecutive losses recently.
            paused_until = self.streak_pause_until.get(sym_u, 0)
            if paused_until and now < paused_until:
                # Report the trigger (running counter is reset to 0 when the
                # pause fires, so reading consec_losses here would always show
                # 0 and confuse log readers). Derive how long ago the pause
                # started from the pause's expiry time.
                pause_started_at = paused_until - self.cfg.trading.streak_loss_pause_seconds
                trigger_age_min = max(0, (now - pause_started_at) // 60)
                self.state.record_skip(
                    f"{sym}: streak-paused for {(paused_until - now) // 60}m more "
                    f"({self.cfg.trading.streak_loss_limit}-loss streak hit "
                    f"{trigger_age_min}m ago)"
                )
                continue

            # Cooldown.
            last = self.last_action_at.get(sym_u, 0)
            if now - last < self.cfg.trading.cooldown_seconds:
                continue

            # Spread filter — reject if order book spread is too wide.
            # Wide spreads cause adverse fills that eat the SL budget.
            meta = self.metas.get(sym_u, _DEFAULT_META)
            max_spread_pct = self._max_entry_spread_pct_for_symbol(sym_u, meta)
            spread_pct = self.ob_feed.get_spread_pct(sym) if self.ob_feed else None
            if spread_pct is not None and spread_pct > max_spread_pct:
                self.state.record_skip(
                    f"{sym}: spread {spread_pct:.3f}% > "
                    f"{max_spread_pct:.3f}% threshold"
                )
                continue

            # Depth filter — only relevant for post-only entries (thin
            # books cause limits to sit forever or get adversely selected).
            # When entering at market (Grok v8), top-of-book takeable
            # liquidity matters but the calibrated post-only threshold
            # is far too aggressive — skip the filter entirely.
            if self.cfg.trading.use_post_only_entries:
                min_depth = self.cfg.trading.symbol_min_depth.get(sym_u, 0.0)
                if min_depth > 0 and self.ob_feed is not None:
                    depth = self.ob_feed.get_depth(sym, top_n=5)
                    if depth is not None:
                        bid_d, ask_d = depth
                        thinnest = min(bid_d, ask_d)
                        if thinnest < min_depth:
                            self.state.record_skip(
                                f"{sym}: thin book ({thinnest:.1f} < {min_depth:.0f} "
                                f"min) — post-only would sit"
                            )
                            continue

            # Klines.
            try:
                rows = self.client.klines(sym, self.cfg.trading.timeframe,
                                          limit=self.cfg.loop.kline_lookback)
            except BitunixError as e:
                log.warning("%s klines failed: %s", sym, e)
                continue
            if len(rows) < 30:
                continue
            rows = sorted(rows, key=lambda r: int(r.get("time") or 0))
            last_bar = int(rows[-1].get("time") or 0)
            if last_bar and self.last_bar_ts.get(sym_u) == last_bar:
                # No fresh bar since we last looked — don't re-fire on the same candle.
                continue
            self.last_bar_ts[sym_u] = last_bar

            opens = [float(r["open"]) for r in rows]
            highs = [float(r["high"]) for r in rows]
            lows = [float(r["low"]) for r in rows]
            closes = [float(r["close"]) for r in rows]

            # Expansion-candle skip — when the most recent bar's range exceeds
            # 2.5× ATR, the next bar is statistically a fakeout/continuation
            # trap. Big bars are usually news-driven or liquidation cascades
            # and the bot's confluence-based signals trigger a chase entry
            # right at the wrong moment. Skip the immediate next bar.
            # Threshold raised 2.0→2.5 (Grok 2026-04-27): the 2.0× cutoff was
            # blocking 100% of bars on thin Sunday markets where ATR multiples
            # of 2.0-2.5× are normal, not extreme.
            atr_arr = atr_fn(np.array(highs), np.array(lows),
                             np.array(closes), self.cfg.strategy.atr_period)
            if len(atr_arr) > 0 and not np.isnan(atr_arr[-1]) and atr_arr[-1] > 0:
                last_range = highs[-1] - lows[-1]
                expansion_ratio = last_range / atr_arr[-1]
                if expansion_ratio >= 2.5:
                    self.state.record_skip(
                        f"{sym}: expansion candle "
                        f"({expansion_ratio:.1f}× ATR), skip next bar"
                    )
                    continue
            # Volume — Bitunix returns base coin volume per bar.
            volumes = [float(r.get("baseVol") or r.get("quoteVol") or 0) for r in rows]
            # Tier-2 inputs (cached): higher-timeframe trend + funding rate.
            htf_closes = self._get_htf_closes(sym)
            funding = self._get_funding_rate(sym)
            # Tier-3 input (live WebSocket): top-N order book imbalance.
            ob_imb = self.ob_feed.get_imbalance(sym) if self.ob_feed else None
            # Wave-3 inputs: BTC leader trend (None for BTC itself), session weight.
            btc_trend = (None if sym_u == self.cfg.strategy.btc_leader_symbol.upper()
                         else self._get_btc_trend())
            sess_w = self._session_weight()

            # Trade-tape inputs (real order flow alpha):
            #   real_cvd_60s          — 60s cumulative volume delta (base coin units)
            #   aggression_10s        — 10s aggression ratio in [-1, +1]
            #   activity_mult         — print-rate vs 5min baseline, clamped [0.85, 1.10]
            #   price_change_10s_pct  — 10s price change % (for absorption detector)
            real_cvd = aggression_10s = activity_mult = price_change_10s_pct = None
            if self.tape_feed is not None:
                real_cvd = self.tape_feed.get_cvd(sym, window_secs=60)
                aggression_10s = self.tape_feed.get_aggression_ratio(sym, window_secs=10)
                activity_mult = self.tape_feed.get_activity_multiplier(sym)
                price_change_10s_pct = self.tape_feed.get_price_change_pct(sym, window_secs=10)

            # Adaptive self-defense — adjust the BASE fire_threshold based
            # on the rolling-20 trade R-tally before regime adaptation.
            # Drawdown raises the bar; hot streaks ease it slightly.
            adaptive_adj = self._adaptive_threshold_adjustment()
            adaptive_base = max(0.0, min(1.0,
                self.cfg.strategy.fire_threshold + adaptive_adj))

            # Signal.
            sig = evaluate(
                opens, highs, lows, closes, self.cfg.strategy,
                volumes=volumes, htf_closes=htf_closes, funding_rate=funding,
                ob_imbalance=ob_imb,
                btc_trend=btc_trend, session_weight=sess_w,
                real_cvd=real_cvd, aggression_10s=aggression_10s,
                activity_mult=activity_mult,
                price_change_10s_pct=price_change_10s_pct,
                fire_threshold_override=adaptive_base,
            )
            if sig is None:
                self.state.record_skip(f"{sym}: evaluate() returned None (ADX below min or insufficient data)")
                continue
            sig_text = (f"{sym} {sig.direction.upper()} score={sig.score:.2f} "
                        f"(pat={sig.pattern_score:.1f}, "
                        f"T:{sig.factor_trend:.2f}/M:{sig.factor_mean_rev:.2f}/"
                        f"F:{sig.factor_flow:.2f}/C:{sig.factor_context:.2f}, "
                        f"sess={sess_w:.2f}) @ "
                        f"{sig.price:.4f} ({', '.join(sig.reasons)})")
            log.info("Signal: %s", sig_text)

            # Determine up-front if this signal will be blocked by any cap.
            # If so, record only the (deduped) skip — don't crowd the activity
            # feed with signal events that won't lead to an order.
            block_reason: str | None = None
            if sig.direction == "long" and long_count >= self.cfg.trading.max_same_direction:
                block_reason = f"{sym}: same-direction cap ({long_count} longs already)"
            elif sig.direction == "short" and short_count >= self.cfg.trading.max_same_direction:
                block_reason = f"{sym}: same-direction cap ({short_count} shorts already)"
            elif n_open >= self.cfg.trading.max_open_positions:
                block_reason = f"{sym}: global cap ({n_open}/{self.cfg.trading.max_open_positions})"

            if block_reason:
                self.state.record_skip(block_reason)
                continue

            # Actionable signal → record for the dashboard.
            self.state.record_signal(sig_text)

            # Risk plan.
            if cached_acct is None:
                try:
                    cached_acct = self.client.account()
                except Exception as e:
                    log.error("account fetch failed: %s", e)
                    self.state.record_error(f"account fetch failed: {e}")
                    return
            free_margin = float(cached_acct.get("available") or 0)
            if free_margin <= 0:
                if not self.cfg.is_live:
                    # Paper mode: pretend we have $1k so the dashboard shows
                    # what the bot WOULD do regardless of real balance.
                    free_margin = 1000.0
                else:
                    self.state.record_skip(f"{sym}: no available margin")
                    return  # no point checking other symbols in live mode

            meta = self.metas.get(sym, _DEFAULT_META)
            # Per-symbol effective leverage: cap config at the symbol's max.
            eff_lev = self._target_leverage_for_symbol(sym, meta, self.cfg.trading.leverage)
            plan = build_order(
                sig,
                free_margin=free_margin,
                trading=self.cfg.trading,
                risk=self.cfg.risk,
                min_volume=meta.min_qty,
                volume_step=meta.base_precision,
                digits=meta.price_precision,
                effective_leverage=eff_lev,
                symbol=sym,
                dd_risk_mult=dd_risk_mult,
            )
            if plan is None:
                self.state.record_skip(f"{sym}: risk manager rejected (volume below min)")
                continue

            # Minimum notional gate — skip trades where fee drag would eat all profit.
            # Sub-$5 notional at 0.1% round-trip = $0.005 fee; a 0.25% SL hit loses
            # more than any realistic TP gain. Saves the account from death-by-fees.
            min_notional = getattr(self.cfg.risk, "min_trade_notional", 0.0)
            trade_notional = plan.volume * plan.price
            if min_notional > 0 and trade_notional < min_notional:
                self.state.record_skip(
                    f"{sym}: notional ${trade_notional:.2f} < min ${min_notional:.2f} — fee drag too high"
                )
                continue

            # Post-signal ticker confirmation (Grok review v8). The signal
            # bar's close is the LAST data we used to evaluate. Before
            # placing the order, fetch the live ticker price and require
            # it to have moved in the trade direction since the bar close.
            # This catches "perfect signal bar that immediately reverses
            # on the next bar" — the exhaustion-fade pattern that the
            # in-bar continuation gate can't catch (it's looking at the
            # bar that just closed, not what's happening now).
            if getattr(self.cfg.strategy, "confirm_with_ticker", False):
                try:
                    ticker = self.client.ticker(sym)
                except BitunixError as e:
                    self.state.record_skip(
                        f"{sym}: ticker confirmation failed ({e.code} {e.msg})"
                    )
                    continue
                except Exception as e:
                    self.state.record_skip(
                        f"{sym}: ticker confirmation network error ({e})"
                    )
                    continue
                try:
                    live_px = float(ticker.get("lastPrice") or 0)
                except (TypeError, ValueError):
                    live_px = 0.0
                bar_close = float(closes[-1])
                if live_px <= 0 or bar_close <= 0:
                    self.state.record_skip(
                        f"{sym}: ticker price unavailable, no confirmation"
                    )
                    continue
                tol_pct = getattr(self.cfg.strategy, "confirmation_tolerance_pct", 0.0)
                tol = bar_close * (tol_pct / 100.0)
                if plan.side == "BUY" and live_px <= (bar_close - tol):
                    self.state.record_skip(
                        f"{sym}: ticker {live_px:.4f} ≤ signal close "
                        f"{bar_close:.4f} (tol={tol:.4f}) — no continuation up, drop long"
                    )
                    continue
                if plan.side == "SELL" and live_px >= (bar_close + tol):
                    self.state.record_skip(
                        f"{sym}: ticker {live_px:.4f} ≥ signal close "
                        f"{bar_close:.4f} (tol={tol:.4f}) — no continuation down, drop short"
                    )
                    continue
                # Ticker confirms direction. Re-derive SL/TP based on the
                # actual fill price (current ticker) so the R-geometry is
                # honored relative to where we're entering, not where the
                # signal triggered.
                rk = self.cfg.risk
                sl_pct = rk.stop_loss_pct
                if rk.use_atr and sig.atr > 0:
                    atr_pct_now = (sig.atr / live_px) * 100.0
                    sl_pct = max(sl_pct, rk.atr_multiplier_sl * atr_pct_now)
                sl_dist = live_px * (sl_pct / 100.0)
                tp_dist = sl_dist * rk.take_profit_r
                if plan.side == "BUY":
                    new_sl = round(live_px - sl_dist, meta.price_precision)
                    new_tp = round(live_px + tp_dist, meta.price_precision)
                else:
                    new_sl = round(live_px + sl_dist, meta.price_precision)
                    new_tp = round(live_px - tp_dist, meta.price_precision)
                # Replace the plan with one calibrated to the live price.
                from .risk import OrderPlan as _OP
                plan = _OP(
                    side=plan.side, volume=plan.volume,
                    price=round(live_px, meta.price_precision),
                    stop_loss=new_sl, take_profit=new_tp,
                    leverage=plan.leverage, notes=plan.notes + ",ticker_confirmed",
                )
                log.info("TICKER CONFIRMS %s %s: bar_close=%s live=%s "
                         "(%+.4f%%) — re-calibrated SL=%s TP=%s",
                         sym, plan.side, bar_close, live_px,
                         (live_px - bar_close) / bar_close * 100,
                         new_sl, new_tp)

            # Execute.
            placed = self._execute(sym, plan)
            if placed:
                # Journal: structured entry-context log for offline analysis.
                # Captures everything we'd want to correlate with outcome —
                # ADX, ATR, spread, depth, tape signals, conviction, sizing.
                conviction_mult = 1.0
                if sig.fire_threshold_used and sig.fire_threshold_used > 0:
                    conviction_mult = max(0.7, min(1.5,
                        sig.score / sig.fire_threshold_used))
                adx_arr = adx_fn(np.array(highs), np.array(lows),
                                 np.array(closes), self.cfg.strategy.adx_period)
                adx_now = (float(adx_arr[-1])
                           if len(adx_arr) > 0 and not np.isnan(adx_arr[-1])
                           else None)
                atr_pct_now = ((float(atr_arr[-1]) / plan.price * 100.0)
                               if len(atr_arr) > 0 and not np.isnan(atr_arr[-1])
                               and plan.price > 0 else None)
                depth_tup = (self.ob_feed.get_depth(sym, top_n=5)
                             if self.ob_feed else None)
                bid_depth = depth_tup[0] if depth_tup else None
                ask_depth = depth_tup[1] if depth_tup else None
                # clientId mirrors what _try_post_only / _place_market built.
                # For entries that went MARKET (post-only path off / not
                # connected), clientId omits "-PO". We use the same minute
                # bucket they did.
                minute_bucket = int(time.time()) // 60
                pl_info = self.pending_limits.get(sym_u)
                if pl_info is not None:
                    entry_mechanism = "MAKER_LIMIT_POST_ONLY"
                    limit_price_used = pl_info.get("limit_px")
                    dynamic_timeout = pl_info.get("timeout_secs")
                    tob_bid_at_entry = pl_info.get("tob_bid")
                    tob_ask_at_entry = pl_info.get("tob_ask")
                    cid_suffix = "-PO"
                    order_type_logged = "LIMIT"
                else:
                    entry_mechanism = "MARKET"
                    limit_price_used = None
                    dynamic_timeout = None
                    # Fall back to current top-of-book snapshot if the OB
                    # feed is connected (entry just happened so this is
                    # within ~1 tick of the order's market context).
                    tob_now = (self.ob_feed.get_top_of_book(sym)
                               if self.ob_feed else None)
                    tob_bid_at_entry = tob_now[0] if tob_now else None
                    tob_ask_at_entry = tob_now[1] if tob_now else None
                    cid_suffix = ""
                    order_type_logged = "MARKET"
                client_id_journal = f"bot-{sym}-{minute_bucket}-{plan.side}{cid_suffix}"
                self.journal.record_entry(
                    symbol=sym,
                    side=plan.side,
                    client_id=client_id_journal,
                    order_type=order_type_logged,
                    score=sig.score,
                    threshold_used=sig.fire_threshold_used,
                    conviction_mult=conviction_mult,
                    indicator_count=sig.indicator_score,
                    pattern_score=sig.pattern_score,
                    reasons=sig.reasons,
                    factor_trend=sig.factor_trend,
                    factor_mean_rev=sig.factor_mean_rev,
                    factor_flow=sig.factor_flow,
                    factor_context=sig.factor_context,
                    atr_pct=atr_pct_now,
                    adx=adx_now,
                    spread_pct=spread_pct,
                    bid_depth=bid_depth,
                    ask_depth=ask_depth,
                    aggression_10s=aggression_10s,
                    real_cvd=real_cvd,
                    activity_mult=activity_mult,
                    session_weight=sess_w,
                    adaptive_adj=adaptive_adj,
                    recent_trade_r_sum=(sum(self.recent_trade_r)
                                        if len(self.recent_trade_r) > 0 else None),
                    entry_mechanism=entry_mechanism,
                    limit_price=limit_price_used,
                    tob_bid=tob_bid_at_entry,
                    tob_ask=tob_ask_at_entry,
                    dynamic_timeout_secs=dynamic_timeout,
                    entry_price=plan.price,
                    stop_loss=plan.stop_loss,
                    take_profit=plan.take_profit,
                    notional=plan.volume * plan.price,
                    leverage=plan.leverage,
                )

                self.last_action_at[sym_u] = now
                per_sym_count[sym_u] = per_sym_count.get(sym_u, 0) + 1
                n_open += 1
                if sig.direction == "long":
                    long_count += 1
                else:
                    short_count += 1
                # Reduce optimistic free margin in cache so subsequent symbols
                # don't oversize against the same dollars.
                used_margin = (plan.volume * plan.price) / max(plan.leverage, 1)
                cached_acct["available"] = str(max(0.0, free_margin - used_margin))
                if n_open >= self.cfg.trading.max_open_positions:
                    log.info("Hit global cap %d after placing %s; halting tick",
                             n_open, sym)
                    return

    # ------------------------------------------------------------------ position management

    def _manage_open_positions(self, open_positions: list[dict[str, Any]]) -> None:
        """Delegate to PositionManager (extracted to its own module per
        Grok holistic review). Behavior unchanged; the wrapper preserves
        the existing method signature so test fixtures and external
        callers don't have to know about the split."""
        self.position_manager.manage(open_positions)


    # ------------------------------------------------------------------ entry execution
    #
    # Delegates to OrderExecutor (Grok holistic review — module split).
    # The wrapper methods preserve existing signatures so test fixtures
    # and call sites in _tick don't have to change. The legacy
    # implementations have been moved to bitunix_bot/order_executor.py.

    def _execute(self, symbol: str, plan: OrderPlan, *, force_market: bool = False) -> bool:
        return self.order_executor.execute(symbol, plan, force_market=force_market)

    def _check_pending_limits(self, all_open_positions: list[dict[str, Any]]) -> None:
        self.order_executor.check_pending_limits(all_open_positions)
