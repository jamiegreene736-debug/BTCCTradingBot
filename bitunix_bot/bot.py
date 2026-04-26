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
from .position_manager import PositionManager
from .strategy import Signal, evaluate
from .symbol_meta import DEFAULT_META as _DEFAULT_META
from .symbol_meta import SymbolMeta
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

    # ------------------------------------------------------------------ setup

    def _resolve_symbol_meta(self) -> None:
        try:
            pairs = self.client.trading_pairs()
        except Exception as e:
            log.warning("trading_pairs() failed: %s — using defaults for all", e)
            for s in self.cfg.trading.symbols:
                self.metas[s] = _DEFAULT_META
            return

        by_name = {str(r.get("symbol", "")).upper(): r for r in pairs}
        for sym in self.cfg.trading.symbols:
            row = by_name.get(sym.upper())
            if not row:
                log.warning("Symbol %s not in trading_pairs; using defaults", sym)
                self.metas[sym] = _DEFAULT_META
                continue
            raw_base = row.get("basePrecision")
            if isinstance(raw_base, (int, float)) and raw_base >= 1:
                base_step = 10 ** (-int(raw_base))
            else:
                base_step = float(raw_base) if raw_base else 0.001
            price_prec = int(row.get("quotePrecision") or row.get("pricePrecision") or 2)
            min_qty = float(row.get("minTradeVolume") or base_step)
            max_lev = int(row.get("maxLeverage") or 100)
            self.metas[sym] = SymbolMeta(base_step, price_prec, min_qty, max_lev)
            log.info("Meta %s: step=%s priceDigits=%s minQty=%s maxLev=%s",
                     sym, base_step, price_prec, min_qty, max_lev)

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
            eff_lev = min(self.cfg.trading.leverage, meta.max_leverage)
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
        try:
            open_px = float(p.get("avgOpenPrice") or 0)
            qty = float(p.get("qty") or 0)
            realized = float(p.get("realizedPNL") or 0)
            fee = float(p.get("fee") or 0)
            funding = float(p.get("funding") or 0)
        except (TypeError, ValueError):
            return 0.0
        if open_px <= 0 or qty <= 0 or sl_pct_default <= 0:
            return 0.0
        sl_dist = open_px * sl_pct_default / 100.0
        risk_dollars = qty * sl_dist
        if risk_dollars <= 0:
            return 0.0
        return (realized + fee + funding) / risk_dollars

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
            # Net PnL = realizedPNL + fee + funding (Bitunix excludes fees from
            # realizedPNL per spec; fee/funding are signed).
            def _f(v):
                try:
                    return float(v) if v not in (None, "", "null") else 0.0
                except (ValueError, TypeError):
                    return 0.0
            realized = _f(p.get("realizedPNL"))
            fee = _f(p.get("fee"))
            funding = _f(p.get("funding"))
            net = realized + fee + funding

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
            entry_px = _f(p.get("avgOpenPrice"))
            exit_px = _f(p.get("avgClosePrice")) or None
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

    def _tick(self) -> None:
        # 0. Update streak-loss state from newly-closed positions.
        self._update_streak_state()

        # 0b. Daily drawdown — gradual throttle on risk sizing as DD deepens,
        # full halt at the configured threshold. Existing positions still
        # managed normally.
        dd_risk_mult = self._daily_dd_risk_multiplier()
        dd_halted = dd_risk_mult <= 0.0

        # 0c. Liquidation-cascade check — halt new entries when BTC moves
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

        # Time-based exit: force-close stale positions, but only if the
        # position is at a loss when time_exit_only_if_losing=True. Winners
        # past max-age stay alive under the SL ratchet — proven signals
        # deserve the chance to harvest more profit.
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
            spread_pct = self.ob_feed.get_spread_pct(sym) if self.ob_feed else None
            if spread_pct is not None and spread_pct > self.cfg.trading.max_entry_spread_pct:
                self.state.record_skip(
                    f"{sym}: spread {spread_pct:.3f}% > "
                    f"{self.cfg.trading.max_entry_spread_pct:.3f}% threshold"
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
            # 2× ATR, the next bar is statistically a fakeout/continuation
            # trap. Big bars are usually news-driven or liquidation cascades
            # and the bot's confluence-based signals trigger a chase entry
            # right at the wrong moment. Skip the immediate next bar.
            atr_arr = atr_fn(np.array(highs), np.array(lows),
                             np.array(closes), self.cfg.strategy.atr_period)
            if len(atr_arr) > 0 and not np.isnan(atr_arr[-1]) and atr_arr[-1] > 0:
                last_range = highs[-1] - lows[-1]
                expansion_ratio = last_range / atr_arr[-1]
                if expansion_ratio >= 2.0:
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
            eff_lev = min(self.cfg.trading.leverage, meta.max_leverage)
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
                if plan.side == "BUY" and live_px <= bar_close:
                    self.state.record_skip(
                        f"{sym}: ticker {live_px:.4f} ≤ signal close "
                        f"{bar_close:.4f} — no continuation up, drop long"
                    )
                    continue
                if plan.side == "SELL" and live_px >= bar_close:
                    self.state.record_skip(
                        f"{sym}: ticker {live_px:.4f} ≥ signal close "
                        f"{bar_close:.4f} — no continuation down, drop short"
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

    def _execute(self, symbol: str, plan: OrderPlan) -> bool:
        return self.order_executor.execute(symbol, plan)

    def _check_pending_limits(self, all_open_positions: list[dict[str, Any]]) -> None:
        self.order_executor.check_pending_limits(all_open_positions)

