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
from .strategy import Signal, evaluate
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


@dataclass
class SymbolMeta:
    base_precision: float     # qty step (e.g. 0.001)
    price_precision: int      # digits for price (e.g. 1 for BTCUSDT)
    min_qty: float
    max_leverage: int = 100   # Bitunix caps differ per symbol


_DEFAULT_META = SymbolMeta(base_precision=0.001, price_precision=2, min_qty=0.001, max_leverage=100)


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
        # Partial-TP tracking: positionId set of those that already had
        # the +1R partial close fired. Prevents double-firing.
        self.partial_tp_done: set[str] = set()
        # Per-position max favorable R seen so far. Used by the stale-trade
        # early exit (cut trades that haven't moved in N minutes). Cleared
        # passively as positions disappear from pending_positions.
        self.position_max_favor: dict[str, float] = {}
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

            # Adaptive self-defense: append trade R to the rolling tally
            # before any other state updates. If this drives the tally below
            # the drawdown threshold, the next signal evaluation will see
            # a higher fire_threshold via _adaptive_threshold_adjustment.
            trade_r = self._compute_trade_r(p, self.cfg.risk.stop_loss_pct)
            self.recent_trade_r.append(trade_r)

            # Flat-trade detection. The BE ratchet at +1R favorable + price
            # reversal frequently produces "near-zero" exits where realized
            # loss ≈ fee/rebate offset → net within $0.001 of break-even.
            # These aren't real losses (the bot did its job — locked the
            # ratchet to BE — and the trade just didn't continue) and
            # shouldn't count toward the streak / mini-cooldown counters.
            # |trade_r| < FLAT_R_THRESHOLD = "essentially flat"
            FLAT_R_THRESHOLD = 0.10   # within ±10% of one R = flat
            is_flat = abs(trade_r) < FLAT_R_THRESHOLD

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
                max_favor_r=self.position_max_favor.get(pid_closed),
                net_pnl=net,
                realized_pnl=realized,
                fee=fee,
                funding=funding,
            )
            # Clear per-position state — the position is gone.
            self.position_max_favor.pop(pid_closed, None)
            self.partial_tp_done.discard(pid_closed)
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
        """Ratchet SL forward as price moves favorably.

        For each open position:
          - Derive current price from entry + unrealizedPNL/qty (no extra API call)
          - Compute R-multiple favorable: (current - entry) / original_sl_distance
          - If trailing region: SL = current_price ∓ trailing_distance_r × sl_dist
          - Else if break-even region: SL = entry ± buffer
          - Only update if new SL is MORE favorable than the existing SL
            (never moves a stop AGAINST you).
        """
        try:
            tpsl_rows = self.client.pending_tpsl()
        except Exception as e:
            log.warning("pending_tpsl fetch failed (skipping SL management): %s", e)
            return
        # Bitunix stores TPSL as separate trigger orders — find the SL-only
        # AND TP-only rows per position so we can target each by its own id.
        sl_orders: dict[str, dict[str, Any]] = {}    # positionId -> SL trigger row
        tp_orders: dict[str, dict[str, Any]] = {}    # positionId -> TP trigger row
        for r in tpsl_rows:
            pid = str(r.get("positionId") or "")
            if r.get("slPrice"):
                sl_orders[pid] = r
            if r.get("tpPrice"):
                tp_orders[pid] = r

        rk = self.cfg.risk
        for p in open_positions:
            try:
                pid = str(p.get("positionId") or "")
                if not pid:
                    continue
                symbol = str(p.get("symbol") or "")
                side = str(p.get("side") or "").upper()
                is_long = side in ("BUY", "LONG")
                entry = float(p.get("avgOpenPrice") or 0)
                qty = float(p.get("qty") or 0)
                upnl = float(p.get("unrealizedPNL") or 0)
                if entry <= 0 or qty <= 0:
                    continue

                # Derive current price from PnL.
                # long:  upnl = (current - entry) * qty   →   current = entry + upnl/qty
                # short: upnl = (entry - current) * qty   →   current = entry - upnl/qty
                price_delta = upnl / qty if qty else 0.0
                current_price = entry + price_delta if is_long else entry - price_delta

                sl_order = sl_orders.get(pid)
                if not sl_order:
                    continue  # no SL trigger order; can't safely manage
                current_sl = float(sl_order["slPrice"])
                sl_order_id = str(sl_order["id"])

                # Symbol meta needed early for both partial-TP qty quantization
                # and SL price rounding.
                meta = self.metas.get(symbol, _DEFAULT_META)

                # R-multiple is measured from the ORIGINAL config-based SL
                # distance, not the current SL — otherwise sl_distance shrinks
                # as we ratchet, and r_favor inflates artificially.
                sl_distance = entry * (rk.stop_loss_pct / 100.0)
                if sl_distance <= 0:
                    continue
                if is_long:
                    r_favor = (current_price - entry) / sl_distance
                else:
                    r_favor = (entry - current_price) / sl_distance

                # Track per-position max favorable R for the stale-trade exit.
                prev_max = self.position_max_favor.get(pid, float("-inf"))
                if r_favor > prev_max:
                    self.position_max_favor[pid] = r_favor

                # Stale-trade early exit — if a position has aged past
                # stale_exit_min without ever reaching stale_exit_max_favor_r,
                # flash-close it. A 1m scalp signal that hasn't moved in 6
                # minutes has lost its edge; pay the small loss and free the
                # slot. Distinct from time_exit_only_if_losing (90-min profit-
                # aware) and from tape-driven exit (immediate flow-flip).
                if rk.stale_exit_enabled:
                    ctime_ms_se = int(p.get("ctime") or 0)
                    age_min_se = (time.time() * 1000 - ctime_ms_se) / 60000.0 \
                                 if ctime_ms_se else 0
                    max_seen = self.position_max_favor.get(pid, r_favor)
                    if (age_min_se >= rk.stale_exit_min
                            and max_seen < rk.stale_exit_max_favor_r):
                        try:
                            self.client.flash_close_position(pid)
                            log.info("STALE EXIT %s: age=%.1fm max_favor=%.2fR "
                                     "current=%.2fR", symbol, age_min_se,
                                     max_seen, r_favor)
                            self.state.record_order(
                                f"{symbol} STALE_EXIT positionId={pid} "
                                f"age={age_min_se:.0f}m max={max_seen:+.2f}R"
                            )
                            continue
                        except BitunixError as e:
                            log.warning("Stale exit failed for %s: %s",
                                        symbol, e)
                            # Fall through to normal management.

                # Tape-driven exit — DISABLED by default after live data
                # showed it firing within 10-15 seconds of entry on
                # microstructure noise, closing trades at 25-50% of full
                # SL distance instead of letting them develop. Aggression
                # naturally swings in 10s windows; the ±0.50 threshold
                # catches normal noise, not real regime flips.
                #
                # Re-enable via cfg.risk.tape_exit_enabled=True only after
                # you've reviewed the journal and tuned threshold +
                # min_hold_secs guard. The min-hold prevents sub-30s
                # exits where the tape barely settled after the maker fill.
                if (rk.tape_exit_enabled
                        and r_favor < 1.0
                        and self.tape_feed is not None):
                    # Min hold gate — let the entry settle before measuring flow.
                    ctime_ms_te = int(p.get("ctime") or 0)
                    age_s_te = (time.time() * 1000 - ctime_ms_te) / 1000.0 \
                               if ctime_ms_te else 0
                    min_hold = float(getattr(rk, "tape_exit_min_hold_secs", 30))
                    threshold = float(getattr(rk, "tape_exit_threshold", 0.50))
                    if age_s_te >= min_hold:
                        agg = self.tape_feed.get_aggression_ratio(symbol, window_secs=10)
                        if agg is not None:
                            flipped = ((is_long and agg <= -threshold)
                                       or (not is_long and agg >= threshold))
                            if flipped:
                                try:
                                    self.client.flash_close_position(pid)
                                    log.info("TAPE EXIT %s %s: flow flipped "
                                             "(agg=%+.2f, r=%.2f, age=%.0fs)",
                                             symbol, "LONG" if is_long else "SHORT",
                                             agg, r_favor, age_s_te)
                                    self.state.record_order(
                                        f"{symbol} TAPE_EXIT positionId={pid} "
                                        f"agg={agg:+.2f} r={r_favor:.2f}"
                                    )
                                    continue   # skip SL/TP management for closed pos
                                except BitunixError as e:
                                    log.warning("Tape exit failed for %s: %s",
                                            symbol, e)
                                # Fall through to normal SL management.

                # Partial TP at +1R favorable — close partial_tp_close_pct% of the
                # position at market when r_favor first reaches partial_tp_at_r.
                # Locks in a guaranteed fee-clearing profit on half the position
                # while the runner trails toward the full TP target. Done ONCE
                # per position; tracked in self.partial_tp_done.
                if (rk.partial_tp_enabled
                        and r_favor >= rk.partial_tp_at_r
                        and pid not in self.partial_tp_done):
                    partial_qty = qty * (rk.partial_tp_close_pct / 100.0)
                    # Quantize to symbol's step.
                    step = meta.base_precision
                    n_steps = int(partial_qty / step) if step > 0 else 0
                    partial_qty_q = round(n_steps * step, 6)
                    if partial_qty_q >= meta.min_qty:
                        opp_side = "SELL" if is_long else "BUY"
                        try:
                            self.client.place_order(
                                symbol=symbol,
                                side=opp_side,
                                qty=str(partial_qty_q),
                                order_type="MARKET",
                                trade_side="CLOSE",
                                reduce_only=True,
                                client_id=f"ptp-{pid}",
                            )
                            self.partial_tp_done.add(pid)
                            log.info("Partial TP fired %s: closed %s @ market (r=%.2f)",
                                     symbol, partial_qty_q, r_favor)
                            self.state.record_order(
                                f"{symbol} PARTIAL_TP closed {partial_qty_q} "
                                f"({rk.partial_tp_close_pct:.0f}% of position) at r={r_favor:.2f}"
                            )
                        except BitunixError as e:
                            log.warning("Partial TP failed for %s: %s", symbol, e)
                            self.partial_tp_done.add(pid)  # avoid retry storm

                # Decide new SL: trailing region wins over break-even region.
                new_sl: float | None = None
                reason = ""
                if rk.trailing_activate_r > 0 and r_favor >= rk.trailing_activate_r:
                    trail_dist = rk.trailing_distance_r * sl_distance
                    new_sl = (current_price - trail_dist) if is_long else (current_price + trail_dist)
                    reason = f"trail (r={r_favor:.2f})"
                elif rk.breakeven_at_r > 0 and r_favor >= rk.breakeven_at_r:
                    buffer = entry * (rk.breakeven_buffer_pct / 100.0)
                    new_sl = (entry + buffer) if is_long else (entry - buffer)
                    reason = f"breakeven (r={r_favor:.2f})"

                if new_sl is None:
                    continue

                # Only ratchet forward — never move SL against the position.
                if is_long and new_sl <= current_sl:
                    continue
                if (not is_long) and new_sl >= current_sl:
                    continue

                # Round to symbol's price precision.
                meta = self.metas.get(symbol, _DEFAULT_META)
                new_sl_rounded = round(new_sl, meta.price_precision)
                if (is_long and new_sl_rounded <= current_sl) or \
                   ((not is_long) and new_sl_rounded >= current_sl):
                    continue

                # Push update — modify the SPECIFIC SL trigger order by its id.
                # The position-level modify endpoint doesn't actually persist;
                # this targets the SL trigger directly.
                #
                # Race-condition fix (Grok review, Option C): r_favor is the
                # MAX favor reached, but current_price for SL placement is
                # the LIVE ticker. So a trade can hit r=0.37 (qualifying for
                # BE) while live price has retraced to r=0.21 — the
                # entry-buffer SL we just computed lands on the wrong side
                # of last price. Bitunix correctly rejects with code 30030.
                # On that specific failure, retry with SL clamped to
                # current_price ± 1 tick — locks in whatever profit is
                # still on the table at the moment of the API call.
                try:
                    self.client.modify_tpsl_order(
                        order_id=sl_order_id,
                        sl_price=str(new_sl_rounded),
                        sl_qty=sl_order.get("slQty"),
                        sl_stop_type=sl_order.get("slStopType") or "LAST_PRICE",
                        sl_order_type=sl_order.get("slOrderType") or "MARKET",
                    )
                    log.info(
                        "SL ratchet %s: %s → %s (%s, entry=%s, current=%.6f)",
                        symbol, current_sl, new_sl_rounded, reason, entry, current_price,
                    )
                    self.state.record_order(
                        f"{symbol} SL {current_sl} → {new_sl_rounded} ({reason})"
                    )
                except BitunixError as e_inner:
                    # Bitunix code 30030: "SL price must be greater/less than
                    # last price" — the intended new_sl is on the wrong side
                    # of live price due to retracement. Retry with a tight
                    # clamp at current_price ± 1 tick (still better than
                    # original SL because we already passed the ratchet check).
                    if e_inner.code != 30030:
                        raise
                    tick = 10 ** -meta.price_precision
                    clamp_sl = (current_price - tick) if is_long else (current_price + tick)
                    clamp_sl_rounded = round(clamp_sl, meta.price_precision)
                    # Still must ratchet forward — if even the clamp isn't
                    # better than current_sl, skip rather than move backward.
                    if (is_long and clamp_sl_rounded <= current_sl) or \
                       ((not is_long) and clamp_sl_rounded >= current_sl):
                        log.info(
                            "SL clamp skipped %s: %s not better than current %s "
                            "(price retraced past ratchet point)",
                            symbol, clamp_sl_rounded, current_sl,
                        )
                        continue
                    self.client.modify_tpsl_order(
                        order_id=sl_order_id,
                        sl_price=str(clamp_sl_rounded),
                        sl_qty=sl_order.get("slQty"),
                        sl_stop_type=sl_order.get("slStopType") or "LAST_PRICE",
                        sl_order_type=sl_order.get("slOrderType") or "MARKET",
                    )
                    log.info(
                        "SL clamped %s: %s → %s (current=%.6f, %s — retraced past intended %s)",
                        symbol, current_sl, clamp_sl_rounded, current_price,
                        reason, new_sl_rounded,
                    )
                    self.state.record_order(
                        f"{symbol} SL {current_sl} → {clamp_sl_rounded} "
                        f"({reason}, clamped from {new_sl_rounded})"
                    )
            except BitunixError as e_sl:
                # SL ratchet may benignly fail if the position closed.
                msg = (e_sl.msg or "").lower()
                benign = any(s in msg for s in
                             ("not found", "not exist", "closed", "expired", "canceled"))
                if not benign:
                    log.warning("SL update failed for %s: %s", p.get("symbol"), e_sl)
                    self.state.record_error(f"SL update failed: {e_sl.code} {e_sl.msg}")
                continue   # skip TP adjustment too — position likely gone

            # ---------- ADAPTIVE TP TIGHTENING ----------
            # As the trade ages, ratchet TP DOWN (closer to current price)
            # toward a fee-aware floor. Always profitable; never below
            # round-trip fees. Original TP only loosens, never tightens
            # past floor.
            try:
                if not rk.adaptive_tp_enabled:
                    continue
                tp_order = tp_orders.get(pid)
                if not tp_order:
                    continue  # position has no TP trigger — skip
                current_tp = float(tp_order["tpPrice"])
                tp_order_id = str(tp_order["id"])
                # Position age in minutes from ctime (Bitunix ms timestamp).
                ctime_ms = int(p.get("ctime") or 0)
                age_min = (time.time() * 1000 - ctime_ms) / 60000.0 if ctime_ms else 0
                desired_r = adaptive_tp_r(
                    age_minutes=age_min,
                    original_tp_r=rk.take_profit_r,
                    fee_pct=rk.round_trip_fee_pct,
                    sl_pct=rk.stop_loss_pct,
                    floor_r=rk.adaptive_tp_floor_r,
                )
                desired_tp = (entry + desired_r * sl_distance) if is_long \
                             else (entry - desired_r * sl_distance)
                desired_tp_rounded = round(desired_tp, meta.price_precision)
                # Only TIGHTEN (move TP toward entry). Never expand TP.
                tighter = (desired_tp_rounded < current_tp) if is_long \
                          else (desired_tp_rounded > current_tp)
                if not tighter:
                    continue
                # And only tighten meaningfully (>= 1 price tick).
                if abs(desired_tp_rounded - current_tp) < 10 ** (-meta.price_precision):
                    continue
                self.client.modify_tpsl_order(
                    order_id=tp_order_id,
                    tp_price=str(desired_tp_rounded),
                    tp_qty=tp_order.get("tpQty"),
                    tp_stop_type=tp_order.get("tpStopType") or "LAST_PRICE",
                    tp_order_type=tp_order.get("tpOrderType") or "MARKET",
                )
                log.info("TP tighten %s: %s → %s (age=%.1fm desired=%.2fR)",
                         symbol, current_tp, desired_tp_rounded, age_min, desired_r)
                self.state.record_order(
                    f"{symbol} TP {current_tp} → {desired_tp_rounded} "
                    f"(age={age_min:.0f}m, target={desired_r:.2f}R)"
                )
            except BitunixError as e:
                # "Order not found" / "position closed" between our pending_tpsl
                # read and the modify call is benign — the position closed
                # naturally (TP/SL fired). Don't crowd the activity log with errors.
                msg = (e.msg or "").lower()
                benign = any(s in msg for s in
                             ("not found", "not exist", "closed", "expired", "canceled"))
                if benign:
                    log.info("SL update skipped for %s (position closed): %s",
                             p.get("symbol"), e.msg)
                else:
                    log.warning("SL update failed for %s: %s", p.get("symbol"), e)
                    self.state.record_error(f"SL update failed: {e.code} {e.msg}")
            except Exception as e:
                log.exception("Position management error for %s: %s", p.get("symbol"), e)

    def _execute(self, symbol: str, plan: OrderPlan) -> bool:
        # Tape veto — "don't fight the flow". If the most recent 10s of trade
        # tape is contrary to our intended direction at ≥0.30 magnitude
        # (~65/35 split or worse), skip the trade. Threshold tightened to
        # ±0.30 (Grok review v7) after live data showed Trade #1 firing
        # SHORT with agg=+0.342 — slipped through the previous ±0.45 gate
        # then immediately reversed for a loss. Lagging trend indicators
        # were drowning out the leading flow signal. At ±0.30 we trade
        # less but with dramatically higher signal quality.
        if self.tape_feed is not None:
            agg = self.tape_feed.get_aggression_ratio(symbol, window_secs=10)
            if agg is not None:
                if plan.side == "BUY" and agg <= -0.30:
                    self.state.record_skip(
                        f"{symbol}: tape veto — long signal but {agg:+.2f} sell flow"
                    )
                    return False
                if plan.side == "SELL" and agg >= 0.30:
                    self.state.record_skip(
                        f"{symbol}: tape veto — short signal but {agg:+.2f} buy flow"
                    )
                    return False

        prefix = "LIVE" if self.cfg.is_live else "PAPER"
        order_text = (f"{prefix} {symbol} {plan.side} qty={plan.volume} "
                      f"entry~{plan.price} SL={plan.stop_loss} TP={plan.take_profit} "
                      f"lev={plan.leverage}x")
        log.info("ORDER %s [%s]", order_text, plan.notes)
        if not self.cfg.is_live:
            self.state.record_order(order_text + " (paper)")
            return True

        # Try post-only maker entry first (saves ~0.04% per round-trip in fees).
        # If the OB feed isn't ready or post-only is rejected, fall through to
        # market. The pending limit is tracked; _check_pending_limits will
        # cancel + market-fallback if it doesn't fill within timeout.
        if (self.cfg.trading.use_post_only_entries
                and self.ob_feed
                and self.ob_feed.is_connected()):
            if self._try_post_only(symbol, plan, order_text):
                return True
            log.info("Post-only path failed/skipped for %s; using market", symbol)

        return self._place_market(symbol, plan, order_text)

    def _try_post_only(self, symbol: str, plan: OrderPlan, order_text: str) -> bool:
        """Attempt a POST_ONLY limit entry at top-of-book. Returns True if
        successfully placed (tracked for timeout sweep), False on any failure
        so caller can fall through to market."""
        sym_u = symbol.upper()
        tob = self.ob_feed.get_top_of_book(sym_u)
        if not tob:
            return False
        bid, ask = tob
        is_long = plan.side == "BUY"
        meta = self.metas.get(symbol, _DEFAULT_META)
        # Long → place at best bid (rest on the book as a maker).
        # Short → place at best ask.
        limit_px = round(bid if is_long else ask, meta.price_precision)
        # If our chosen price would already be a taker (rare race), bail to market.
        if (is_long and limit_px > ask) or ((not is_long) and limit_px < bid):
            return False

        # Dynamic timeout — high tape activity = price moves fast = either
        # fill quickly or skip. Dead market = give the limit more time to
        # be hit. Inverted scaling so high activity → shorter timeout.
        # Wider clamp range here ([0.5, 2.0]) than the score-multiplier
        # version ([0.85, 1.10]) so we get meaningful timeout variation.
        base_timeout = self.cfg.trading.post_only_timeout_secs
        timeout_secs = base_timeout
        if self.tape_feed is not None:
            activity = self.tape_feed.get_activity_multiplier(
                sym_u, clamp_min=0.5, clamp_max=2.0
            )
            if activity is not None and activity > 0:
                # High activity → shorter timeout. Clamp final value to [4, 12]s.
                timeout_secs = max(4, min(12, int(round(base_timeout / activity))))

        minute_bucket = int(time.time()) // 60
        client_id = f"bot-{symbol}-{minute_bucket}-{plan.side}-PO"
        try:
            resp = self.client.place_order(
                symbol=symbol,
                side=plan.side,
                qty=str(plan.volume),
                order_type="LIMIT",
                price=str(limit_px),
                trade_side="OPEN",
                tp_price=str(plan.take_profit),
                sl_price=str(plan.stop_loss),
                client_id=client_id,
            )
            order_id = resp.get("orderId")
            if not order_id:
                return False
            self.pending_limits[sym_u] = {
                "symbol": symbol,
                "order_id": str(order_id),
                "place_ts": int(time.time()),
                "plan": plan,
                "order_text": order_text,
                "limit_px": limit_px,
                "timeout_secs": timeout_secs,
                # Top-of-book snapshot at placement — feeds the journal so
                # downstream analysis can spot adverse-selection patterns
                # (limit always at the bid/ask side that price walks away from).
                "tob_bid": float(bid),
                "tob_ask": float(ask),
            }
            log.info("MAKER %s LIMIT @ %s POST_ONLY orderId=%s timeout=%ds",
                     symbol, limit_px, order_id, timeout_secs)
            self.state.record_order(
                f"{symbol} MAKER {plan.side} qty={plan.volume} @ {limit_px} "
                f"(POST_ONLY, t/o={timeout_secs}s)"
            )
            return True
        except BitunixError as e:
            # POST_ONLY rejection (would-cross), validation, etc — fall through.
            log.info("Post-only rejected for %s: %s — falling back to market", symbol, e.msg)
            return False
        except Exception as e:
            log.warning("Post-only network error for %s: %s", symbol, e)
            return False

    def _place_market(self, symbol: str, plan: OrderPlan, order_text: str) -> bool:
        """Market entry with deterministic clientId and full error handling."""
        minute_bucket = int(time.time()) // 60
        client_id = f"bot-{symbol}-{minute_bucket}-{plan.side}"
        try:
            resp = self.client.place_order(
                symbol=symbol,
                side=plan.side,
                qty=str(plan.volume),
                order_type="MARKET",
                trade_side="OPEN",
                tp_price=str(plan.take_profit),
                sl_price=str(plan.stop_loss),
                client_id=client_id,
            )
            log.info("Placed orderId=%s clientId=%s", resp.get("orderId"), resp.get("clientId"))
            self.state.record_order(f"{order_text} → orderId={resp.get('orderId')}")
            return True
        except BitunixError as e:
            log.error("Order rejected: %s (payload=%s)", e, e.payload)
            self.state.record_error(f"{symbol} order rejected: {e.code} {e.msg}")
            return False
        except Exception as e:
            log.error("place_order network/unknown failure for %s: %s", symbol, e)
            self.state.record_error(f"{symbol} order failure (unknown state): {e}")
            return False

    def _check_pending_limits(self, all_open_positions: list[dict[str, Any]]) -> None:
        """Sweep pending post-only limit entries:
          - if a position now exists for the symbol → entry filled, clear tracking
          - if timeout exceeded → cancel limit and skip (no market fallback)
        """
        if not self.pending_limits:
            return
        now = int(time.time())
        open_by_sym = {str(p.get("symbol", "")).upper() for p in all_open_positions}
        for sym_u, info in list(self.pending_limits.items()):
            # Filled? A position now exists for this symbol.
            if sym_u in open_by_sym:
                log.info("MAKER fill confirmed for %s (orderId=%s)", sym_u, info["order_id"])
                self.state.record_order(f"{sym_u} MAKER FILLED orderId={info['order_id']}")
                del self.pending_limits[sym_u]
                continue
            # Per-order timeout (computed at placement time from activity).
            timeout = info.get("timeout_secs", self.cfg.trading.post_only_timeout_secs)
            # Timed out → cancel and SKIP. Do NOT market-fallback.
            #
            # Pro-desk rule: if your maker bid wasn't hit in the timeout window,
            # the price moved AWAY from your bid. For a long, that means price
            # went UP — marketing in now means CHASING, paying the worst possible
            # price on a setup that already invalidated. Trust the next signal.
            #
            # The cooldown gets refreshed so the per-symbol loop later in this
            # tick doesn't immediately re-fire the same trade.
            age = now - info["place_ts"]
            if age >= timeout:
                try:
                    self.client.cancel_order(symbol=info["symbol"], order_id=info["order_id"])
                except Exception as e:
                    log.warning("Cancel pending limit for %s failed: %s", sym_u, e)
                log.info("MAKER timeout %s after %ds — signal failed, skip "
                         "(no market fallback)", sym_u, age)
                self.state.record_skip(
                    f"{sym_u}: post-only didn't fill in {age}s — signal invalidated"
                )
                del self.pending_limits[sym_u]
                # Refresh cooldown so we don't re-fire on the same bar.
                self.last_action_at[sym_u] = now
