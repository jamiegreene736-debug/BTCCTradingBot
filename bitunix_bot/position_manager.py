"""Position lifecycle management — SL ratchet, BE protect, trailing,
stale exit, tape-driven exit, partial TP, adaptive TP tightening.

Extracted from bot.py (Grok holistic review item) to give the position-
management subsystem a clear module boundary. The class holds no state
of its own — it's a behavior bundle that operates on the BitunixBot's
state. The bot is referenced as `self._bot` so all the existing
attributes (client, state, journal, cfg, metas, position_max_favor,
partial_tp_done, tape_feed) remain accessible via composition.

Design choice: pure delegation, no state migration. The risk dicts
(`position_max_favor`, `partial_tp_done`) stay on the bot so other call
sites don't have to change. The PositionManager just operates on them
through the bot reference.
"""
from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING, Any

from .client import BitunixError
from .risk import adaptive_tp_r, parse_timeframe_minutes
from .symbol_meta import DEFAULT_META as _DEFAULT_META

if TYPE_CHECKING:
    from .bot import BitunixBot

log = logging.getLogger(__name__)


class PositionManager:
    """Wraps position-management lifecycle behavior."""

    def __init__(self, bot: "BitunixBot") -> None:
        self._bot = bot
        # Per-position max favorable R seen so far. Used by the stale-trade
        # early exit (cut trades that haven't moved in N minutes) and read
        # by _tick when a position closes to record max_favor in the journal.
        # Cleared via clear_position_state(pid) when a position disappears
        # from pending_positions.
        self.position_max_favor: dict[str, float] = {}
        # Partial-TP tracking: positionId set of those that already had
        # the +Nx partial close fired. Prevents double-firing.
        self.partial_tp_done: set[str] = set()

    # ------------------------------------------------------------------
    # State accessors (used from bot.py during position-close cleanup
    # and journal recording)
    # ------------------------------------------------------------------

    def get_max_favor(self, pid: str) -> float | None:
        """Return the max favorable R seen on this position, or None
        if never tracked."""
        return self.position_max_favor.get(pid)

    def clear_position_state(self, pid: str) -> None:
        """Clear all per-position state when a position disappears
        (closed via SL/TP/manual). Idempotent — safe to call on
        already-cleared pids."""
        self.position_max_favor.pop(pid, None)
        self.partial_tp_done.discard(pid)

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def manage(self, open_positions: list[dict[str, Any]]) -> None:
        """Ratchet SL forward as price moves favorably.

        For each open position:
          - Derive current price from entry + unrealizedPNL/qty (no extra API call)
          - Compute R-multiple favorable: (current - entry) / original_sl_distance
          - Stale-exit if aged past stale_exit_min without reaching threshold favor
          - Tape-exit if flow flips (opt-in)
          - Partial TP at +1R if enabled
          - Trailing region: SL = current_price ∓ trailing_distance_r × sl_dist
          - BE region: SL = entry ± buffer
          - Adaptive TP tightening over time
          - Only ratchet SL forward — never against the position
        """
        bot = self._bot
        try:
            tpsl_rows = bot.client.pending_tpsl()
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

        rk = bot.cfg.risk
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
                meta = bot.metas.get(symbol, _DEFAULT_META)

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
                # flash-close it. Distinct from time_exit_only_if_losing
                # (long-window profit-aware) and from tape-driven exit
                # (immediate, flow-flip based).
                if rk.stale_exit_enabled:
                    ctime_ms_se = int(p.get("ctime") or 0)
                    age_min_se = (time.time() * 1000 - ctime_ms_se) / 60000.0 \
                                 if ctime_ms_se else 0
                    max_seen = self.position_max_favor.get(pid, r_favor)
                    if (age_min_se >= rk.stale_exit_min
                            and max_seen < rk.stale_exit_max_favor_r):
                        try:
                            bot.client.flash_close_position(pid)
                            log.info("STALE EXIT %s: age=%.1fm max_favor=%.2fR "
                                     "current=%.2fR", symbol, age_min_se,
                                     max_seen, r_favor)
                            bot.state.record_order(
                                f"{symbol} STALE_EXIT positionId={pid} "
                                f"age={age_min_se:.0f}m max={max_seen:+.2f}R"
                            )
                            continue
                        except BitunixError as e:
                            log.warning("Stale exit failed for %s: %s",
                                        symbol, e)
                            # Fall through to normal management.

                # Tape-driven exit — opt-in. Catches regime-flip via 10s
                # tape aggression. Has min-hold guard so we don't kill
                # entries before the maker fill settles.
                if (rk.tape_exit_enabled
                        and r_favor < 1.0
                        and bot.tape_feed is not None):
                    ctime_ms_te = int(p.get("ctime") or 0)
                    age_s_te = (time.time() * 1000 - ctime_ms_te) / 1000.0 \
                               if ctime_ms_te else 0
                    min_hold = float(getattr(rk, "tape_exit_min_hold_secs", 30))
                    threshold = float(getattr(rk, "tape_exit_threshold", 0.50))
                    if age_s_te >= min_hold:
                        agg = bot.tape_feed.get_aggression_ratio(symbol, window_secs=10)
                        if agg is not None:
                            flipped = ((is_long and agg <= -threshold)
                                       or (not is_long and agg >= threshold))
                            if flipped:
                                try:
                                    bot.client.flash_close_position(pid)
                                    log.info("TAPE EXIT %s %s: flow flipped "
                                             "(agg=%+.2f, r=%.2f, age=%.0fs)",
                                             symbol, "LONG" if is_long else "SHORT",
                                             agg, r_favor, age_s_te)
                                    bot.state.record_order(
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
                # Done ONCE per position; tracked in self.partial_tp_done.
                if (rk.partial_tp_enabled
                        and r_favor >= rk.partial_tp_at_r
                        and pid not in self.partial_tp_done):
                    partial_qty = qty * (rk.partial_tp_close_pct / 100.0)
                    step = meta.base_precision
                    n_steps = int(partial_qty / step) if step > 0 else 0
                    partial_qty_q = round(n_steps * step, 6)
                    if partial_qty_q >= meta.min_qty:
                        opp_side = "SELL" if is_long else "BUY"
                        try:
                            bot.client.place_order(
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
                            bot.state.record_order(
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
                meta = bot.metas.get(symbol, _DEFAULT_META)
                new_sl_rounded = round(new_sl, meta.price_precision)
                if (is_long and new_sl_rounded <= current_sl) or \
                   ((not is_long) and new_sl_rounded >= current_sl):
                    continue

                # Push update — modify the SPECIFIC SL trigger order by its id.
                # Race-condition fix: when r_favor is the MAX favor reached
                # but live price has retraced, the entry-buffer SL we just
                # computed lands on the wrong side of last price. Bitunix
                # rejects with code 30030. Retry with SL clamped to
                # current_price ± 1 tick — locks in remaining profit.
                try:
                    bot.client.modify_tpsl_order(
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
                    bot.state.record_order(
                        f"{symbol} SL {current_sl} → {new_sl_rounded} ({reason})"
                    )
                except BitunixError as e_inner:
                    if e_inner.code == 30028:
                        # Price already moved past the intended SL — update is moot.
                        # Original SL remains in place; position is still protected.
                        log.info(
                            "SL update moot for %s: price %.6f already past target %s (%s), ignoring",
                            symbol, current_price, new_sl_rounded, reason,
                        )
                        continue
                    if e_inner.code != 30030:
                        raise
                    tick = 10 ** -meta.price_precision
                    clamp_sl = (current_price - tick) if is_long else (current_price + tick)
                    clamp_sl_rounded = round(clamp_sl, meta.price_precision)
                    if (is_long and clamp_sl_rounded <= current_sl) or \
                       ((not is_long) and clamp_sl_rounded >= current_sl):
                        log.info(
                            "SL clamp skipped %s: %s not better than current %s "
                            "(price retraced past ratchet point)",
                            symbol, clamp_sl_rounded, current_sl,
                        )
                        continue
                    bot.client.modify_tpsl_order(
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
                    bot.state.record_order(
                        f"{symbol} SL {current_sl} → {clamp_sl_rounded} "
                        f"({reason}, clamped from {new_sl_rounded})"
                    )
            except BitunixError as e_sl:
                msg = (e_sl.msg or "").lower()
                benign = any(s in msg for s in
                             ("not found", "not exist", "closed", "expired", "canceled"))
                if not benign:
                    log.warning("SL update failed for %s: %s", p.get("symbol"), e_sl)
                    bot.state.record_error(f"SL update failed: {e_sl.code} {e_sl.msg}")
                continue   # skip TP adjustment too — position likely gone

            # ---------- ADAPTIVE TP TIGHTENING ----------
            try:
                if not rk.adaptive_tp_enabled:
                    continue
                tp_order = tp_orders.get(pid)
                if not tp_order:
                    continue
                current_tp = float(tp_order["tpPrice"])
                tp_order_id = str(tp_order["id"])
                ctime_ms = int(p.get("ctime") or 0)
                age_min = (time.time() * 1000 - ctime_ms) / 60000.0 if ctime_ms else 0
                desired_r = adaptive_tp_r(
                    age_minutes=age_min,
                    original_tp_r=rk.take_profit_r,
                    fee_pct=rk.round_trip_fee_pct,
                    sl_pct=rk.stop_loss_pct,
                    floor_r=rk.adaptive_tp_floor_r,
                    bar_minutes=parse_timeframe_minutes(bot.cfg.trading.timeframe),
                )
                desired_tp = (entry + desired_r * sl_distance) if is_long \
                             else (entry - desired_r * sl_distance)
                desired_tp_rounded = round(desired_tp, meta.price_precision)
                tighter = (desired_tp_rounded < current_tp) if is_long \
                          else (desired_tp_rounded > current_tp)
                if not tighter:
                    continue
                if abs(desired_tp_rounded - current_tp) < 10 ** (-meta.price_precision):
                    continue
                bot.client.modify_tpsl_order(
                    order_id=tp_order_id,
                    tp_price=str(desired_tp_rounded),
                    tp_qty=tp_order.get("tpQty"),
                    tp_stop_type=tp_order.get("tpStopType") or "LAST_PRICE",
                    tp_order_type=tp_order.get("tpOrderType") or "MARKET",
                )
                log.info("TP tighten %s: %s → %s (age=%.1fm desired=%.2fR)",
                         symbol, current_tp, desired_tp_rounded, age_min, desired_r)
                bot.state.record_order(
                    f"{symbol} TP {current_tp} → {desired_tp_rounded} "
                    f"(age={age_min:.0f}m, target={desired_r:.2f}R)"
                )
            except BitunixError as e:
                msg = (e.msg or "").lower()
                benign = any(s in msg for s in
                             ("not found", "not exist", "closed", "expired", "canceled"))
                if benign:
                    log.info("SL update skipped for %s (position closed): %s",
                             p.get("symbol"), e.msg)
                else:
                    log.warning("SL update failed for %s: %s", p.get("symbol"), e)
                    bot.state.record_error(f"SL update failed: {e.code} {e.msg}")
            except Exception as e:
                log.exception("Position management error for %s: %s", p.get("symbol"), e)
