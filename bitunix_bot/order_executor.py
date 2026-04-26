"""Entry-side order execution — tape veto, maker-first post-only path,
market fallback, and pending-limit timeout sweeping.

Extracted from bot.py (Grok holistic review item) so the entry
plumbing has a clear module boundary alongside PositionManager. The
class holds no state of its own — it operates on the BitunixBot's
existing attributes (cfg, client, state, metas, ob_feed, tape_feed,
pending_limits, last_action_at) via composition through `self._bot`.

Design choice: identical pattern to PositionManager. State stays on
the bot for compat with existing call sites (e.g. _tick reads
pending_limits to count against max_open_positions); OrderExecutor
just manages that state through the bot reference.
"""
from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING, Any

from .client import BitunixError
from .risk import OrderPlan
from .symbol_meta import DEFAULT_META as _DEFAULT_META

if TYPE_CHECKING:
    from .bot import BitunixBot

log = logging.getLogger(__name__)


class OrderExecutor:
    """Wraps entry-execution behavior: tape veto, maker-first, market
    fallback, and pending-limit lifecycle."""

    def __init__(self, bot: "BitunixBot") -> None:
        self._bot = bot

    # ------------------------------------------------------------------
    # Public entry points
    # ------------------------------------------------------------------

    def execute(self, symbol: str, plan: OrderPlan) -> bool:
        """Execute an entry plan.

        Tape veto first ("don't fight the flow") — if the most recent
        10s of trade tape is contrary to our intended direction at
        ≥0.30 magnitude (~65/35 split or worse), skip the trade.

        Then try post-only maker entry (saves ~0.04% per round-trip
        in fees). If the OB feed isn't ready or post-only is rejected,
        fall through to market.
        """
        bot = self._bot
        if bot.tape_feed is not None:
            agg = bot.tape_feed.get_aggression_ratio(symbol, window_secs=10)
            if agg is not None:
                if plan.side == "BUY" and agg <= -0.30:
                    bot.state.record_skip(
                        f"{symbol}: tape veto — long signal but {agg:+.2f} sell flow"
                    )
                    return False
                if plan.side == "SELL" and agg >= 0.30:
                    bot.state.record_skip(
                        f"{symbol}: tape veto — short signal but {agg:+.2f} buy flow"
                    )
                    return False

        prefix = "LIVE" if bot.cfg.is_live else "PAPER"
        order_text = (f"{prefix} {symbol} {plan.side} qty={plan.volume} "
                      f"entry~{plan.price} SL={plan.stop_loss} TP={plan.take_profit} "
                      f"lev={plan.leverage}x")
        log.info("ORDER %s [%s]", order_text, plan.notes)
        if not bot.cfg.is_live:
            bot.state.record_order(order_text + " (paper)")
            return True

        # Try post-only maker entry first. If the OB feed isn't ready
        # or post-only is rejected, fall through to market.
        if (bot.cfg.trading.use_post_only_entries
                and bot.ob_feed
                and bot.ob_feed.is_connected()):
            if self._try_post_only(symbol, plan, order_text):
                return True
            log.info("Post-only path failed/skipped for %s; using market", symbol)

        return self._place_market(symbol, plan, order_text)

    def check_pending_limits(self, all_open_positions: list[dict[str, Any]]) -> None:
        """Sweep pending post-only limit entries.

          - if a position now exists for the symbol → entry filled, clear tracking
          - if timeout exceeded → cancel limit and skip (no market fallback)
        """
        bot = self._bot
        if not bot.pending_limits:
            return
        now = int(time.time())
        open_by_sym = {str(p.get("symbol", "")).upper() for p in all_open_positions}
        for sym_u, info in list(bot.pending_limits.items()):
            # Filled? A position now exists for this symbol.
            if sym_u in open_by_sym:
                log.info("MAKER fill confirmed for %s (orderId=%s)", sym_u, info["order_id"])
                bot.state.record_order(f"{sym_u} MAKER FILLED orderId={info['order_id']}")
                del bot.pending_limits[sym_u]
                continue
            # Per-order timeout (computed at placement time from activity).
            timeout = info.get("timeout_secs", bot.cfg.trading.post_only_timeout_secs)
            # Timed out → cancel and SKIP. Do NOT market-fallback.
            # Pro-desk rule: if your maker bid wasn't hit in the timeout
            # window, the price moved AWAY from your bid. For a long, that
            # means price went UP — marketing in now means CHASING. Trust
            # the next signal.
            age = now - info["place_ts"]
            if age >= timeout:
                try:
                    bot.client.cancel_order(symbol=info["symbol"], order_id=info["order_id"])
                except Exception as e:
                    log.warning("Cancel pending limit for %s failed: %s", sym_u, e)
                log.info("MAKER timeout %s after %ds — signal failed, skip "
                         "(no market fallback)", sym_u, age)
                bot.state.record_skip(
                    f"{sym_u}: post-only didn't fill in {age}s — signal invalidated"
                )
                del bot.pending_limits[sym_u]
                # Refresh cooldown so we don't re-fire on the same bar.
                bot.last_action_at[sym_u] = now

    # ------------------------------------------------------------------
    # Internal: entry mechanism implementations
    # ------------------------------------------------------------------

    def _try_post_only(self, symbol: str, plan: OrderPlan, order_text: str) -> bool:
        """Attempt a POST_ONLY limit entry. Returns True if successfully
        placed (tracked for timeout sweep), False on any failure so the
        caller can fall through to market.

        Aggressive maker (Grok holistic review): when spread > 1 tick,
        step INSIDE the spread to become the new top of book — fills
        sooner than passively joining the existing TOB. When spread is
        exactly 1 tick, join existing TOB (stepping inside would cross
        to taker).
        """
        bot = self._bot
        sym_u = symbol.upper()
        tob = bot.ob_feed.get_top_of_book(sym_u)
        if not tob:
            return False
        bid, ask = tob
        is_long = plan.side == "BUY"
        meta = bot.metas.get(symbol, _DEFAULT_META)
        tick_size = 10 ** -meta.price_precision if meta.price_precision >= 0 else 0
        spread = ask - bid
        if tick_size > 0 and spread > tick_size * 1.5:
            # Spread > 1 tick: step inside by 1 tick to become new TOB.
            limit_px = round((bid + tick_size) if is_long else (ask - tick_size),
                             meta.price_precision)
        else:
            # Tight spread or unknown precision: join existing TOB.
            limit_px = round(bid if is_long else ask, meta.price_precision)
        # If our chosen price would already be a taker (rare race), bail to market.
        if (is_long and limit_px >= ask) or ((not is_long) and limit_px <= bid):
            return False

        # Dynamic timeout — high tape activity → shorter timeout. Clamp
        # range scales with base_timeout so 15m-timeframe configs get
        # a meaningful range.
        base_timeout = bot.cfg.trading.post_only_timeout_secs
        timeout_secs = base_timeout
        if bot.tape_feed is not None:
            activity = bot.tape_feed.get_activity_multiplier(
                sym_u, clamp_min=0.5, clamp_max=2.0
            )
            if activity is not None and activity > 0:
                lo = max(4, base_timeout // 2)
                hi = max(lo + 1, int(base_timeout * 1.5))
                timeout_secs = max(lo, min(hi, int(round(base_timeout / activity))))

        minute_bucket = int(time.time()) // 60
        client_id = f"bot-{symbol}-{minute_bucket}-{plan.side}-PO"
        try:
            resp = bot.client.place_order(
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
            bot.pending_limits[sym_u] = {
                "symbol": symbol,
                "order_id": str(order_id),
                "place_ts": int(time.time()),
                "plan": plan,
                "order_text": order_text,
                "limit_px": limit_px,
                "timeout_secs": timeout_secs,
                # Top-of-book snapshot at placement — feeds the journal so
                # downstream analysis can spot adverse-selection patterns.
                "tob_bid": float(bid),
                "tob_ask": float(ask),
            }
            log.info("MAKER %s LIMIT @ %s POST_ONLY orderId=%s timeout=%ds",
                     symbol, limit_px, order_id, timeout_secs)
            bot.state.record_order(
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
        bot = self._bot
        minute_bucket = int(time.time()) // 60
        client_id = f"bot-{symbol}-{minute_bucket}-{plan.side}"
        try:
            resp = bot.client.place_order(
                symbol=symbol,
                side=plan.side,
                qty=str(plan.volume),
                order_type="MARKET",
                trade_side="OPEN",
                tp_price=str(plan.take_profit),
                sl_price=str(plan.stop_loss),
                client_id=client_id,
            )
            log.info("Placed orderId=%s clientId=%s",
                     resp.get("orderId"), resp.get("clientId"))
            bot.state.record_order(f"{order_text} → orderId={resp.get('orderId')}")
            return True
        except BitunixError as e:
            log.error("Order rejected: %s (payload=%s)", e, e.payload)
            bot.state.record_error(f"{symbol} order rejected: {e.code} {e.msg}")
            return False
        except Exception as e:
            log.error("place_order network/unknown failure for %s: %s", symbol, e)
            bot.state.record_error(f"{symbol} order failure (unknown state): {e}")
            return False
