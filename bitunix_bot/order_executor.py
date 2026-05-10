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
import math
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

    def execute(self, symbol: str, plan: OrderPlan, *, force_market: bool = False) -> bool:
        """Execute an entry plan.

        Tape veto first ("don't fight the flow") — if the most recent
        10s of trade tape is contrary to our intended direction at
        ≥0.30 magnitude (~65/35 split or worse), skip the trade.

        Then try post-only maker entry unless the caller explicitly needs a
        market entry (the pump-fade scalp does). If the OB feed isn't ready
        or post-only is rejected, fall through to market.
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
            realism = self._paper_realism_report(symbol, plan, force_market=force_market)
            bot.state.record_order(
                order_text + " (paper) | " + realism["summary"],
                extra={"paper_realism": realism},
            )
            return True

        # Try post-only maker entry first. If the OB feed isn't ready
        # or post-only is rejected, fall through to market.
        if (not force_market
                and bot.cfg.trading.use_post_only_entries
                and bot.ob_feed
                and bot.ob_feed.is_connected()):
            if self._try_post_only(symbol, plan, order_text):
                return True
            log.info("Post-only path failed/skipped for %s; using market", symbol)

        return self._place_market(symbol, plan, order_text)

    # ------------------------------------------------------------------
    # Paper realism
    # ------------------------------------------------------------------

    @staticmethod
    def _ratio(value: float | None, denom: float) -> float | None:
        if value is None or denom <= 0:
            return None
        return value / denom

    @staticmethod
    def _impact_pct(spread_pct: float | None, depth_ratio: float | None) -> float:
        """Conservative top-of-book impact model for paper-mode scoring."""
        if spread_pct is None:
            return 0.0
        spread = max(0.0, float(spread_pct))
        if depth_ratio is None:
            return spread
        if depth_ratio >= 10:
            return spread * 0.25
        if depth_ratio >= 3:
            return spread * 0.50
        if depth_ratio >= 1:
            return spread
        # If visible depth cannot cover the order, model a punitive partial
        # sweep. This is not a price forecast; it is a "paper is too rosy"
        # penalty for comparing setups.
        return spread * min(5.0, 1.0 / max(depth_ratio, 0.05))

    @staticmethod
    def _fmt_ratio(value: float | None) -> str:
        return "--" if value is None or not math.isfinite(value) else f"{value:.1f}x"

    def _paper_realism_report(
        self,
        symbol: str,
        plan: OrderPlan,
        *,
        force_market: bool = False,
    ) -> dict[str, Any]:
        """Score a paper entry against live-execution frictions.

        Paper mode still does not send orders, but this report keeps it honest:
        spread, top-of-book depth, partial-fill risk, exit liquidity, fee drag,
        estimated impact, and rough isolated-liquidation pressure are included
        in the event payload and summary text.
        """
        bot = self._bot
        sym_u = symbol.upper()
        entry_style = (
            "MARKET"
            if force_market or not bot.cfg.trading.use_post_only_entries
            else "POST_ONLY_LIMIT"
        )
        notional = max(0.0, float(plan.volume or 0.0) * float(plan.price or 0.0))
        reward_price_pct = (
            abs(float(plan.take_profit) - float(plan.price)) / float(plan.price) * 100.0
            if plan.price else 0.0
        )
        risk_price_pct = (
            abs(float(plan.stop_loss) - float(plan.price)) / float(plan.price) * 100.0
            if plan.price else 0.0
        )
        fee_pct = max(0.0, float(getattr(bot.cfg.risk, "round_trip_fee_pct", 0.0) or 0.0))
        warnings: list[str] = []

        bid = ask = spread_pct = None
        bid_depth = ask_depth = None
        ob_connected = bool(bot.ob_feed and bot.ob_feed.is_connected())
        if not ob_connected:
            warnings.append("order book feed unavailable; fill/slippage unknown")
        elif bot.ob_feed is not None:
            tob = bot.ob_feed.get_top_of_book(sym_u)
            if tob:
                bid, ask = float(tob[0]), float(tob[1])
            else:
                warnings.append("top-of-book unavailable")
            spread_pct = bot.ob_feed.get_spread_pct(sym_u)
            depth = bot.ob_feed.get_depth(sym_u, top_n=5)
            if depth:
                bid_depth, ask_depth = float(depth[0]), float(depth[1])
            else:
                warnings.append("top-5 depth unavailable")

        max_spread_pct = None
        try:
            meta = bot.metas.get(sym_u, _DEFAULT_META)
            max_spread_pct = bot._max_entry_spread_pct_for_symbol(sym_u, meta)
        except Exception:
            max_spread_pct = float(getattr(bot.cfg.trading, "max_entry_spread_pct", 0.0) or 0.0)
        if spread_pct is not None and max_spread_pct and spread_pct > max_spread_pct:
            warnings.append(f"spread {spread_pct:.3f}% above {max_spread_pct:.3f}% entry threshold")

        market_entry_depth = ask_depth if plan.side == "BUY" else bid_depth
        passive_queue_depth = bid_depth if plan.side == "BUY" else ask_depth
        exit_depth = bid_depth if plan.side == "BUY" else ask_depth
        entry_depth_ratio = (
            self._ratio(market_entry_depth, plan.volume)
            if entry_style == "MARKET"
            else self._ratio(passive_queue_depth, plan.volume)
        )
        exit_depth_ratio = self._ratio(exit_depth, plan.volume)

        if entry_depth_ratio is not None:
            if entry_depth_ratio < 1:
                warnings.append("entry size exceeds visible top-5 depth; partial fill likely")
            elif entry_depth_ratio < 3:
                warnings.append("entry depth cushion below 3x size")
        if exit_depth_ratio is not None:
            if exit_depth_ratio < 1:
                warnings.append("exit size exceeds visible top-5 depth; TP/SL may slip")
            elif exit_depth_ratio < 3:
                warnings.append("exit depth cushion below 3x size")
        if entry_style == "POST_ONLY_LIMIT" and entry_depth_ratio is not None and entry_depth_ratio < 4:
            warnings.append("post-only queue is large versus size; maker fill may be late or missed")

        entry_impact_pct = (
            self._impact_pct(spread_pct, entry_depth_ratio)
            if entry_style == "MARKET"
            else 0.0
        )
        # Native TP/SL on perps usually acts like trigger liquidity when it
        # fires. Paper assumes the exit has to cross the book.
        exit_impact_pct = self._impact_pct(spread_pct, exit_depth_ratio)
        round_trip_cost_pct = fee_pct + entry_impact_pct + exit_impact_pct
        fee_usdt = notional * fee_pct / 100.0
        impact_usdt = notional * (entry_impact_pct + exit_impact_pct) / 100.0
        gross_tp_usdt = notional * reward_price_pct / 100.0
        gross_sl_usdt = notional * risk_price_pct / 100.0
        net_tp_usdt = gross_tp_usdt - fee_usdt - impact_usdt
        net_sl_usdt = -(gross_sl_usdt + fee_usdt + impact_usdt)
        net_tp_margin_pct = (reward_price_pct - round_trip_cost_pct) * plan.leverage
        net_sl_margin_pct = -(risk_price_pct + round_trip_cost_pct) * plan.leverage

        if net_tp_margin_pct <= 0:
            warnings.append("TP target does not clear estimated fees/impact")
        fee_drag_margin_pct = fee_pct * plan.leverage
        if fee_drag_margin_pct >= 15:
            warnings.append(f"fee drag is high at {fee_drag_margin_pct:.1f}% of margin")
        liquidation_move_pct = 100.0 / max(1, int(plan.leverage or 1))
        sl_liq_ratio = risk_price_pct / liquidation_move_pct if liquidation_move_pct > 0 else None
        if sl_liq_ratio is not None and sl_liq_ratio >= 0.65:
            warnings.append("stop is close to rough isolated liquidation buffer")

        if any("exceeds visible" in w or "does not clear" in w for w in warnings):
            status = "fail"
        elif warnings:
            status = "warn"
        else:
            status = "ok"

        summary = (
            f"realism={status} {entry_style} "
            f"netTP={net_tp_margin_pct:+.1f}%m netSL={net_sl_margin_pct:+.1f}%m "
            f"liq entry={self._fmt_ratio(entry_depth_ratio)} exit={self._fmt_ratio(exit_depth_ratio)} "
            f"fees={fee_drag_margin_pct:.1f}%m"
        )
        if warnings:
            summary += " warnings=" + "; ".join(warnings[:3])

        return {
            "status": status,
            "symbol": sym_u,
            "entry_style": entry_style,
            "entryStyle": entry_style,
            "side": plan.side,
            "qty": plan.volume,
            "notional": round(notional, 6),
            "leverage": plan.leverage,
            "bid": bid,
            "ask": ask,
            "spread_pct": round(spread_pct, 6) if spread_pct is not None else None,
            "spreadPct": round(spread_pct, 6) if spread_pct is not None else None,
            "bid_depth_top5": bid_depth,
            "bidDepthTop5": bid_depth,
            "ask_depth_top5": ask_depth,
            "askDepthTop5": ask_depth,
            "entry_depth_ratio": round(entry_depth_ratio, 4) if entry_depth_ratio is not None else None,
            "entryDepthRatio": round(entry_depth_ratio, 4) if entry_depth_ratio is not None else None,
            "exit_depth_ratio": round(exit_depth_ratio, 4) if exit_depth_ratio is not None else None,
            "exitDepthRatio": round(exit_depth_ratio, 4) if exit_depth_ratio is not None else None,
            "fee_pct_notional": round(fee_pct, 6),
            "feePctNotional": round(fee_pct, 6),
            "fee_drag_margin_pct": round(fee_drag_margin_pct, 4),
            "feeDragMarginPct": round(fee_drag_margin_pct, 4),
            "entry_impact_pct": round(entry_impact_pct, 6),
            "entryImpactPct": round(entry_impact_pct, 6),
            "exit_impact_pct": round(exit_impact_pct, 6),
            "exitImpactPct": round(exit_impact_pct, 6),
            "round_trip_cost_pct": round(round_trip_cost_pct, 6),
            "roundTripCostPct": round(round_trip_cost_pct, 6),
            "reward_price_pct": round(reward_price_pct, 6),
            "rewardPricePct": round(reward_price_pct, 6),
            "risk_price_pct": round(risk_price_pct, 6),
            "riskPricePct": round(risk_price_pct, 6),
            "estimated_net_tp_usdt": round(net_tp_usdt, 6),
            "estimatedNetTpUsdt": round(net_tp_usdt, 6),
            "estimated_net_sl_usdt": round(net_sl_usdt, 6),
            "estimatedNetSlUsdt": round(net_sl_usdt, 6),
            "estimated_net_tp_margin_pct": round(net_tp_margin_pct, 4),
            "estimatedNetTpMarginPct": round(net_tp_margin_pct, 4),
            "estimated_net_sl_margin_pct": round(net_sl_margin_pct, 4),
            "estimatedNetSlMarginPct": round(net_sl_margin_pct, 4),
            "liquidation_move_pct_rough": round(liquidation_move_pct, 6),
            "liquidationMovePctRough": round(liquidation_move_pct, 6),
            "sl_to_liquidation_buffer_ratio": round(sl_liq_ratio, 4) if sl_liq_ratio is not None else None,
            "slToLiquidationBufferRatio": round(sl_liq_ratio, 4) if sl_liq_ratio is not None else None,
            "warnings": warnings,
            "summary": summary,
        }

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
