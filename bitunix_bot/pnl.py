"""Closed-position P&L helpers.

Bitunix history rows are not perfectly consistent across endpoints. Live rows
seen in production use entryPrice/closePrice and return fee as a positive
absolute cost while realizedPNL is already fee-adjusted. Older fixtures and
some docs use avgOpenPrice/avgClosePrice with signed fee. Keep the normalization
in one place so dashboard stats, journal exits, and streak gates agree.
"""
from __future__ import annotations

from typing import Any


def as_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value) if value not in (None, "", "null") else default
    except (TypeError, ValueError):
        return default


def first_float(*values: Any) -> float:
    for value in values:
        out = as_float(value, default=0.0)
        if out > 0:
            return out
    return 0.0


def position_side(p: dict[str, Any]) -> str:
    raw = str(p.get("side") or p.get("positionSide") or "").upper()
    if raw in ("BUY", "LONG"):
        return "LONG"
    if raw in ("SELL", "SHORT"):
        return "SHORT"
    return raw


def position_entry_price(p: dict[str, Any]) -> float:
    return first_float(
        p.get("avgOpenPrice"),
        p.get("entryPrice"),
        p.get("openPrice"),
        p.get("avg_open_price"),
        p.get("entry_price"),
        p.get("open_price"),
    )


def position_exit_price(p: dict[str, Any]) -> float:
    return first_float(
        p.get("avgClosePrice"),
        p.get("closePrice"),
        p.get("exitPrice"),
        p.get("avg_close_price"),
        p.get("close_price"),
        p.get("exit_price"),
    )


def position_qty(p: dict[str, Any]) -> float:
    return first_float(p.get("qty"), p.get("size"), p.get("volume"), p.get("maxQty"))


def position_gross_pnl(p: dict[str, Any]) -> float | None:
    side = position_side(p)
    entry = position_entry_price(p)
    exit_px = position_exit_price(p)
    qty = position_qty(p)
    if entry <= 0 or exit_px <= 0 or qty <= 0:
        return None
    if side == "SHORT":
        return (entry - exit_px) * qty
    if side == "LONG":
        return (exit_px - entry) * qty
    return None


def position_price_pnl_pct(p: dict[str, Any]) -> float | None:
    side = position_side(p)
    entry = position_entry_price(p)
    exit_px = position_exit_price(p)
    if entry <= 0 or exit_px <= 0:
        return None
    if side == "SHORT":
        return (entry - exit_px) / entry * 100.0
    if side == "LONG":
        return (exit_px - entry) / entry * 100.0
    return None


def closed_position_net_pnl(p: dict[str, Any]) -> float:
    """Return actual realized P&L after fees/funding when possible.

    Live Bitunix history rows observed on 2026-05-04 show:
      gross move P&L = (entry - close) * qty for shorts
      realizedPNL    = gross move P&L - positive fee

    Older tests and signed-fee schemas show:
      realizedPNL excludes fees, fee is negative, net = realized + fee

    The positive-fee branch therefore treats realizedPNL as actual unless the
    row clearly looks like the older "realized equals gross" schema.
    """
    realized = as_float(p.get("realizedPNL") or p.get("realizedPnl") or p.get("realized_pnl"))
    fee = as_float(p.get("fee"))
    funding = as_float(p.get("funding"))
    gross = position_gross_pnl(p)

    if fee > 0:
        if gross is not None:
            tolerance = max(1e-8, abs(fee) * 0.05, abs(gross) * 0.001)
            if abs(realized - gross) <= tolerance:
                return realized - fee + funding
            return realized + funding

        # Compatibility for rows with no close price where old code modeled a
        # near-flat BE exit as realized + positive fee ~= 0.
        if realized < 0 and abs(abs(realized) - fee) <= max(0.001, abs(fee) * 0.02):
            return realized + fee + funding
        return realized + funding

    return realized + fee + funding
