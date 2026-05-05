"""Shared symbol-metadata types.

Extracted from bot.py so multiple modules (bot.py, position_manager.py,
order_executor.py, etc.) can reference SymbolMeta without circular imports.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class SymbolMeta:
    base_precision: float     # qty step (e.g. 0.001)
    price_precision: int      # digits for price (e.g. 1 for BTCUSDT)
    min_qty: float
    max_leverage: int = 100   # Bitunix caps differ per symbol


# Defensive fallback when symbol metadata isn't available (e.g. paper-mode
# tests, or before _resolve_symbol_meta has populated the cache). Picks
# precision values that work for most majors without overflowing the API.
DEFAULT_META = SymbolMeta(
    base_precision=0.001,
    price_precision=2,
    min_qty=0.001,
    max_leverage=100,
)


def _first_float(row: dict[str, Any], *keys: str, default: float = 0.0) -> float:
    for key in keys:
        value = row.get(key)
        if value in (None, ""):
            continue
        try:
            return float(value)
        except (TypeError, ValueError):
            continue
    return default


def _first_int(row: dict[str, Any], *keys: str, default: int = 0) -> int:
    for key in keys:
        value = row.get(key)
        if value in (None, ""):
            continue
        try:
            return int(float(value))
        except (TypeError, ValueError):
            continue
    return default


def row_symbol(row: dict[str, Any]) -> str:
    return str(
        row.get("symbol")
        or row.get("pair")
        or row.get("contractCode")
        or ""
    ).strip().upper()


def row_quote_volume_usdt(row: dict[str, Any]) -> float:
    """Best-effort 24h quote volume from Bitunix pair/ticker payloads."""
    return _first_float(
        row,
        "quoteVol",
        "quoteVolume",
        "quoteVolume24h",
        "turnover24h",
        "turnover",
        "volumeUSDT",
        "usdtVolume",
        "amount24h",
        "quote_volume",
    )


def row_open_interest_usdt(row: dict[str, Any]) -> float:
    return _first_float(
        row,
        "openInterestUSDT",
        "openInterestValue",
        "openInterestUsd",
        "openInterest",
        "oi",
    )


def row_max_leverage(row: dict[str, Any]) -> int:
    return _first_int(row, "maxLeverage", "max_leverage", "leverage", default=100)


def parse_symbol_meta(row: dict[str, Any]) -> SymbolMeta:
    raw_base = row.get("basePrecision")
    try:
        raw_base_num = float(raw_base)
        raw_base_text = str(raw_base).strip()
        if raw_base_num >= 1 or raw_base_text.isdigit():
            base_step = 10 ** (-int(raw_base_num))
        elif raw_base_num > 0:
            base_step = raw_base_num
        else:
            base_step = 0.001
    except (TypeError, ValueError):
        base_step = 0.001
    price_prec = _first_int(row, "quotePrecision", "pricePrecision", default=2)
    min_qty = _first_float(row, "minTradeVolume", "minQty", "minVolume", default=base_step)
    max_lev = row_max_leverage(row) or 100
    return SymbolMeta(base_step, price_prec, min_qty, max_lev)


def row_is_tradeable_usdt_perp(row: dict[str, Any]) -> bool:
    sym = row_symbol(row)
    if not sym.endswith("USDT"):
        return False
    status = str(
        row.get("status")
        or row.get("tradeStatus")
        or row.get("state")
        or row.get("symbolStatus")
        or "TRADING"
    ).upper()
    return status in {"TRADING", "OPEN", "ONLINE", "ENABLE", "ENABLED", "1", "TRUE"}


def auto_symbol_risk_mult(
    symbol: str,
    *,
    quote_volume_usdt: float = 0.0,
    max_leverage: int = 100,
    atr_pct: float | None = None,
) -> float:
    """Conservative default sizing for dynamically discovered symbols.

    BTC/ETH can carry more size; lower-volume or high-ATR alts are scaled
    down so broad scanning does not become broad correlated risk.
    """
    sym = symbol.upper()
    if sym == "BTCUSDT":
        base = 1.0
    elif sym == "ETHUSDT":
        base = 0.8
    elif max_leverage >= 100 and quote_volume_usdt >= 50_000_000:
        base = 0.5
    elif quote_volume_usdt >= 10_000_000:
        base = 0.35
    else:
        base = 0.25

    if max_leverage < 75:
        base *= 0.85
    if atr_pct is not None and atr_pct > 0:
        if atr_pct >= 1.0:
            base *= 0.6
        elif atr_pct >= 0.6:
            base *= 0.8

    return round(max(0.20, min(1.0, base)), 3)


def select_dynamic_symbols(
    rows: list[dict[str, Any]],
    *,
    min_quote_volume_usdt: float,
    min_open_interest_usdt: float,
    min_leverage: int,
    max_symbols: int,
    keep_symbols: list[str] | None = None,
) -> list[str]:
    keep = [s.upper() for s in (keep_symbols or [])]
    ranked: list[tuple[float, str]] = []
    for row in rows:
        sym = row_symbol(row)
        if not row_is_tradeable_usdt_perp(row):
            continue
        quote_vol = row_quote_volume_usdt(row)
        oi = row_open_interest_usdt(row)
        max_lev = row_max_leverage(row)
        if min_quote_volume_usdt > 0 and quote_vol <= 0:
            continue
        if quote_vol and quote_vol < min_quote_volume_usdt:
            continue
        if min_open_interest_usdt > 0 and oi and oi < min_open_interest_usdt:
            continue
        if max_lev < min_leverage:
            continue
        ranked.append((quote_vol, sym))

    ranked.sort(key=lambda item: (-item[0], item[1]))
    selected = []
    seen = set()
    for sym in keep + [sym for _, sym in ranked]:
        if sym in seen:
            continue
        selected.append(sym)
        seen.add(sym)
        if max_symbols > 0 and len(selected) >= max_symbols:
            break
    return selected
