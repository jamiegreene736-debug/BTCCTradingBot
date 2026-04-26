"""Shared symbol-metadata types.

Extracted from bot.py so multiple modules (bot.py, position_manager.py,
order_executor.py, etc.) can reference SymbolMeta without circular imports.
"""
from __future__ import annotations

from dataclasses import dataclass


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
