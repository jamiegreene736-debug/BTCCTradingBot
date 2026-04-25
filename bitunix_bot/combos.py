"""Combo-bonus scoring — recognize high-EV multi-component setups.

Each combo defines a list of "required reason markers" (substrings or
prefixes that must appear in the per-side reasons list). When all markers
are present on one side, the combo "fires" and contributes an extra
indicator vote on that side, plus a CMB:<name> reason for visibility.

The point: an indicator confluence that hits 3 specific high-quality
agreement patterns (e.g. squeeze + volume + supertrend, or DIV + S/R +
volume) is meaningfully more reliable than 3 random votes lining up.
This rewards genuine confluence over coincidence.

Combos defined here are based on patterns repeatedly cited in:
  - freqtrade NostalgiaForInfinity buy_condition_* tags
  - bookmap.com / TradingView scalp playbooks
  - Wyckoff / SMC literature

ADDING NEW COMBOS: append to COMBO_RECIPES with all-substring matchers.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

Direction = Literal["bullish", "bearish"]


@dataclass(frozen=True)
class Combo:
    name: str
    direction: Direction
    weight: float                       # bonus weight (typically 1.0 = one extra vote)
    detail: str


@dataclass(frozen=True)
class _Recipe:
    name: str
    direction: Direction
    long_match: tuple[tuple[str, ...], ...]
    """Required substring groups. Each inner tuple is OR (any one matches);
    the outer tuple is AND (all groups must match for the combo to fire).

    Example: ((\"ema_stack_up\",), (\"vol_spike\", \"mfi(\")) means
    EITHER the ema_stack_up vote AND EITHER vol_spike OR mfi must be present.
    """
    short_match: tuple[tuple[str, ...], ...] = ()


COMBO_RECIPES: list[_Recipe] = [
    # 1. Trend pullback continuation: established trend + pullback to EMA
    #    confirmed by volume / MFI. Most-cited scalp setup.
    _Recipe(
        name="trend_pullback",
        direction="bullish",
        long_match=(
            ("ema_stack_up",),
            ("cross_above_ema_fast",),
            ("vol_spike", "mfi("),
        ),
        short_match=(),
    ),
    _Recipe(
        name="trend_pullback",
        direction="bearish",
        long_match=(),
        short_match=(
            ("ema_stack_down",),
            ("cross_below_ema_fast",),
            ("vol_spike", "mfi("),
        ),
    ),
    # 2. Compression breakout: TTM Squeeze release + volume + supertrend agree.
    _Recipe(
        name="squeeze_breakout",
        direction="bullish",
        long_match=(
            ("squeeze_up",),
            ("vol_spike",),
            ("supertrend_up",),
        ),
        short_match=(),
    ),
    _Recipe(
        name="squeeze_breakout",
        direction="bearish",
        long_match=(),
        short_match=(
            ("squeeze_down",),
            ("vol_spike",),
            ("supertrend_down",),
        ),
    ),
    # 3. SMC reversal: liquidity sweep OR FVG + divergence + S/R bounce.
    #    Three independent confirmations of a turn.
    _Recipe(
        name="smc_reversal",
        direction="bullish",
        long_match=(
            ("SMC:liquidity_sweep_bullish", "SMC:fvg_bullish"),
            ("DIV:rsi_bullish", "DIV:macd_bullish", "DIV:obv_bullish", "DIV:cvd_bullish"),
            ("sr_bounce_support",),
        ),
        short_match=(),
    ),
    _Recipe(
        name="smc_reversal",
        direction="bearish",
        long_match=(),
        short_match=(
            ("SMC:liquidity_sweep_bearish", "SMC:fvg_bearish"),
            ("DIV:rsi_bearish", "DIV:macd_bearish", "DIV:obv_bearish", "DIV:cvd_bearish"),
            ("sr_reject_resistance",),
        ),
    ),
    # 4. Mean-reversion at extreme: BB outside + RSI extreme + S/R level.
    _Recipe(
        name="bb_extreme_revert",
        direction="bullish",
        long_match=(
            ("below_bb_mid",),  # we don't have explicit "below BB lower" — basis is fine as a proxy
            ("DIV:rsi_bullish",),
            ("sr_bounce_support",),
        ),
        short_match=(),
    ),
    _Recipe(
        name="bb_extreme_revert",
        direction="bearish",
        long_match=(),
        short_match=(
            ("above_bb_mid",),
            ("DIV:rsi_bearish",),
            ("sr_reject_resistance",),
        ),
    ),
    # 5. Crowd contrarian: high funding + opposite divergence + volume spike.
    _Recipe(
        name="crowd_contrarian",
        direction="bullish",
        long_match=(
            ("funding-",),  # crowded shorts
            ("DIV:rsi_bullish", "DIV:macd_bullish", "DIV:obv_bullish", "DIV:cvd_bullish"),
            ("vol_spike",),
        ),
        short_match=(),
    ),
    _Recipe(
        name="crowd_contrarian",
        direction="bearish",
        long_match=(),
        short_match=(
            ("funding+",),  # crowded longs
            ("DIV:rsi_bearish", "DIV:macd_bearish", "DIV:obv_bearish", "DIV:cvd_bearish"),
            ("vol_spike",),
        ),
    ),
]


def _match_recipe(reasons: list[str], groups: tuple[tuple[str, ...], ...]) -> bool:
    """All groups must have at least one matching reason."""
    if not groups:
        return False
    for group in groups:
        if not any(any(marker in r for r in reasons) for marker in group):
            return False
    return True


def detect(long_reasons: list[str], short_reasons: list[str]) -> list[Combo]:
    """Return all combo hits. Each fires at most once per direction."""
    out: list[Combo] = []
    seen: set[tuple[str, str]] = set()
    for recipe in COMBO_RECIPES:
        key = (recipe.name, recipe.direction)
        if key in seen:
            continue
        if recipe.direction == "bullish" and recipe.long_match:
            if _match_recipe(long_reasons, recipe.long_match):
                out.append(Combo(recipe.name, "bullish", 1.0,
                                  f"3+ confirming votes co-fire"))
                seen.add(key)
        elif recipe.direction == "bearish" and recipe.short_match:
            if _match_recipe(short_reasons, recipe.short_match):
                out.append(Combo(recipe.name, "bearish", 1.0,
                                  f"3+ confirming votes co-fire"))
                seen.add(key)
    return out
