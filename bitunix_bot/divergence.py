"""Divergence detection — RSI / MACD / OBV bullish, bearish, and hidden.

Divergence is when price makes a new extreme but the oscillator doesn't —
indicates exhaustion / reversal. Hidden divergence is the inverse and
indicates trend continuation.

Algorithm: find the two most recent pivot lows (for bullish divergence)
or pivot highs (for bearish). Compare price extremes against oscillator
extremes at the same pivots. A pivot is a fractal — local extreme over
N bars on each side.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

Direction = Literal["bullish", "bearish", "hidden_bullish", "hidden_bearish"]


@dataclass(frozen=True)
class Divergence:
    name: str           # "rsi_bullish", "macd_hidden_bearish", etc.
    direction: Direction
    strength: float     # 1.0 for regular, 0.7 for hidden (less reliable)
    detail: str


def _last_two_pivots(
    arr: np.ndarray,
    extrema: Literal["high", "low"],
    lookback: int = 5,
    min_separation: int = 5,
) -> list[int] | None:
    """Return the indices of the two most recent fractal pivots in `arr`,
    or None if there aren't two with sufficient separation."""
    n = len(arr)
    if n < 2 * lookback + 1:
        return None
    pivots: list[int] = []
    cmp = (lambda a, b: a > b) if extrema == "high" else (lambda a, b: a < b)
    for i in range(n - lookback - 1, lookback - 1, -1):
        v = arr[i]
        if np.isnan(v):
            continue
        left = arr[i - lookback : i]
        right = arr[i + 1 : i + 1 + lookback]
        if any(np.isnan(x) for x in left) or any(np.isnan(x) for x in right):
            continue
        if extrema == "high":
            if v > left.max() and v > right.max():
                pivots.append(i)
        else:
            if v < left.min() and v < right.min():
                pivots.append(i)
        if len(pivots) >= 2:
            break
    if len(pivots) < 2:
        return None
    if abs(pivots[0] - pivots[1]) < min_separation:
        return None
    return [pivots[1], pivots[0]]  # chronological order: older, then newer


def detect_divergences(
    closes: np.ndarray,
    highs: np.ndarray,
    lows: np.ndarray,
    rsi_v: np.ndarray,
    macd_line: np.ndarray,
    obv_v: np.ndarray,
    pivot_lookback: int = 5,
) -> list[Divergence]:
    """Run all divergence detectors. Returns hits at the most recent confirmed
    pivot — note that pivots are confirmed only `pivot_lookback` bars after
    they form, so this is a slightly-lagging indicator by design (no false
    early signals)."""
    out: list[Divergence] = []
    for osc_name, osc_v in (("rsi", rsi_v), ("macd", macd_line), ("obv", obv_v)):
        # Bullish: price LL but oscillator HL
        low_pivots = _last_two_pivots(lows, "low", pivot_lookback)
        if low_pivots:
            i_old, i_new = low_pivots
            if (lows[i_new] < lows[i_old]
                and not np.isnan(osc_v[i_new]) and not np.isnan(osc_v[i_old])
                and osc_v[i_new] > osc_v[i_old]):
                out.append(Divergence(
                    f"{osc_name}_bullish_div", "bullish", 1.0,
                    f"price LL {lows[i_new]:.4f}<{lows[i_old]:.4f}; "
                    f"{osc_name} HL {osc_v[i_new]:.2f}>{osc_v[i_old]:.2f}",
                ))
            # Hidden bullish: price HL but oscillator LL — continuation in uptrend
            elif (lows[i_new] > lows[i_old]
                  and not np.isnan(osc_v[i_new]) and not np.isnan(osc_v[i_old])
                  and osc_v[i_new] < osc_v[i_old]):
                out.append(Divergence(
                    f"{osc_name}_hidden_bullish_div", "hidden_bullish", 0.7,
                    f"price HL>{lows[i_old]:.4f}; {osc_name} LL<{osc_v[i_old]:.2f}",
                ))
        # Bearish: price HH but oscillator LH
        high_pivots = _last_two_pivots(highs, "high", pivot_lookback)
        if high_pivots:
            i_old, i_new = high_pivots
            if (highs[i_new] > highs[i_old]
                and not np.isnan(osc_v[i_new]) and not np.isnan(osc_v[i_old])
                and osc_v[i_new] < osc_v[i_old]):
                out.append(Divergence(
                    f"{osc_name}_bearish_div", "bearish", 1.0,
                    f"price HH {highs[i_new]:.4f}>{highs[i_old]:.4f}; "
                    f"{osc_name} LH {osc_v[i_new]:.2f}<{osc_v[i_old]:.2f}",
                ))
            # Hidden bearish: price LH but oscillator HH — continuation in downtrend
            elif (highs[i_new] < highs[i_old]
                  and not np.isnan(osc_v[i_new]) and not np.isnan(osc_v[i_old])
                  and osc_v[i_new] > osc_v[i_old]):
                out.append(Divergence(
                    f"{osc_name}_hidden_bearish_div", "hidden_bearish", 0.7,
                    f"price LH<{highs[i_old]:.4f}; {osc_name} HH>{osc_v[i_old]:.2f}",
                ))
    return out


def score(hits: list[Divergence]) -> tuple[int, int]:
    """Return (long_count, short_count) — caller treats each as one indicator
    vote per side. Hidden divergences vote for trend continuation."""
    long = sum(1 for h in hits if h.direction in ("bullish", "hidden_bullish"))
    short = sum(1 for h in hits if h.direction in ("bearish", "hidden_bearish"))
    return long, short
