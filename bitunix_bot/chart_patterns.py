"""Multi-bar chart pattern detection.

Patterns rely on swing highs/lows from levels.find_swings, then check
geometric relationships between recent swings to identify:

  - Double top / Double bottom    (2 swing extrema at similar price)
  - Head and shoulders            (3 swing highs: peak-PEAK-peak with center
                                    higher and outer two at similar level)
  - Inverse H&S                   (mirror)
  - Ascending triangle            (rising swing lows + flat resistance)
  - Descending triangle           (falling swing highs + flat support)
  - Symmetric triangle            (lower highs + higher lows converging)
  - Bull flag                     (sharp uptrend impulse + tight pullback channel)
  - Bear flag                     (mirror)

All detectors return (direction, name, strength) tuples or None. They
operate on the LAST N bars and use the most recent swings as anchors.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from .levels import Swing, find_swings

Direction = Literal["bullish", "bearish"]


@dataclass(frozen=True)
class ChartPattern:
    name: str
    direction: Direction
    strength: float
    detail: str


# ---------------------------------------------------------------- helpers

def _pct_diff(a: float, b: float) -> float:
    """Absolute % difference between two prices (relative to mean)."""
    if a + b == 0:
        return 0.0
    return abs(a - b) / ((a + b) / 2.0) * 100.0


def _linfit(xs: list[int], ys: list[float]) -> tuple[float, float]:
    """Simple least-squares slope/intercept. Returns (slope_per_bar, intercept)."""
    if len(xs) < 2:
        return 0.0, ys[0] if ys else 0.0
    x = np.array(xs, dtype=float)
    y = np.array(ys, dtype=float)
    s, i = np.polyfit(x, y, 1)
    return float(s), float(i)


# ---------------------------------------------------------------- detectors

def detect_double_top(
    swing_highs: list[Swing],
    closes: np.ndarray,
    tol_pct: float = 0.4,
    min_separation: int = 5,
) -> ChartPattern | None:
    """Two swing highs at similar price level + close below the trough between."""
    if len(swing_highs) < 2:
        return None
    second, first = swing_highs[-1], swing_highs[-2]
    if second.idx - first.idx < min_separation:
        return None
    if _pct_diff(first.price, second.price) > tol_pct:
        return None
    # Confirm: current close has broken below the lowest close between the peaks.
    trough = float(closes[first.idx : second.idx + 1].min())
    if closes[-1] < trough:
        return ChartPattern(
            "double_top", "bearish", 1.4,
            f"peaks~{(first.price+second.price)/2:.4f}",
        )
    return None


def detect_double_bottom(
    swing_lows: list[Swing],
    closes: np.ndarray,
    tol_pct: float = 0.4,
    min_separation: int = 5,
) -> ChartPattern | None:
    if len(swing_lows) < 2:
        return None
    second, first = swing_lows[-1], swing_lows[-2]
    if second.idx - first.idx < min_separation:
        return None
    if _pct_diff(first.price, second.price) > tol_pct:
        return None
    peak = float(closes[first.idx : second.idx + 1].max())
    if closes[-1] > peak:
        return ChartPattern(
            "double_bottom", "bullish", 1.4,
            f"lows~{(first.price+second.price)/2:.4f}",
        )
    return None


def detect_head_and_shoulders(
    swing_highs: list[Swing],
    swing_lows: list[Swing],
    closes: np.ndarray,
    shoulder_tol_pct: float = 0.6,
    head_premium_pct: float = 0.3,
) -> ChartPattern | None:
    """3 swing highs: left shoulder, head (higher), right shoulder (~= left).
    Confirmed when close breaks below neckline (lowest of the two intervening lows).
    """
    if len(swing_highs) < 3:
        return None
    left_s, head, right_s = swing_highs[-3], swing_highs[-2], swing_highs[-1]
    # head must be the highest
    if not (head.price > left_s.price and head.price > right_s.price):
        return None
    # head must exceed shoulders by a meaningful margin
    if (head.price - max(left_s.price, right_s.price)) / head.price * 100.0 < head_premium_pct:
        return None
    # shoulders at similar price
    if _pct_diff(left_s.price, right_s.price) > shoulder_tol_pct:
        return None
    # find neckline — the two lows between the shoulders
    neckline_lows = [s for s in swing_lows
                      if left_s.idx < s.idx < right_s.idx]
    if not neckline_lows:
        return None
    neckline = min(s.price for s in neckline_lows)
    if closes[-1] < neckline:
        return ChartPattern(
            "head_and_shoulders", "bearish", 1.5,
            f"head={head.price:.4f},neckline={neckline:.4f}",
        )
    return None


def detect_inverse_head_and_shoulders(
    swing_highs: list[Swing],
    swing_lows: list[Swing],
    closes: np.ndarray,
    shoulder_tol_pct: float = 0.6,
    head_premium_pct: float = 0.3,
) -> ChartPattern | None:
    if len(swing_lows) < 3:
        return None
    left_s, head, right_s = swing_lows[-3], swing_lows[-2], swing_lows[-1]
    if not (head.price < left_s.price and head.price < right_s.price):
        return None
    if (min(left_s.price, right_s.price) - head.price) / head.price * 100.0 < head_premium_pct:
        return None
    if _pct_diff(left_s.price, right_s.price) > shoulder_tol_pct:
        return None
    neckline_highs = [s for s in swing_highs
                       if left_s.idx < s.idx < right_s.idx]
    if not neckline_highs:
        return None
    neckline = max(s.price for s in neckline_highs)
    if closes[-1] > neckline:
        return ChartPattern(
            "inverse_head_and_shoulders", "bullish", 1.5,
            f"head={head.price:.4f},neckline={neckline:.4f}",
        )
    return None


def detect_triangle(
    swing_highs: list[Swing],
    swing_lows: list[Swing],
    closes: np.ndarray,
    min_swings: int = 3,
    flat_slope_pct_per_bar: float = 0.02,
) -> ChartPattern | None:
    """Identify ascending / descending / symmetric triangle from recent swings.

    Triggers ONLY on breakout — close breaks the converging band.
    """
    if len(swing_highs) < min_swings or len(swing_lows) < min_swings:
        return None
    sh = swing_highs[-min_swings:]
    sl = swing_lows[-min_swings:]
    high_slope, high_intercept = _linfit([s.idx for s in sh], [s.price for s in sh])
    low_slope, low_intercept = _linfit([s.idx for s in sl], [s.price for s in sl])

    last_idx = len(closes) - 1
    proj_high = high_slope * last_idx + high_intercept
    proj_low = low_slope * last_idx + low_intercept
    p = float(closes[-1])

    # Slope expressed as % of avg price per bar.
    avg_price = (proj_high + proj_low) / 2.0
    high_slope_pct = (high_slope / avg_price) * 100.0 if avg_price else 0.0
    low_slope_pct = (low_slope / avg_price) * 100.0 if avg_price else 0.0

    # Ascending: flat top + rising bottom + breakout above top
    if abs(high_slope_pct) < flat_slope_pct_per_bar and low_slope_pct > flat_slope_pct_per_bar:
        if p > proj_high:
            return ChartPattern("ascending_triangle", "bullish", 1.2,
                                f"breakout>{proj_high:.4f}")
    # Descending: falling top + flat bottom + breakdown below bottom
    if abs(low_slope_pct) < flat_slope_pct_per_bar and high_slope_pct < -flat_slope_pct_per_bar:
        if p < proj_low:
            return ChartPattern("descending_triangle", "bearish", 1.2,
                                f"breakdown<{proj_low:.4f}")
    # Symmetric: lower highs + higher lows
    if high_slope_pct < -flat_slope_pct_per_bar and low_slope_pct > flat_slope_pct_per_bar:
        if p > proj_high:
            return ChartPattern("symmetric_triangle", "bullish", 1.0,
                                f"upbreak>{proj_high:.4f}")
        if p < proj_low:
            return ChartPattern("symmetric_triangle", "bearish", 1.0,
                                f"downbreak<{proj_low:.4f}")
    return None


def detect_flag(
    closes: np.ndarray,
    impulse_window: int = 10,
    pullback_window: int = 8,
    impulse_pct: float = 1.5,
    pullback_max_pct: float = 0.6,
) -> ChartPattern | None:
    """Bull/bear flag: a sharp impulse followed by a tight retrace.

    Bull: large green move, then 4-8 bars of mild pullback ranging in a
    tight channel. Continuation breakout above the pullback high = bull flag.
    """
    n = len(closes)
    need = impulse_window + pullback_window
    if n < need + 2:
        return None

    impulse_start = closes[-need - 1]
    impulse_end = closes[-pullback_window - 1]
    move = (impulse_end - impulse_start) / impulse_start * 100.0

    pull_window = closes[-pullback_window - 1 :]
    pull_range = float(pull_window.max() - pull_window.min())
    pull_range_pct = (pull_range / impulse_end) * 100.0
    last = float(closes[-1])
    pull_high = float(pull_window.max())
    pull_low = float(pull_window.min())

    if move > impulse_pct and pull_range_pct < pullback_max_pct and last > pull_high:
        return ChartPattern("bull_flag", "bullish", 1.2,
                            f"impulse={move:+.2f}%,pull<{pullback_max_pct}%")
    if move < -impulse_pct and pull_range_pct < pullback_max_pct and last < pull_low:
        return ChartPattern("bear_flag", "bearish", 1.2,
                            f"impulse={move:+.2f}%,pull<{pullback_max_pct}%")
    return None


# ---------------------------------------------------------------- aggregator

def detect_all(
    highs: np.ndarray,
    lows: np.ndarray,
    closes: np.ndarray,
    swing_lookback: int = 5,
) -> list[ChartPattern]:
    """Run every chart-pattern detector. Returns list of hits at the LAST bar."""
    hits: list[ChartPattern] = []
    if len(closes) < 2 * swing_lookback + 5:
        return hits
    sh, sl = find_swings(highs, lows, swing_lookback)

    for fn in (
        lambda: detect_double_top(sh, closes),
        lambda: detect_double_bottom(sl, closes),
        lambda: detect_head_and_shoulders(sh, sl, closes),
        lambda: detect_inverse_head_and_shoulders(sh, sl, closes),
        lambda: detect_triangle(sh, sl, closes),
        lambda: detect_flag(closes),
    ):
        try:
            r = fn()
            if r:
                hits.append(r)
        except Exception:
            # Defensive: any detector bug should not kill the tick.
            continue
    return hits


def score(hits: list[ChartPattern]) -> tuple[float, float]:
    """Return (bullish_total, bearish_total) summed strengths."""
    bull = sum(p.strength for p in hits if p.direction == "bullish")
    bear = sum(p.strength for p in hits if p.direction == "bearish")
    return bull, bear
