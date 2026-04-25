"""Classical Japanese candlestick pattern detection.

Detects the patterns a human chart trader would look for, with directional
weights based on standard pattern reliability rankings (Bulkowski, Nison,
freqtrade community norms). Returns a list of `Pattern` hits at the LATEST
candle, each carrying:

  - name: pattern identifier
  - direction: "bullish" | "bearish" | "neutral"
  - strength: 0.4 - 1.5  (higher = more reliable historically)

Aggregator returns (bullish_score, bearish_score) summed strengths so the
caller can normalize and combine with other signals.

Patterns covered:
  Single candle: hammer, shooting_star, hanging_man, inverted_hammer,
                 doji, marubozu (bull/bear), spinning_top, pin_bar
  Two candle:    bullish_engulfing, bearish_engulfing, piercing_line,
                 dark_cloud_cover, bullish_harami, bearish_harami,
                 tweezer_top, tweezer_bottom
  Three candle:  morning_star, evening_star, three_white_soldiers,
                 three_black_crows, three_inside_up, three_inside_down
  Multi:         inside_bar, outside_bar
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

Direction = Literal["bullish", "bearish", "neutral"]


@dataclass(frozen=True)
class Pattern:
    name: str
    direction: Direction
    strength: float


# ----------------------------------------------------------------- helpers

def _body(o: float, c: float) -> float:
    return abs(c - o)


def _upper_wick(o: float, h: float, c: float) -> float:
    return h - max(o, c)


def _lower_wick(o: float, l: float, c: float) -> float:
    return min(o, c) - l


def _range(h: float, l: float) -> float:
    return h - l


def _is_bull(o: float, c: float) -> bool:
    return c > o


def _is_bear(o: float, c: float) -> bool:
    return c < o


def _trend_up(closes: np.ndarray, lookback: int = 10) -> bool:
    """Simple trend check: close[-1] > average of last `lookback` closes."""
    if len(closes) < lookback + 1:
        return False
    return closes[-1] > closes[-lookback - 1]


def _trend_down(closes: np.ndarray, lookback: int = 10) -> bool:
    if len(closes) < lookback + 1:
        return False
    return closes[-1] < closes[-lookback - 1]


# ----------------------------------------------------------------- single-candle

def is_hammer(o: float, h: float, l: float, c: float) -> bool:
    """Small body, long lower wick (>= 2x body), tiny upper wick."""
    rng = _range(h, l)
    if rng <= 0:
        return False
    b = _body(o, c)
    return (b / rng < 0.35
            and _lower_wick(o, l, c) >= 2 * b
            and _upper_wick(o, h, c) <= b)


def is_shooting_star(o: float, h: float, l: float, c: float) -> bool:
    """Small body, long upper wick (>= 2x body), tiny lower wick."""
    rng = _range(h, l)
    if rng <= 0:
        return False
    b = _body(o, c)
    return (b / rng < 0.35
            and _upper_wick(o, h, c) >= 2 * b
            and _lower_wick(o, l, c) <= b)


def is_doji(o: float, h: float, l: float, c: float, threshold: float = 0.1) -> bool:
    rng = _range(h, l)
    if rng <= 0:
        return False
    return _body(o, c) / rng < threshold


def is_marubozu(o: float, h: float, l: float, c: float, threshold: float = 0.85) -> bool:
    """Body fills almost the whole range — almost no wicks."""
    rng = _range(h, l)
    if rng <= 0:
        return False
    return _body(o, c) / rng > threshold


def is_spinning_top(o: float, h: float, l: float, c: float) -> bool:
    rng = _range(h, l)
    if rng <= 0:
        return False
    b = _body(o, c)
    return (b / rng < 0.35
            and _lower_wick(o, l, c) > b
            and _upper_wick(o, h, c) > b)


def is_pin_bar_bull(o: float, h: float, l: float, c: float) -> bool:
    """Long lower wick rejection, body in upper third of range."""
    rng = _range(h, l)
    if rng <= 0:
        return False
    return (_lower_wick(o, l, c) > 0.6 * rng
            and min(o, c) > l + 0.5 * rng)


def is_pin_bar_bear(o: float, h: float, l: float, c: float) -> bool:
    rng = _range(h, l)
    if rng <= 0:
        return False
    return (_upper_wick(o, h, c) > 0.6 * rng
            and max(o, c) < l + 0.5 * rng)


# ----------------------------------------------------------------- two-candle

def is_bullish_engulfing(o1: float, c1: float, o2: float, c2: float) -> bool:
    """Prev bearish, current bullish, current body engulfs prev body."""
    return _is_bear(o1, c1) and _is_bull(o2, c2) and o2 <= c1 and c2 >= o1


def is_bearish_engulfing(o1: float, c1: float, o2: float, c2: float) -> bool:
    return _is_bull(o1, c1) and _is_bear(o2, c2) and o2 >= c1 and c2 <= o1


def is_piercing_line(o1: float, c1: float, o2: float, c2: float) -> bool:
    """Prev bearish, current bullish that opens below prev close, closes
    above prev midpoint."""
    if not _is_bear(o1, c1) or not _is_bull(o2, c2):
        return False
    prev_mid = (o1 + c1) / 2
    return o2 < c1 and c2 > prev_mid and c2 < o1


def is_dark_cloud_cover(o1: float, c1: float, o2: float, c2: float) -> bool:
    if not _is_bull(o1, c1) or not _is_bear(o2, c2):
        return False
    prev_mid = (o1 + c1) / 2
    return o2 > c1 and c2 < prev_mid and c2 > o1


def is_bullish_harami(o1: float, c1: float, o2: float, c2: float) -> bool:
    """Large bearish, then small bullish whose body sits inside the prev body."""
    if not _is_bear(o1, c1) or not _is_bull(o2, c2):
        return False
    return (o2 > c1 and c2 < o1
            and _body(o2, c2) < _body(o1, c1) * 0.6)


def is_bearish_harami(o1: float, c1: float, o2: float, c2: float) -> bool:
    if not _is_bull(o1, c1) or not _is_bear(o2, c2):
        return False
    return (o2 < c1 and c2 > o1
            and _body(o2, c2) < _body(o1, c1) * 0.6)


def is_tweezer_bottom(l1: float, l2: float, tol_pct: float = 0.001) -> bool:
    if max(l1, l2) <= 0:
        return False
    return abs(l1 - l2) / max(l1, l2) < tol_pct


def is_tweezer_top(h1: float, h2: float, tol_pct: float = 0.001) -> bool:
    if max(h1, h2) <= 0:
        return False
    return abs(h1 - h2) / max(h1, h2) < tol_pct


def is_inside_bar(h1: float, l1: float, h2: float, l2: float) -> bool:
    """Current bar's high/low fully inside prev bar's high/low."""
    return h2 < h1 and l2 > l1


def is_outside_bar_bull(o1: float, h1: float, l1: float, c1: float,
                       o2: float, h2: float, l2: float, c2: float) -> bool:
    return h2 > h1 and l2 < l1 and _is_bull(o2, c2)


def is_outside_bar_bear(o1: float, h1: float, l1: float, c1: float,
                       o2: float, h2: float, l2: float, c2: float) -> bool:
    return h2 > h1 and l2 < l1 and _is_bear(o2, c2)


# ----------------------------------------------------------------- three-candle

def is_morning_star(o1: float, c1: float, o2: float, c2: float, o3: float, c3: float) -> bool:
    """Large bear, small body (any direction), large bull closing above the
    midpoint of bar 1."""
    if not _is_bear(o1, c1):
        return False
    if _body(o2, c2) > _body(o1, c1) * 0.5:
        return False
    if not _is_bull(o3, c3):
        return False
    prev_mid = (o1 + c1) / 2
    return c3 > prev_mid


def is_evening_star(o1: float, c1: float, o2: float, c2: float, o3: float, c3: float) -> bool:
    if not _is_bull(o1, c1):
        return False
    if _body(o2, c2) > _body(o1, c1) * 0.5:
        return False
    if not _is_bear(o3, c3):
        return False
    prev_mid = (o1 + c1) / 2
    return c3 < prev_mid


def is_three_white_soldiers(o1: float, c1: float, o2: float, c2: float, o3: float, c3: float) -> bool:
    return (_is_bull(o1, c1) and _is_bull(o2, c2) and _is_bull(o3, c3)
            and c2 > c1 and c3 > c2
            and o2 > o1 and o3 > o2
            and o2 < c1 and o3 < c2)  # opens within prior bodies


def is_three_black_crows(o1: float, c1: float, o2: float, c2: float, o3: float, c3: float) -> bool:
    return (_is_bear(o1, c1) and _is_bear(o2, c2) and _is_bear(o3, c3)
            and c2 < c1 and c3 < c2
            and o2 < o1 and o3 < o2
            and o2 > c1 and o3 > c2)


def is_three_inside_up(o1: float, c1: float, o2: float, c2: float, o3: float, c3: float) -> bool:
    """Bullish harami followed by bullish close above c1."""
    return is_bullish_harami(o1, c1, o2, c2) and _is_bull(o3, c3) and c3 > o1


def is_three_inside_down(o1: float, c1: float, o2: float, c2: float, o3: float, c3: float) -> bool:
    return is_bearish_harami(o1, c1, o2, c2) and _is_bear(o3, c3) and c3 < o1


# ----------------------------------------------------------------- aggregator

# Standard reliability weights (relative; final scoring normalizes).
PATTERN_WEIGHTS: dict[str, float] = {
    # Three-candle patterns are the most reliable.
    "morning_star":          1.5,
    "evening_star":          1.5,
    "three_white_soldiers":  1.5,
    "three_black_crows":     1.5,
    "three_inside_up":       1.3,
    "three_inside_down":     1.3,
    # Strong two-candle reversals.
    "bullish_engulfing":     1.4,
    "bearish_engulfing":     1.4,
    # Single-candle reversals (need confirmation but historically reliable).
    "hammer":                1.0,
    "shooting_star":         1.0,
    "hanging_man":           0.9,
    "inverted_hammer":       0.9,
    "pin_bar_bull":          1.1,
    "pin_bar_bear":          1.1,
    # Two-candle moderate.
    "piercing_line":         1.0,
    "dark_cloud_cover":      1.0,
    "bullish_harami":        0.8,
    "bearish_harami":        0.8,
    "tweezer_bottom":        0.7,
    "tweezer_top":           0.7,
    "outside_bar_bull":      0.9,
    "outside_bar_bear":      0.9,
    # Continuation / strength.
    "marubozu_bull":         0.8,
    "marubozu_bear":         0.8,
    # Indecision (count as neutral; light influence).
    "doji":                  0.4,
    "spinning_top":          0.3,
    "inside_bar":            0.4,
}


def detect(
    opens: np.ndarray,
    highs: np.ndarray,
    lows: np.ndarray,
    closes: np.ndarray,
) -> list[Pattern]:
    """Detect all patterns at the LAST candle. Returns Pattern hits."""
    n = len(closes)
    if n < 1:
        return []

    hits: list[Pattern] = []
    o, h, l, c = opens[-1], highs[-1], lows[-1], closes[-1]
    bull_now = _is_bull(o, c)

    # ------ single-candle ------
    if is_hammer(o, h, l, c):
        # Hammer in downtrend = bullish reversal; in uptrend = "hanging man" (bearish).
        if _trend_down(closes):
            hits.append(Pattern("hammer", "bullish", PATTERN_WEIGHTS["hammer"]))
        elif _trend_up(closes):
            hits.append(Pattern("hanging_man", "bearish", PATTERN_WEIGHTS["hanging_man"]))
        else:
            hits.append(Pattern("hammer", "bullish", PATTERN_WEIGHTS["hammer"] * 0.6))
    if is_shooting_star(o, h, l, c):
        if _trend_up(closes):
            hits.append(Pattern("shooting_star", "bearish", PATTERN_WEIGHTS["shooting_star"]))
        elif _trend_down(closes):
            hits.append(Pattern("inverted_hammer", "bullish", PATTERN_WEIGHTS["inverted_hammer"]))
        else:
            hits.append(Pattern("shooting_star", "bearish", PATTERN_WEIGHTS["shooting_star"] * 0.6))
    if is_pin_bar_bull(o, h, l, c):
        hits.append(Pattern("pin_bar_bull", "bullish", PATTERN_WEIGHTS["pin_bar_bull"]))
    if is_pin_bar_bear(o, h, l, c):
        hits.append(Pattern("pin_bar_bear", "bearish", PATTERN_WEIGHTS["pin_bar_bear"]))
    if is_doji(o, h, l, c):
        hits.append(Pattern("doji", "neutral", PATTERN_WEIGHTS["doji"]))
    if is_spinning_top(o, h, l, c):
        hits.append(Pattern("spinning_top", "neutral", PATTERN_WEIGHTS["spinning_top"]))
    if is_marubozu(o, h, l, c):
        if bull_now:
            hits.append(Pattern("marubozu_bull", "bullish", PATTERN_WEIGHTS["marubozu_bull"]))
        else:
            hits.append(Pattern("marubozu_bear", "bearish", PATTERN_WEIGHTS["marubozu_bear"]))

    # ------ two-candle ------
    if n >= 2:
        o1, h1, l1, c1 = opens[-2], highs[-2], lows[-2], closes[-2]
        if is_bullish_engulfing(o1, c1, o, c):
            hits.append(Pattern("bullish_engulfing", "bullish", PATTERN_WEIGHTS["bullish_engulfing"]))
        if is_bearish_engulfing(o1, c1, o, c):
            hits.append(Pattern("bearish_engulfing", "bearish", PATTERN_WEIGHTS["bearish_engulfing"]))
        if is_piercing_line(o1, c1, o, c):
            hits.append(Pattern("piercing_line", "bullish", PATTERN_WEIGHTS["piercing_line"]))
        if is_dark_cloud_cover(o1, c1, o, c):
            hits.append(Pattern("dark_cloud_cover", "bearish", PATTERN_WEIGHTS["dark_cloud_cover"]))
        if is_bullish_harami(o1, c1, o, c):
            hits.append(Pattern("bullish_harami", "bullish", PATTERN_WEIGHTS["bullish_harami"]))
        if is_bearish_harami(o1, c1, o, c):
            hits.append(Pattern("bearish_harami", "bearish", PATTERN_WEIGHTS["bearish_harami"]))
        if is_tweezer_bottom(l1, l):
            hits.append(Pattern("tweezer_bottom", "bullish", PATTERN_WEIGHTS["tweezer_bottom"]))
        if is_tweezer_top(h1, h):
            hits.append(Pattern("tweezer_top", "bearish", PATTERN_WEIGHTS["tweezer_top"]))
        if is_inside_bar(h1, l1, h, l):
            hits.append(Pattern("inside_bar", "neutral", PATTERN_WEIGHTS["inside_bar"]))
        if is_outside_bar_bull(o1, h1, l1, c1, o, h, l, c):
            hits.append(Pattern("outside_bar_bull", "bullish", PATTERN_WEIGHTS["outside_bar_bull"]))
        if is_outside_bar_bear(o1, h1, l1, c1, o, h, l, c):
            hits.append(Pattern("outside_bar_bear", "bearish", PATTERN_WEIGHTS["outside_bar_bear"]))

    # ------ three-candle ------
    if n >= 3:
        o0, c0 = opens[-3], closes[-3]
        o1, c1 = opens[-2], closes[-2]
        if is_morning_star(o0, c0, o1, c1, o, c):
            hits.append(Pattern("morning_star", "bullish", PATTERN_WEIGHTS["morning_star"]))
        if is_evening_star(o0, c0, o1, c1, o, c):
            hits.append(Pattern("evening_star", "bearish", PATTERN_WEIGHTS["evening_star"]))
        if is_three_white_soldiers(o0, c0, o1, c1, o, c):
            hits.append(Pattern("three_white_soldiers", "bullish", PATTERN_WEIGHTS["three_white_soldiers"]))
        if is_three_black_crows(o0, c0, o1, c1, o, c):
            hits.append(Pattern("three_black_crows", "bearish", PATTERN_WEIGHTS["three_black_crows"]))
        if is_three_inside_up(o0, c0, o1, c1, o, c):
            hits.append(Pattern("three_inside_up", "bullish", PATTERN_WEIGHTS["three_inside_up"]))
        if is_three_inside_down(o0, c0, o1, c1, o, c):
            hits.append(Pattern("three_inside_down", "bearish", PATTERN_WEIGHTS["three_inside_down"]))

    return hits


def score(hits: list[Pattern]) -> tuple[float, float]:
    """Aggregate hit list into (bullish_total, bearish_total) raw strength sums.
    Neutrals count slightly toward both sides (indecision pulls toward reversal
    of whichever direction is otherwise winning)."""
    bull = sum(p.strength for p in hits if p.direction == "bullish")
    bear = sum(p.strength for p in hits if p.direction == "bearish")
    neutral = sum(p.strength for p in hits if p.direction == "neutral")
    # Neutrals add small weight to both sides — they're indecision, not noise.
    return bull + neutral * 0.25, bear + neutral * 0.25
