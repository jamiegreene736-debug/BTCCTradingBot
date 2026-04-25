"""Smart Money Concepts — algorithmically tractable subset.

Implements:
  - Fair Value Gap (FVG): 3-candle imbalance pattern. Bullish FVG = candle 1's
    high < candle 3's low (price gapped up between bars 1 and 3, and bar 2
    didn't fill it). When price returns and "respects" the FVG (touches but
    doesn't fully fill), it's a high-quality pullback entry.
  - Liquidity sweep / stop hunt: price wicks beyond a recent swing extreme
    but closes back inside, indicating that the move was a fakeout to grab
    liquidity rather than a real breakout.

Skipped (per audit — too subjective on 1m):
  - Full BOS/CHoCH state machine
  - Order block detection (heuristics too noisy for scalping timeframes)
  - Wyckoff Spring (rare on 1m, hard to confirm)
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from .levels import find_swings

Direction = Literal["bullish", "bearish"]


@dataclass(frozen=True)
class SmcSignal:
    name: str
    direction: Direction
    strength: float
    detail: str


# ---------------------------------------------------------------- FVG

def detect_recent_fvg(
    highs: np.ndarray,
    lows: np.ndarray,
    closes: np.ndarray,
    max_bars_back: int = 20,
    proximity_pct: float = 0.15,
) -> SmcSignal | None:
    """Look back up to `max_bars_back` for a Fair Value Gap, then check if
    the current price is RETURNING to it (within proximity_pct). The play is
    that price often respects FVGs as support (bullish FVG) or resistance
    (bearish FVG) when it revisits them.

    Bullish FVG: highs[i-2] < lows[i] (gap up between bar i-2 and bar i,
    bar i-1 doesn't fill it). Vote LONG when price returns to the gap zone.
    """
    n = len(closes)
    if n < 4:
        return None
    p = float(closes[-1])
    # Walk back through 3-candle windows; most recent FVG wins.
    for i in range(n - 1, max(2, n - max_bars_back) - 1, -1):
        # bar i-2, i-1, i
        h0, l0 = highs[i - 2], lows[i - 2]
        h2, l2 = highs[i], lows[i]
        # Bullish FVG: gap up — bar i's low > bar i-2's high
        if l2 > h0:
            gap_high = l2
            gap_low = h0
            # If price has returned to the gap zone (touched but not exceeded), vote.
            if gap_low <= p <= gap_high:
                return SmcSignal(
                    "fvg_bullish", "bullish", 1.1,
                    f"FVG @ [{gap_low:.4f}, {gap_high:.4f}]",
                )
            # If price is just below or above with proximity, also vote.
            if abs(p - gap_high) / p * 100 <= proximity_pct and p <= gap_high:
                return SmcSignal("fvg_bullish_near", "bullish", 0.8,
                                 f"approaching bull FVG @ {gap_high:.4f}")
        # Bearish FVG: gap down — bar i's high < bar i-2's low
        if h2 < l0:
            gap_high = l0
            gap_low = h2
            if gap_low <= p <= gap_high:
                return SmcSignal(
                    "fvg_bearish", "bearish", 1.1,
                    f"FVG @ [{gap_low:.4f}, {gap_high:.4f}]",
                )
            if abs(p - gap_low) / p * 100 <= proximity_pct and p >= gap_low:
                return SmcSignal("fvg_bearish_near", "bearish", 0.8,
                                 f"approaching bear FVG @ {gap_low:.4f}")
    return None


# ---------------------------------------------------------------- liquidity sweep

def detect_liquidity_sweep(
    highs: np.ndarray,
    lows: np.ndarray,
    closes: np.ndarray,
    swing_lookback: int = 5,
    max_age_bars: int = 30,
) -> SmcSignal | None:
    """Stop-hunt fade: current bar wicked beyond a recent swing extreme but
    closed back inside.

    Bullish (long) sweep: bar's low pierced a recent swing low BUT close is
    above the swing low. Implies sellers got their stops hit and buyers
    stepped in. Fade short→long.
    Bearish (short) sweep: bar's high pierced a recent swing high but close
    is below it.
    """
    n = len(closes)
    if n < swing_lookback * 2 + 5:
        return None
    sh, sl = find_swings(highs, lows, swing_lookback)
    cur_low = float(lows[-1])
    cur_high = float(highs[-1])
    cur_close = float(closes[-1])
    cur_open_idx = n - 1

    # Look at recent (within max_age_bars) swing lows for a bullish sweep.
    recent_sl = [s for s in sl if (cur_open_idx - s.idx) <= max_age_bars]
    if recent_sl:
        nearest_sl = min(recent_sl, key=lambda s: abs(cur_low - s.price))
        if cur_low < nearest_sl.price and cur_close > nearest_sl.price:
            return SmcSignal(
                "liquidity_sweep_bullish", "bullish", 1.2,
                f"swept low {nearest_sl.price:.4f}, closed back at {cur_close:.4f}",
            )

    recent_sh = [s for s in sh if (cur_open_idx - s.idx) <= max_age_bars]
    if recent_sh:
        nearest_sh = min(recent_sh, key=lambda s: abs(cur_high - s.price))
        if cur_high > nearest_sh.price and cur_close < nearest_sh.price:
            return SmcSignal(
                "liquidity_sweep_bearish", "bearish", 1.2,
                f"swept high {nearest_sh.price:.4f}, closed back at {cur_close:.4f}",
            )

    return None


def detect_all(
    highs: np.ndarray,
    lows: np.ndarray,
    closes: np.ndarray,
    swing_lookback: int = 5,
) -> list[SmcSignal]:
    out: list[SmcSignal] = []
    fvg = detect_recent_fvg(highs, lows, closes)
    if fvg:
        out.append(fvg)
    sweep = detect_liquidity_sweep(highs, lows, closes, swing_lookback)
    if sweep:
        out.append(sweep)
    return out


def score(hits: list[SmcSignal]) -> tuple[float, float]:
    bull = sum(h.strength for h in hits if h.direction == "bullish")
    bear = sum(h.strength for h in hits if h.direction == "bearish")
    return bull, bear
