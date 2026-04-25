"""Swing point detection + support/resistance level clustering.

Foundation module for:
  1. Direct S/R votes in the strategy (price-near-level → directional bias)
  2. Multi-bar chart patterns in patterns_chart.py (H&S, double top/bottom,
     triangles — all require swing points as anchors)

Approach: fractal swing detection. A swing high at index `i` is a bar
whose `high` is strictly greater than the highs of `lookback` bars on
each side. Swing low is the mirror. Then cluster nearby swings into
horizontal levels.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Swing:
    idx: int           # bar index in the input arrays
    price: float       # high (for swing_high) or low (for swing_low)
    kind: str          # "high" or "low"


@dataclass(frozen=True)
class Level:
    price: float       # cluster center (mean of constituent swings)
    touches: int       # how many swings cluster here
    last_idx: int      # most recent swing index in the cluster
    kind: str          # "support" (built from swing lows) | "resistance" (highs)


def find_swings(
    highs: np.ndarray,
    lows: np.ndarray,
    lookback: int = 5,
) -> tuple[list[Swing], list[Swing]]:
    """Detect swing highs and swing lows via fractal lookback.

    A swing requires `lookback` confirming bars on EACH side, so the most
    recent `lookback` bars are never confirmed swings (they may still
    become one as new bars arrive).
    """
    n = len(highs)
    swing_highs: list[Swing] = []
    swing_lows: list[Swing] = []
    if n < 2 * lookback + 1:
        return swing_highs, swing_lows

    for i in range(lookback, n - lookback):
        h, l = highs[i], lows[i]
        left_h = highs[i - lookback : i]
        right_h = highs[i + 1 : i + 1 + lookback]
        left_l = lows[i - lookback : i]
        right_l = lows[i + 1 : i + 1 + lookback]
        if h > left_h.max() and h > right_h.max():
            swing_highs.append(Swing(idx=i, price=float(h), kind="high"))
        if l < left_l.min() and l < right_l.min():
            swing_lows.append(Swing(idx=i, price=float(l), kind="low"))
    return swing_highs, swing_lows


def cluster_levels(
    swings: list[Swing],
    tolerance_pct: float = 0.3,
    min_touches: int = 2,
) -> list[Level]:
    """Cluster swings whose prices are within tolerance_pct of each other.

    A level is "real" only if at least `min_touches` swings cluster — a
    single high/low that was never revisited isn't a confirmed S/R.
    """
    if not swings:
        return []
    # Sort by price ascending. Walk the list and merge into clusters.
    sorted_swings = sorted(swings, key=lambda s: s.price)
    clusters: list[list[Swing]] = []
    cur: list[Swing] = [sorted_swings[0]]
    for sw in sorted_swings[1:]:
        cluster_avg = sum(c.price for c in cur) / len(cur)
        if abs(sw.price - cluster_avg) / cluster_avg * 100.0 <= tolerance_pct:
            cur.append(sw)
        else:
            clusters.append(cur)
            cur = [sw]
    clusters.append(cur)

    out: list[Level] = []
    for cluster in clusters:
        if len(cluster) < min_touches:
            continue
        kind = "resistance" if cluster[0].kind == "high" else "support"
        out.append(
            Level(
                price=sum(c.price for c in cluster) / len(cluster),
                touches=len(cluster),
                last_idx=max(c.idx for c in cluster),
                kind=kind,
            )
        )
    # Strongest first (most touches; ties broken by most recent).
    out.sort(key=lambda lv: (lv.touches, lv.last_idx), reverse=True)
    return out


def nearest_level(
    price: float,
    levels: list[Level],
    kind: str | None = None,
    above: bool | None = None,
) -> Level | None:
    """Return the nearest level matching the filters, or None if no match.

    `kind`: "support" / "resistance" / None (any).
    `above`: True (level above price), False (below), None (either).
    """
    candidates = [
        lv for lv in levels
        if (kind is None or lv.kind == kind)
        and (above is None or (lv.price > price) == above)
    ]
    if not candidates:
        return None
    return min(candidates, key=lambda lv: abs(lv.price - price))


def proximity_to_level(
    price: float,
    level: Level,
) -> float:
    """Distance to a level as a fraction of price (0.003 = 0.3% away)."""
    return abs(price - level.price) / price


def detect_sr_signal(
    highs: np.ndarray,
    lows: np.ndarray,
    closes: np.ndarray,
    swing_lookback: int = 5,
    cluster_tol_pct: float = 0.3,
    min_touches: int = 2,
    proximity_pct: float = 0.3,
) -> tuple[str | None, str | None]:
    """High-level helper: return (long_reason, short_reason) for the current bar.

    Long: close just bounced off a support level (still within proximity AND
    closing above it after dipping near).
    Short: close just rejected at resistance (within proximity AND closing
    below it after pushing near).
    Either side may be None if no qualifying level is nearby.
    """
    if len(closes) < 2 * swing_lookback + 5:
        return None, None
    sh, sl = find_swings(highs, lows, swing_lookback)
    levels = cluster_levels(sh + sl, cluster_tol_pct, min_touches)
    if not levels:
        return None, None

    p = float(closes[-1])
    prev = float(closes[-2])

    long_reason = short_reason = None

    sup = nearest_level(p, levels, kind="support", above=False)
    if sup and proximity_to_level(p, sup) * 100 <= proximity_pct:
        # Bounce confirmation: dipped near support and now closing above it.
        if prev <= sup.price * 1.001 and p > sup.price:
            long_reason = f"sr_bounce_support({sup.price:.4f},×{sup.touches})"

    res = nearest_level(p, levels, kind="resistance", above=True)
    if res and proximity_to_level(p, res) * 100 <= proximity_pct:
        # Rejection: pushed near resistance and now closing below.
        if prev >= res.price * 0.999 and p < res.price:
            short_reason = f"sr_reject_resistance({res.price:.4f},×{res.touches})"

    return long_reason, short_reason
