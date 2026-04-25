"""Signal generation — pure technicals: candlestick patterns + indicators.

Per user direction: candlestick patterns AND other technicals (RSI, MACD,
EMA, BB, ADX, Supertrend) ARE the decision. No other factors matter.

NO HARD GATES — every signal comes purely from technical agreement.

Architecture:

  SCORING (combined directional score):

    indicator_score = (count of rules agreeing) / 7              # 0-1
    pattern_score   = min(1, sum(pattern strengths) / norm)      # 0-1
    combined        = pattern_weight * pattern + (1-pw) * ind    # 0-1

  Fire if combined >= fire_threshold for one direction AND that side wins.

INDICATOR RULES (counted, non-pattern half):
  1. EMA stack (fast > mid > slow for long; reverse for short)
  2. Close cross of EMA fast (bar-to-bar)
  3. RSI in long/short window
  4. MACD line vs signal AND histogram rising
  5. Bollinger basis (close above/below mid)
  6. Supertrend direction
  7. ADX trend strength (counts for whichever side wins on other rules)

PATTERNS: 23 classical candlestick patterns from bitunix_bot.patterns,
weighted 0.4–1.5 by historical reliability. See patterns.py.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Literal

import numpy as np

from . import patterns
from .config import StrategyCfg
from .indicators import adx as adx_fn
from .indicators import atr as atr_fn
from .indicators import bollinger, ema, macd, rsi
from .indicators import supertrend as supertrend_fn

log = logging.getLogger(__name__)

Direction = Literal["long", "short"]


@dataclass
class Signal:
    direction: Direction
    score: float                       # combined 0-1
    indicator_score: int               # raw indicator confluence count
    pattern_score: float               # raw pattern strength sum
    reasons: list[str]                 # human-readable mix of indicators + patterns
    pattern_hits: list[str] = field(default_factory=list)
    price: float = 0.0
    atr: float = 0.0

    @property
    def side_code(self) -> str:
        return "BUY" if self.direction == "long" else "SELL"


def evaluate(
    opens: list[float],
    highs: list[float],
    lows: list[float],
    closes: list[float],
    cfg: StrategyCfg,
) -> Signal | None:
    if len(closes) < max(cfg.ema_slow, cfg.macd_slow, cfg.bb_period, cfg.atr_period) + 5:
        return None

    o = np.array(opens, dtype=float)
    h = np.array(highs, dtype=float)
    l = np.array(lows, dtype=float)
    c = np.array(closes, dtype=float)

    # ------------------------------------------------------------------ indicators
    ema_f = ema(c, cfg.ema_fast)
    ema_m = ema(c, cfg.ema_mid)
    ema_s = ema(c, cfg.ema_slow)
    rsi_v = rsi(c, cfg.rsi_period)
    macd_l, macd_s, macd_h = macd(c, cfg.macd_fast, cfg.macd_slow, cfg.macd_signal)
    bb_up, bb_mid, bb_lo = bollinger(c, cfg.bb_period, cfg.bb_std)
    atr_v = atr_fn(h, l, c, cfg.atr_period)
    adx_v = adx_fn(h, l, c, cfg.adx_period)
    _, st_dir = supertrend_fn(h, l, c, cfg.supertrend_period, cfg.supertrend_mult)

    i = len(c) - 1
    p = c[i]
    if p <= 0:
        return None

    # ATR is still required for SL/TP sizing (not as a gate).
    atr_now = atr_v[i] if i < len(atr_v) else np.nan
    if np.isnan(atr_now):
        return None
    atr_pct = (atr_now / p) * 100.0

    # ------------------------------------------------------------------ indicators
    long_reasons: list[str] = []
    short_reasons: list[str] = []

    # 1. EMA stack
    if ema_f[i] > ema_m[i] > ema_s[i]:
        long_reasons.append("ema_stack_up")
    if ema_f[i] < ema_m[i] < ema_s[i]:
        short_reasons.append("ema_stack_down")

    # 2. Close cross of ema_fast
    if i >= 1:
        if c[i - 1] <= ema_f[i - 1] and c[i] > ema_f[i]:
            long_reasons.append("cross_above_ema_fast")
        if c[i - 1] >= ema_f[i - 1] and c[i] < ema_f[i]:
            short_reasons.append("cross_below_ema_fast")

    # 3. RSI window
    r = rsi_v[i]
    if not np.isnan(r):
        if cfg.rsi_long_min <= r <= cfg.rsi_long_max:
            long_reasons.append(f"rsi({r:.0f})")
        if cfg.rsi_short_min <= r <= cfg.rsi_short_max:
            short_reasons.append(f"rsi({r:.0f})")

    # 4. MACD
    if i >= 1 and not np.isnan(macd_h[i - 1]):
        if macd_l[i] > macd_s[i] and macd_h[i] > macd_h[i - 1]:
            long_reasons.append("macd_up")
        if macd_l[i] < macd_s[i] and macd_h[i] < macd_h[i - 1]:
            short_reasons.append("macd_down")

    # 5. Bollinger basis
    if not np.isnan(bb_mid[i]):
        if p > bb_mid[i]:
            long_reasons.append("above_bb_mid")
        if p < bb_mid[i]:
            short_reasons.append("below_bb_mid")

    # 6. Supertrend regime
    if i < len(st_dir) and not np.isnan(st_dir[i]):
        if st_dir[i] == 1:
            long_reasons.append("supertrend_up")
        elif st_dir[i] == -1:
            short_reasons.append("supertrend_down")

    # 7. ADX trend strength — counts for whichever side is otherwise winning.
    a_now = adx_v[i] if i < len(adx_v) else np.nan
    if not np.isnan(a_now) and a_now >= cfg.adx_min:
        # ADX is direction-agnostic; it amplifies whichever side has more votes.
        if len(long_reasons) > len(short_reasons):
            long_reasons.append(f"adx({a_now:.0f})")
        elif len(short_reasons) > len(long_reasons):
            short_reasons.append(f"adx({a_now:.0f})")
        else:
            # Tie — ADX doesn't break it, but record context.
            pass

    indicator_long = len(long_reasons)
    indicator_short = len(short_reasons)
    INDICATOR_MAX = 7

    # ------------------------------------------------------------------ patterns
    hits = patterns.detect(o, h, l, c)
    pattern_long, pattern_short = patterns.score(hits)
    pattern_long_n = min(1.0, pattern_long / cfg.pattern_norm)
    pattern_short_n = min(1.0, pattern_short / cfg.pattern_norm)

    # ------------------------------------------------------------------ combine
    pw = cfg.pattern_weight
    combined_long = pw * pattern_long_n + (1 - pw) * (indicator_long / INDICATOR_MAX)
    combined_short = pw * pattern_short_n + (1 - pw) * (indicator_short / INDICATOR_MAX)

    pat_long_names = [p.name for p in hits if p.direction == "bullish"]
    pat_short_names = [p.name for p in hits if p.direction == "bearish"]
    pat_neutral_names = [p.name for p in hits if p.direction == "neutral"]

    def _build(direction: Direction) -> Signal:
        is_long = direction == "long"
        score_combined = combined_long if is_long else combined_short
        ind_score = indicator_long if is_long else indicator_short
        pat_score = pattern_long if is_long else pattern_short
        ind_reasons = long_reasons if is_long else short_reasons
        pat_names = pat_long_names if is_long else pat_short_names
        all_reasons = (
            [f"atr({atr_pct:.2f}%)"]
            + [f"PAT:{n}" for n in pat_names]
            + (["NEU:" + ",".join(pat_neutral_names)] if pat_neutral_names else [])
            + ind_reasons
        )
        return Signal(
            direction=direction,
            score=score_combined,
            indicator_score=ind_score,
            pattern_score=pat_score,
            reasons=all_reasons,
            pattern_hits=pat_names,
            price=float(p),
            atr=float(atr_now),
        )

    if combined_long >= cfg.fire_threshold and combined_long > combined_short:
        return _build("long")
    if combined_short >= cfg.fire_threshold and combined_short > combined_long:
        return _build("short")
    return None
