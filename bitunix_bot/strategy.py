"""Signal generation — pure-technical confluence.

Inputs: rolling candles.
Output: a Signal(direction, confluence_score, atr) or None.

Architecture: hard gates first, then confluence count.

HARD GATE:
  ADX > adx_min  — if false, NO signal regardless of anything else.
                   This is how freqtrade strategies actually wire ADX
                   (see Strategy005, FSupertrendStrategy). Without it,
                   pure noise can still hit confluence on lagging
                   indicators alone.

CONFLUENCE RULES (require min_confluence to agree):
  1. EMA trend:    ema_fast > ema_mid > ema_slow            (trend stack)
  2. Close cross:  close crossed above ema_fast             (entry trigger)
  3. RSI:          rsi in [rsi_long_min, rsi_long_max]      (momentum window)
  4. MACD:         macd line > signal AND hist > prev hist  (rising momentum)
  5. Bollinger:    close > bb_mid                           (above basis)
  6. Supertrend:   direction = +1 (or -1 short)             (regime filter)

Recommended min_confluence is 3-of-6 for aggressive, 4-of-6 for selective.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Literal

import numpy as np

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
    score: int            # how many rules agreed
    reasons: list[str]
    price: float          # close of last bar
    atr: float            # for ATR-based SL sizing

    @property
    def side_code(self) -> str:
        # Bitunix: "BUY" / "SELL".
        return "BUY" if self.direction == "long" else "SELL"


def evaluate(
    highs: list[float],
    lows: list[float],
    closes: list[float],
    cfg: StrategyCfg,
) -> Signal | None:
    if len(closes) < max(cfg.ema_slow, cfg.macd_slow, cfg.bb_period, cfg.atr_period) + 5:
        return None

    h = np.array(highs, dtype=float)
    l = np.array(lows, dtype=float)
    c = np.array(closes, dtype=float)

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

    # HARD GATE 1: no trades in chop (ADX-based).
    a_now = adx_v[i] if i < len(adx_v) else np.nan
    if np.isnan(a_now) or a_now < cfg.adx_min:
        return None

    # HARD GATE 2: no trades in dead markets (ATR/price too low to ever clear
    # fees + reach TP within reasonable time). 0.08% default.
    atr_now = atr_v[i] if i < len(atr_v) else np.nan
    if np.isnan(atr_now) or p <= 0:
        return None
    atr_pct = (atr_now / p) * 100.0
    if atr_pct < cfg.min_atr_pct:
        return None

    long_reasons, short_reasons = [f"adx({a_now:.0f})"], [f"adx({a_now:.0f})"]

    # 1. EMA trend.
    if ema_f[i] > ema_m[i] > ema_s[i]:
        long_reasons.append("ema_stack_up")
    if ema_f[i] < ema_m[i] < ema_s[i]:
        short_reasons.append("ema_stack_down")

    # 2. Close cross of ema_fast.
    if i >= 1:
        if c[i - 1] <= ema_f[i - 1] and c[i] > ema_f[i]:
            long_reasons.append("cross_above_ema_fast")
        if c[i - 1] >= ema_f[i - 1] and c[i] < ema_f[i]:
            short_reasons.append("cross_below_ema_fast")

    # 3. RSI window.
    r = rsi_v[i]
    if not np.isnan(r):
        if cfg.rsi_long_min <= r <= cfg.rsi_long_max:
            long_reasons.append(f"rsi_ok({r:.1f})")
        if cfg.rsi_short_min <= r <= cfg.rsi_short_max:
            short_reasons.append(f"rsi_ok({r:.1f})")

    # 4. MACD.
    if i >= 1 and not np.isnan(macd_h[i - 1]):
        if macd_l[i] > macd_s[i] and macd_h[i] > macd_h[i - 1]:
            long_reasons.append("macd_up")
        if macd_l[i] < macd_s[i] and macd_h[i] < macd_h[i - 1]:
            short_reasons.append("macd_down")

    # 5. Bollinger basis.
    if not np.isnan(bb_mid[i]):
        if p > bb_mid[i]:
            long_reasons.append("above_bb_mid")
        if p < bb_mid[i]:
            short_reasons.append("below_bb_mid")

    # 6. Supertrend regime.
    if i < len(st_dir) and not np.isnan(st_dir[i]):
        if st_dir[i] == 1:
            long_reasons.append("supertrend_up")
        elif st_dir[i] == -1:
            short_reasons.append("supertrend_down")

    # ADX is a hard gate (see top), not counted in confluence.
    long_score = len(long_reasons) - 1   # subtract the adx prefix
    short_score = len(short_reasons) - 1

    if long_score >= cfg.min_confluence and long_score > short_score:
        return Signal("long", long_score, long_reasons, p, float(atr_v[i]) if not np.isnan(atr_v[i]) else 0.0)
    if short_score >= cfg.min_confluence and short_score > long_score:
        return Signal("short", short_score, short_reasons, p, float(atr_v[i]) if not np.isnan(atr_v[i]) else 0.0)
    return None
