"""Signal generation — pure-technical confluence.

Inputs: rolling candles.
Output: a Signal(direction, confluence_score, atr) or None.

Rules (long side — short is mirror):
  1. EMA trend:    ema_fast > ema_mid > ema_slow             (trend stack)
  2. Close cross:  close crossed above ema_fast              (entry trigger)
  3. RSI:          rsi in [rsi_long_min, rsi_long_max]       (momentum window)
  4. MACD:         macd line > signal AND hist > prev hist   (rising momentum)
  5. Bollinger:    close > bb_mid                            (above basis)
  6. ADX:          adx > adx_min                             (trend strength filter)
  7. Supertrend:   supertrend direction = +1 (or -1 short)   (regime filter)

Require `min_confluence` of these 7 rules to fire. ADX + Supertrend are the
biggest whipsaw killers; recommended floor is 4-of-7 for live trading.
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

    long_reasons, short_reasons = [], []

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

    # 6. ADX — trend strength filter (single-sided: counts for whichever side
    # is otherwise winning, since ADX is direction-agnostic).
    a = adx_v[i] if i < len(adx_v) else np.nan
    if not np.isnan(a) and a >= cfg.adx_min:
        long_reasons.append(f"adx_strong({a:.0f})")
        short_reasons.append(f"adx_strong({a:.0f})")

    # 7. Supertrend regime.
    if i < len(st_dir) and not np.isnan(st_dir[i]):
        if st_dir[i] == 1:
            long_reasons.append("supertrend_up")
        elif st_dir[i] == -1:
            short_reasons.append("supertrend_down")

    long_score = len(long_reasons)
    short_score = len(short_reasons)

    if long_score >= cfg.min_confluence and long_score > short_score:
        return Signal("long", long_score, long_reasons, p, float(atr_v[i]) if not np.isnan(atr_v[i]) else 0.0)
    if short_score >= cfg.min_confluence and short_score > long_score:
        return Signal("short", short_score, short_reasons, p, float(atr_v[i]) if not np.isnan(atr_v[i]) else 0.0)
    return None
