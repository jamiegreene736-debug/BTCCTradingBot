"""Signal generation — pure technicals: candlestick patterns + indicators.

Per user direction: candlestick patterns AND other technicals (RSI, MACD,
EMA, BB, ADX, Supertrend, VWAP, Stoch RSI, Volume) ARE the decision.
No other factors matter.

NO HARD GATES — every signal comes purely from technical agreement.

Architecture:

  SCORING (combined directional score):

    indicator_score = (count of rules agreeing) / 10             # 0-1
    pattern_score   = min(1, sum(pattern strengths) / norm)      # 0-1
    combined        = pattern_weight * pattern + (1-pw) * ind    # 0-1

  Fire if combined >= fire_threshold for one direction AND that side wins.

INDICATOR RULES (counted, non-pattern half — 19 votes):
  1.  EMA stack (fast > mid > slow for long; reverse for short)
  2.  Close cross of EMA fast (bar-to-bar)
  3.  RSI in long/short window
  4.  MACD line vs signal AND histogram rising
  5.  Bollinger basis (close above/below mid)
  6.  Supertrend direction
  7.  VWAP (close above/below institutional volume-weighted price)
  8.  Stoch RSI cross (more sensitive than raw RSI)
  9.  Volume spike (votes for whichever side wins; like ADX)
  10. ADX trend strength (votes for whichever side wins)
  11. HTF (higher-timeframe) trend — close vs EMA on htf_timeframe
  12. Funding rate (contrarian — vote against crowded positioning)
  13. Support/Resistance bounce/rejection (swing-detected levels)
  14. Order book imbalance (live WebSocket depth feed)
  15. RSI divergence (regular = reversal, hidden = continuation)
  16. MACD divergence
  17. OBV divergence (volume-confirmed momentum reversal)
  18. SMC Fair Value Gap (3-candle imbalance respected on revisit)
  19. SMC Liquidity sweep (wick-through-then-reverse stop hunt fade)

PATTERNS (combined into pattern_score):
  - 23 candlestick patterns (1-3 bar) — see patterns.py
  - 7 multi-bar chart patterns — see chart_patterns.py:
      double_top, double_bottom, head_and_shoulders,
      inverse_head_and_shoulders, ascending/descending/symmetric_triangle,
      bull_flag, bear_flag
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Literal

import numpy as np

from . import chart_patterns
from . import divergence as div_mod
from . import levels as levels_mod
from . import patterns
from . import smc as smc_mod
from .config import StrategyCfg
from .indicators import adx as adx_fn
from .indicators import atr as atr_fn
from .indicators import bollinger, ema, macd, mfi, obv, rsi
from .indicators import stoch_rsi as stoch_rsi_fn
from .indicators import supertrend as supertrend_fn
from .indicators import volume_ma as volume_ma_fn
from .indicators import vwap as vwap_fn

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
    volumes: list[float] | None = None,
    htf_closes: list[float] | None = None,
    funding_rate: float | None = None,
    ob_imbalance: float | None = None,
) -> Signal | None:
    if len(closes) < max(cfg.ema_slow, cfg.macd_slow, cfg.bb_period, cfg.atr_period) + 5:
        return None

    o = np.array(opens, dtype=float)
    h = np.array(highs, dtype=float)
    l = np.array(lows, dtype=float)
    c = np.array(closes, dtype=float)
    v = np.array(volumes if volumes is not None else [0.0] * len(c), dtype=float)

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
    vol_ma_v = volume_ma_fn(v, cfg.volume_ma_period) if v.sum() > 0 else None
    vwap_v = vwap_fn(h, l, c, v) if v.sum() > 0 else None
    stoch_k, stoch_d = stoch_rsi_fn(c, cfg.rsi_period, cfg.stoch_rsi_period,
                                     cfg.stoch_rsi_k, cfg.stoch_rsi_d)

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

    # 7. VWAP — close above/below institutional volume-weighted reference.
    if vwap_v is not None and i < len(vwap_v) and not np.isnan(vwap_v[i]):
        if p > vwap_v[i]:
            long_reasons.append("above_vwap")
        elif p < vwap_v[i]:
            short_reasons.append("below_vwap")

    # 8. Stochastic RSI — momentum oscillator. Bull cross in oversold, bear
    # cross in overbought. More sensitive than raw RSI.
    if i < len(stoch_k) and i >= 1:
        k_now, k_prev = stoch_k[i], stoch_k[i - 1]
        d_now, d_prev = stoch_d[i], stoch_d[i - 1]
        if not (np.isnan(k_now) or np.isnan(d_now) or np.isnan(k_prev) or np.isnan(d_prev)):
            # Bullish cross: %K crosses above %D AND we're not already overbought.
            if k_prev <= d_prev and k_now > d_now and k_now < 80:
                long_reasons.append(f"stoch_bull({k_now:.0f})")
            # Bearish cross: %K crosses below %D AND not already oversold.
            if k_prev >= d_prev and k_now < d_now and k_now > 20:
                short_reasons.append(f"stoch_bear({k_now:.0f})")

    # 9. Volume confirmation — a volume spike (current > N× average) votes
    # for whichever side is otherwise winning. Like ADX, direction-agnostic.
    if vol_ma_v is not None and i < len(vol_ma_v):
        vma = vol_ma_v[i]
        if not np.isnan(vma) and vma > 0:
            ratio = v[i] / vma
            if ratio >= cfg.volume_spike_multiplier:
                if len(long_reasons) > len(short_reasons):
                    long_reasons.append(f"vol_spike({ratio:.1f}x)")
                elif len(short_reasons) > len(long_reasons):
                    short_reasons.append(f"vol_spike({ratio:.1f}x)")

    # 10. ADX trend strength — counts for whichever side is otherwise winning.
    a_now = adx_v[i] if i < len(adx_v) else np.nan
    if not np.isnan(a_now) and a_now >= cfg.adx_min:
        if len(long_reasons) > len(short_reasons):
            long_reasons.append(f"adx({a_now:.0f})")
        elif len(short_reasons) > len(long_reasons):
            short_reasons.append(f"adx({a_now:.0f})")

    # 11. Higher-timeframe trend confirmation. Don't trade against the HTF trend.
    if htf_closes is not None and len(htf_closes) >= cfg.htf_ema_period + 1:
        htf_arr = np.array(htf_closes, dtype=float)
        htf_ema = ema(htf_arr, cfg.htf_ema_period)
        if not np.isnan(htf_ema[-1]):
            if htf_arr[-1] > htf_ema[-1]:
                long_reasons.append(f"htf_uptrend({cfg.htf_timeframe})")
            elif htf_arr[-1] < htf_ema[-1]:
                short_reasons.append(f"htf_downtrend({cfg.htf_timeframe})")

    # 12. Funding rate — contrarian. High positive funding (crowded longs)
    # votes SHORT; high negative funding (crowded shorts) votes LONG.
    if funding_rate is not None and abs(funding_rate) >= cfg.funding_threshold:
        if funding_rate > 0:
            short_reasons.append(f"funding+{funding_rate*100:.3f}%")
        else:
            long_reasons.append(f"funding{funding_rate*100:.3f}%")

    # 13. Support/Resistance — bounce off swing-detected support votes long;
    # rejection at swing-detected resistance votes short.
    sr_long, sr_short = levels_mod.detect_sr_signal(
        h, l, c,
        swing_lookback=cfg.swing_lookback,
        cluster_tol_pct=cfg.sr_cluster_tol_pct,
        min_touches=cfg.sr_min_touches,
        proximity_pct=cfg.sr_proximity_pct,
    )
    if sr_long:
        long_reasons.append(sr_long)
    if sr_short:
        short_reasons.append(sr_short)

    # 14. Order book imbalance (top N levels via WebSocket depth feed).
    # imbalance = (bid_vol - ask_vol) / (bid_vol + ask_vol). Strongly positive
    # = bid pressure → vote LONG. Strongly negative = ask pressure → vote SHORT.
    if ob_imbalance is not None:
        if ob_imbalance >= cfg.ob_imbalance_threshold:
            long_reasons.append(f"ob_imb+{ob_imbalance:.2f}")
        elif ob_imbalance <= -cfg.ob_imbalance_threshold:
            short_reasons.append(f"ob_imb{ob_imbalance:.2f}")

    # 15. RSI divergence — bullish: price LL but RSI HL (reversal up).
    #     bearish: price HH but RSI LH. Hidden divergences = continuation.
    # 16. MACD divergence — same logic against MACD line.
    # 17. OBV divergence — same logic against on-balance volume.
    # All three computed in one pass via div_mod.detect_divergences.
    obv_v = obv(c, v) if v.sum() > 0 else np.zeros(len(c))
    div_hits = div_mod.detect_divergences(c, h, l, rsi_v, macd_l, obv_v,
                                           pivot_lookback=cfg.swing_lookback)
    seen_div = set()
    for d in div_hits:
        # One vote per oscillator+side to avoid double-counting hidden+regular.
        osc = d.name.split("_")[0]  # "rsi", "macd", "obv"
        bull_side = d.direction in ("bullish", "hidden_bullish")
        key = (osc, bull_side)
        if key in seen_div:
            continue
        seen_div.add(key)
        if bull_side:
            long_reasons.append(f"DIV:{d.name}")
        else:
            short_reasons.append(f"DIV:{d.name}")

    # 18. SMC: Fair Value Gap respect — price returning to a recent FVG.
    # 19. SMC: Liquidity sweep — wick-through-then-close-inside fade pattern.
    smc_hits = smc_mod.detect_all(h, l, c, swing_lookback=cfg.swing_lookback)
    for s in smc_hits:
        tag = f"SMC:{s.name}"
        if s.direction == "bullish":
            long_reasons.append(tag)
        else:
            short_reasons.append(tag)

    indicator_long = len(long_reasons)
    indicator_short = len(short_reasons)
    INDICATOR_MAX = 19

    # ------------------------------------------------------------------ patterns
    candle_hits = patterns.detect(o, h, l, c)
    chart_hits = chart_patterns.detect_all(h, l, c, cfg.swing_lookback)
    candle_long, candle_short = patterns.score(candle_hits)
    chart_long, chart_short = chart_patterns.score(chart_hits)
    pattern_long = candle_long + chart_long
    pattern_short = candle_short + chart_short
    pattern_long_n = min(1.0, pattern_long / cfg.pattern_norm)
    pattern_short_n = min(1.0, pattern_short / cfg.pattern_norm)

    # ------------------------------------------------------------------ combine
    pw = cfg.pattern_weight
    combined_long = pw * pattern_long_n + (1 - pw) * (indicator_long / INDICATOR_MAX)
    combined_short = pw * pattern_short_n + (1 - pw) * (indicator_short / INDICATOR_MAX)

    pat_long_names = [p.name for p in candle_hits if p.direction == "bullish"]
    pat_short_names = [p.name for p in candle_hits if p.direction == "bearish"]
    pat_neutral_names = [p.name for p in candle_hits if p.direction == "neutral"]
    chart_long_names = [p.name for p in chart_hits if p.direction == "bullish"]
    chart_short_names = [p.name for p in chart_hits if p.direction == "bearish"]

    def _build(direction: Direction) -> Signal:
        is_long = direction == "long"
        score_combined = combined_long if is_long else combined_short
        ind_score = indicator_long if is_long else indicator_short
        pat_score = pattern_long if is_long else pattern_short
        ind_reasons = long_reasons if is_long else short_reasons
        candle_names = pat_long_names if is_long else pat_short_names
        cp_names = chart_long_names if is_long else chart_short_names
        all_reasons = (
            [f"atr({atr_pct:.2f}%)"]
            + [f"PAT:{n}" for n in candle_names]
            + [f"CP:{n}" for n in cp_names]
            + (["NEU:" + ",".join(pat_neutral_names)] if pat_neutral_names else [])
            + ind_reasons
        )
        return Signal(
            direction=direction,
            score=score_combined,
            indicator_score=ind_score,
            pattern_score=pat_score,
            reasons=all_reasons,
            pattern_hits=candle_names + cp_names,
            price=float(p),
            atr=float(atr_now),
        )

    if combined_long >= cfg.fire_threshold and combined_long > combined_short:
        return _build("long")
    if combined_short >= cfg.fire_threshold and combined_short > combined_long:
        return _build("short")
    return None
