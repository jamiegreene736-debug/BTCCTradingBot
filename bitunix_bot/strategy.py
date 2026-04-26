"""Signal generation — pure technicals: candlestick patterns + indicators.

Per user direction: candlestick patterns AND other technicals (RSI, MACD,
EMA, BB, ADX, Supertrend, VWAP, Stoch RSI, Volume) ARE the decision.
No other factors matter.

NO HARD GATES — every signal comes purely from technical agreement.

Architecture:

  SCORING (combined directional score, factor-group based):

    Each indicator vote is classified into ONE of 4 factor groups:
      trend      — directional/trend-following (EMA, MACD, supertrend, …)
      mean_rev   — reversal/exhaustion/level (RSI, BB, divergence, S/R, …)
      flow       — order-flow / tape (CVD, aggression, OB, absorption)
      context    — regime / vol / session (ADX, vol_spike, funding, squeeze)

    Within each group, votes are DEDUPLICATED by category (so 5 correlated
    trend votes don't inflate confidence) and CAPPED at saturation. Each
    group outputs a value in [0, 1].

    factor_score    = Σ weight[g] * min(1, count[g] / saturation[g])    # weighted avg
    pattern_score   = min(1, sum(pattern strengths) / norm)              # 0-1
    combined        = pattern_weight * pattern + (1-pw) * factor_score   # 0-1

  Fire if combined >= effective_fire_threshold for one direction AND that
  side wins. effective_fire_threshold is regime-adaptive (ADX-based) plus
  adaptive self-defense (recent R-tally based — see bot._adaptive_threshold_adjustment).

INDICATOR RULES (counted, non-pattern half — 25 votes):
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
  20. MFI (volume-weighted RSI window)
  21. TTM Squeeze (BB inside Keltner = compression; vote on release direction)
  22. BTC-leader (alts vote with BTC's 1m EMA trend; passed in by bot)
  23. CVD trend — REAL CVD from trade tape (sum of buy/sell aggressor volume
      over 60s) when available; falls back to candle-CVD proxy when offline.
  24. Aggression burst — 10s tape aggression ratio ≥0.40 magnitude. Catches
      momentum surges that happen BETWEEN bar closes (tape-only signal).
  25. Absorption — extreme tape aggression (|ratio| ≥0.55) WITHOUT price
      movement (|10s ΔP| <0.15%). Big players defending the level — votes
      OPPOSITE the aggressive flow direction (classic exhaustion signal).

COMBO-BONUS LAYER (extra vote when 3+ co-fire on the same side):
  - trend_pullback:   ema_stack + ema_cross + (vol_spike OR mfi)
  - squeeze_breakout: squeeze + vol_spike + supertrend
  - smc_reversal:     (FVG OR liquidity_sweep) + divergence + S/R
  - bb_extreme_revert: BB-position + RSI-divergence + S/R
  - crowd_contrarian: funding (opposite side) + divergence + vol_spike

GLOBAL MULTIPLIERS:
  - session_weight: 0.7 dead hours, 1.0 normal, 1.2 high-edge overlaps.
    Multiplied into combined score after pattern+indicator combine.
  - REGIME WEIGHT (ADX-derived): in trending markets (ADX>28), trend-aligned
    signals (ema_stack, supertrend, htf, btc_leader, macd, trend_pullback,
    squeeze_breakout combos) each add +4% to combined score. In ranging
    markets (ADX<18), mean-reversion signals (divergences, S/R, FVG, liquidity
    sweep, smc_reversal/bb_extreme_revert combos) each add +4%. Same vote
    framework — just colored to match what the market is doing.

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
from . import combos as combos_mod
from . import divergence as div_mod
from . import levels as levels_mod
from . import patterns
from . import smc as smc_mod
from .config import StrategyCfg
from .indicators import adx as adx_fn
from .indicators import atr as atr_fn
from .indicators import bollinger, cvd, ema, keltner_channels, macd, mfi, obv, rsi
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
    indicator_score: int               # raw indicator confluence count (kept for compat)
    pattern_score: float               # raw pattern strength sum
    reasons: list[str]                 # human-readable mix of indicators + patterns
    pattern_hits: list[str] = field(default_factory=list)
    price: float = 0.0
    atr: float = 0.0
    # Effective fire threshold used to admit this signal — propagated to the
    # risk manager so conviction-based sizing can scale risk by score/threshold.
    # None when not set (legacy callers); build_order skips conviction-mult.
    fire_threshold_used: float | None = None
    # Per-factor breakdown (each 0..1 normalized) — captured for the journal
    # and dashboard so we can later analyze WHICH factor categories actually
    # predict outcomes. Source of truth for strategy quality calibration.
    factor_trend: float = 0.0
    factor_mean_rev: float = 0.0
    factor_flow: float = 0.0
    factor_context: float = 0.0
    # Last bar's high/low — used by risk.build_order to anchor SL beyond
    # the entry bar's structural extreme rather than pure %-of-entry.
    # Captured at signal-emit time so SL placement reflects what actually
    # just happened on the chart (Grok holistic review).
    last_bar_high: float = 0.0
    last_bar_low: float = 0.0
    # Current VWAP value at signal-emit time. Used as an additional SL
    # anchor when VWAP is on the protective side of entry (below for long,
    # above for short) — getting stopped at VWAP means price returned to
    # fair value and broke through, structural invalidation of the trend
    # thesis (Grok holistic review).
    vwap: float = 0.0

    @property
    def side_code(self) -> str:
        return "BUY" if self.direction == "long" else "SELL"


## ---------------------------------------------------------------- factor groups
#
# Each indicator vote is classified into one of four factor groups:
#   trend      — directional/trend-following votes
#   mean_rev   — reversal/exhaustion/level votes
#   flow       — order-flow / tape-derived votes (highest scalper edge)
#   context    — regime / volatility / session modulators
#
# Within each group, multiple votes are DEDUPLICATED by category so highly
# correlated signals (e.g. 5 trend indicators all firing on the same move)
# count as ONE category, not five. This kills the confluence-inflation
# problem where the same underlying signal is over-counted.
#
# Group scores are capped at saturation (configurable per group), then
# weighted-averaged into the indicator-half of combined_score. Each group
# outputs a value in [0, 1].

# (prefix, group, dedup_key) — first match wins. Order matters: more
# specific prefixes must come before generic ones.
_FACTOR_CLASSIFICATIONS: tuple[tuple[str, str, str], ...] = (
    # ----- Trend group -----
    ("ema_stack_",          "trend",    "ema_stack"),
    ("cross_above_ema",     "trend",    "ema_cross"),
    ("cross_below_ema",     "trend",    "ema_cross"),
    ("macd_up",             "trend",    "macd"),
    ("macd_down",           "trend",    "macd"),
    ("supertrend_",         "trend",    "supertrend"),
    ("above_vwap",          "trend",    "vwap"),
    ("below_vwap",          "trend",    "vwap"),
    ("above_bb_mid",        "trend",    "bb_mid"),
    ("below_bb_mid",        "trend",    "bb_mid"),
    ("htf_uptrend",         "trend",    "htf"),
    ("htf_downtrend",       "trend",    "htf"),
    ("btc_leader_",         "trend",    "btc_leader"),
    ("CMB:trend_pullback",  "trend",    "trend_pullback_combo"),
    ("CMB:squeeze_breakout","trend",    "squeeze_breakout_combo"),
    # ----- Mean-Reversion group -----
    ("rsi(",                "mean_rev", "rsi"),
    ("stoch_",              "mean_rev", "stoch"),
    ("DIV:rsi",             "mean_rev", "div_rsi"),
    ("DIV:macd",            "mean_rev", "div_macd"),
    ("DIV:obv",             "mean_rev", "div_volume"),
    ("DIV:cvd",             "mean_rev", "div_volume"),
    ("SMC:fvg",             "mean_rev", "smc_fvg"),
    ("SMC:liquidity_sweep", "mean_rev", "smc_sweep"),
    ("mfi(",                "mean_rev", "mfi"),
    ("sr_bounce",           "mean_rev", "sr"),
    ("sr_reject",           "mean_rev", "sr"),
    ("CMB:smc_reversal",    "mean_rev", "smc_reversal_combo"),
    ("CMB:bb_extreme_revert","mean_rev","bb_extreme_combo"),
    # ----- Flow group -----
    ("ob_imb",              "flow",     "ob_imb"),
    ("cvd_real",            "flow",     "cvd"),
    ("cvd_proxy",           "flow",     "cvd"),
    ("agg+",                "flow",     "agg"),
    ("agg-",                "flow",     "agg"),
    ("absorb",              "flow",     "absorb"),
    # ----- Context group -----
    ("vol_spike",           "context",  "vol_spike"),
    ("adx(",                "context",  "adx"),
    ("funding",             "context",  "funding"),
    ("squeeze_up",          "context",  "squeeze"),
    ("squeeze_down",        "context",  "squeeze"),
    ("CMB:crowd_contrarian","context",  "crowd_combo"),
)

_FACTOR_GROUPS = ("trend", "mean_rev", "flow", "context")


def _classify_reason(reason: str) -> tuple[str, str] | None:
    """Return (group, dedup_key) for a reason tag, or None if uncategorized."""
    for prefix, group, key in _FACTOR_CLASSIFICATIONS:
        if reason.startswith(prefix):
            return (group, key)
    return None


def factor_score_breakdown(
    reasons: list[str],
    saturation: dict[str, int],
) -> dict[str, float]:
    """Compute per-group normalized scores [0, 1] for a reasons list.

    Returns dict keyed by group name. Counts are deduplicated within each
    group by category — e.g. one score for "ema_stack" regardless of how
    many ema_stack_up tags appear. Saturation caps each group at the
    configured count (e.g. 6 trend votes → 1.0).
    """
    seen: set[tuple[str, str]] = set()
    counts = {grp: 0 for grp in _FACTOR_GROUPS}
    for r in reasons:
        cls = _classify_reason(r)
        if cls is None:
            continue
        if cls in seen:
            continue
        seen.add(cls)
        counts[cls[0]] += 1
    out: dict[str, float] = {}
    for grp in _FACTOR_GROUPS:
        sat = max(1, int(saturation.get(grp, 1)))
        out[grp] = min(1.0, counts[grp] / sat)
    return out


def factor_score_weighted(
    breakdown: dict[str, float],
    weights: dict[str, float],
) -> float:
    """Weighted average of factor-group scores. Returns 0..max(weights_sum)."""
    total = 0.0
    for grp in _FACTOR_GROUPS:
        total += weights.get(grp, 0.0) * breakdown.get(grp, 0.0)
    return total


def _continuation_confirmed(
    direction: Direction,
    highs: np.ndarray,
    lows: np.ndarray,
    closes: np.ndarray,
    aggression_10s: float | None = None,
    real_cvd: float | None = None,
) -> bool:
    """Block 'exhaustion entry' setups (Grok reviews v6 + v7).

    Live data showed the bot firing SHORT on bearish reversal candle
    patterns + lagging bearish trend indicators at local lows — exactly
    when the down-move was already exhausted. 0/48 trades hit positive
    realized PnL because the entry direction was systematically wrong.
    The fundamental issue: confirmation signals are by definition
    LAGGING, so by the time they all align, the move has resolved.

    Confirmation requires SIGNS of continuation in trade direction:

      (a) Last bar's close in the top 25% of its range (long) /
          bottom 25% (short). Bar itself impulsive in our direction.

      (b) Last close beyond prior close in trade direction. Net price
          movement on the entry bar matches our bet.

      (c) Tape alignment when available (Grok v7) — the leading signal:
          aggression_10s in trade direction at ≥0.25 magnitude AND
          real_cvd sign matches direction. If tape data is None,
          gracefully degrades (allows). This catches cases where
          lagging trend confluence aligns but live flow is contrary.

    Returns True if all checks pass. Insufficient data → True (allow).
    """
    if len(closes) < 2:
        return True
    last_high = float(highs[-1])
    last_low = float(lows[-1])
    last_close = float(closes[-1])
    prev_close = float(closes[-2])
    bar_range = last_high - last_low
    if bar_range <= 0:
        return True
    close_pos = (last_close - last_low) / bar_range  # 0=at low, 1=at high
    if direction == "long":
        # Bar mechanics
        if close_pos < 0.75:
            return False
        if last_close <= prev_close:
            return False
        # Tape alignment — only applied when data available (graceful degrade)
        if aggression_10s is not None and aggression_10s < 0.25:
            return False
        if real_cvd is not None and real_cvd < 0:
            return False
    else:
        if close_pos > 0.25:
            return False
        if last_close >= prev_close:
            return False
        if aggression_10s is not None and aggression_10s > -0.25:
            return False
        if real_cvd is not None and real_cvd > 0:
            return False
    return True


def _effective_fire_threshold(adx_val: float, base_threshold: float) -> float:
    """Regime-adaptive fire threshold.

    Lowers the bar in trending markets (ADX > 28) so high-quality trend-
    aligned signals — which already get the +4% per-vote regime boost —
    fire more readily. Raises the bar in chop (ADX < 22) where most
    scalpers bleed; forces stronger pattern + divergence confluence.
    Returns base_threshold when ADX is mid-range or unavailable.

    Ranging band tightened from <18 to <22 after observing live drawdown
    sessions where BTC was clearly chopping (multi-hour consolidation,
    weak directional moves) but ADX sat in the 18-25 "neutral" zone, so
    the bot kept firing marginal signals at the base threshold and
    bleeding fees. <22 covers genuine chop AND weak-trend conditions
    where 1m signals are statistically unreliable.

    Empirical scalping research consistently shows that pure-confluence
    strategies perform best when selectivity matches the market regime.
    """
    if not np.isnan(adx_val):
        if adx_val > 28:
            return max(0.0, base_threshold - 0.05)
        if adx_val < 22:
            return min(1.0, base_threshold + 0.08)
    return base_threshold


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
    btc_trend: int | None = None,    # +1 BTC up, -1 BTC down, 0/None unknown
    session_weight: float | None = None,  # 0.0–1.5; reweights combined score
    real_cvd: float | None = None,        # tape-derived 60s CVD (base-coin units)
    aggression_10s: float | None = None,  # tape-derived 10s aggression in [-1,+1]
    activity_mult: float | None = None,   # tape-derived score multiplier (~0.85–1.10)
    price_change_10s_pct: float | None = None,  # tape-derived 10s price change %
    fire_threshold_override: float | None = None,  # adaptive self-defense override
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
    # All three (and CVD when available) computed in one pass.
    obv_v = obv(c, v) if v.sum() > 0 else np.zeros(len(c))
    cvd_v = cvd(h, l, c, v) if v.sum() > 0 else None
    div_hits = div_mod.detect_divergences(c, h, l, rsi_v, macd_l, obv_v,
                                           pivot_lookback=cfg.swing_lookback,
                                           cvd_v=cvd_v)
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

    # 20. MFI (Money Flow Index) — volume-weighted RSI. Long when not
    # overbought, short when not oversold. Stronger than raw RSI because
    # it confirms with volume.
    if v.sum() > 0:
        mfi_v = mfi(h, l, c, v, cfg.mfi_period)
        mfi_now = mfi_v[i] if i < len(mfi_v) else np.nan
        if not np.isnan(mfi_now):
            if mfi_now < cfg.mfi_long_max:
                long_reasons.append(f"mfi({mfi_now:.0f})")
            if mfi_now > cfg.mfi_short_min:
                short_reasons.append(f"mfi({mfi_now:.0f})")

    # 21. TTM Squeeze — when BB are INSIDE Keltner Channels, market is
    # compressed. We vote in the direction of momentum (close vs Keltner
    # midline) — typical TTM Squeeze release direction tells us where the
    # break is more likely to go.
    k_up, k_mid, k_lo = keltner_channels(h, l, c, cfg.keltner_period,
                                          cfg.keltner_atr_multiplier)
    if (i < len(bb_up) and i < len(k_up)
            and not np.isnan(bb_up[i]) and not np.isnan(k_up[i])):
        squeeze_on = (bb_up[i] < k_up[i]) and (bb_lo[i] > k_lo[i])
        if squeeze_on:
            # Direction by close vs Keltner mid.
            if not np.isnan(k_mid[i]):
                if p > k_mid[i]:
                    long_reasons.append("squeeze_up")
                elif p < k_mid[i]:
                    short_reasons.append("squeeze_down")

    # 22. BTC leader gate — alts shouldn't fight BTC's intraday trend.
    # +1 = BTC uptrend (vote LONG); -1 = BTC downtrend (vote SHORT).
    # The bot passes None for BTC's own evaluation (no self-confirmation).
    if btc_trend == 1:
        long_reasons.append("btc_leader_up")
    elif btc_trend == -1:
        short_reasons.append("btc_leader_down")

    # 23. CVD trend confirmation. Prefer REAL CVD from the trade-tape feed
    # (sum of buy/sell aggressor volume) when available — that's true order
    # flow, the alpha that 1m candles can't capture. Fall back to the
    # candle-derived CVD proxy (`cvd_v`) when the tape feed is offline or
    # too cold to read. Real CVD is direction-only; the candle proxy
    # additionally requires price agreement to fire.
    if real_cvd is not None and abs(real_cvd) > 0:
        if real_cvd > 0:
            long_reasons.append(f"cvd_real+{real_cvd:.2f}")
        else:
            short_reasons.append(f"cvd_real{real_cvd:.2f}")
    elif cvd_v is not None and len(cvd_v) >= 20:
        cvd_recent = cvd_v[-1] - cvd_v[-20]   # last 20-bar net delta
        price_change = c[i] - c[max(0, i - 20)]
        if cvd_recent > 0 and price_change > 0:
            long_reasons.append(f"cvd_proxy+({cvd_recent:.0f})")
        elif cvd_recent < 0 and price_change < 0:
            short_reasons.append(f"cvd_proxy({cvd_recent:.0f})")

    # 24. Aggression burst — short-window (10s) flow lopsidedness from the
    # trade tape. Strong directional aggression (≥70/30 split) is real-time
    # momentum that the candle-based indicators can't see because it
    # happens BETWEEN bar closes. Especially valuable on 1m where a single
    # bar can hide a major directional surge or reversal in its body.
    if aggression_10s is not None:
        if aggression_10s >= 0.40:        # ≥70/30 buy-side
            long_reasons.append(f"agg+{aggression_10s:.2f}")
        elif aggression_10s <= -0.40:     # ≥70/30 sell-side
            short_reasons.append(f"agg{aggression_10s:.2f}")

    # 25. Absorption — extreme tape aggression (≥77/23 split) WITHOUT
    # corresponding price movement (<0.15% in same 10s) means big players
    # are absorbing the aggressive flow at a level. Classic exhaustion
    # signal: aggressive buyers can't push price up = sellers defending
    # = imminent reversal DOWN. (And mirror for buy absorption of sell flow.)
    # Votes the OPPOSITE of the aggressive flow direction.
    if (aggression_10s is not None and price_change_10s_pct is not None
            and abs(aggression_10s) >= 0.55
            and abs(price_change_10s_pct) < 0.15):
        if aggression_10s > 0:
            # Strong buy aggression absorbed at level → sellers defending → SHORT.
            short_reasons.append(f"absorb(buyflow@{aggression_10s:+.2f})")
        else:
            # Strong sell aggression absorbed at level → buyers defending → LONG.
            long_reasons.append(f"absorb(sellflow@{aggression_10s:+.2f})")

    # ----------------- COMBO-BONUS LAYER -----------------
    # Award an extra vote when 3+ specific high-quality reasons co-fire.
    # See combos.py for the recipe definitions. Each combo fires at most
    # once per side per evaluation.
    combo_hits = combos_mod.detect(long_reasons, short_reasons)
    for combo in combo_hits:
        tag = f"CMB:{combo.name}"
        if combo.direction == "bullish":
            long_reasons.append(tag)
        else:
            short_reasons.append(tag)

    indicator_long = len(long_reasons)
    indicator_short = len(short_reasons)

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
    # Factor-group scoring replaces raw vote counting. Each side's votes are
    # classified into 4 groups (trend / mean_rev / flow / context), capped
    # within each group (so 6 correlated trend votes don't inflate score),
    # and weighted-averaged. This kills the "1 weak signal counted 6 times"
    # problem that creates over-confidence in chop.
    factor_breakdown_long = factor_score_breakdown(long_reasons, cfg.factor_saturation)
    factor_breakdown_short = factor_score_breakdown(short_reasons, cfg.factor_saturation)
    factor_long = factor_score_weighted(factor_breakdown_long, cfg.factor_weights)
    factor_short = factor_score_weighted(factor_breakdown_short, cfg.factor_weights)

    pw = cfg.pattern_weight
    combined_long = pw * pattern_long_n + (1 - pw) * factor_long
    combined_short = pw * pattern_short_n + (1 - pw) * factor_short

    # Trend-dominance dampener (Grok review v6 fix). When one side has
    # strong trend confluence (factor_trend ≥ 0.67 = 4+ of 6 distinct
    # trend categories firing), the OPPOSITE side's combined score is
    # halved. Blocks the "established uptrend, but tweezer_top + double_top
    # → fire SHORT" exhaustion-fade pattern that bled the prior session.
    # Reversal trades aren't outright blocked (still allowed if they
    # outscore the trend side), just dampened.
    TREND_DOMINANCE_THRESHOLD = 0.67
    if factor_breakdown_long.get("trend", 0.0) >= TREND_DOMINANCE_THRESHOLD:
        combined_short *= 0.5
    if factor_breakdown_short.get("trend", 0.0) >= TREND_DOMINANCE_THRESHOLD:
        combined_long *= 0.5

    # ----------------- REGIME-AWARE WEIGHTING -----------------
    # ADX-based market regime: trend (>28), neutral, or range (<18). In a
    # trending regime, trend-aligned signals deserve a boost; in chop, mean-
    # reversion signals (divergence, S/R bounce, FVG) deserve the boost.
    # Same signal stack, just weighted to match what the market is doing.
    a_for_regime = adx_v[i] if i < len(adx_v) else float("nan")
    if not np.isnan(a_for_regime):
        TREND_TAGS = ("ema_stack_", "supertrend_", "htf_", "btc_leader_",
                      "macd_up", "macd_down", "CMB:trend_pullback",
                      "CMB:squeeze_breakout", "cvd_real", "cvd_proxy")
        REV_TAGS = ("DIV:", "sr_bounce_", "sr_reject_", "SMC:fvg_",
                    "SMC:liquidity_sweep_", "CMB:smc_reversal",
                    "CMB:bb_extreme_revert", "absorb")
        if a_for_regime > 28:
            # Trending: bump trend-aligned, dampen mean-reversion.
            t_long = sum(1 for r in long_reasons if any(t in r for t in TREND_TAGS))
            t_short = sum(1 for r in short_reasons if any(t in r for t in TREND_TAGS))
            combined_long *= 1.0 + 0.04 * t_long
            combined_short *= 1.0 + 0.04 * t_short
        elif a_for_regime < 18:
            # Ranging: bump mean-reversion, dampen trend-following.
            r_long = sum(1 for r in long_reasons if any(t in r for t in REV_TAGS))
            r_short = sum(1 for r in short_reasons if any(t in r for t in REV_TAGS))
            combined_long *= 1.0 + 0.04 * r_long
            combined_short *= 1.0 + 0.04 * r_short

    # Session reweighting: combined scores get multiplied by session_weight
    # (default 1.0). High-edge sessions can boost; dead hours dampen. The
    # bot passes the multiplier in based on UTC hour.
    if session_weight is not None and session_weight > 0:
        combined_long *= session_weight
        combined_short *= session_weight

    # Activity multiplier — derived from trade-tape print rate. Surge in
    # tape activity = real conviction (boost slightly); below-baseline =
    # market asleep (dampen more, asymmetric clamp 0.85–1.10). Stacks with
    # session_weight so dead-hour + low-activity gets meaningful suppression.
    if activity_mult is not None and activity_mult > 0:
        combined_long *= activity_mult
        combined_short *= activity_mult

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
        breakdown = factor_breakdown_long if is_long else factor_breakdown_short
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
            factor_trend=breakdown.get("trend", 0.0),
            factor_mean_rev=breakdown.get("mean_rev", 0.0),
            factor_flow=breakdown.get("flow", 0.0),
            factor_context=breakdown.get("context", 0.0),
            last_bar_high=float(h[-1]),
            last_bar_low=float(l[-1]),
            vwap=float(vwap_v[i]) if (vwap_v is not None and i < len(vwap_v)
                                        and not np.isnan(vwap_v[i])) else 0.0,
        )

    # Hard ADX floor — don't trade in deep chop. Live drawdown analysis
    # showed the bot firing 0.64+ scores at ADX=15-20 and bleeding via
    # 6-min stale exits (price never moved in predicted direction). The
    # regime-adaptive threshold raise wasn't enough; in deep chop the
    # confluence is pattern noise on flat candles. Skip outright.
    min_adx = float(getattr(cfg, "min_adx_for_trade", 0.0) or 0.0)
    if min_adx > 0 and not np.isnan(a_for_regime) and a_for_regime < min_adx:
        return None

    # Base threshold — `fire_threshold_override` lets the bot inject the
    # adaptive self-defense adjustment (rolling-R-tally based) before regime
    # adaptation. Composes additively: drawdown bumps the BASE up, then
    # regime adjusts further (chop adds more, trending eases).
    base_threshold = (fire_threshold_override
                      if fire_threshold_override is not None
                      else cfg.fire_threshold)

    # Regime-adaptive fire threshold — lower bar in trending markets,
    # higher bar in chop. ADX is the same value already used for regime
    # weighting above, so this composes cleanly: trend regimes both pump
    # trend-tagged votes AND lower the bar; chop both dampens trend votes
    # AND raises the bar.
    eff_threshold = _effective_fire_threshold(a_for_regime, base_threshold)

    # ABSORPTION HARD VETO. The absorption vote (#25) fires when extreme
    # aggression (|ratio| ≥ 0.55) coincides with no meaningful price
    # movement — the classic "big flow being absorbed at a level"
    # signature. It votes OPPOSITE the aggressive flow direction, but as
    # a single +1 it gets drowned out by trend/momentum votes that point
    # WITH the aggressive flow.
    #
    # Live data showed the bot shorting BTC at agg=-0.90/-0.92 with
    # essentially no price movement (textbook sell-side absorption →
    # buyers defending the level), getting stopped out as price reversed
    # up. EVERY single trade in the recent 30-trade window had negative
    # realized PnL — the bot was systematically taking trades INTO the
    # absorbing flow.
    #
    # Veto: when absorption fires, BLOCK any signal in the same direction
    # as the aggressive flow. Reversal trades (opposite direction) are
    # still allowed if they otherwise score above threshold.
    #   absorb(buyflow)  in short_reasons → buyers being absorbed → veto LONG
    #   absorb(sellflow) in long_reasons  → sellers being absorbed → veto SHORT
    absorption_vetoes_short = any(r.startswith("absorb(sellflow")
                                   for r in long_reasons)
    absorption_vetoes_long = any(r.startswith("absorb(buyflow")
                                  for r in short_reasons)

    # Continuation gate (Grok holistic review): SOFT penalty, not hard veto.
    # Live data showed the hard-veto version was over-selective — killed
    # genuine setups during minor noise pullbacks AND let weak setups
    # through during high-vol expansions. Now: deduct 0.10 from combined
    # score when continuation is missing, then re-check threshold. Strong
    # signals (score 0.85+) survive the penalty; marginal signals (score
    # 0.75-0.84) get filtered. This is a graceful filter — it weights
    # continuation as ONE input among many rather than an absolute gate.
    CONTINUATION_PENALTY = 0.10

    cont_long_ok = _continuation_confirmed("long", h, l, c,
                                              aggression_10s=aggression_10s,
                                              real_cvd=real_cvd)
    cont_short_ok = _continuation_confirmed("short", h, l, c,
                                               aggression_10s=aggression_10s,
                                               real_cvd=real_cvd)
    effective_long = combined_long - (0.0 if cont_long_ok else CONTINUATION_PENALTY)
    effective_short = combined_short - (0.0 if cont_short_ok else CONTINUATION_PENALTY)

    # Optional signal-inversion gate (forward-looking experiment). All
    # gates above (continuation penalty, absorption, threshold) run on the
    # ORIGINAL direction so we only invert genuine high-conviction setups.
    invert = bool(getattr(cfg, "invert_signals", False))

    def _emit(direction: Direction, cont_ok: bool) -> Signal:
        sig = _build(direction)
        sig.fire_threshold_used = eff_threshold
        if not cont_ok:
            sig.reasons = ["CONT_PENALTY"] + sig.reasons
            log.info("Continuation penalty applied to %s signal (score %.2f → "
                     "%.2f after -%.2f); still cleared threshold %.2f",
                     direction, sig.score, sig.score - CONTINUATION_PENALTY,
                     CONTINUATION_PENALTY, eff_threshold)
        if invert:
            flipped: Direction = "short" if direction == "long" else "long"
            sig.direction = flipped
            sig.reasons = ["INVERTED"] + sig.reasons
            log.info("INVERTED signal: %s → %s (score=%.2f)",
                     direction, flipped, sig.score)
        return sig

    if effective_long >= eff_threshold and effective_long > effective_short:
        if absorption_vetoes_long:
            log.info("ABSORPTION VETO long signal (buyflow being absorbed → "
                     "expect reversal down, not continuation up)")
            return None
        return _emit("long", cont_long_ok)
    if effective_short >= eff_threshold and effective_short > effective_long:
        if absorption_vetoes_short:
            log.info("ABSORPTION VETO short signal (sellflow being absorbed → "
                     "expect reversal up, not continuation down)")
            return None
        return _emit("short", cont_short_ok)
    return None
