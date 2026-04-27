"""Risk manager — fee-aware SL, aggressive TP, leverage-aware sizing.

The user wants:
  * fee-aware stop loss                    -> tight % of price (e.g. 0.40%
                                              chosen to balance noise tolerance
                                              vs fee-as-pct-of-risk)
  * fairly aggressive take profit          -> multiple R (default 2.5R)
  * pro-scalper leverage                   -> 25x sweet spot
  * % risk per trade                       -> caps loss as % of free margin

Sizing math:
  risk_amount_usdt = free_margin * risk_per_trade_pct / 100
  stop_distance    = price * stop_loss_pct / 100      (or atr-based)
  volume_base      = risk_amount_usdt / stop_distance
  required_margin  = (volume_base * price) / leverage
  -> if required_margin > free_margin, scale volume down so it fits.
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass

from .config import RiskCfg, TradingCfg
from .strategy import Signal

log = logging.getLogger(__name__)


@dataclass
class OrderPlan:
    side: str            # "BUY" or "SELL"
    volume: float
    price: float
    stop_loss: float
    take_profit: float
    leverage: int
    notes: str


def adaptive_tp_r(
    age_minutes: float,
    original_tp_r: float,
    fee_pct: float,
    sl_pct: float,
    floor_r: float = 0.7,
    bar_minutes: float = 1.0,
) -> float:
    """Return the desired TP distance in R-multiples based on position age.

    Methodology: as a flat trade ages, lower the bar on what counts as a
    "win" — but never below a level that's still profitable after fees.

    Tier ladder is expressed in BARS (not minutes), then converted to
    minutes via `bar_minutes`. This keeps the ladder semantics consistent
    across timeframes — "5 bars in" means the same thing regardless of
    whether each bar is 1m, 5m, 15m, or 1h:
      0–5 bars:   original (full target — still ambitious)
      5–10 bars:  80% of original
      10–20 bars: 60% of original
      20–40 bars: 40% of original (or floor, whichever is higher)
      40+ bars:   floor_r (minimum profitable after fees)

    Examples:
      bar_minutes=1.0  → tiers at 5/10/20/40 minutes (legacy 1m behavior)
      bar_minutes=15.0 → tiers at 75/150/300/600 minutes (15m timeframe)

    Floor is enforced: TP must stay ≥ floor_r × SL distance, AND must
    cover fees with a small margin (fee_pct + 50% buffer in % of price).
    Returns R-multiple (NOT a price); caller multiplies by SL distance.
    """
    # Floor must beat fees with margin. Convert fee_pct to R units:
    #   1R = sl_pct of price; fees = fee_pct of price → fees in R = fee_pct / sl_pct.
    fee_r = (fee_pct / sl_pct) if sl_pct > 0 else 0.0
    safety_floor = max(floor_r, fee_r * 1.5)

    # Convert bar-based tiers to minutes via timeframe scale.
    bar_min = max(0.001, float(bar_minutes))
    if age_minutes < 5 * bar_min:
        return max(original_tp_r, safety_floor)
    if age_minutes < 10 * bar_min:
        return max(original_tp_r * 0.80, safety_floor)
    if age_minutes < 20 * bar_min:
        return max(original_tp_r * 0.60, safety_floor)
    if age_minutes < 40 * bar_min:
        return max(original_tp_r * 0.40, safety_floor)
    return safety_floor


def parse_timeframe_minutes(tf: str) -> float:
    """Parse a Bitunix timeframe string (e.g. '15m', '1h', '4h', '1d')
    into minutes. Defaults to 1.0 on unrecognized input — caller code
    that uses this for tier scaling falls back to the legacy 1m
    behavior, which is conservative."""
    if not tf or len(tf) < 2:
        return 1.0
    try:
        n = float(tf[:-1])
    except (ValueError, TypeError):
        return 1.0
    unit = tf[-1].lower()
    if unit == "m":
        return n
    if unit == "h":
        return n * 60.0
    if unit == "d":
        return n * 1440.0
    return 1.0


def build_order(
    signal: Signal,
    free_margin: float,
    trading: TradingCfg,
    risk: RiskCfg,
    min_volume: float = 0.01,
    volume_step: float = 0.01,
    digits: int = 2,
    effective_leverage: int | None = None,
    symbol: str | None = None,
    dd_risk_mult: float = 1.0,
) -> OrderPlan | None:
    """Compute volume + SL/TP for an order plan.

    `effective_leverage` overrides `trading.leverage` to handle per-symbol
    caps (Bitunix tops out leverage differently per symbol — e.g. BTCUSDT
    200x but SOLUSDT 75x). If the symbol cap is lower than config, sizing
    must use the cap or we'll silently oversize.

    `symbol` enables correlation-adjusted sizing via
    `trading.symbol_risk_mult` (default 1.0 if symbol not in the map). Used
    to down-weight high-beta alts that are heavily BTC-correlated, keeping
    effective portfolio risk closer to nominal across multi-symbol books.
    """
    price = signal.price
    if price <= 0 or free_margin <= 0:
        return None

    leverage = effective_leverage if effective_leverage is not None else trading.leverage

    # Per-symbol risk multiplier (correlation/beta adjustment).
    sym_mult = 1.0
    if symbol is not None:
        sym_mult = trading.symbol_risk_mult.get(symbol.upper(), 1.0)
        if sym_mult <= 0:
            sym_mult = 1.0  # guard against pathological config

    # Conviction-based dynamic risk multiplier. The composite score already
    # quantifies signal strength on a 0-1 scale; instead of sizing every
    # signal identically, scale risk by score / fire_threshold (clamped
    # 0.7–1.5). High-conviction signals (score well above the threshold)
    # get up to 1.5× risk; marginal signals (right at the threshold) get
    # 0.7× — average risk preserved across the distribution while
    # compounding more aggressively on the best setups.
    #
    # Skipped (multiplier=1.0) when fire_threshold_used is None — preserves
    # legacy callers that build orders without a threshold context.
    conviction_mult = 1.0
    if signal.fire_threshold_used is not None and signal.fire_threshold_used > 0:
        raw_ratio = signal.score / signal.fire_threshold_used
        conviction_mult = max(0.7, min(1.5, raw_ratio))

    # Stop distance in price units. Hybrid: fixed-pct FLOOR + ATR-derived
    # widening + STRUCTURE-anchored extension (Grok holistic review).
    #
    # Three components, take MAX of all three:
    #   (a) fixed_pct floor — protects calm markets from bar-internal noise
    #   (b) atr_multiplier × ATR% — widens in vol regimes (news, cascades)
    #   (c) structure-anchor — distance from entry to last bar's extreme
    #       (low for longs, high for shorts) plus a small buffer. Ensures
    #       SL sits BEYOND the bar that just confirmed our direction —
    #       a real structural invalidation level, not an arbitrary % point.
    #
    # The structure anchor matters most when ATR is small but the entry
    # bar had a long wick. Pure %-of-price SL would land INSIDE the wick
    # and get hit on the next bar's natural retest. The anchor places SL
    # safely beyond the wick.
    if risk.use_atr and signal.atr > 0:
        atr_pct = (signal.atr / price) * 100.0
        atr_derived_pct = risk.atr_multiplier_sl * atr_pct
        stop_dist_pct = max(risk.stop_loss_pct, atr_derived_pct)
    else:
        stop_dist_pct = risk.stop_loss_pct
    stop_dist = price * (stop_dist_pct / 100.0)

    # Structure-anchored component: distance from entry to last bar's
    # opposing extreme + 10bps buffer (so SL is JUST beyond the wick).
    # Falls back to 0 if Signal lacks bar high/low (legacy callers).
    structure_buffer_pct = 0.10  # 10 bps beyond the bar's wick
    last_high = getattr(signal, "last_bar_high", 0.0) or 0.0
    last_low = getattr(signal, "last_bar_low", 0.0) or 0.0
    if last_high > 0 and last_low > 0 and last_high > last_low:
        if signal.direction == "long":
            # SL must be below last_low (long invalidates if price breaks
            # the low of the bar that just gave us our entry signal).
            anchor_dist = (price - last_low) + price * (structure_buffer_pct / 100.0)
        else:
            # SL must be above last_high (short invalidates if price
            # breaks above the entry-bar's high).
            anchor_dist = (last_high - price) + price * (structure_buffer_pct / 100.0)
        anchor_dist = max(anchor_dist, 0.0)
        stop_dist = max(stop_dist, anchor_dist)

    # VWAP anchor (Grok holistic review): when VWAP is on the protective
    # side of entry (below for long, above for short), use it as an
    # additional structural SL. Logic: price extended away from fair
    # value gets stopped if it returns and breaks fair value. SKIP when
    # VWAP is on the SAME side as direction (e.g. long with VWAP above
    # entry = mean-reversion long where VWAP is the target, not the SL).
    sig_vwap = getattr(signal, "vwap", 0.0) or 0.0
    if sig_vwap > 0:
        if signal.direction == "long" and sig_vwap < price:
            # Long above VWAP: SL must reach below-VWAP for thesis-break.
            vwap_anchor = (price - sig_vwap) + price * (structure_buffer_pct / 100.0)
            stop_dist = max(stop_dist, vwap_anchor)
        elif signal.direction == "short" and sig_vwap > price:
            # Short below VWAP: SL must reach above-VWAP for thesis-break.
            vwap_anchor = (sig_vwap - price) + price * (structure_buffer_pct / 100.0)
            stop_dist = max(stop_dist, vwap_anchor)

    # Volume-profile HVN anchor (Grok holistic review): high-volume nodes
    # are price levels with significant historical trading activity —
    # strong structural support/resistance. For SL placement:
    #   long  → SL extends below nearest HVN below entry (support level
    #           breaks = thesis breaks)
    #   short → SL extends above nearest HVN above entry (resistance
    #           level breaks = thesis breaks)
    # When the HVN is on the wrong side (long with hvn_below = 0, etc.),
    # the anchor doesn't fire; falls through to other anchors.
    hvn_below = getattr(signal, "hvn_below", 0.0) or 0.0
    hvn_above = getattr(signal, "hvn_above", 0.0) or 0.0
    if signal.direction == "long" and hvn_below > 0 and hvn_below < price:
        hvn_anchor = (price - hvn_below) + price * (structure_buffer_pct / 100.0)
        stop_dist = max(stop_dist, hvn_anchor)
    elif signal.direction == "short" and hvn_above > 0 and hvn_above > price:
        hvn_anchor = (hvn_above - price) + price * (structure_buffer_pct / 100.0)
        stop_dist = max(stop_dist, hvn_anchor)

    # TP is always an R-multiple of effective SL distance — keeps geometry
    # consistent regardless of which SL formula bound. The legacy
    # atr_multiplier_tp config is retained for backwards-compat but unused
    # in the hybrid path; remove from config to fully decommission.
    tp_dist = stop_dist * risk.take_profit_r

    # Volume-profile TP anchor: when HVN exists in the trade direction,
    # use it as a TP target (resistance for long = where price is likely
    # to bounce off; support for short = where shorts typically cover).
    # We TIGHTEN the TP toward the HVN if the HVN is closer than the
    # R-multiple target — capturing the realistic move instead of
    # holding for an unrealistic R-multiple. NEVER expand TP past the
    # R-multiple (hvn-anchor only constrains down, not up).
    if signal.direction == "long" and hvn_above > 0 and hvn_above > price:
        # Take profit just below the HVN (price likely bounces off resistance).
        hvn_tp_dist = (hvn_above - price) - price * (structure_buffer_pct / 100.0)
        if hvn_tp_dist > 0:
            tp_dist = min(tp_dist, hvn_tp_dist)
    elif signal.direction == "short" and hvn_below > 0 and hvn_below < price:
        hvn_tp_dist = (price - hvn_below) - price * (structure_buffer_pct / 100.0)
        if hvn_tp_dist > 0:
            tp_dist = min(tp_dist, hvn_tp_dist)

    if stop_dist <= 0:
        return None

    # Risk-budgeted volume (base currency units), down-weighted by:
    #   - symbol's correlation/beta multiplier (1.0 for BTC, ~0.70-0.85 alts)
    #   - conviction multiplier (score/threshold ratio, 0.7–1.5)
    #   - daily-drawdown multiplier (1.0 normal → 0.25 deep DD → 0.0 halted)
    # The DD throttle kicks in PRE-halt to protect capital gradually rather
    # than tripping a binary cliff at -8%.
    risk_usdt = (free_margin * (trading.risk_per_trade_pct / 100.0)
                 * sym_mult * conviction_mult * dd_risk_mult)
    if risk_usdt <= 0:
        return None  # halted by DD breaker

    # Fee reserve (Grok holistic review): risk_per_trade_pct represents the
    # intended LOSS budget. But fees are paid REGARDLESS of outcome — so
    # actual loss-on-SL = risk_usdt + round_trip_fee × notional. To keep
    # actual loss ≈ risk_usdt budget, scale risk down by the fee burden in
    # R-units: fee_burden = round_trip_fee_pct / stop_loss_pct. For 0.08%
    # maker fees / 0.25% SL = 0.32 (fees consume 32% of risk budget).
    # Capped at 60% (Grok rescan: was 50%) — leaves a minimum of 40% of
    # the risk budget for actual stop-distance, while reserving more for
    # fees in high-fee regimes (0.20% taker / 0.25% SL = 80% burden,
    # capped to 60%).
    sl_pct_for_fee = max(0.001, risk.stop_loss_pct)
    fee_burden_r = risk.round_trip_fee_pct / sl_pct_for_fee
    fee_reserve_frac = min(0.6, fee_burden_r)
    risk_usdt_after_fees = risk_usdt * (1.0 - fee_reserve_frac)
    volume = risk_usdt_after_fees / stop_dist

    # Cap by available margin at the EFFECTIVE leverage.
    max_vol_by_margin = (free_margin * leverage) / price
    volume = min(volume, max_vol_by_margin * 0.98)  # 2% safety buffer

    # Round DOWN to volume step and clamp to min.
    steps = math.floor(volume / volume_step)
    volume = steps * volume_step
    if volume < min_volume:
        log.info("Order skipped: volume %.6f < min %.6f (risk=%.2f, "
                 "after_fees=%.2f, stop_dist=%.6f)",
                 volume, min_volume, risk_usdt, risk_usdt_after_fees, stop_dist)
        return None

    # SL / TP prices.
    if signal.direction == "long":
        sl = round(price - stop_dist, digits)
        tp = round(price + tp_dist, digits)
    else:
        sl = round(price + stop_dist, digits)
        tp = round(price - tp_dist, digits)

    return OrderPlan(
        side=signal.side_code,
        volume=round(volume, 6),
        price=round(price, digits),
        stop_loss=sl,
        take_profit=tp,
        leverage=leverage,
        notes=f"conf={signal.score} reasons={','.join(signal.reasons)}",
    )
