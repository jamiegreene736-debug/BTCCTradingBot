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
) -> float:
    """Return the desired TP distance in R-multiples based on position age.

    Methodology: as a flat trade ages, lower the bar on what counts as a
    "win" — but never below a level that's still profitable after fees.

    Tier ladder (1m timeframe assumed):
      0–5 min:   original (e.g. 1.5R = full target)
      5–10 min:  80% of original (still ambitious)
      10–20 min: 60% of original
      20–40 min: 40% of original (or floor, whichever is higher)
      40+ min:   floor_r (minimum profitable after fees)

    Floor is enforced: TP must stay ≥ floor_r × SL distance, AND must
    cover fees with a small margin (fee_pct + 50% buffer in % of price).
    Returns R-multiple (NOT a price); caller multiplies by SL distance.
    """
    # Floor must beat fees with margin. Convert fee_pct to R units:
    #   1R = sl_pct of price; fees = fee_pct of price → fees in R = fee_pct / sl_pct.
    fee_r = (fee_pct / sl_pct) if sl_pct > 0 else 0.0
    safety_floor = max(floor_r, fee_r * 1.5)

    if age_minutes < 5:
        return max(original_tp_r, safety_floor)
    if age_minutes < 10:
        return max(original_tp_r * 0.80, safety_floor)
    if age_minutes < 20:
        return max(original_tp_r * 0.60, safety_floor)
    if age_minutes < 40:
        return max(original_tp_r * 0.40, safety_floor)
    return safety_floor


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
    # widening in vol regimes. The floor (`stop_loss_pct`) protects calm-market
    # entries from being stopped out on bar-internal noise; the ATR multiplier
    # widens the stop when realized volatility expands (FOMC, news, cascades)
    # so we don't get wicked out by the same setup that would normally work.
    # Risk per trade stays constant — notional auto-adjusts to whichever
    # SL distance ends up binding.
    #
    # Calm BTC (ATR ~0.05% × 1.2 = 0.06%): floor wins → SL at 0.40%.
    # Vol expansion (ATR ~0.5% × 1.2 = 0.60%): ATR wins → SL widens to 0.60%.
    if risk.use_atr and signal.atr > 0:
        atr_pct = (signal.atr / price) * 100.0
        atr_derived_pct = risk.atr_multiplier_sl * atr_pct
        stop_dist_pct = max(risk.stop_loss_pct, atr_derived_pct)
        stop_dist = price * (stop_dist_pct / 100.0)
    else:
        stop_dist = price * (risk.stop_loss_pct / 100.0)

    # TP is always an R-multiple of effective SL distance — keeps geometry
    # consistent regardless of which SL formula bound. The legacy
    # atr_multiplier_tp config is retained for backwards-compat but unused
    # in the hybrid path; remove from config to fully decommission.
    tp_dist = stop_dist * risk.take_profit_r

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
    volume = risk_usdt / stop_dist

    # Cap by available margin at the EFFECTIVE leverage.
    max_vol_by_margin = (free_margin * leverage) / price
    volume = min(volume, max_vol_by_margin * 0.98)  # 2% safety buffer

    # Round DOWN to volume step and clamp to min.
    steps = math.floor(volume / volume_step)
    volume = steps * volume_step
    if volume < min_volume:
        log.info("Order skipped: volume %.6f < min %.6f (risk=%.2f, stop_dist=%.6f)",
                 volume, min_volume, risk_usdt, stop_dist)
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
