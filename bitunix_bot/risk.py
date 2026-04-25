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
) -> OrderPlan | None:
    """Compute volume + SL/TP for an order plan.

    `effective_leverage` overrides `trading.leverage` to handle per-symbol
    caps (Bitunix tops out leverage differently per symbol — e.g. BTCUSDT
    200x but SOLUSDT 75x). If the symbol cap is lower than config, sizing
    must use the cap or we'll silently oversize.
    """
    price = signal.price
    if price <= 0 or free_margin <= 0:
        return None

    leverage = effective_leverage if effective_leverage is not None else trading.leverage

    # Stop distance in price units.
    if risk.use_atr and signal.atr > 0:
        stop_dist = signal.atr * risk.atr_multiplier_sl
        tp_dist = signal.atr * risk.atr_multiplier_tp
    else:
        stop_dist = price * (risk.stop_loss_pct / 100.0)
        tp_dist = stop_dist * risk.take_profit_r

    if stop_dist <= 0:
        return None

    # Risk-budgeted volume (base currency units).
    risk_usdt = free_margin * (trading.risk_per_trade_pct / 100.0)
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
