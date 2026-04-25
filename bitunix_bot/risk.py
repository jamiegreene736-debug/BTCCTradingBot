"""Risk manager — conservative SL, aggressive TP, leverage-aware sizing.

The user wants:
  * extremely conservative stop loss       -> tight % (e.g. 0.25% of price)
  * fairly aggressive take profit          -> multiple R (default 5R)
  * extremely high leverage                -> amplifies position exposure
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
