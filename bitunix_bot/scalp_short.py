"""Parabolic-exhaustion short scalps: 1-2h holds at up to the tier's leverage.

Deterministic and network-free, like ``intraday``. Shorts only. The candidate
must be extended, climactic and crowded; the trigger is a failed high on the
1m/3m bars; the stop sits just above that high and must fit inside the
estimated isolated-margin liquidation distance at the planned leverage.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .indicators import ema
from .intraday import (
    INTERVALS,
    Candle,
    Check,
    Decision,
    Market,
    TradePlan,
    ema_bias,
    session_vwap,
    trend,
    volatility,
)
from .signal_config import ScalpCfg, SignalsCfg, SignalSettings

PROFILE = "scalp_short"
SETUP_NAME = "Failed high"

SCALP_CHECK_GROUPS: dict[str, str] = {
    "Fresh market data": "market",
    "Liquid market": "market",
    "Spread": "market",
    "Leverage tier": "market",
    "Mark vs last": "market",
    "Funding print window": "market",
    "Extended above value": "market",
    "Climax volume": "market",
    "Crowded longs": "market",
    "BTC context": "market",
    "Failed high trigger": "setup",
    "Flow divergence": "setup",
    "Entry zone": "plan",
    "Stop inside liquidation": "plan",
    "Stop outside noise": "plan",
    "Mean-reversion target": "plan",
    "Reward after costs": "plan",
    "Order size": "plan",
    "Execution depth": "plan",
    "Tracked exposure": "portfolio",
}
SCALP_CHECK_LABELS: tuple[str, ...] = tuple(SCALP_CHECK_GROUPS)
CANDIDATE_LABELS = ("Extended above value", "Climax volume", "Crowded longs")
SETUP_LABELS = ("Failed high trigger", "Flow divergence")
SCALP_PLAN_LABELS: tuple[str, ...] = tuple(
    label for label, group in SCALP_CHECK_GROUPS.items() if group == "plan"
)
WAITING_CANDIDATE = "Waiting for an extended, climactic, crowded market"
WAITING_TRIGGER = "Waiting for a completed failed-high bar"


def scalp_check(label: str, passed: bool, detail: str, *, waiting: bool = False) -> Check:
    return Check(label, passed, detail, SCALP_CHECK_GROUPS[label], waiting)


def scalp_waiting(label: str, detail: str) -> Check:
    return scalp_check(label, False, detail, waiting=True)


def order_scalp_checks(checks: list[Check]) -> list[Check]:
    by_label = {item.label: item for item in checks}
    return [
        by_label[label]
        if label in by_label
        else scalp_waiting(label, "Waiting for evaluation")
        for label in SCALP_CHECK_LABELS
    ]


def blank_scalp_checklist(detail: str) -> list[Check]:
    return order_scalp_checks(
        [scalp_waiting(label, detail) for label in SCALP_CHECK_LABELS]
    )


def volume_delta(candle: Candle) -> float:
    """Candle-based proxy for aggressor flow: volume weighted by close location.

    A close at the high counts as full buying, at the low as full selling. The
    public kline feed has no taker split, so this is the deterministic stand-in
    for cumulative volume delta.
    """
    span = candle.high - candle.low
    if span <= 0:
        return 0.0
    return candle.volume * ((2 * candle.close - candle.high - candle.low) / span)


@dataclass(frozen=True)
class FailedHigh:
    spike_index: int
    spike_high: float
    stop: float
    base: float
    flow_delta: float


def find_failed_high(bars: list[Candle], atr_value: float, cfg: ScalpCfg) -> FailedHigh | None:
    """A spike high that price retested and rejected on a completed bar.

    The spike is the highest high of the lookback window. It must be old enough
    for a rejection to have printed and young enough to still be a scalp. The
    last completed bar must close below the spike bar's body without having
    printed a new high. Stop: the spike high plus a small ATR buffer.
    """
    if len(bars) < cfg.spike_lookback + 25:
        return None
    window = bars[-cfg.spike_lookback :]
    offset = len(bars) - len(window)
    spike_local = max(range(len(window)), key=lambda i: window[i].high)
    spike_index = offset + spike_local
    age = len(bars) - 1 - spike_index
    if not cfg.spike_min_age_bars <= age <= cfg.spike_max_age_bars:
        return None
    spike = bars[spike_index]
    after = bars[spike_index + 1 :]
    last = bars[-1]
    if any(c.high >= spike.high for c in after):
        return None
    body_low = min(spike.open, spike.close)
    if not (last.close < body_low and last.close < last.open):
        return None
    base = min(c.low for c in bars[max(0, spike_index - 20) : spike_index])
    flow = sum(volume_delta(c) for c in bars[spike_index:])
    return FailedHigh(
        spike_index,
        spike.high,
        spike.high + cfg.stop_atr_buffer * atr_value,
        base,
        flow,
    )


def climax_volume(bars: list[Candle], cfg: ScalpCfg) -> float:
    window = bars[-cfg.climax_lookback :]
    before = bars[-(cfg.climax_lookback + 20) : -cfg.climax_lookback]
    baseline = sum(c.volume for c in before) / len(before) if before else 0.0
    if baseline <= 0:
        return 0.0
    return max(c.volume for c in window) / baseline


def scalp_targets(
    levels: list[float],
    entry: float,
    stop_distance: float,
    cost_pct: float,
    travel: float,
    cfg: ScalpCfg,
) -> list[float]:
    """Nearest mean-reversion levels below entry that clear R inside the travel budget."""
    risk_fraction = (stop_distance / entry * 100 + cost_pct) / 100
    if risk_fraction <= 0 or travel <= 0:
        return []
    chosen: list[float] = []
    for price in sorted({p for p in levels if 0 < entry - p <= travel}, reverse=True):
        reward = (entry - price) / entry - cost_pct / 100
        if reward / risk_fraction >= cfg.min_reward_risk:
            chosen.append(price)
    return chosen


def liquidation_fit(
    market: Market,
    entry: float,
    stop: float,
    notional: float,
    cost_pct: float,
    atr_value: float,
    leverage: int,
    buffer_pct: float,
) -> tuple[int, float | None, object]:
    """Highest leverage whose estimated short liquidation stays above the stop."""
    tier = next(
        (
            t
            for t in sorted(market.tiers, key=lambda t: t.minimum, reverse=True)
            if t.minimum <= notional <= t.maximum
        ),
        None,
    )
    if tier is None:
        return 0, None, None
    adverse_basis = min(0.0, -(market.mark - market.price))
    buffer = max(entry * buffer_pct / 100, atr_value * 0.5)
    max_leverage, liquidation = 0, None
    for level in range(1, tier.max_leverage + 1):
        estimated = entry * (1 + 1 / level - cost_pct / 100) / (1 + tier.maintenance_rate)
        if (estimated - stop) + adverse_basis >= buffer:
            max_leverage = level
        if level == leverage:
            liquidation = estimated
    return max_leverage, liquidation, tier


def build_scalp_plan(
    market: Market,
    bars: list[Candle],
    quarter: list[Candle],
    hourly: list[Candle],
    setup: FailedHigh,
    now: int,
    settings: SignalSettings,
    cfg: SignalsCfg,
) -> tuple[TradePlan | None, list[Check]]:
    scalp = cfg.scalp
    entry = market.bid
    atr_value = volatility(bars)
    anchor = bars[-1].close
    # Do not chase lower than 0.15 ATR below the trigger close; a bounce of up
    # to 0.25 ATR is still the rejection zone.
    low, high = anchor - 0.15 * atr_value, anchor + 0.25 * atr_value
    stop_distance = setup.stop - entry
    checks = [
        scalp_check("Entry zone", low <= entry <= high, "Sell the rejection; do not chase lower"),
    ]
    if stop_distance <= 0:
        blocked = "Price is back above the spike high; the failed high is invalid"
        return None, checks + [
            scalp_check(label, False, blocked)
            for label in SCALP_PLAN_LABELS
            if label != "Entry zone"
        ]
    stop_pct = stop_distance / entry * 100
    # Shorts receive positive funding; only a negative print is a cost.
    funding_pct = max(0.0, -market.funding_rate) * 100
    funding_payments = 0
    end = now + settings.hold_hours * 3600
    if market.next_funding <= end and market.funding_interval_hours > 0:
        funding_payments = 1 + int(
            (end - market.next_funding) // (market.funding_interval_hours * 3600)
        )
    funding_cost_pct = funding_pct * funding_payments
    cost_pct = cfg.round_trip_fee_pct + cfg.slippage_pct + funding_cost_pct
    risk_fraction = (stop_pct + cost_pct) / 100
    notional = min(
        settings.planning_equity * settings.risk_pct / 100 / risk_fraction,
        settings.planning_equity * settings.leverage * 0.9,
    )
    qty = math.floor(notional / entry / market.quantity_step) * market.quantity_step
    notional = qty * entry
    max_leverage, liquidation, tier = liquidation_fit(
        market,
        entry,
        setup.stop,
        notional,
        cost_pct,
        atr_value,
        settings.leverage,
        scalp.liquidation_buffer_pct,
    )
    hourly_atr = volatility(hourly)
    travel = scalp.travel_atr_multiple * hourly_atr
    q_closes = np.array([c.close for c in quarter])
    h_closes = np.array([c.close for c in hourly])
    vwap = session_vwap(quarter, now)
    levels = [float(ema(q_closes, 20)[-1]), float(ema(h_closes, 20)[-1]), setup.base]
    if vwap is not None:
        levels.append(vwap)
    targets = scalp_targets(levels, entry, stop_distance, cost_pct, travel, scalp)
    if targets:
        reward = (entry - targets[0]) / entry - cost_pct / 100
        ratio = reward / risk_fraction
    else:
        reward, ratio = 0.0, 0.0
    stop_fits = (
        tier is not None
        and settings.leverage <= max_leverage
        and stop_pct <= scalp.max_stop_pct
    )
    if tier is None:
        liquidation_detail = "No position tier covers the planned notional"
    elif settings.leverage > max_leverage:
        liquidation_detail = (
            f"{settings.leverage}x would liquidate before the stop; "
            f"estimated maximum {max_leverage}x for a {stop_pct:.2f}% stop"
        )
    elif stop_pct > scalp.max_stop_pct:
        liquidation_detail = (
            f"Stop {stop_pct:.2f}% exceeds the {scalp.max_stop_pct:g}% scalp limit"
        )
    else:
        liquidation_detail = (
            f"Stop {stop_pct:.2f}% sits inside the {settings.leverage}x liquidation "
            f"distance; ceiling {max_leverage}x"
        )
    checks.extend(
        [
            scalp_check("Stop inside liquidation", stop_fits, liquidation_detail),
            scalp_check(
                "Stop outside noise",
                stop_distance >= scalp.min_stop_atr * atr_value,
                f"Stop must allow at least {scalp.min_stop_atr:g} ATR of the trigger bars",
            ),
            scalp_check(
                "Mean-reversion target",
                bool(targets),
                (
                    f"Nearest of VWAP, 15m EMA20, 1h EMA20 or spike base within {scalp.travel_atr_multiple:g} hourly ATR"
                    if targets
                    else f"No VWAP, EMA or spike-base target clears {scalp.min_reward_risk:g}R within {scalp.travel_atr_multiple:g} hourly ATR"
                ),
            ),
            scalp_check(
                "Reward after costs",
                bool(targets) and ratio >= scalp.min_reward_risk,
                f"{ratio:.2f}R net; need {scalp.min_reward_risk:g}R",
            ),
            scalp_check(
                "Order size",
                qty >= market.min_quantity and qty > 0,
                "Quantity must meet the exchange minimum",
            ),
            scalp_check(
                "Execution depth",
                min(market.bid_depth_usdt, market.ask_depth_usdt)
                >= notional * scalp.min_depth_ratio,
                f"Both sides need at least {scalp.min_depth_ratio:g} times the planned notional",
            ),
        ]
    )
    if not targets:
        return None, checks
    interval = INTERVALS[scalp.trigger_interval]
    return TradePlan(
        "short",
        entry,
        low,
        high,
        setup.stop,
        targets[0],
        targets[1] if len(targets) > 1 else None,
        qty,
        notional,
        notional / settings.leverage,
        notional * risk_fraction,
        notional * risk_fraction / settings.planning_equity * 100,
        ratio,
        stop_pct,
        cost_pct,
        funding_cost_pct,
        funding_payments,
        liquidation,
        max_leverage,
        settings.leverage,
        settings.hold_hours,
        bars[-1].time + interval + scalp.entry_expiry_seconds,
        min(0.0, -(market.mark - market.price)),
        PROFILE,
        scalp.trigger_interval,
    ), checks


def evaluate_scalp_short(
    market: Market,
    frames: dict[str, list[Candle]],
    btc_hourly: list[Candle] | None,
    settings: SignalSettings,
    cfg: SignalsCfg,
    now: int,
    oi_change_pct: float | None = None,
) -> Decision:
    scalp = cfg.scalp
    result = Decision(market.symbol, as_of=market.as_of, price=market.price)
    bars = frames[scalp.trigger_interval]
    quarter, hourly, four_hour = frames["15m"], frames["1h"], frames["4h"]
    interval = INTERVALS[scalp.trigger_interval]
    result.bar_time = bars[-1].time
    atr_value = volatility(bars)
    hourly_atr = volatility(hourly)
    vwap = session_vwap(quarter, now)
    h_ema20 = float(ema(np.array([c.close for c in hourly]), 20)[-1])
    back_1h = max(1, 3600 // interval)
    gain_1h = (bars[-1].close / bars[-1 - back_1h].close - 1) * 100
    gain_4h = (quarter[-1].close / quarter[-17].close - 1) * 100
    extension = (market.price - h_ema20) / hourly_atr if hourly_atr > 0 else 0.0
    climax = climax_volume(bars, scalp)
    funding_pct = market.funding_rate * 100
    spread_pct = (market.ask - market.bid) / market.price * 100
    basis_pct = abs(market.mark - market.price) / market.price * 100
    funding_eta = market.next_funding - now
    tier_max = max(t.max_leverage for t in market.tiers) if market.tiers else 0
    result.metrics = {
        "profile": PROFILE,
        "trigger_interval": scalp.trigger_interval,
        "trend_1h": trend(hourly),
        "trend_4h": ema_bias(four_hour),
        "gain_1h_pct": gain_1h,
        "gain_4h_pct": gain_4h,
        "extension_atr": extension,
        "climax_volume": climax,
        "atr_pct": atr_value / market.price * 100,
        "hourly_atr_pct": hourly_atr / market.price * 100,
        "vwap": vwap,
        "funding_rate_pct": funding_pct,
        "next_funding": market.next_funding,
        "open_interest": market.open_interest,
        "oi_change_pct": oi_change_pct,
        "spread_pct": spread_pct,
        "tier_max_leverage": tier_max,
    }
    crowded_by_funding = funding_pct >= scalp.min_funding_rate_pct
    crowded_by_oi = oi_change_pct is not None and oi_change_pct >= scalp.min_oi_change_pct
    oi_text = (
        f"OI {oi_change_pct:+.2f}% over {scalp.oi_window_seconds // 60}m"
        if oi_change_pct is not None
        else "OI history warming up"
    )
    checks = [
        scalp_check(
            "Fresh market data",
            0 <= now - market.as_of <= cfg.max_data_age_seconds,
            "Market snapshot must be fresh",
        ),
        scalp_check(
            "Liquid market",
            market.quote_volume >= cfg.min_quote_volume,
            "24h quote volume must pass the liquidity floor",
        ),
        scalp_check(
            "Spread",
            0 < market.bid <= market.ask and spread_pct <= scalp.max_spread_pct,
            f"Spread {spread_pct:.3f}%; max {scalp.max_spread_pct:g}% for the leverage",
        ),
        scalp_check(
            "Leverage tier",
            tier_max >= settings.leverage,
            f"Exchange allows up to {tier_max}x here; planning {settings.leverage}x",
        ),
        scalp_check(
            "Mark vs last",
            basis_pct <= cfg.max_mark_basis_pct,
            f"Mark is {basis_pct:.3f}% from last; max {cfg.max_mark_basis_pct:g}%",
        ),
        scalp_check(
            "Funding print window",
            funding_eta > cfg.funding_blackout_seconds,
            f"Next funding in {max(0, funding_eta)}s; wait if ≤{cfg.funding_blackout_seconds}s",
        ),
        scalp_check(
            "Extended above value",
            gain_1h >= scalp.min_gain_1h_pct
            and gain_4h >= scalp.min_gain_4h_pct
            and extension >= scalp.min_extension_atr,
            f"+{gain_1h:.2f}% 1h, +{gain_4h:.2f}% 4h, {extension:.1f} hourly ATR above the 1h EMA20; "
            f"need {scalp.min_gain_1h_pct:g}% / {scalp.min_gain_4h_pct:g}% / {scalp.min_extension_atr:g}",
        ),
        scalp_check(
            "Climax volume",
            climax >= scalp.min_climax_volume,
            f"Peak bar {climax:.1f}× the prior baseline; need {scalp.min_climax_volume:g}×",
        ),
        scalp_check(
            "Crowded longs",
            crowded_by_funding or crowded_by_oi,
            f"Funding {funding_pct:+.4f}% (need ≥{scalp.min_funding_rate_pct:g}%) or {oi_text} "
            f"(need ≥{scalp.min_oi_change_pct:g}%)",
        ),
    ]
    if market.symbol == "BTCUSDT":
        checks.append(
            scalp_check("BTC context", True, "BTC is the benchmark; no relative gate")
        )
    elif btc_hourly and len(btc_hourly) >= 3 and len(hourly) >= 3:
        relative = (
            (hourly[-1].close / hourly[-3].close - 1)
            - (btc_hourly[-1].close / btc_hourly[-3].close - 1)
        ) * 100
        result.metrics["relative_strength_pct"] = relative
        checks.append(
            scalp_check(
                "BTC context",
                relative >= 0,
                f"Alt outran BTC by {relative:+.2f}% over 2h; fade the idiosyncratic pump, not a BTC move",
            )
        )
    else:
        checks.append(scalp_check("BTC context", False, "BTC candles unavailable"))
    candidate = all(
        item.passed for item in checks if item.label in CANDIDATE_LABELS
    )
    if not candidate:
        checks.extend(
            scalp_waiting(label, WAITING_CANDIDATE)
            for label in SETUP_LABELS + SCALP_PLAN_LABELS
        )
        checks.append(scalp_check("Tracked exposure", True, "No conflicting tracked exposure"))
        result.checks = order_scalp_checks(checks)
        result.reasons = [item.detail for item in result.checks if not item.passed and not item.waiting] or [
            WAITING_CANDIDATE
        ]
        return result
    result.side = "short"
    result.state = "WATCH_SHORT"
    setup = find_failed_high(bars, atr_value, scalp)
    checks.append(
        scalp_check(
            "Failed high trigger",
            setup is not None,
            f"Waiting for a completed {scalp.trigger_interval} close back through the spike body",
        )
    )
    if setup:
        result.setup = SETUP_NAME
        result.signal_id = f"{market.symbol}:short:{SETUP_NAME}:{result.bar_time}"
        result.metrics.update(
            {
                "spike_high": setup.spike_high,
                "spike_base": setup.base,
                "flow_delta": setup.flow_delta,
                "relative_volume": climax,
            }
        )
        checks.append(
            scalp_check(
                "Flow divergence",
                setup.flow_delta < 0,
                f"Close-weighted volume delta since the spike {setup.flow_delta:+.0f}; sellers must dominate",
            )
        )
        result.plan, plan_checks = build_scalp_plan(
            market, bars, quarter, hourly, setup, now, settings, cfg
        )
        checks.extend(plan_checks)
    else:
        checks.extend(
            scalp_waiting(label, WAITING_TRIGGER)
            for label in ("Flow divergence",) + SCALP_PLAN_LABELS
        )
    checks.append(scalp_check("Tracked exposure", True, "No conflicting tracked exposure"))
    result.checks = order_scalp_checks(checks)
    if (
        result.plan
        and now < result.plan.expires_at
        and all(item.passed for item in result.checks)
    ):
        result.state = "ENTER_SHORT"
    result.reasons = [item.detail for item in result.checks if not item.passed] or [
        f"{SETUP_NAME} confirmed on a completed {scalp.trigger_interval} candle"
    ]
    return result
