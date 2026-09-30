"""Deterministic intraday signals. No network calls, orders, or probability scores."""

from __future__ import annotations

import itertools
import math
from collections.abc import Sequence
from dataclasses import dataclass, field, replace
from typing import Literal

import numpy as np

from .indicators import atr, ema
from .signal_config import (
    PROFILES,
    SignalsCfg,
    SignalSettings,
    TrendCfg,
    min_hourly_atr_pct,
)

Side = Literal["long", "short"]
INTERVALS = {"1m": 60, "3m": 180, "5m": 300, "15m": 900, "1h": 3600, "4h": 14400}

PROFILE = "trend"
LEGACY_PROFILES: tuple[str, ...] = ("swing",)
TREND_PROFILES: tuple[str, ...] = (PROFILE, *LEGACY_PROFILES)
STRUCTURE_INTERVAL = "15m"
BIAS_INTERVAL = "1h"


def number(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (str, int, float)):
        raise ValueError("Expected a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError("Non-finite market value")
    return result


@dataclass(frozen=True)
class Candle:
    time: int
    open: float
    high: float
    low: float
    close: float
    volume: float


MAX_FILLED_GAP_BARS = 10


def closed_candles(
    rows: list[dict[str, object]],
    interval: str,
    now: int,
    *,
    allow_gaps: bool = False,
    fill_gaps: bool = False,
) -> list[Candle]:
    """Parse completed candles. ``fill_gaps`` inserts flat zero-volume candles for
    up to ten missing bars, which the sub-15m feed omits when nothing traded."""
    seconds = INTERVALS[interval]
    candles: dict[int, Candle] = {}
    for row in rows:
        timestamp = number(row.get("time"))
        timestamp = timestamp / 1000 if timestamp > 10_000_000_000 else timestamp
        if timestamp <= 0 or timestamp != int(timestamp) or int(timestamp) % seconds:
            raise ValueError(f"Invalid {interval} candle timestamp")
        if timestamp + seconds > now:
            continue
        candle = Candle(
            int(timestamp),
            number(row.get("open")),
            number(row.get("high")),
            number(row.get("low")),
            number(row.get("close")),
            # Bitunix's live payload and official example use quoteVol for coin
            # quantity and baseVol for turnover, despite the field names.
            number(row.get("quoteVol", row.get("baseVol"))),
        )
        if (
            min(candle.open, candle.low, candle.close) <= 0
            or candle.volume < 0
            or candle.high < max(candle.close, candle.low)
            or candle.low > candle.close
        ):
            raise ValueError(f"Invalid {interval} OHLCV")
        # The exchange can carry the previous close as open outside the traded
        # high/low. Include that opening gap in our range and volatility estimate.
        candle = replace(
            candle, high=max(candle.high, candle.open), low=min(candle.low, candle.open)
        )
        if candle.time in candles and candles[candle.time] != candle:
            raise ValueError(f"Conflicting {interval} candle duplicates")
        candles[candle.time] = candle
    result = sorted(candles.values(), key=lambda c: c.time)
    if fill_gaps:
        filled: list[Candle] = []
        for candle in result:
            if filled and 1 < (candle.time - filled[-1].time) // seconds <= MAX_FILLED_GAP_BARS + 1:
                prior = filled[-1]
                for missing in range(prior.time + seconds, candle.time, seconds):
                    filled.append(
                        Candle(missing, prior.close, prior.close, prior.close, prior.close, 0.0)
                    )
            filled.append(candle)
        result = filled
    if len(result) < 64:
        raise ValueError(f"Warming up: need 64 completed {interval} candles")
    if result[-1].time != (now // seconds - 1) * seconds:
        raise ValueError(f"Latest completed {interval} candle is missing")
    if not allow_gaps and any(
        b.time - a.time != seconds for a, b in itertools.pairwise(result)
    ):
        raise ValueError(f"Gap in {interval} candle history")
    return result


def swing_levels(candles: list[Candle]) -> tuple[list[float], list[float]]:
    highs, lows = [], []
    # Two bars on either side must have closed before a pivot is usable.
    for i in range(2, len(candles) - 2):
        others = candles[i - 2 : i] + candles[i + 1 : i + 3]
        if all(candles[i].high > c.high for c in others):
            highs.append(candles[i].high)
        if all(candles[i].low < c.low for c in others):
            lows.append(candles[i].low)
    return highs, lows


def _ema_stack(candles: list[Candle]) -> str:
    closes = np.array([c.close for c in candles])
    fast, slow = ema(closes, 20), ema(closes, 50)
    if closes[-1] > fast[-1] > slow[-1] and fast[-1] > fast[-4]:
        return "long"
    if closes[-1] < fast[-1] < slow[-1] and fast[-1] < fast[-4]:
        return "short"
    return "mixed"


def ema_bias(candles: list[Candle]) -> str:
    """Directional bias of the higher frame (1h for the trend profile).

    Confirmed swings on the bias frame need two closed bars on each side and
    often arrive after the move a 1-2h trade can capture. EMA stack plus slope
    is the bias; the structure frame still has to confirm with HH/HL.
    """
    return _ema_stack(candles)


def trend(candles: list[Candle]) -> str:
    """Working trend of the structure frame: EMA stack, slope, and confirmed HH/HL or LH/LL."""
    highs, lows = swing_levels(candles[-64:])
    if min(len(highs), len(lows)) < 2:
        return "mixed"
    stacked = _ema_stack(candles)
    if stacked == "long" and highs[-1] > highs[-2] and lows[-1] > lows[-2]:
        return "long"
    if stacked == "short" and highs[-1] < highs[-2] and lows[-1] < lows[-2]:
        return "short"
    return "mixed"


def volatility(candles: list[Candle]) -> float:
    highs = np.array([c.high for c in candles])
    lows = np.array([c.low for c in candles])
    closes = np.array([c.close for c in candles])
    return float(atr(highs, lows, closes, period=14)[-1])


PROJECTION_HORIZON_SECONDS = 300
PROJECTION_LOOKBACK_SECONDS = 3600


def next_price_projection(
    candles: list[Candle],
    interval: str,
    price: float | None,
    horizon_seconds: int = PROJECTION_HORIZON_SECONDS,
) -> dict[str, float | int | str] | None:
    """Projected price over the next few minutes, refreshed every scan.

    The last hour's drift (least-squares slope of the trigger-interval closes)
    is extrapolated over the horizon and clamped to one trigger ATR scaled by
    the square root of the bars ahead; the same ATR band sits either side of
    the result. It is a volatility envelope around recent momentum anchored to
    the live price, not a forecast of direction.
    """
    seconds = INTERVALS.get(interval)
    if not seconds or len(candles) < 15:
        return None
    anchor = (
        float(price)
        if isinstance(price, (int, float))
        and not isinstance(price, bool)
        and math.isfinite(price)
        and price > 0
        else candles[-1].close
    )
    atr_value = volatility(candles)
    if not (anchor > 0 and math.isfinite(atr_value) and atr_value > 0):
        return None
    bars_ahead = horizon_seconds / seconds
    lookback = max(6, min(len(candles), round(PROJECTION_LOOKBACK_SECONDS / seconds)))
    closes = np.array([c.close for c in candles[-lookback:]])
    slope = float(np.polyfit(np.arange(len(closes)), closes, 1)[0])
    envelope = atr_value * math.sqrt(bars_ahead)
    drift = max(-envelope, min(envelope, slope * bars_ahead))
    expected = anchor + drift
    return {
        "horizon_minutes": round(horizon_seconds / 60),
        "price": expected,
        "low": expected - envelope,
        "high": expected + envelope,
        "drift_pct": drift / anchor * 100,
        "basis": f"1h drift on {interval} closes, ±1 {interval} ATR",
    }


def session_vwap(candles: list[Candle], now: int) -> float | None:
    session = [c for c in candles if c.time >= now // 86400 * 86400]
    total = sum(c.volume for c in session)
    return (
        sum((c.high + c.low + c.close) / 3 * c.volume for c in session) / total
        if total > 0
        else None
    )


def _ema20(candles: list[Candle]) -> float:
    return float(ema(np.array([c.close for c in candles]), 20)[-1])


@dataclass(frozen=True)
class Tier:
    minimum: float
    maximum: float
    maintenance_rate: float
    max_leverage: int


@dataclass(frozen=True)
class Market:
    symbol: str
    price: float
    mark: float
    bid: float
    ask: float
    bid_depth_usdt: float
    ask_depth_usdt: float
    quote_volume: float
    funding_rate: float
    funding_interval_hours: float
    next_funding: int
    tiers: list[Tier]
    quantity_step: float
    min_quantity: float
    as_of: int
    open_interest: float | None = None


@dataclass
class Check:
    label: str
    passed: bool
    detail: str
    group: str = "market"
    waiting: bool = False


CHECK_GROUPS: dict[str, str] = {
    "Fresh market data": "market",
    "Liquid market": "market",
    "Spread": "market",
    "Hold-window volatility": "market",
    "Mark vs last": "market",
    "Funding print window": "market",
    "1h bias / 15m structure": "market",
    "BTC context": "market",
    "Not extended": "market",
    "Blow-off guard": "market",
    "Crowding headwind": "market",
    "Completed trigger candle": "setup",
    "Volume confirmation": "setup",
    "Entry zone": "plan",
    "Stop size": "plan",
    "Target inside hold budget": "plan",
    "Reward after costs": "plan",
    "Funding drag": "plan",
    "Order size": "plan",
    "Execution depth": "plan",
    "Leverage ceiling": "plan",
    "Tracked exposure": "portfolio",
}
CHECKLIST_LABELS: tuple[str, ...] = tuple(CHECK_GROUPS)
PLAN_LABELS: tuple[str, ...] = tuple(
    label for label, group in CHECK_GROUPS.items() if group == "plan"
)
WAITING_ALIGNMENT = "Waiting for 1h EMA bias and confirmed 15m structure"


def waiting_setup(interval: str) -> str:
    return f"Waiting for a completed {interval} setup"


def make_check(label: str, passed: bool, detail: str, *, waiting: bool = False) -> Check:
    return Check(label, passed, detail, CHECK_GROUPS[label], waiting)


def waiting_check(label: str, detail: str) -> Check:
    return make_check(label, False, detail, waiting=True)


def order_checks(checks: list[Check]) -> list[Check]:
    by_label = {item.label: item for item in checks}
    return [
        by_label[label]
        if label in by_label
        else waiting_check(label, "Waiting for evaluation")
        for label in CHECKLIST_LABELS
    ]


def blank_checklist(detail: str) -> list[Check]:
    return order_checks([waiting_check(label, detail) for label in CHECKLIST_LABELS])


def upsert_check(checks: list[Check], label: str, passed: bool, detail: str) -> None:
    replacement = make_check(label, passed, detail)
    for index, item in enumerate(checks):
        if item.label == label:
            checks[index] = replacement
            return
    checks.append(replacement)


@dataclass
class TradePlan:
    side: Side
    entry: float
    entry_low: float
    entry_high: float
    stop: float
    target: float
    target2: float | None
    quantity: float
    notional: float
    margin: float
    risk_usdt: float
    risk_pct: float
    net_reward_risk: float
    stop_pct: float
    cost_pct: float
    funding_cost_pct: float
    funding_payments: int
    liquidation_estimate: float | None
    max_leverage: int
    leverage: int
    hold_hours: int
    expires_at: int
    adverse_mark_basis: float = 0.0
    profile: str = PROFILE
    trigger_interval: str = "5m"
    # Where the setup's structure put the stop before the leverage fit; equal
    # to ``stop`` when no tightening was needed.
    structural_stop: float | None = None
    # Share of the isolated margin lost at the stop, costs included.
    margin_loss_pct: float = 0.0


@dataclass
class Decision:
    symbol: str
    state: str = "WAIT"
    side: str = ""
    setup: str = ""
    signal_id: str = ""
    as_of: int = 0
    bar_time: int = 0
    state_since: int = 0
    # Scan time at which the checklist was last run for this symbol. Symbols
    # outside the hot set keep an older value until their rotation comes round.
    evaluated_at: int = 0
    price: float | None = None
    plan: TradePlan | None = None
    checks: list[Check] = field(default_factory=list)
    reasons: list[str] = field(default_factory=list)
    metrics: dict[str, float | str | None] = field(default_factory=dict)
    # Price levels that would move this card to the next state, for the
    # overlay's "what does it need" line: {label, price, price2?}.
    actions: list[dict[str, object]] = field(default_factory=list)
    # Set when a safety gate failed: the market is dangerous at the planned
    # leverage (too volatile, extended, crowded or blown off). The state is
    # AVOID, the row ranks last and no entry levels are shown.
    avoid: str = ""
    # Where price is projected to sit over the next few minutes; see
    # next_price_projection. None until enough trigger candles exist.
    projection: dict[str, float | int | str] | None = None


def action(label: str, price: float, price2: float | None = None) -> dict[str, object]:
    item: dict[str, object] = {"label": label, "price": float(price)}
    if price2 is not None:
        item["price2"] = float(price2)
    return item


def _pullback_levels(
    bars: list[Candle],
    structure: list[Candle],
    side: Side,
    vwap: float | None,
    extra_levels: Sequence[float],
) -> list[float]:
    """Raw-price pullback levels: EMA20 of both frames, the last three structure
    swing lows (highs for shorts), session VWAP and any caller-supplied levels."""
    levels = [_ema20(structure), _ema20(bars)]
    highs, lows = swing_levels(structure[-64:])
    levels.extend(lows[-3:] if side == "long" else highs[-3:])
    if vwap is not None:
        levels.append(vwap)
    levels.extend(extra_levels)
    return levels


def watch_actions(
    bars: list[Candle],
    structure: list[Candle],
    side: Side,
    vwap: float | None,
    interval: str,
    extra_levels: Sequence[float] = (),
) -> list[dict[str, object]]:
    """Levels a WATCH card needs: the breakout boundary and the nearest pullback level."""
    price = bars[-1].close
    levels = _pullback_levels(bars, structure, side, vwap, extra_levels)
    if side == "long":
        boundary = max(c.high for c in bars[-26:-6])
        pullback = [level for level in levels if level < price]
        nearest = max(pullback) if pullback else None
        items = [action(f"Needs a {interval} close above", max(boundary, price))]
    else:
        boundary = min(c.low for c in bars[-26:-6])
        pullback = [level for level in levels if level > price]
        nearest = min(pullback) if pullback else None
        items = [action(f"Needs a {interval} close below", min(boundary, price))]
    if nearest is not None:
        items.append(action("or a pullback to", nearest))
    return items


def normalize(candles: list[Candle], side: Side) -> list[Candle]:
    if side == "long":
        return candles
    return [
        Candle(c.time, -c.open, -c.low, -c.high, -c.close, c.volume) for c in candles
    ]


@dataclass(frozen=True)
class Setup:
    name: str
    stop: float
    relative_volume: float


def _relative_volume(bars: list[Candle]) -> float:
    baseline = sum(c.volume for c in bars[-21:-1]) / 20
    return bars[-1].volume / baseline if baseline > 0 else 0


def _reclaim(last: Candle, prior: Candle) -> bool:
    # Close through the prior close, not the prior high: chasing the high at
    # 50x puts the structural stop too far from a 1-2h entry.
    span = last.high - last.low
    return last.close > max(prior.close, last.open) and (
        span <= 0 or last.close >= last.low + 0.4 * span
    )


def find_setup(
    candles: list[Candle],
    structure: list[Candle],
    side: Side,
    atr_value: float,
    vwap: float | None,
    t: TrendCfg,
    *,
    extra_levels: Sequence[float] = (),
) -> Setup | None:
    """Completed continuation trigger on the trigger frame; ``structure`` is the
    15m frame. ``extra_levels`` are raw prices (e.g. the 1h EMA20) that join the
    pullback level set; they are side-normalised here like VWAP."""
    sign = 1 if side == "long" else -1
    bars = normalize(candles, side)
    last = bars[-1]
    # A breakout must precede the retest. Its range excludes every breakout/retest bar.
    boundary = max(c.high for c in bars[-26:-6])
    for i in range(len(bars) - 5, len(bars) - 1):
        baseline = sum(c.volume for c in bars[i - 20 : i]) / 20
        relative = bars[i].volume / baseline if baseline > 0 else 0
        if (
            bars[i - 1].close <= boundary < bars[i].close
            and relative >= t.breakout_volume_min
        ):
            retest = bars[i + 1 :]
            if (
                min(c.low for c in retest) >= boundary - 0.5 * atr_value
                and last.low <= boundary + 0.25 * atr_value
                and last.close > max(boundary, last.open, bars[-2].close)
            ):
                return Setup(
                    "Breakout & retest",
                    sign
                    * (min(c.low for c in retest) - t.stop_atr_buffer * atr_value),
                    relative,
                )
    levels = [
        sign * level
        for level in _pullback_levels(candles, structure, side, vwap, extra_levels)
    ]
    pullback_low = min(c.low for c in bars[-4:-1])
    touched = any(
        abs(pullback_low - level) <= 0.5 * atr_value and last.close > level
        for level in levels
    )
    pulled_back = any(b.close < a.close for a, b in zip(bars[-5:-1], bars[-4:-1]))
    if touched and pulled_back and _reclaim(last, bars[-2]):
        return Setup(
            "Trend pullback",
            sign * (min(c.low for c in bars[-4:]) - t.stop_atr_buffer * atr_value),
            _relative_volume(bars),
        )
    return _impulse_continuation(bars, side, atr_value, t)


def _impulse_continuation(
    bars: list[Candle], side: Side, atr_value: float, t: TrendCfg
) -> Setup | None:
    sign = 1 if side == "long" else -1
    last = bars[-1]
    if not _reclaim(last, bars[-2]):
        return None
    for idx in range(len(bars) - 8, len(bars) - 3):
        impulse = bars[idx]
        if impulse.close - impulse.open < t.impulse_atr_min * atr_value:
            continue
        pullback = bars[idx + 1 : -1]
        if len(pullback) < 2:
            continue
        pullback_low = min(c.low for c in pullback)
        if pullback_low < impulse.low - 0.35 * atr_value:
            continue
        if impulse.high - pullback_low < 0.3 * atr_value:
            continue
        eased = any(
            b.close < a.close for a, b in zip(pullback, pullback[1:])
        ) or pullback[0].close < impulse.close
        if not eased:
            continue
        if last.low < impulse.low - 0.35 * atr_value:
            continue
        return Setup(
            "Impulse continuation",
            sign * (min(pullback_low, last.low) - t.stop_atr_buffer * atr_value),
            _relative_volume(bars),
        )
    return None


def travel_budget(hourly_atr: float, hold_minutes: int, multiple: float) -> float:
    """Price distance a hold window can carry: multiple x ATR(1h) x sqrt(hold / 60 min)."""
    return multiple * hourly_atr * math.sqrt(hold_minutes / 60)


def select_targets(
    levels: list[float],
    entry: float,
    side: Side,
    stop_distance: float,
    cost_pct: float,
    travel: float,
    min_reward_risk: float,
) -> list[float]:
    """Nearest structural targets that clear the profile R inside the hold-window travel budget."""
    sign = 1 if side == "long" else -1
    risk_fraction = (stop_distance / entry * 100 + cost_pct) / 100
    if risk_fraction <= 0 or travel <= 0:
        return []
    chosen: list[float] = []
    for price in sorted(
        {p for p in levels if 0 < sign * (p - entry) <= travel},
        reverse=side == "short",
    ):
        reward = sign * (price - entry) / entry - cost_pct / 100
        if reward / risk_fraction >= min_reward_risk:
            chosen.append(price)
    return chosen


def funding_cost(
    market: Market, side: Side, now: int, hold_minutes: int
) -> tuple[float, int]:
    if market.next_funding < now or market.funding_interval_hours <= 0:
        raise ValueError("Funding schedule is missing or stale")
    end = now + hold_minutes * 60
    payments = (
        0
        if market.next_funding > end
        else 1
        + int((end - market.next_funding) // (market.funding_interval_hours * 3600))
    )
    # Do not subsidize a setup with uncertain future funding receipts.
    rate = max(0.0, market.funding_rate * (1 if side == "long" else -1))
    return rate * payments * 100, payments


def estimate_liquidation(
    entry: float, leverage: int, cost_pct: float, maintenance_rate: float, side: str
) -> float:
    """Isolated-margin liquidation estimate after round-trip costs."""
    sign = 1 if side == "long" else -1
    return (
        entry
        * (1 - sign / leverage + sign * cost_pct / 100)
        / (1 - sign * maintenance_rate)
    )


def select_tier(tiers: Sequence[Tier], notional: float) -> Tier | None:
    """The highest-minimum position tier whose range contains ``notional``."""
    return next(
        (
            t
            for t in sorted(tiers, key=lambda t: t.minimum, reverse=True)
            if t.minimum <= notional <= t.maximum
        ),
        None,
    )


def liquidation_fit(
    market: Market,
    entry: float,
    stop: float,
    notional: float,
    cost_pct: float,
    atr_value: float,
    leverage: int,
    buffer_pct: float,
    side: str = "short",
    cap: int | None = None,
) -> tuple[int, float | None, Tier | None]:
    """Highest leverage whose estimated liquidation stays beyond the stop.

    Returns (ceiling, liquidation at ``leverage`` or None past the pair cap, tier).
    ``cap`` bounds the search (the profile's maximum); the tier cap always applies.
    """
    sign = 1 if side == "long" else -1
    return fit_leverage(
        market.tiers,
        entry,
        stop,
        notional,
        cost_pct,
        atr_value,
        leverage,
        buffer_pct,
        side,
        cap=cap,
        adverse_basis=min(0.0, sign * (market.mark - market.price)),
    )


def fit_leverage(
    tiers: Sequence[Tier],
    entry: float,
    stop: float,
    notional: float,
    cost_pct: float,
    atr_value: float,
    leverage: int,
    buffer_pct: float,
    side: str = "short",
    *,
    cap: int | None = None,
    adverse_basis: float = 0.0,
) -> tuple[int, float | None, Tier | None]:
    """``liquidation_fit`` over a bare tier list (no Market needed).

    ``adverse_basis`` is the mark-vs-last basis working against the trade
    (``<= 0``); pass 0 when only the mark price is known.
    """
    sign = 1 if side == "long" else -1
    tier = select_tier(tiers, notional)
    if tier is None:
        return 0, None, None
    buffer = max(entry * buffer_pct / 100, atr_value * 0.5)
    max_leverage, liquidation = 0, None
    for level in range(1, min(cap or tier.max_leverage, tier.max_leverage) + 1):
        estimated = estimate_liquidation(
            entry, level, cost_pct, tier.maintenance_rate, side
        )
        if sign * (stop - estimated) + adverse_basis >= buffer:
            max_leverage = level
        if level == leverage:
            liquidation = estimated
    return max_leverage, liquidation, tier


def size_notional(
    planning_equity: float,
    risk_pct: float,
    leverage: int,
    risk_fraction: float,
    entry: float,
    quantity_step: float,
) -> tuple[float, float]:
    """Risk-based size rounded to the quantity step: (quantity, notional).

    Leverage only caps the notional at 90% of the buying power; it never
    enlarges the position.
    """
    notional = min(
        planning_equity * risk_pct / 100 / risk_fraction,
        planning_equity * leverage * 0.9,
    )
    qty = math.floor(notional / entry / quantity_step) * quantity_step
    return qty, qty * entry


@dataclass(frozen=True)
class StopBudget:
    """How far from entry a stop may sit at the leverage in use.

    ``distance`` is the widest allowed stop in price units: the smaller of the
    room inside the estimated liquidation (less the safety buffer) and the
    room the planned maximum margin loss leaves, costs included. ``binding``
    names which of the two set it. ``liquidation`` is the estimate itself.
    """

    distance: float
    liquidation: float
    binding: str


def stop_budget(
    entry: float,
    side: str,
    leverage: int,
    maintenance_rate: float,
    cost_pct: float,
    buffer: float,
    adverse_basis: float,
    max_margin_loss_pct: float,
) -> StopBudget:
    sign = 1 if side == "long" else -1
    liquidation = estimate_liquidation(entry, leverage, cost_pct, maintenance_rate, side)
    inside_liquidation = sign * (entry - liquidation) - buffer + adverse_basis
    inside_margin = entry * (max_margin_loss_pct / leverage - cost_pct) / 100
    if inside_margin < inside_liquidation:
        return StopBudget(inside_margin, liquidation, "margin")
    return StopBudget(inside_liquidation, liquidation, "liquidation")


def margin_loss_pct(stop_pct: float, cost_pct: float, leverage: int) -> float:
    """Share of the isolated margin lost when the stop fills, costs included."""
    return (stop_pct + cost_pct) * leverage


def fit_stop(entry: float, side: str, structural_stop: float, budget: StopBudget) -> float:
    """Pull the structural stop toward entry until it fits the budget."""
    sign = 1 if side == "long" else -1
    if sign * (entry - structural_stop) <= budget.distance:
        return structural_stop
    return entry - sign * max(budget.distance, 0.0)


@dataclass(frozen=True)
class StopFit:
    """A structural stop fitted to the planned leverage, with its sizing."""

    stop: float
    quantity: float
    notional: float
    tier: Tier | None
    budget: StopBudget | None
    # Highest leverage (up to the profile cap) whose budget still holds the
    # untightened structural stop.
    ceiling: int


def fit_plan_stop(
    tiers: Sequence[Tier],
    entry: float,
    side: str,
    structural_stop: float,
    cost_pct: float,
    atr_value: float,
    buffer_pct: float,
    adverse_basis: float,
    settings: SignalSettings,
    quantity_step: float,
    *,
    cap: int | None = None,
) -> StopFit:
    """Fit the stop to the leverage in use and size the plan to it.

    The stop must sit inside the estimated liquidation distance and inside the
    planned maximum margin loss. A tighter stop raises the notional for the
    same planned loss, which can move the maintenance tier, so the tier and
    the stop are settled together.
    """
    sign = 1 if side == "long" else -1
    buffer = max(entry * buffer_pct / 100, atr_value * 0.5)
    stop = structural_stop
    tier: Tier | None = None
    budget: StopBudget | None = None
    qty = notional = 0.0
    for _ in range(3):
        stop_pct = sign * (entry - stop) / entry * 100
        qty, notional = size_notional(
            settings.planning_equity,
            settings.risk_pct,
            settings.leverage,
            max((stop_pct + cost_pct) / 100, 1e-9),
            entry,
            quantity_step,
        )
        next_tier = select_tier(tiers, notional)
        if next_tier is None or next_tier == tier:
            tier = next_tier
            break
        tier = next_tier
        budget = stop_budget(
            entry,
            side,
            settings.leverage,
            tier.maintenance_rate,
            cost_pct,
            buffer,
            adverse_basis,
            settings.max_margin_loss_pct,
        )
        stop = fit_stop(entry, side, structural_stop, budget)
    ceiling = 0
    if tier is not None:
        for level in range(1, min(cap or tier.max_leverage, tier.max_leverage) + 1):
            candidate = stop_budget(
                entry,
                side,
                level,
                tier.maintenance_rate,
                cost_pct,
                buffer,
                adverse_basis,
                settings.max_margin_loss_pct,
            )
            if sign * (entry - structural_stop) <= candidate.distance:
                ceiling = level
    return StopFit(stop, qty, notional, tier, budget, ceiling)


def leverage_stop_detail(
    leverage: int,
    structural_pct: float,
    stop_pct: float,
    loss_pct: float,
    budget: StopBudget,
    max_leverage: int,
    fits: bool,
) -> str:
    """One line for the checklist on what the leverage did to the stop."""
    ceiling = (
        f"structure fits up to {max_leverage}x untightened"
        if max_leverage
        else "structure fits no leverage"
    )
    if not fits:
        return (
            f"{leverage}x leaves only a {max(0.0, stop_pct):.2f}% stop, inside market noise; "
            f"reduce leverage ({ceiling})"
        )
    if stop_pct + 1e-9 < structural_pct:
        cap = (
            "planned margin loss"
            if budget.binding == "margin"
            else "estimated liquidation buffer"
        )
        return (
            f"Stop tightened from {structural_pct:.2f}% to {stop_pct:.2f}% for {leverage}x by the {cap}; "
            f"a stop-out loses {loss_pct:.0f}% of margin ({ceiling})"
        )
    return (
        f"Structural stop fits {leverage}x; a stop-out loses {loss_pct:.0f}% of margin "
        f"({ceiling})"
    )


def _target_levels(
    fifteen: list[Candle], hourly: list[Candle], side: Side, now: int
) -> list[float]:
    """Structural levels in the trade direction: 15m (24h) and 1h (4 days) swings,
    the prior UTC day's extreme and the current session's extreme."""
    levels: list[float] = []
    for bars in (fifteen[-96:], hourly[-96:]):
        highs, lows = swing_levels(bars)
        levels.extend(highs if side == "long" else lows)
    midnight = now // 86400 * 86400
    previous_day = [c for c in fifteen if midnight - 86400 <= c.time < midnight]
    session = [c for c in fifteen if c.time >= midnight]
    for bars in (previous_day if len(previous_day) == 96 else [], session if len(session) >= 4 else []):
        if bars:
            levels.append(
                max(c.high for c in bars) if side == "long" else min(c.low for c in bars)
            )
    return levels


def build_plan(
    market: Market,
    five: list[Candle],
    fifteen: list[Candle],
    hourly: list[Candle],
    setup: Setup,
    side: Side,
    now: int,
    settings: SignalSettings,
    cfg: SignalsCfg,
) -> tuple[TradePlan | None, list[Check]]:
    t = cfg.trend
    sign = 1 if side == "long" else -1
    entry = market.ask if side == "long" else market.bid
    stop_distance = sign * (entry - setup.stop)
    if stop_distance <= 0:
        blocked = "Price has passed the setup's stop"
        return None, [make_check(label, False, blocked) for label in PLAN_LABELS]
    atr_value = volatility(five)
    atr_15m = volatility(fifteen)
    hourly_atr = volatility(hourly)
    hold_minutes = settings.hold_hours * 60
    anchor = five[-1].close
    funding_pct, payments = funding_cost(market, side, now, hold_minutes)
    cost_pct = cfg.round_trip_fee_pct + cfg.slippage_pct + funding_pct
    adverse_basis = min(0.0, sign * (market.mark - market.price))
    structural_pct = stop_distance / entry * 100
    fit = fit_plan_stop(
        market.tiers,
        entry,
        side,
        setup.stop,
        cost_pct,
        atr_value,
        t.liquidation_buffer_pct,
        adverse_basis,
        settings,
        market.quantity_step,
        cap=PROFILES[PROFILE][2],
    )
    stop, qty, notional, tier, budget = fit.stop, fit.quantity, fit.notional, fit.tier, fit.budget
    max_leverage = fit.ceiling
    stop_distance = sign * (entry - stop)
    stop_pct = stop_distance / entry * 100
    risk_fraction = (stop_pct + cost_pct) / 100
    lev = settings.leverage
    loss_pct = margin_loss_pct(stop_pct, cost_pct, lev)
    tightened = stop_pct + 1e-9 < structural_pct
    low, high = sorted(
        (
            anchor - sign * t.entry_pullback_atr * atr_value,
            anchor
            + sign
            * min(
                t.entry_chase_atr * atr_value,
                t.entry_chase_risk_fraction * stop_distance,
            ),
        )
    )
    floor = max(
        t.min_stop_atr * atr_value,
        t.min_stop_atr_15m * atr_15m,
        entry * t.min_stop_pct / 100,
    )
    travel = travel_budget(hourly_atr, hold_minutes, t.travel_atr_multiple)
    targets = select_targets(
        _target_levels(fifteen, hourly, side, now),
        entry,
        side,
        stop_distance,
        cost_pct,
        travel,
        t.min_reward_risk,
    )
    missing_target = (
        f"No structural target that clears {t.min_reward_risk:g}R inside the "
        f"{hold_minutes}-minute travel budget ({travel / entry * 100:.2f}%)"
    )
    if not targets:
        ratio = 0.0
        reward_detail = missing_target
    else:
        reward = sign * (targets[0] - entry) / entry - cost_pct / 100
        ratio = reward / risk_fraction
        reward_detail = f"{ratio:.2f}R net; need {t.min_reward_risk:g}R"
    stop_ok = floor <= stop_distance and stop_pct <= t.max_stop_pct
    fits = (
        tier is not None
        and budget is not None
        and lev <= tier.max_leverage
        and stop_distance >= floor
    )
    if tier is None or budget is None:
        leverage_detail = "Notional falls outside every position tier"
    elif lev > tier.max_leverage:
        leverage_detail = f"Pair allows {tier.max_leverage}x; planned {lev}x"
    else:
        leverage_detail = leverage_stop_detail(
            lev, structural_pct, stop_pct, loss_pct, budget, max_leverage, fits
        ) + f"; pair cap {tier.max_leverage}x"
    liquidation = budget.liquidation if budget is not None else None
    checks = [
        make_check(
            "Entry zone", low <= entry <= high, "Wait for the entry zone; do not chase"
        ),
        make_check(
            "Stop size",
            stop_ok,
            f"Stop {stop_pct:.2f}% must sit between {floor / entry * 100:.2f}% (5m noise) "
            f"and {t.max_stop_pct:g}% ({lev}x limit)"
            + (f"; tightened from {structural_pct:.2f}% for {lev}x" if tightened else ""),
        ),
        make_check(
            "Target inside hold budget",
            bool(targets),
            (
                f"Structural target inside the {hold_minutes}-minute travel budget "
                f"({travel / entry * 100:.2f}%)"
                if targets
                else missing_target
            ),
        ),
        make_check(
            "Reward after costs",
            bool(targets) and ratio >= t.min_reward_risk,
            reward_detail,
        ),
        make_check(
            "Funding drag",
            funding_pct <= t.max_funding_cost_pct,
            f"{funding_pct:.3f}% projected funding over {hold_minutes} min; max {t.max_funding_cost_pct:g}%",
        ),
        make_check(
            "Order size",
            qty >= market.min_quantity and qty > 0,
            "Quantity must meet the exchange minimum",
        ),
        make_check(
            "Execution depth",
            min(market.bid_depth_usdt, market.ask_depth_usdt)
            >= notional * t.min_depth_ratio,
            f"Both sides need at least {t.min_depth_ratio:g} times the planned notional",
        ),
        make_check("Leverage ceiling", fits, leverage_detail),
    ]
    if not targets:
        return None, checks
    return TradePlan(
        side,
        entry,
        low,
        high,
        stop,
        targets[0],
        targets[1] if len(targets) > 1 else None,
        qty,
        notional,
        notional / lev,
        notional * risk_fraction,
        notional * risk_fraction / settings.planning_equity * 100,
        ratio,
        stop_pct,
        cost_pct,
        funding_pct,
        payments,
        liquidation,
        max_leverage,
        lev,
        settings.hold_hours,
        five[-1].time + INTERVALS[t.trigger_interval] + t.entry_expiry_seconds,
        adverse_basis,
        PROFILE,
        t.trigger_interval,
        structural_stop=setup.stop,
        margin_loss_pct=loss_pct,
    ), checks


def evaluate_intraday(
    market: Market,
    frames: dict[str, list[Candle]],
    btc_hourly: list[Candle] | None,
    settings: SignalSettings,
    cfg: SignalsCfg,
    now: int,
    oi_change_pct: float | None = None,
) -> Decision:
    t = cfg.trend
    result = Decision(market.symbol, as_of=market.as_of, price=market.price)
    five, fifteen, hourly = (
        frames[t.trigger_interval],
        frames[STRUCTURE_INTERVAL],
        frames[BIAS_INTERVAL],
    )
    bias, structure = ema_bias(hourly), trend(fifteen)
    result.bar_time = five[-1].time
    atr_value, atr_15m, hourly_atr = (
        volatility(five),
        volatility(fifteen),
        volatility(hourly),
    )
    vwap = session_vwap(fifteen, now)
    hold_minutes = settings.hold_hours * 60
    hourly_atr_pct = hourly_atr / market.price * 100
    h_ema20 = _ema20(hourly)
    # Positive when price sits above the 1h EMA20; gates read it in the trade direction.
    extension = (market.price - h_ema20) / hourly_atr if hourly_atr > 0 else 0.0
    hourly_volume = market.quote_volume / 24
    session_volume = sum(c.volume for c in fifteen[-4:])
    session_ratio = session_volume / hourly_volume if hourly_volume > 0 else 0.0
    spread_pct = (market.ask - market.bid) / market.price * 100
    basis_pct = abs(market.mark - market.price) / market.price * 100
    funding_eta = market.next_funding - now
    atr_floor = min_hourly_atr_pct(t, hold_minutes)
    result.metrics = {
        "profile": PROFILE,
        "trigger_interval": t.trigger_interval,
        "hold_minutes": hold_minutes,
        "trend_1h": bias,
        "trend_15m": structure,
        "atr_pct": atr_value / market.price * 100,
        "atr_15m_pct": atr_15m / market.price * 100,
        "hourly_atr_pct": hourly_atr_pct,
        "extension_atr": extension,
        "session_volume_ratio": session_ratio,
        "vwap": vwap,
        "funding_rate_pct": market.funding_rate * 100,
        "next_funding": market.next_funding,
        "open_interest": market.open_interest,
        "oi_change_pct": oi_change_pct,
        "spread_pct": spread_pct,
        "tier_max_leverage": max(x.max_leverage for x in market.tiers)
        if market.tiers
        else 0,
    }
    result.checks = [
        make_check(
            "Fresh market data",
            0 <= now - market.as_of <= cfg.max_data_age_seconds,
            "Market snapshot must be fresh",
        ),
        make_check(
            "Liquid market",
            market.quote_volume >= cfg.min_quote_volume
            and session_volume >= t.session_volume_ratio * hourly_volume,
            f"24h quote volume must pass the liquidity floor and the last hour's session "
            f"volume must be at least {t.session_volume_ratio:g}x the 24h hourly average "
            f"({session_ratio:.2f}x)",
        ),
        make_check(
            "Spread",
            0 < market.bid <= market.ask and spread_pct <= t.max_spread_pct,
            f"Spread {spread_pct:.3f}%; max {t.max_spread_pct:g}% for the leverage",
        ),
        make_check(
            "Hold-window volatility",
            atr_floor <= hourly_atr_pct <= t.max_hourly_atr_pct,
            f"1h ATR {hourly_atr_pct:.2f}% must sit in {atr_floor:.2f}-{t.max_hourly_atr_pct:g}% "
            f"to carry a {t.min_reward_risk:g}R target inside {hold_minutes} min "
            f"at {settings.leverage}x",
        ),
        make_check(
            "Mark vs last",
            basis_pct <= t.max_mark_basis_pct,
            f"Mark is {basis_pct:.3f}% from last; max {t.max_mark_basis_pct:g}%",
        ),
        make_check(
            "Funding print window",
            funding_eta > t.funding_blackout_seconds,
            f"Next funding in {max(0, funding_eta)}s; wait if ≤{t.funding_blackout_seconds}s",
        ),
        make_check(
            "1h bias / 15m structure",
            bias == structure and bias in ("long", "short"),
            f"1h bias {bias}; 15m structure {structure}",
        ),
    ]
    if bias != structure or bias not in ("long", "short"):
        result.checks.extend(
            waiting_check(label, WAITING_ALIGNMENT)
            for label in CHECKLIST_LABELS
            if label
            not in {item.label for item in result.checks}
            and label != "Tracked exposure"
        )
        result.checks.append(
            make_check(
                "Tracked exposure", True, "No conflicting tracked exposure"
            )
        )
        result.checks = order_checks(result.checks)
        result.reasons = [
            "Wait for 1h EMA bias and confirmed 15m structure in the same direction"
        ]
        return result
    side: Side = "long" if bias == "long" else "short"
    opposite = "short" if side == "long" else "long"
    sign = 1 if side == "long" else -1
    result.side = side
    result.state = f"WATCH_{side.upper()}"
    if market.symbol == "BTCUSDT":
        result.checks.append(
            make_check(
                "BTC context",
                True,
                "BTC is the benchmark; no extra relative-strength gate",
            )
        )
    elif btc_hourly and len(btc_hourly) >= 50 and len(hourly) >= 3:
        btc_bias = ema_bias(btc_hourly)
        relative = (
            (hourly[-1].close / hourly[-3].close - 1)
            - (btc_hourly[-1].close / btc_hourly[-3].close - 1)
        ) * 100
        result.metrics.update(
            {"btc_trend": btc_bias, "relative_strength_pct": relative}
        )
        result.checks.append(
            make_check(
                "BTC context",
                btc_bias != opposite and sign * relative >= 0,
                f"BTC 1h bias {btc_bias} must not oppose the trade and the 2h relative "
                f"strength ({relative:+.2f}%) must favour it",
            )
        )
    else:
        result.metrics.update({"btc_trend": "unavailable", "relative_strength_pct": None})
        result.checks.append(make_check("BTC context", False, "BTC candles unavailable"))
    directional_extension = sign * extension
    gain_1h = sign * (market.price / hourly[-2].close - 1) * 100 if len(hourly) >= 2 else 0.0
    gain_4h = sign * (market.price / hourly[-5].close - 1) * 100 if len(hourly) >= 5 else 0.0
    result.metrics.update({"gain_1h_pct": gain_1h, "gain_4h_pct": gain_4h})
    crowded = sign * market.funding_rate * 100 > t.max_funding_rate_pct or (
        oi_change_pct is not None
        and oi_change_pct >= t.max_oi_change_pct
        and directional_extension >= t.crowd_extension_atr
    )
    oi_text = (
        f"OI {oi_change_pct:+.2f}% over 1h" if oi_change_pct is not None else "OI history warming up"
    )
    result.checks.extend(
        [
            make_check(
                "Not extended",
                directional_extension <= t.max_extension_atr,
                f"{directional_extension:+.1f} hourly ATR from the 1h EMA20 in the trade "
                f"direction; max {t.max_extension_atr:g}",
            ),
            make_check(
                "Blow-off guard",
                gain_1h <= t.max_gain_1h_pct and gain_4h <= t.max_gain_4h_pct,
                f"Already {gain_1h:+.1f}% over 1h and {gain_4h:+.1f}% over 4h in the trade "
                f"direction; max {t.max_gain_1h_pct:g}% / {t.max_gain_4h_pct:g}% at "
                f"{settings.leverage}x",
            ),
            make_check(
                "Crowding headwind",
                not crowded,
                f"Funding {market.funding_rate * 100:+.4f}% per print (max "
                f"{t.max_funding_rate_pct:g}% against the trade); {oi_text} with "
                f"{directional_extension:+.1f} ATR extension (crowded at ≥{t.max_oi_change_pct:g}% "
                f"and ≥{t.crowd_extension_atr:g} ATR)",
            ),
        ]
    )
    setup = find_setup(
        five, fifteen, side, atr_value, vwap, t, extra_levels=[h_ema20]
    )
    result.checks.append(
        make_check(
            "Completed trigger candle",
            setup is not None,
            f"Waiting for a completed {t.trigger_interval} pullback reclaim, "
            "impulse continuation, or breakout retest",
        )
    )
    if setup:
        result.setup = setup.name
        result.signal_id = f"{market.symbol}:{side}:{setup.name}:{result.bar_time}"
        result.metrics["relative_volume"] = setup.relative_volume
        result.checks.append(
            make_check(
                "Volume confirmation",
                setup.relative_volume >= t.relative_volume_min,
                f"{setup.relative_volume:.2f} times baseline volume",
            )
        )
        result.plan, checks = build_plan(
            market, five, fifteen, hourly, setup, side, now, settings, cfg
        )
        result.checks.extend(checks)
    else:
        result.checks.extend(
            waiting_check(label, waiting_setup(t.trigger_interval))
            for label in ("Volume confirmation",) + PLAN_LABELS
        )
    if result.plan:
        result.actions = [
            action(f"Enter {side}", result.plan.entry_low, result.plan.entry_high)
        ]
    else:
        result.actions = watch_actions(
            five, fifteen, side, vwap, t.trigger_interval, [h_ema20]
        )
    result.checks.append(
        make_check("Tracked exposure", True, "No conflicting tracked exposure")
    )
    result.checks = order_checks(result.checks)
    if (
        result.plan
        and now < result.plan.expires_at
        and all(item.passed for item in result.checks)
    ):
        result.state = f"ENTER_{side.upper()}"
    result.reasons = [item.detail for item in result.checks if not item.passed] or [
        f"{result.setup} confirmed on a completed {t.trigger_interval} candle"
    ]
    # Safety gates: a failure here means the market can move through the
    # whole margin inside the hold, so the card must not read as a setup.
    danger = {item.label: item for item in result.checks if not item.passed}
    if hourly_atr_pct > t.max_hourly_atr_pct:
        result.avoid = (
            f"1h ATR {hourly_atr_pct:.2f}% is above the {t.max_hourly_atr_pct:g}% ceiling: "
            f"one average hour moves {hourly_atr_pct * settings.leverage:.0f}% of margin at "
            f"{settings.leverage}x"
        )
    else:
        for label in ("Blow-off guard", "Not extended", "Crowding headwind"):
            if label in danger:
                result.avoid = danger[label].detail
                break
    if result.avoid:
        result.state = "AVOID"
        result.actions = []
        result.reasons = [f"Do not {side} at {settings.leverage}x: {result.avoid}"]
    return result
