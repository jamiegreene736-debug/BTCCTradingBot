"""Deterministic intraday signals. No network calls, orders, or probability scores."""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass, field, replace
from typing import Literal

import numpy as np

from .indicators import atr, ema
from .signal_config import SignalsCfg, SignalSettings

Side = Literal["long", "short"]
INTERVALS = {"15m": 900, "1h": 3600, "4h": 14400}


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


def closed_candles(
    rows: list[dict[str, object]], interval: str, now: int, *, allow_gaps: bool = False
) -> list[Candle]:
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
    """4h directional filter for ≤24h holds.

    Confirmed 4h swings need two closed bars on each side (16 hours) and often
    arrive after the move a 12/24h trade can capture. EMA stack plus slope is
    the bias; 1h structure still has to confirm.
    """
    return _ema_stack(candles)


def trend(candles: list[Candle]) -> str:
    """1h working trend: EMA stack, slope, and confirmed HH/HL or LH/LL."""
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


def session_vwap(candles: list[Candle], now: int) -> float | None:
    session = [c for c in candles if c.time >= now // 86400 * 86400]
    total = sum(c.volume for c in session)
    return (
        sum((c.high + c.low + c.close) / 3 * c.volume for c in session) / total
        if total > 0
        else None
    )


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


@dataclass
class Decision:
    symbol: str
    state: str = "WAIT"
    side: str = ""
    setup: str = ""
    signal_id: str = ""
    as_of: int = 0
    bar_time: int = 0
    price: float | None = None
    plan: TradePlan | None = None
    checks: list[Check] = field(default_factory=list)
    reasons: list[str] = field(default_factory=list)
    metrics: dict[str, float | str | None] = field(default_factory=dict)


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
    # Close through the prior close, not the prior high — chasing the high is
    # late for a 25-40x entry and spends the 24h hold on already-extended price.
    span = last.high - last.low
    return last.close > max(prior.close, last.open) and (
        span <= 0 or last.close >= last.low + 0.4 * span
    )


def find_setup(
    candles: list[Candle],
    hourly: list[Candle],
    side: Side,
    atr_value: float,
    vwap: float | None,
    cfg: SignalsCfg,
) -> Setup | None:
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
            and relative >= cfg.breakout_volume_min
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
                    * (min(c.low for c in retest) - cfg.stop_atr_buffer * atr_value),
                    relative,
                )
    h_closes = np.array([c.close for c in hourly])
    m_closes = np.array([c.close for c in candles])
    _, support_lows = swing_levels(normalize(hourly[-64:], side))
    levels = [
        sign * float(ema(h_closes, 20)[-1]),
        sign * float(ema(m_closes, 20)[-1]),
    ]
    levels.extend(support_lows[-3:])
    if vwap is not None:
        levels.append(sign * vwap)
    pullback_low = min(c.low for c in bars[-4:-1])
    touched = any(
        abs(pullback_low - level) <= 0.5 * atr_value and last.close > level
        for level in levels
    )
    pulled_back = any(b.close < a.close for a, b in zip(bars[-5:-1], bars[-4:-1]))
    if touched and pulled_back and _reclaim(last, bars[-2]):
        return Setup(
            "Trend pullback",
            sign * (min(c.low for c in bars[-4:]) - cfg.stop_atr_buffer * atr_value),
            _relative_volume(bars),
        )
    return _impulse_continuation(bars, side, atr_value, cfg)


def _impulse_continuation(
    bars: list[Candle], side: Side, atr_value: float, cfg: SignalsCfg
) -> Setup | None:
    sign = 1 if side == "long" else -1
    last = bars[-1]
    if not _reclaim(last, bars[-2]):
        return None
    for idx in range(len(bars) - 8, len(bars) - 3):
        impulse = bars[idx]
        if impulse.close - impulse.open < cfg.impulse_atr_min * atr_value:
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
            sign * (min(pullback_low, last.low) - cfg.stop_atr_buffer * atr_value),
            _relative_volume(bars),
        )
    return None


def select_targets(
    levels: list[float],
    entry: float,
    side: Side,
    stop_distance: float,
    cost_pct: float,
    hourly_atr: float,
    four_atr: float,
    cfg: SignalsCfg,
) -> list[float]:
    """Nearest structural targets that still clear 2R and a ≤24h travel budget."""
    sign = 1 if side == "long" else -1
    risk_fraction = (stop_distance / entry * 100 + cost_pct) / 100
    if risk_fraction <= 0:
        return []
    min_distance = (cfg.min_reward_risk * risk_fraction + cost_pct / 100) * entry
    max_distance = max(
        cfg.max_target_atr_multiple * hourly_atr,
        cfg.max_target_4h_atr_multiple * four_atr,
        min_distance,
    )
    chosen: list[float] = []
    for price in sorted(
        {p for p in levels if 0 < sign * (p - entry) <= max_distance},
        reverse=side == "short",
    ):
        reward = sign * (price - entry) / entry - cost_pct / 100
        if reward / risk_fraction >= cfg.min_reward_risk:
            chosen.append(price)
    return chosen


def funding_cost(
    market: Market, side: Side, now: int, hold_hours: int
) -> tuple[float, int]:
    if market.next_funding < now or market.funding_interval_hours <= 0:
        raise ValueError("Funding schedule is missing or stale")
    end = now + hold_hours * 3600
    payments = (
        0
        if market.next_funding > end
        else 1
        + int((end - market.next_funding) // (market.funding_interval_hours * 3600))
    )
    # Do not subsidize a setup with uncertain future funding receipts.
    rate = max(0.0, market.funding_rate * (1 if side == "long" else -1))
    return rate * payments * 100, payments


def build_plan(
    market: Market,
    candles: list[Candle],
    hourly: list[Candle],
    four_hour: list[Candle],
    setup: Setup,
    side: Side,
    now: int,
    settings: SignalSettings,
    cfg: SignalsCfg,
) -> tuple[TradePlan | None, list[Check]]:
    sign = 1 if side == "long" else -1
    entry = market.ask if side == "long" else market.bid
    stop_distance = sign * (entry - setup.stop)
    atr_value = volatility(candles)
    anchor = candles[-1].close
    low, high = sorted(
        (anchor - sign * 0.15 * atr_value, anchor + sign * 0.2 * atr_value)
    )
    checks = [
        Check(
            "Entry zone", low <= entry <= high, "Wait for the entry zone; do not chase"
        ),
        Check(
            "Stop outside normal noise",
            stop_distance >= 0.75 * atr_value,
            "Structural stop must allow at least 0.75 ATR",
        ),
    ]
    if stop_distance <= 0:
        return None, checks + [
            Check("Invalidation", False, "Price has passed the setup's stop")
        ]
    levels: list[float] = []
    for bars in (hourly[-96:], four_hour[-96:]):
        highs, lows = swing_levels(bars)
        levels.extend(highs if side == "long" else lows)
    previous_day = [
        c
        for c in candles
        if now // 86400 * 86400 - 86400 <= c.time < now // 86400 * 86400
    ]
    if len(previous_day) == 96:
        levels.append(
            max(c.high for c in previous_day)
            if side == "long"
            else min(c.low for c in previous_day)
        )
    funding_pct, payments = funding_cost(market, side, now, settings.hold_hours)
    cost_pct = cfg.round_trip_fee_pct + cfg.slippage_pct + funding_pct
    stop_pct = stop_distance / entry * 100
    risk_fraction = (stop_pct + cost_pct) / 100
    targets = select_targets(
        levels,
        entry,
        side,
        stop_distance,
        cost_pct,
        volatility(hourly),
        volatility(four_hour),
        cfg,
    )
    if not targets:
        return None, checks + [
            Check(
                "Structural target",
                False,
                "No confirmed target that clears 2R inside the ≤24h travel budget",
            )
        ]
    notional = min(
        settings.planning_equity * settings.risk_pct / 100 / risk_fraction,
        settings.planning_equity * settings.leverage * 0.9,
    )
    qty = math.floor(notional / entry / market.quantity_step) * market.quantity_step
    notional = qty * entry
    tier = next(
        (
            t
            for t in sorted(market.tiers, key=lambda t: t.minimum, reverse=True)
            if t.minimum <= notional <= t.maximum
        ),
        None,
    )
    reward = sign * (targets[0] - entry) / entry - cost_pct / 100
    ratio = reward / risk_fraction
    max_leverage, liquidation = 0, None
    if tier:
        adverse_basis = min(0, sign * (market.mark - market.price))
        buffer = max(entry * cfg.liquidation_buffer_pct / 100, atr_value * 0.5)
        for leverage in range(1, min(40, tier.max_leverage) + 1):
            estimated = (
                entry
                * (1 - sign / leverage + sign * cost_pct / 100)
                / (1 - sign * tier.maintenance_rate)
            )
            if sign * (setup.stop - estimated) + adverse_basis >= buffer:
                max_leverage = leverage
            if leverage == settings.leverage:
                liquidation = estimated
    usable_band = min(max_leverage, 40) if max_leverage >= 25 else max_leverage
    if tier is not None and settings.leverage <= max_leverage:
        leverage_detail = (
            f"{settings.leverage}x clears the isolated-margin buffer; "
            f"{usable_band}x is the highest 25-40x leverage that still fits this stop"
            if max_leverage >= 25
            else "Estimated isolated-margin buffer passes; verify exchange liquidation price"
        )
    else:
        leverage_detail = (
            f"Reduce leverage or skip; estimated maximum {max_leverage}x"
        )
    checks.extend(
        [
            Check(
                "Reward after costs",
                ratio >= cfg.min_reward_risk,
                f"{ratio:.2f}R net; need {cfg.min_reward_risk:g}R",
            ),
            Check(
                "Funding drag",
                funding_pct <= cfg.max_funding_cost_pct,
                f"{funding_pct:.3f}% projected funding over the hold; max {cfg.max_funding_cost_pct:g}%",
            ),
            Check(
                "Order size",
                qty >= market.min_quantity and qty > 0,
                "Quantity must meet the exchange minimum",
            ),
            Check(
                "Execution depth",
                min(market.bid_depth_usdt, market.ask_depth_usdt)
                >= notional * cfg.min_depth_ratio,
                f"Both sides need at least {cfg.min_depth_ratio:g} times the planned notional",
            ),
            Check(
                "Leverage buffer",
                tier is not None and settings.leverage <= max_leverage,
                leverage_detail,
            ),
        ]
    )
    return TradePlan(
        side,
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
        funding_pct,
        payments,
        liquidation,
        max_leverage,
        settings.leverage,
        settings.hold_hours,
        candles[-1].time + 1800,
        min(0, sign * (market.mark - market.price)),
    ), checks


def evaluate_intraday(
    market: Market,
    frames: dict[str, list[Candle]],
    btc_hourly: list[Candle] | None,
    settings: SignalSettings,
    cfg: SignalsCfg,
    now: int,
) -> Decision:
    result = Decision(market.symbol, as_of=market.as_of, price=market.price)
    bars, hourly, four_hour = frames["15m"], frames["1h"], frames["4h"]
    one, four_bias, four_structure = (
        trend(hourly),
        ema_bias(four_hour),
        trend(four_hour),
    )
    result.bar_time = bars[-1].time
    atr_value, vwap = volatility(bars), session_vwap(bars, now)
    hourly_atr_pct = volatility(hourly) / market.price * 100
    result.metrics = {
        "trend_1h": one,
        "trend_4h": four_bias,
        "trend_4h_structure": four_structure,
        "atr_pct": atr_value / market.price * 100,
        "hourly_atr_pct": hourly_atr_pct,
        "vwap": vwap,
        "funding_rate_pct": market.funding_rate * 100,
        "next_funding": market.next_funding,
        "open_interest": market.open_interest,
        "spread_pct": (market.ask - market.bid) / market.price * 100,
    }
    result.checks = [
        Check(
            "Fresh market data",
            0 <= now - market.as_of <= cfg.max_data_age_seconds,
            "Market snapshot must be fresh",
        ),
        Check(
            "Liquid market",
            market.quote_volume >= cfg.min_quote_volume,
            "24h quote volume must pass the liquidity floor",
        ),
        Check(
            "Spread",
            0 < market.bid <= market.ask
            and (market.ask - market.bid) / market.price * 100 <= cfg.max_spread_pct,
            "Spread must fit the execution limit",
        ),
        Check(
            "Hold-window volatility",
            cfg.min_hourly_atr_pct <= hourly_atr_pct <= cfg.max_hourly_atr_pct,
            f"1h ATR {hourly_atr_pct:.2f}% must fit a 25-40x, ≤24h trade",
        ),
        Check(
            "4h bias / 1h structure",
            one == four_bias and one != "mixed",
            f"4h bias {four_bias}; 1h structure {one}",
        ),
    ]
    if one != four_bias or one not in ("long", "short"):
        result.reasons = [
            "Wait for 4h EMA bias and confirmed 1h structure in the same direction"
        ]
        return result
    side: Side = "long" if one == "long" else "short"
    sign = 1 if side == "long" else -1
    result.side = side
    result.state = f"WATCH_{side.upper()}"
    if market.symbol != "BTCUSDT":
        btc_trend = trend(btc_hourly) if btc_hourly else "unavailable"
        relative = (
            (
                (hourly[-1].close / hourly[-7].close - 1)
                - (btc_hourly[-1].close / btc_hourly[-7].close - 1)
            )
            * 100
            if btc_hourly
            else None
        )
        result.metrics.update(
            {"btc_trend": btc_trend, "relative_strength_pct": relative}
        )
        result.checks.append(
            Check(
                "BTC context",
                btc_trend == side and relative is not None and sign * relative >= 0,
                "Require aligned BTC direction and matching 6h relative strength",
            )
        )
    setup = find_setup(bars, hourly, side, atr_value, vwap, cfg)
    result.checks.append(
        Check(
            "Completed 15m trigger",
            setup is not None,
            "Waiting for a pullback reclaim, impulse continuation, or breakout retest",
        )
    )
    if setup:
        result.setup = setup.name
        result.signal_id = f"{market.symbol}:{side}:{setup.name}:{result.bar_time}"
        result.metrics["relative_volume"] = setup.relative_volume
        result.checks.append(
            Check(
                "Volume confirmation",
                setup.relative_volume >= cfg.relative_volume_min,
                f"{setup.relative_volume:.2f} times baseline volume",
            )
        )
        result.plan, checks = build_plan(
            market, bars, hourly, four_hour, setup, side, now, settings, cfg
        )
        result.checks.extend(checks)
        if result.plan and now >= result.plan.expires_at:
            result.checks.append(
                Check("Entry expiry", False, "Entry window has expired")
            )
        if result.plan and all(c.passed for c in result.checks):
            result.state = f"ENTER_{side.upper()}"
    result.reasons = [c.detail for c in result.checks if not c.passed] or [
        f"{result.setup} confirmed on a completed 15m candle"
    ]
    return result
