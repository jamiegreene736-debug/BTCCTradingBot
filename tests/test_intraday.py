"""Regression coverage for entry validity, risk accounting, lifecycle and read-only execution."""

from __future__ import annotations

import base64
import math
import re
from dataclasses import asdict, fields, replace
from unittest.mock import MagicMock, patch

import pytest
import requests

from bitunix_bot import intraday, scalp_short
from bitunix_bot.client import BitunixClient, BitunixError
from bitunix_bot.config import load
from bitunix_bot.dashboard import create_app
from bitunix_bot.intraday import (
    CHECKLIST_LABELS,
    Candle,
    Market,
    Tier,
    closed_candles,
    ema_bias,
    estimate_liquidation,
    evaluate_intraday,
    find_setup,
    funding_cost,
    liquidation_fit,
    make_check,
    select_targets,
    travel_budget,
    trend,
    waiting_check,
)
from bitunix_bot.signal_config import SignalsCfg, SignalSettings, TrendCfg
from bitunix_bot.signal_scanner import SignalScanner, parse_open_position
from bitunix_bot.signal_store import (
    HOLD_CHECK_LABELS,
    SignalStore,
    TrackedTrade,
    evaluate_exit,
)

NOW = 1_800_000_060
PRICE = 98.7
# Structural pullback low of the happy-path 5m trigger: stop 98.36 (0.35%).
PULLBACK_LOW = 98.395
# A 15m swing high 1.25% above the entry: inside the 2h travel budget
# (1.5 x ATR1h x sqrt(2) = 1.38%), outside the 1h budget (0.97%).
TARGET_HIGH = 99.93


def hourly_bars(drift=0.08, amp=0.25, span=0.32):
    """100 EMA-stacked 1h bars ending at PRICE: ATR14 = 2 x span (0.65%)."""
    return [
        Candle(
            NOW // 3600 * 3600 - (100 - i) * 3600,
            c - 0.05,
            c + span,
            c - span,
            c,
            1000.0,
        )
        for i, c in (
            (i, PRICE - (99 - i) * drift + amp * math.sin((i - 99) * math.pi / 6))
            for i in range(100)
        )
    ]


def fifteen_bars(drift=0.03, amp=0.12, span=0.15, volume=700_000.0):
    """200 sine-drift 15m bars with HH/HL; per-bar volume clears the session gate."""
    return [
        Candle(
            NOW // 900 * 900 - (200 - i) * 900,
            c - 0.02,
            c + span,
            c - span,
            c,
            volume,
        )
        for i, c in (
            (i, PRICE - (199 - i) * drift + amp * math.sin((i - 199) * math.pi / 8))
            for i in range(200)
        )
    ]


def pullback_bars(pullback_low=PULLBACK_LOW, price=PRICE):
    """200 flat 5m bars (ATR ~0.17) ending in a 4-bar pullback and reclaim."""
    base = pullback_low + 0.005
    bars = [
        Candle(
            NOW // 300 * 300 - (200 - i) * 300,
            base,
            base + 0.085,
            base - 0.085,
            base,
            100.0,
        )
        for i in range(200)
    ]
    bars[-4:] = [
        replace(
            bars[-4], open=base, high=base + 0.2, low=pullback_low + 0.005, close=base + 0.15
        ),
        replace(
            bars[-3],
            open=base + 0.15,
            high=base + 0.17,
            low=pullback_low + 0.02,
            close=base + 0.05,
        ),
        replace(
            bars[-2], open=base + 0.05, high=base + 0.10, low=pullback_low, close=base + 0.03
        ),
        replace(
            bars[-1],
            open=base + 0.03,
            high=price + 0.02,
            low=base + 0.02,
            close=price,
            volume=200.0,
        ),
    ]
    return bars


def mirror(frames):
    return {
        key: [
            Candle(c.time, 200 - c.open, 200 - c.low, 200 - c.high, 200 - c.close, c.volume)
            for c in values
        ]
        for key, values in frames.items()
    }


def market_frames(side="long", pullback_low=PULLBACK_LOW, hourly=None):
    frames = {
        "5m": pullback_bars(pullback_low),
        "15m": fifteen_bars(),
        "1h": hourly or hourly_bars(),
    }
    if side == "short":
        frames = mirror(frames)
    price = frames["5m"][-1].close
    market = Market(
        "BTCUSDT",
        price,
        price,
        price - 0.005,
        price + 0.005,
        1_000_000,
        1_000_000,
        100_000_000,
        0,
        8,
        NOW + 3600,
        [Tier(0, 50_000, 0.004, 125)],
        0.001,
        0.001,
        NOW,
    )
    return market, frames


def candle_rows(bars):
    return [
        {
            "time": c.time * 1000,
            "open": c.open,
            "high": c.high,
            "low": c.low,
            "close": c.close,
            "baseVol": c.volume,
        }
        for c in bars
    ]


def ready_decision(side="long", settings=None, **overrides):
    market, frames = market_frames(side, **overrides)
    # Lift one confirmed 15m swing high (low for shorts) inside the 2h travel
    # budget; it sits outside the 64-bar structure window so trend() is unchanged.
    frames["15m"][120] = replace(
        frames["15m"][120],
        **({"high": TARGET_HIGH} if side == "long" else {"low": 200 - TARGET_HIGH}),
    )
    decision = evaluate_intraday(
        market, frames, None, settings or SignalSettings(), SignalsCfg(), NOW
    )
    return decision, market, frames


@pytest.mark.parametrize("side", ["long", "short"])
def test_completed_pullback_produces_entry_with_structural_stop(side):
    decision, _, frames = ready_decision(side)
    assert decision.state == f"ENTER_{side.upper()}", decision.reasons
    plan = decision.plan
    assert plan is not None
    assert plan.net_reward_risk >= 1.5
    assert plan.risk_usdt <= 5
    sign = 1 if side == "long" else -1
    assert sign * (plan.entry - plan.stop) > 0
    assert sign * (plan.target - plan.entry) > 0
    assert plan.leverage == 50 and plan.hold_hours == 2
    assert plan.profile == "trend" and plan.trigger_interval == "5m"
    assert plan.max_leverage >= 50
    assert plan.liquidation_estimate is not None
    assert sign * (plan.stop - plan.liquidation_estimate) >= 0.0025 * plan.entry
    assert plan.stop_pct <= 0.60
    assert plan.stop_pct == pytest.approx(0.35, abs=0.01)
    assert plan.margin == pytest.approx(plan.notional / 50)
    assert plan.expires_at == frames["5m"][-1].time + 300 + 600
    assert decision.metrics["trend_15m"] == side
    assert not hasattr(decision, "confidence")
    assert decision.actions == [
        {
            "label": f"Enter {side}",
            "price": decision.plan.entry_low,
            "price2": decision.plan.entry_high,
        }
    ]


def test_partial_candle_cannot_change_closed_signal():
    _, frames = market_frames()
    rows = candle_rows(frames["5m"])
    rows += [
        {
            "time": NOW // 300 * 300 * 1000,
            "open": 99,
            "high": 200,
            "low": 1,
            "close": 190,
            "baseVol": 999999,
        }
    ]
    assert closed_candles(rows, "5m", NOW) == frames["5m"]
    assert closed_candles(list(reversed(rows)), "5m", NOW) == frames["5m"]


@pytest.mark.parametrize("fault", ["gap", "nan", "stale", "duplicate", "bad_ohlc"])
def test_invalid_candles_fail_closed(fault):
    _, frames = market_frames()
    rows = candle_rows(frames["5m"])
    if fault == "gap":
        rows.pop(130)
    if fault == "nan":
        rows[-1]["close"] = "nan"
    if fault == "stale":
        rows.pop()
    if fault == "duplicate":
        rows.append({**rows[-1], "close": 98.5})
    if fault == "bad_ohlc":
        rows[-1]["low"] = 101
    with pytest.raises(ValueError):
        closed_candles(rows, "5m", NOW)


@pytest.mark.parametrize("side", ["long", "short"])
def test_breakout_needs_later_retest_and_volume(side):
    _, frames = market_frames()
    bars = [
        replace(c, open=99.5, close=99.5, high=100, low=99, volume=100)
        for c in frames["5m"]
    ]
    bars[-3] = replace(
        bars[-3], open=99.5, close=100.7, high=100.8, low=99.4, volume=250
    )
    bars[-2] = replace(bars[-2], open=100.7, close=100.3, high=100.8, low=100.1)
    bars[-1] = replace(bars[-1], open=100.3, close=100.6, high=100.7, low=100.05)
    if side == "short":
        bars = [
            Candle(
                c.time, 200 - c.open, 200 - c.low, 200 - c.high, 200 - c.close, c.volume
            )
            for c in bars
        ]
    structure = frames["15m"] if side == "long" else mirror(frames)["15m"]
    setup = find_setup(bars, structure, side, 1, None, SignalsCfg().trend)
    assert setup is not None and setup.name == "Breakout & retest"
    bars[-3] = replace(bars[-3], volume=100)
    setup = find_setup(bars, structure, side, 1, None, SignalsCfg().trend)
    assert setup is None or setup.name != "Breakout & retest"


def test_costs_and_actual_funding_schedule_can_block_trade():
    _, market, frames = ready_decision()
    market = replace(
        market, funding_rate=0.01, funding_interval_hours=1, next_funding=NOW + 10
    )
    decision = evaluate_intraday(
        market, frames, None, SignalSettings(), SignalsCfg(), NOW
    )
    assert not decision.state.startswith("ENTER")
    assert any(c.label == "Funding drag" and not c.passed for c in decision.checks)
    # Two hourly prints fall inside a 120-minute hold.
    assert funding_cost(market, "long", NOW, 120) == (2.0, 2)
    assert funding_cost(market, "short", NOW, 120) == (0, 2)


def test_funding_cost():
    market, _ = market_frames()
    rate = 0.0001
    market = replace(market, funding_rate=rate, funding_interval_hours=8)
    assert funding_cost(replace(market, next_funding=NOW + 3600), "long", NOW, 120) == (
        pytest.approx(rate * 100),
        1,
    )
    assert funding_cost(replace(market, next_funding=NOW + 7201), "long", NOW, 120) == (
        0.0,
        0,
    )
    hourly = replace(market, funding_interval_hours=1, next_funding=NOW + 1800)
    assert funding_cost(hourly, "long", NOW, 120) == (pytest.approx(rate * 200), 2)
    assert funding_cost(hourly, "long", NOW, 60) == (pytest.approx(rate * 100), 1)


def test_exact_funding_settlement_is_charged():
    market, _ = market_frames()
    market = replace(
        market, funding_rate=0.0001, next_funding=NOW, funding_interval_hours=8
    )
    cost, payments = funding_cost(market, "long", NOW, 120)
    assert payments == 1 and cost == pytest.approx(0.01)


@pytest.mark.parametrize(
    "change",
    [
        {"quote_volume": 1},
        {"bid_depth_usdt": 1},
        {"bid": 98},
        {"as_of": NOW - 100},
        {"tiers": []},
    ],
)
def test_execution_and_data_gates_block_entries(change):
    _, market, frames = ready_decision()
    decision = evaluate_intraday(
        replace(market, **change), frames, None, SignalSettings(), SignalsCfg(), NOW
    )
    assert not decision.state.startswith("ENTER")


def test_higher_leverage_tightens_stop_to_the_margin_budget():
    _, market, frames = ready_decision()
    market = replace(market, tiers=[Tier(0, 50_000, 0.004, 125)])

    def plan_at(leverage, max_margin_loss_pct):
        return evaluate_intraday(
            market,
            frames,
            None,
            SignalSettings(leverage=leverage, max_margin_loss_pct=max_margin_loss_pct),
            SignalsCfg(),
            NOW,
        )

    low, high = plan_at(20, 25), plan_at(50, 25)
    assert low.state == "ENTER_LONG", low.reasons
    assert high.state == "ENTER_LONG", high.reasons
    assert low.plan.stop == low.plan.structural_stop
    assert high.plan.structural_stop == low.plan.structural_stop
    assert high.plan.stop > low.plan.stop
    assert high.plan.margin_loss_pct == pytest.approx(25, abs=0.01)
    assert low.plan.margin_loss_pct < 25
    assert high.plan.liquidation_estimate < high.plan.stop
    detail = next(c for c in high.checks if c.label == "Leverage ceiling").detail
    assert "tightened" in detail and "planned margin loss" in detail
    assert "tightened from" in next(c for c in high.checks if c.label == "Stop size").detail
    # Sized to the tighter stop: same planned loss, larger notional.
    assert high.plan.notional > low.plan.notional
    assert high.plan.risk_usdt == pytest.approx(low.plan.risk_usdt, rel=0.05)
    cramped = plan_at(50, 15)
    assert cramped.state == "WATCH_LONG"
    labels = {c.label: c for c in cramped.checks}
    assert not labels["Stop size"].passed
    assert not labels["Leverage ceiling"].passed
    assert "reduce leverage" in labels["Leverage ceiling"].detail


def test_leverage_that_cannot_open_is_blocked_not_widened():
    _, market, frames = ready_decision()
    # A 2.5% maintenance tier: 50x cannot post enough margin at all.
    market = replace(market, tiers=[Tier(0, 50_000, 0.025, 125)])
    low = evaluate_intraday(
        market, frames, None, SignalSettings(leverage=20), SignalsCfg(), NOW
    )
    high = evaluate_intraday(
        market, frames, None, SignalSettings(leverage=50), SignalsCfg(), NOW
    )
    assert low.state == "ENTER_LONG", low.reasons
    assert high.state == "WATCH_LONG"
    ceiling = next(c for c in high.checks if c.label == "Leverage ceiling")
    assert not ceiling.passed and "reduce leverage" in ceiling.detail
    assert high.plan.max_leverage < 50
    assert high.plan.structural_stop == low.plan.stop
    assert high.plan.stop >= high.plan.structural_stop


def test_stop_never_sits_beyond_the_liquidation_buffer():
    _, market, frames = ready_decision()
    market = replace(market, tiers=[Tier(0, 50_000, 0.004, 125)])
    loose = evaluate_intraday(
        market,
        frames,
        None,
        SignalSettings(leverage=100, max_margin_loss_pct=100),
        SignalsCfg(),
        NOW,
    )
    plan = loose.plan
    buffer = plan.entry * SignalsCfg().trend.liquidation_buffer_pct / 100
    assert plan.stop - plan.liquidation_estimate >= buffer - 1e-9
    assert plan.stop >= plan.structural_stop


def test_planning_settings_accept_and_bound_the_margin_loss_budget():
    base = {"planning_equity": 1000, "risk_pct": 0.5, "leverage": 75, "hold_hours": 2, "profile": "scalp"}
    assert SignalSettings.from_dict(base).max_margin_loss_pct == 50
    saved = SignalSettings.from_dict({**base, "max_margin_loss_pct": 35})
    assert saved.max_margin_loss_pct == 35 and saved.leverage == 75
    assert SignalSettings.from_dict(asdict(saved)) == saved
    for bad in (5, 101, "50", True):
        with pytest.raises(ValueError):
            SignalSettings.from_dict({**base, "max_margin_loss_pct": bad})


def test_select_targets_skips_too_close_and_beyond_hold_budget():
    # 100.2 nets under 1.5R, 140 is beyond the travel budget; 101.2 clears both.
    assert select_targets(
        [100.2, 101.2, 140.0], 100.0, "long", 0.35, 0.18, 1.273, 1.5
    ) == [101.2]
    assert select_targets([100.2, 101.2, 140.0], 100.0, "long", 0.35, 0.18, 0.90, 1.5) == []
    assert select_targets([99.8, 98.8, 60.0], 100.0, "short", 0.35, 0.18, 1.273, 1.5) == [98.8]


def test_1h_ema_bias_can_enter_without_confirmed_1h_swings():
    _, market, frames = ready_decision()
    # A straight 1h ramp has no pivots, so trend() is mixed while the EMA bias is long.
    frames["1h"] = [
        replace(
            c,
            open=PRICE - (99 - i) * 0.08 - 0.05,
            close=PRICE - (99 - i) * 0.08,
            high=PRICE - (99 - i) * 0.08 + 0.32,
            low=PRICE - (99 - i) * 0.08 - 0.32,
        )
        for i, c in enumerate(frames["1h"])
    ]
    assert trend(frames["1h"]) == "mixed"
    assert ema_bias(frames["1h"]) == "long"
    decision = evaluate_intraday(
        market, frames, None, SignalSettings(), SignalsCfg(), NOW
    )
    assert decision.state == "ENTER_LONG", decision.reasons
    assert decision.metrics["trend_1h"] == "long"
    assert decision.metrics["trend_15m"] == "long"
    assert "trend_4h" not in decision.metrics


def test_opposite_1h_bias_blocks_entry():
    _, market, frames = ready_decision()
    frames["1h"] = [
        replace(
            c,
            open=110 - i * 0.08 + 0.05,
            close=110 - i * 0.08,
            high=110 - i * 0.08 + 0.32,
            low=110 - i * 0.08 - 0.32,
        )
        for i, c in enumerate(frames["1h"])
    ]
    assert ema_bias(frames["1h"]) == "short"
    decision = evaluate_intraday(
        market, frames, None, SignalSettings(), SignalsCfg(), NOW
    )
    assert decision.state == "WAIT" and decision.side == ""
    alignment = next(c for c in decision.checks if c.label == "1h bias / 15m structure")
    assert not alignment.passed and "1h bias short; 15m structure long" == alignment.detail
    assert all(
        c.waiting for c in decision.checks if c.label in ("Entry zone", "BTC context")
    )


def test_impulse_continuation_after_pullback():
    _, frames = market_frames()
    bars = [
        replace(c, open=80.4, close=80.5, high=80.7, low=80.3, volume=100)
        for c in frames["5m"]
    ]
    bars[-6] = replace(
        bars[-6], open=105.5, close=107.2, high=107.3, low=105.4, volume=220
    )
    bars[-5] = replace(bars[-5], open=107.2, close=106.8, high=107.25, low=106.7)
    bars[-4] = replace(bars[-4], open=106.8, close=106.5, high=106.9, low=106.4)
    bars[-3] = replace(bars[-3], open=106.5, close=106.35, high=106.6, low=106.25)
    bars[-2] = replace(bars[-2], open=106.35, close=106.4, high=106.55, low=106.25)
    bars[-1] = replace(
        bars[-1], open=106.4, close=106.85, high=106.9, low=106.35, volume=180
    )
    setup = find_setup(bars, frames["15m"], "long", 0.7, None, SignalsCfg().trend)
    assert setup is not None and setup.name == "Impulse continuation"


def test_stop_covers_costs_at_one_r():
    trade, decision, _ = new_trade()
    risk = abs(trade.plan.entry - trade.plan.stop)
    decision.price = trade.plan.entry + 1.05 * risk
    evaluate_exit(trade, decision, [], NOW + 10, SignalsCfg())
    assert trade.state == "HOLD_LONG"
    covered = trade.plan.entry * (1 + trade.plan.cost_pct / 100)
    assert trade.current_stop == pytest.approx(covered)
    assert trade.current_stop > trade.plan.stop


def test_alt_requires_btc_context_and_relative_strength():
    _, market, frames = ready_decision()
    decision = evaluate_intraday(
        replace(market, symbol="ETHUSDT"),
        frames,
        None,
        SignalSettings(),
        SignalsCfg(),
        NOW,
    )
    assert decision.state == "WATCH_LONG"
    assert any(c.label == "BTC context" and not c.passed for c in decision.checks)


def new_trade(side="long"):
    decision, _, frames = ready_decision(side)
    assert decision.plan
    trade = TrackedTrade(
        "test",
        "BTCUSDT",
        "paper",
        NOW,
        decision.plan,
        decision.plan.stop,
        decision.plan.entry,
        state=f"HOLD_{side.upper()}",
        checked_at=NOW,
    )
    return trade, decision, frames


def flat_bars(entry, start, minutes, interval=300):
    """Completed trigger bars pinned at the entry price: no progress in either direction."""
    return [
        Candle(start // interval * interval + k * interval, entry, entry, entry, entry, 100)
        for k in range(minutes * 60 // interval)
    ]


@pytest.mark.parametrize("side", ["long", "short"])
@pytest.mark.parametrize(
    "exit_type", ["stop", "target", "time", "structure", "stale", "late"]
)
def test_exit_rules_are_position_specific_and_sticky(side, exit_type):
    trade, decision, frames = new_trade(side)
    sign = 1 if side == "long" else -1
    now = NOW + 20
    bars = frames["5m"]
    if exit_type == "stop":
        decision.price = trade.plan.stop
    if exit_type == "target":
        decision.price = trade.plan.target
    if exit_type == "time":
        now = NOW + 2 * 3600
    if exit_type == "structure":
        decision.metrics["trend_15m"] = "short" if side == "long" else "long"
    if exit_type == "stale":
        # 35% of the 120-minute hold: 42 minutes; one minute earlier still holds.
        decision.price = trade.plan.entry
        bars = flat_bars(trade.plan.entry, NOW, 42)
        decision.as_of = NOW + 41 * 60
        evaluate_exit(trade, decision, bars, NOW + 41 * 60, SignalsCfg())
        assert trade.state == f"HOLD_{side.upper()}"
        now = NOW + 42 * 60 + 60
    if exit_type == "late":
        # Best progress cleared the stale review (0.3R) but the trade sits at
        # +0.2R when 75% of the hold has gone.
        risk = abs(trade.plan.entry - trade.plan.stop)
        trade.best_price = trade.plan.entry + sign * 0.4 * risk
        decision.price = trade.plan.entry + sign * 0.2 * risk
        bars = []
        now = NOW + 90 * 60
    decision.as_of = now
    evaluate_exit(trade, decision, bars, now, SignalsCfg())
    assert trade.state == f"EXIT_{side.upper()}", trade.reason
    assert trade.suggestion == f"CLOSE_{side.upper()}"
    assert trade.hold_confidence is not None and trade.hold_confidence <= 10
    assert [item.label for item in trade.checks] == list(HOLD_CHECK_LABELS)
    if exit_type == "late":
        assert "Hold window closing" in trade.reason
    if exit_type == "structure":
        assert "15m structure reversed" in trade.reason
    hold_left = next(c for c in trade.checks if c.label == "Hold time remaining")
    if exit_type == "time":
        assert hold_left.detail == "Maximum holding time reached"
    else:
        assert re.fullmatch(r"\d+ of 120 min hold left", hold_left.detail)
    decision.price = trade.plan.entry
    decision.metrics["trend_1h"] = side
    decision.metrics["trend_15m"] = side
    evaluate_exit(trade, decision, [], now + 1, SignalsCfg())
    assert trade.state == f"EXIT_{side.upper()}"
    assert trade.closed_at is None


def test_stale_market_causes_review_and_time_exit_still_works():
    trade, decision, _ = new_trade()
    evaluate_exit(trade, decision, [], NOW + 100, SignalsCfg())
    assert trade.state == "REVIEW"
    evaluate_exit(trade, None, [], NOW + 2 * 3600, SignalsCfg())
    assert trade.state == "EXIT_LONG"


def test_pre_entry_wick_does_not_trigger_exit():
    trade, decision, _ = new_trade()
    bar = Candle(NOW // 900 * 900, 99, 120, 10, 99, 100)
    evaluate_exit(trade, decision, [bar], NOW + 10, SignalsCfg())
    assert trade.state == "HOLD_LONG"


def test_hold_suggestion_is_live_with_fixed_close_checks():
    trade, decision, _ = new_trade("short")
    evaluate_exit(trade, decision, [], NOW + 10, SignalsCfg())
    assert trade.state == "HOLD_SHORT"
    assert trade.suggestion == "HOLD_SHORT"
    assert [item.label for item in trade.checks] == list(HOLD_CHECK_LABELS)
    assert trade.hold_confidence >= 70
    assert trade.reason.startswith("Live:")
    assert "15m short" in trade.reason
    assert "min left" in trade.reason


def test_hold_reason_and_check_use_minutes():
    trade, decision, _ = new_trade()
    now = NOW + 30 * 60
    decision.as_of = now
    evaluate_exit(trade, decision, [], now, SignalsCfg())
    assert trade.state == "HOLD_LONG"
    assert trade.reason.endswith("90 min left")
    hold_left = next(c for c in trade.checks if c.label == "Hold time remaining")
    assert hold_left.detail == "90 of 120 min hold left"
    progress = next(c for c in trade.checks if c.label == "Progress vs review window")
    assert progress.detail.endswith("need 0.3R within 42m")
    one_hour = replace(trade.plan, hold_hours=1)
    trade = TrackedTrade("t1", "BTCUSDT", "paper", NOW, one_hour, one_hour.stop, one_hour.entry)
    now = NOW + 15 * 60
    decision.as_of = now
    evaluate_exit(trade, decision, [], now, SignalsCfg())
    assert trade.state == "HOLD_LONG"
    assert trade.reason.endswith("45 min left")
    progress = next(c for c in trade.checks if c.label == "Progress vs review window")
    assert progress.detail.endswith("need 0.3R within 21m")


def test_soft_hold_failures_suggest_close_without_latching_exit():
    trade, decision, _ = new_trade("short")
    risk = abs(trade.plan.entry - trade.plan.stop)
    decision.price = trade.plan.entry + 0.6 * risk
    decision.metrics["trend_1h"] = "long"
    decision.metrics["vwap"] = decision.price - 1
    decision.metrics["funding_rate_pct"] = -0.05
    evaluate_exit(trade, decision, [], NOW + 10, SignalsCfg())
    assert trade.state == "HOLD_SHORT"
    assert trade.suggestion == "CONSIDER_CLOSE"
    assert trade.hold_confidence < 70
    assert any(item.label == "Bias intact" and not item.passed for item in trade.checks)
    assert any(item.label == "Structure intact" and item.passed for item in trade.checks)
    assert any(
        item.label == "Drawdown contained" and not item.passed for item in trade.checks
    )
    evaluate_exit(trade, decision, [], NOW + 15, SignalsCfg())
    assert trade.state == "HOLD_SHORT"
    assert trade.suggestion == "CONSIDER_CLOSE"


@pytest.mark.parametrize("side", ["long", "short"])
def test_hope_hold_and_liquidation_latch_exit(side):
    trade, decision, _ = new_trade(side)
    sign = 1 if side == "long" else -1
    risk = abs(trade.plan.entry - trade.plan.stop)
    decision.price = trade.plan.entry - sign * 0.8 * risk
    decision.as_of = NOW + 10
    evaluate_exit(trade, decision, [], NOW + 10, SignalsCfg())
    assert trade.state == f"EXIT_{side.upper()}"
    assert trade.suggestion == f"CLOSE_{side.upper()}"
    assert "Do not wait for a reversal" in trade.reason
    decision.price = trade.plan.entry
    evaluate_exit(trade, decision, [], NOW + 15, SignalsCfg())
    assert trade.state == f"EXIT_{side.upper()}"

    trade, decision, _ = new_trade(side)
    trade.plan = replace(
        trade.plan, liquidation_estimate=decision.price - sign * decision.price * 0.0030
    )
    evaluate_exit(trade, decision, [], NOW + 10, SignalsCfg())
    assert trade.state == f"HOLD_{side.upper()}"
    trade.plan = replace(
        trade.plan, liquidation_estimate=decision.price - sign * decision.price * 0.0020
    )
    evaluate_exit(trade, decision, [], NOW + 15, SignalsCfg())
    assert trade.state == f"EXIT_{side.upper()}"
    assert "liquidation" in trade.reason.lower()


def test_liquidation_buffer_exit_uses_profile_buffer():
    cfg = SignalsCfg()
    assert cfg.trend.liquidation_buffer_pct == 0.25
    assert cfg.scalp.liquidation_buffer_pct == 0.15
    trade, decision, _ = new_trade()
    price = decision.price
    trade.plan = replace(trade.plan, liquidation_estimate=price * (1 - 0.0020))
    evaluate_exit(trade, decision, [], NOW + 10, cfg)
    assert trade.state == "EXIT_LONG" and "liquidation" in trade.reason.lower()

    trade, decision, _ = new_trade()
    scalp_plan = replace(
        trade.plan, profile="scalp", trigger_interval="1m", liquidation_estimate=price * (1 - 0.0020)
    )
    trade = TrackedTrade("s", "BTCUSDT", "paper", NOW, scalp_plan, scalp_plan.stop, scalp_plan.entry)
    evaluate_exit(trade, decision, [], NOW + 10, cfg)
    assert trade.state == "HOLD_LONG"
    buffer_check = next(c for c in trade.checks if c.label == "Liquidation buffer")
    assert buffer_check.passed and "need ≥0.15%" in buffer_check.detail
    trade.plan = replace(scalp_plan, liquidation_estimate=price * (1 - 0.0010))
    evaluate_exit(trade, decision, [], NOW + 15, cfg)
    assert trade.state == "EXIT_LONG" and "liquidation" in trade.reason.lower()


def test_late_hold_exit():
    cfg = SignalsCfg()
    late_at = NOW + round(cfg.trend.late_hold_fraction * 120) * 60
    assert late_at == NOW + 90 * 60
    trade, decision, _ = new_trade()
    risk = abs(trade.plan.entry - trade.plan.stop)
    # Best progress cleared the stale review; the live price has faded to +0.2R.
    trade.best_price = trade.plan.entry + 0.4 * risk
    decision.price = trade.plan.entry + 0.2 * risk
    decision.as_of = late_at - 60
    evaluate_exit(trade, decision, [], late_at - 60, cfg)
    assert trade.state == "HOLD_LONG", trade.reason
    decision.as_of = late_at
    evaluate_exit(trade, decision, [], late_at, cfg)
    assert trade.state == "EXIT_LONG"
    assert trade.reason == "Hold window closing without progress; close on Bitunix"
    assert trade.suggestion == "CLOSE_LONG"

    trade, decision, _ = new_trade()
    decision.price = trade.plan.entry + 0.6 * risk
    decision.as_of = late_at
    evaluate_exit(trade, decision, [], late_at, cfg)
    assert trade.state == "HOLD_LONG", trade.reason
    assert trade.reason.endswith("30 min left")


def test_unconfirmed_stop_blocks_hold_and_losing_unprotected_exits():
    trade, decision, _ = new_trade()
    trade.kind = "manual"
    trade.exchange_stop_confirmed = False
    evaluate_exit(trade, decision, [], NOW + 10, SignalsCfg())
    assert trade.state == "HOLD_LONG"
    assert trade.suggestion == "SET_STOP"
    assert "Set the Bitunix stop" in trade.reason
    assert any(
        item.label == "Exchange protective stop" and not item.passed
        for item in trade.checks
    )

    risk = abs(trade.plan.entry - trade.plan.stop)
    decision.price = trade.plan.entry - 0.55 * risk
    evaluate_exit(trade, decision, [], NOW + 15, SignalsCfg())
    assert trade.state == "EXIT_LONG"
    assert "Unprotected" in trade.reason


def test_confirm_stop_allows_hold_on_manual_track(tmp_path):
    scanner, decision = scanner_with_entry(tmp_path)
    with patch("time.time", return_value=NOW):
        trade = scanner.track(
            {
                "signal_id": decision.signal_id,
                "kind": "manual",
                "entry": decision.plan.entry,
                "quantity": decision.plan.quantity,
            }
        )
    assert trade.exchange_stop_confirmed is False
    with patch("time.time", return_value=NOW + 10):
        confirmed = scanner.confirm_stop({"id": trade.id})
        evaluate_exit(
            confirmed, decision, [], NOW + 10, SignalsCfg()
        )
    assert confirmed.exchange_stop_confirmed is True
    assert confirmed.suggestion == "HOLD_LONG"


def test_place_stop_sends_position_tpsl_and_never_opens_or_closes(tmp_path):
    row = {
        "positionId": "HYPE1",
        "symbol": "HYPEUSDT",
        "qty": "36.59",
        "side": "SHORT",
        "avgOpenPrice": "80.37",
        "markPrice": "80.574",
        "unrealizedPNL": "-7.318",
        "leverage": 40,
        "ctime": (NOW - 120) * 1000,
    }
    scanner, _ = _live_scanner(tmp_path, [row])
    parsed = parse_open_position(row)
    assert parsed is not None
    with patch("time.time", return_value=NOW):
        scanner._sync_exchange_positions([parsed], scanner.decisions, NOW, fetch_ok=True)
        trade = scanner.store.trades(active_only=True)[0]
        scanner.client.pending_tpsl.return_value = []
        scanner.client.place_qty_tpsl.return_value = {"orderId": "SL1"}
        placed = scanner.place_stop({"id": trade.id})
    assert placed.exchange_stop_confirmed is True
    scanner.client.place_qty_tpsl.assert_called_once()
    args = scanner.client.place_qty_tpsl.call_args.args
    assert args[0] == "HYPEUSDT" and args[1] == "HYPE1"
    assert float(args[2]) == pytest.approx(trade.current_stop)
    assert float(args[3]) == pytest.approx(36.59)
    scanner.client.place_order.assert_not_called()
    scanner.client.flash_close_position.assert_not_called()


def test_place_stop_rounds_to_quote_precision(tmp_path):
    row = {
        "positionId": "XAG1",
        "symbol": "XAGUSDT",
        "qty": "12.3",
        "side": "LONG",
        "avgOpenPrice": "65.5",
        "markPrice": "66.48",
        "unrealizedPNL": "12.05",
        "leverage": 40,
    }
    scanner, _ = _live_scanner(tmp_path, [row])
    scanner._cache["pairs"] = (
        NOW,
        [
            {
                "symbol": "XAGUSDT",
                "quotePrecision": 2,
                "basePrecision": 3,
                "minTradeVolume": "0.1",
            }
        ],
    )
    parsed = parse_open_position(row)
    assert parsed is not None
    with patch("time.time", return_value=NOW):
        scanner._sync_exchange_positions([parsed], scanner.decisions, NOW, fetch_ok=True)
        trade = scanner.store.trades(active_only=True)[0]
        trade.current_stop = 65.5025
        scanner.store.save_trade(trade)
        scanner.client.pending_tpsl.return_value = []
        scanner.client.place_qty_tpsl.return_value = {"orderId": "SLXAG"}
        placed = scanner.place_stop({"id": trade.id})
    assert placed.exchange_stop_confirmed is True
    args = scanner.client.place_qty_tpsl.call_args.args
    assert args[0] == "XAGUSDT" and args[1] == "XAG1"
    assert args[2] in {"65.5", "65.50"}
    assert args[2] != "65.5025"
    assert float(args[3]) == pytest.approx(12.3)


def test_place_stop_falls_back_when_quantity_stop_already_exists(tmp_path):
    row = {
        "positionId": "HYPE1",
        "symbol": "HYPEUSDT",
        "qty": "36.59",
        "side": "SHORT",
        "avgOpenPrice": "80.37",
        "markPrice": "80.574",
        "unrealizedPNL": "-7.318",
        "leverage": 40,
    }
    scanner, _ = _live_scanner(tmp_path, [row])
    parsed = parse_open_position(row)
    assert parsed is not None
    with patch("time.time", return_value=NOW):
        scanner._sync_exchange_positions([parsed], scanner.decisions, NOW, fetch_ok=True)
        trade = scanner.store.trades(active_only=True)[0]
        scanner.client.pending_tpsl.return_value = []
        scanner.client.place_qty_tpsl.side_effect = BitunixError(
            30019, "already exist", {}
        )
        scanner.client.place_position_tpsl.return_value = {"orderId": "SL2"}
        placed = scanner.place_stop({"id": trade.id})
    assert placed.exchange_stop_confirmed is True
    scanner.client.place_position_tpsl.assert_called_once()
    scanner.client.place_order.assert_not_called()


def test_place_stop_tightens_a_wider_existing_stop(tmp_path):
    row = {
        "positionId": "HYPE1",
        "symbol": "HYPEUSDT",
        "qty": "36.59",
        "side": "SHORT",
        "avgOpenPrice": "80.37",
        "markPrice": "80.574",
        "unrealizedPNL": "-7.318",
        "leverage": 40,
    }
    scanner, _ = _live_scanner(tmp_path, [row])
    parsed = parse_open_position(row)
    assert parsed is not None
    with patch("time.time", return_value=NOW):
        scanner._sync_exchange_positions([parsed], scanner.decisions, NOW, fetch_ok=True)
        trade = scanner.store.trades(active_only=True)[0]
        scanner.client.pending_tpsl.return_value = [
            {
                "positionId": "HYPE1",
                "id": "TPSL1",
                "slPrice": "90",
                "slQty": "36.59",
                "slStopType": "LAST_PRICE",
                "slOrderType": "MARKET",
            }
        ]
        scanner.place_stop({"id": trade.id})
    scanner.client.place_position_tpsl.assert_not_called()
    scanner.client.modify_tpsl_order.assert_called_once()
    assert scanner.client.modify_tpsl_order.call_args.kwargs["sl_price"] == (
        f"{trade.current_stop:.8f}".rstrip("0").rstrip(".")
    )


def test_place_stop_and_confirm_stop_evaluate_on_trigger_frame(tmp_path):
    row = {
        "positionId": "HYPE1",
        "symbol": "HYPEUSDT",
        "qty": "36.59",
        "side": "SHORT",
        "avgOpenPrice": "80.37",
        "markPrice": "80.574",
        "unrealizedPNL": "-7.318",
        "leverage": 40,
        "ctime": (NOW - 120) * 1000,
    }
    scanner, decision = _live_scanner(tmp_path, [row])
    # Only the trigger frame is held for every tracked symbol; no 15m frame.
    scanner.frames = {
        symbol: {"5m": frames["5m"]} for symbol, frames in scanner.frames.items()
    }
    scanner.frames["HYPEUSDT"] = {"5m": []}
    parsed = parse_open_position(row)
    assert parsed is not None
    with patch("time.time", return_value=NOW):
        manual = scanner.track(
            {
                "signal_id": decision.signal_id,
                "kind": "manual",
                "entry": decision.plan.entry,
                "quantity": decision.plan.quantity,
            }
        )
        confirmed = scanner.confirm_stop({"id": manual.id})
        scanner._sync_exchange_positions([parsed], scanner.decisions, NOW, fetch_ok=True)
        trade = next(t for t in scanner.store.trades(active_only=True) if t.symbol == "HYPEUSDT")
        assert trade.plan.trigger_interval == "5m"
        scanner.client.pending_tpsl.return_value = []
        scanner.client.place_qty_tpsl.return_value = {"orderId": "SL1"}
        placed = scanner.place_stop({"id": trade.id})
    assert placed.exchange_stop_confirmed and confirmed.exchange_stop_confirmed
    assert confirmed.state == "HOLD_LONG" and confirmed.suggestion == "HOLD_LONG"
    assert all(
        call.args[1] != "15m" for call in scanner.client.klines.call_args_list
    )


def _hype_position(side: str, leverage: int):
    parsed = parse_open_position(
        {
            "positionId": f"HYPE-{side}-{leverage}",
            "symbol": "HYPEUSDT",
            "qty": "36.59",
            "side": side.upper(),
            "avgOpenPrice": "80.37",
            "markPrice": "80.37",
            "unrealizedPNL": "0",
            "leverage": leverage,
        }
    )
    assert parsed is not None
    return parsed


def _imported_trade(scanner, plan, side, position_id):
    return TrackedTrade(
        f"exchange:{position_id}",
        "HYPEUSDT",
        "exchange",
        NOW,
        plan,
        plan.stop,
        80.37,
        state=f"HOLD_{side.upper()}",
        reason="imported",
        checked_at=NOW,
        mark_price=80.37,
        unrealized_pnl=0.0,
        exchange_position_id=position_id,
    )


def test_imported_position_fallback_plan_is_fifty_x_safe(tmp_path):
    scanner, _ = _live_scanner(tmp_path, [])
    scanner.client.position_tiers.return_value = []  # tiers unknown
    for side in ("long", "short"):
        parsed = _hype_position(side, 50)
        with patch("time.time", return_value=NOW):
            plan = scanner._plan_for_position(parsed, None, SignalSettings(), NOW)
        sign = 1 if side == "long" else -1
        cost_pct = SignalsCfg().round_trip_fee_pct + SignalsCfg().slippage_pct
        assert plan.profile == "trend" and plan.trigger_interval == "5m"
        assert plan.leverage == 50 and plan.hold_hours == 2
        assert 0 < plan.stop_pct <= 0.60
        # A 1% maintenance guess still leaves the 0.20% stop a buffer inside.
        assert plan.liquidation_estimate == pytest.approx(
            estimate_liquidation(80.37, 50, cost_pct, 0.01, side)
        )
        assert sign * (plan.stop - plan.liquidation_estimate) > 0
        assert plan.max_leverage >= 50
        assert sign * (plan.target - plan.entry) > 0
        assert plan.net_reward_risk >= SignalsCfg().trend.min_reward_risk - 1e-9
        assert plan.expires_at == NOW + 1800


@pytest.mark.parametrize("leverage", [70, 75, 85, 100, 125])
@pytest.mark.parametrize("side", ["long", "short"])
def test_imported_position_without_tiers_starts_in_hold_at_high_leverage(
    tmp_path, leverage, side
):
    """A guessed maintenance tier must never raise a sticky liquidation EXIT."""
    scanner, _ = _live_scanner(tmp_path, [])
    scanner.client.position_tiers.return_value = []
    parsed = _hype_position(side, leverage)
    settings = SignalSettings(leverage=min(leverage, 100))
    with patch("time.time", return_value=NOW):
        plan = scanner._plan_for_position(parsed, None, settings, NOW)
    sign = 1 if side == "long" else -1
    buffer = SignalsCfg().trend.liquidation_buffer_pct
    assert plan.leverage == leverage
    assert plan.liquidation_estimate is not None
    assert sign * (plan.stop - plan.liquidation_estimate) / plan.entry * 100 >= buffer
    assert plan.max_leverage >= leverage
    trade = _imported_trade(scanner, plan, side, parsed.position_id)
    decision = intraday.Decision(
        "HYPEUSDT", state="WAIT", as_of=NOW, price=80.37, evaluated_at=NOW
    )
    for now in (NOW, NOW + 60):
        evaluate_exit(trade, decision, [], now, SignalsCfg())
        assert trade.state == f"HOLD_{side.upper()}", trade.reason
        liq = next(c for c in trade.checks if c.label == "Liquidation buffer")
        assert liq.passed, liq.detail


def test_imported_position_uses_the_pair_tier_when_available(tmp_path):
    scanner, _ = _live_scanner(tmp_path, [])
    scanner.client.position_tiers.return_value = [
        {"startValue": 0, "endValue": 50000, "maintenanceMarginRate": 0.005, "leverage": 100}
    ]
    cost_pct = SignalsCfg().round_trip_fee_pct + SignalsCfg().slippage_pct
    for side in ("long", "short"):
        parsed = _hype_position(side, 50)
        with patch("time.time", return_value=NOW):
            plan = scanner._plan_for_position(parsed, None, SignalSettings(), NOW)
        assert plan.liquidation_estimate == pytest.approx(
            estimate_liquidation(80.37, 50, cost_pct, 0.005, side)
        )
        # 0.20% stop + 0.25% buffer on a 0.5% tier: ceiling 88x, not the echoed 50x.
        assert plan.max_leverage == 88
    scanner.client.position_tiers.assert_called_once_with("HYPEUSDT")
    # Above the pair cap the estimate still uses the real tier rate, and a real
    # tier that puts 100x inside the buffer is a genuine alarm.
    parsed = _hype_position("long", 125)
    with patch("time.time", return_value=NOW):
        plan = scanner._plan_for_position(parsed, None, SignalSettings(), NOW)
    assert plan.liquidation_estimate == pytest.approx(
        estimate_liquidation(80.37, 125, cost_pct, 0.005, "long")
    )
    assert plan.max_leverage == 88 < plan.leverage
    trade = _imported_trade(scanner, plan, "long", parsed.position_id)
    decision = intraday.Decision(
        "HYPEUSDT", state="WAIT", as_of=NOW, price=80.37, evaluated_at=NOW
    )
    evaluate_exit(trade, decision, [], NOW, SignalsCfg())
    assert trade.state == "EXIT_LONG" and "liquidation buffer" in trade.reason


def test_imported_position_stop_fits_its_real_leverage(tmp_path):
    scanner, _ = _live_scanner(tmp_path, [])
    scanner.client.position_tiers.return_value = [
        {"startValue": 0, "endValue": 50000, "maintenanceMarginRate": 0.004, "leverage": 125}
    ]
    settings = SignalSettings(leverage=50, max_margin_loss_pct=40)
    for side in ("long", "short"):
        parsed = _hype_position(side, 75)
        sign = 1 if side == "long" else -1
        with patch("time.time", return_value=NOW):
            plan = scanner._plan_for_position(parsed, None, settings, NOW)
        assert plan.leverage == 75
        assert sign * (plan.stop - plan.liquidation_estimate) > 0
        assert plan.margin_loss_pct <= 40 + 1e-9
        assert plan.stop_pct * 75 < 100


def test_place_stop_refuses_paper_and_missing_keys(tmp_path):
    scanner, decision = scanner_with_entry(tmp_path)
    with patch("time.time", return_value=NOW):
        paper = scanner.track(
            {
                "signal_id": decision.signal_id,
                "kind": "paper",
                "entry": decision.plan.entry,
                "quantity": decision.plan.quantity,
            }
        )
    with pytest.raises(ValueError, match="Paper"):
        scanner.place_stop({"id": paper.id})
    other, decision = scanner_with_entry(tmp_path / "manual")
    with patch("time.time", return_value=NOW):
        manual = other.track(
            {
                "signal_id": decision.signal_id,
                "kind": "manual",
                "entry": decision.plan.entry,
                "quantity": decision.plan.quantity,
            }
        )
    with pytest.raises(ValueError, match="API keys"):
        other.place_stop({"id": manual.id})


def test_paper_track_does_not_require_exchange_stop(tmp_path):
    scanner, decision = scanner_with_entry(tmp_path)
    with patch("time.time", return_value=NOW):
        trade = scanner.track(
            {
                "signal_id": decision.signal_id,
                "kind": "paper",
                "entry": decision.plan.entry,
                "quantity": decision.plan.quantity,
            }
        )
    evaluate_exit(trade, decision, [], NOW + 10, SignalsCfg())
    assert trade.exchange_stop_confirmed is True
    assert trade.suggestion == "HOLD_LONG"


def test_trailing_stop_never_widens():
    trade, decision, _ = new_trade()
    original = trade.current_stop
    risk = trade.plan.entry - original
    decision.price = trade.plan.entry + 1.2 * risk
    evaluate_exit(trade, decision, [], NOW + 5, SignalsCfg())
    covered = trade.plan.entry * (1 + trade.plan.cost_pct / 100)
    assert trade.current_stop == pytest.approx(covered)
    # Trailing activates at 1.25R; the trail (last - 1R) only replaces the
    # cost-covered stop once it sits above it.
    decision.price = trade.plan.entry + 1.3 * risk
    evaluate_exit(trade, decision, [], NOW + 10, SignalsCfg())
    assert trade.current_stop == pytest.approx(max(covered, decision.price - risk))
    assert trade.current_stop > original
    decision.price = trade.plan.entry + 1.6 * risk
    evaluate_exit(trade, decision, [], NOW + 12, SignalsCfg())
    tightened = trade.current_stop
    assert tightened > covered
    assert tightened == pytest.approx(decision.price - risk)
    decision.price -= 0.1
    evaluate_exit(trade, decision, [], NOW + 15, SignalsCfg())
    assert trade.current_stop >= tightened and trade.plan.stop == original


@pytest.mark.parametrize(
    "values",
    [
        {"planning_equity": float("nan")},
        {"leverage": 101},
        {"leverage": 19},
        {"hold_hours": 24},
        {"hold_hours": 3},
        {"risk_pct": 0},
        {"leverage": 50.5},
    ],
)
def test_invalid_planning_values_rejected(values):
    with pytest.raises(ValueError):
        SignalSettings.from_dict({**asdict(SignalSettings()), **values})


def scanner_with_entry(tmp_path):
    scanner = SignalScanner(
        MagicMock(spec=BitunixClient),
        SignalsCfg(),
        SignalStore(str(tmp_path / "signals.db")),
    )
    decision, _, frames = ready_decision()
    scanner.decisions[decision.symbol] = decision
    scanner.frames[decision.symbol] = frames
    return scanner, decision


def test_tracking_is_idempotent_persistent_and_does_not_send_orders(tmp_path):
    scanner, decision = scanner_with_entry(tmp_path)
    values = {
        "signal_id": decision.signal_id,
        "kind": "paper",
        "entry": decision.plan.entry,
        "quantity": decision.plan.quantity,
    }
    with patch("time.time", return_value=NOW):
        first = scanner.track(values)
        assert scanner.track(values).id == first.id
    assert len(SignalStore(str(tmp_path / "signals.db")).trades()) == 1
    scanner.client.place_order.assert_not_called()
    with patch("time.time", return_value=NOW + 60):
        closed = scanner.close_track({"id": first.id, "exit_price": first.plan.target})
    assert closed.closed_at and closed.estimated_net_pnl > 0
    assert (
        scanner.close_track({"id": first.id, "exit_price": 1}).exit_price
        == first.plan.target
    )


def test_settings_invalidate_old_entries_not_existing_plans(tmp_path):
    scanner, decision = scanner_with_entry(tmp_path)
    with patch("time.time", return_value=NOW):
        trade = scanner.track(
            {
                "signal_id": decision.signal_id,
                "kind": "manual",
                "entry": decision.plan.entry,
                "quantity": decision.plan.quantity,
            }
        )
    scanner.update_settings(asdict(SignalSettings(leverage=100)))
    assert (
        not scanner.decisions
        and scanner.store.trades()[0].plan.leverage == trade.plan.leverage
    )


def test_read_only_client_refuses_every_exchange_post():
    client = BitunixClient("", "", read_only=True)
    client.session = MagicMock()
    with pytest.raises(ValueError, match="disabled"):
        client.set_leverage("BTCUSDT", 40)
    with pytest.raises(ValueError, match="disabled"):
        client.flash_close_position("123")
    client.session.post.assert_not_called()


def test_read_only_client_allows_only_protective_stop_posts():
    client = BitunixClient("key", "secret", read_only=True)
    response = MagicMock()
    response.json.return_value = {"code": 0, "data": {"orderId": "SL1"}}
    client.session = MagicMock()
    client.session.post.return_value = response
    client.place_qty_tpsl("BTCUSDT", "1", "97.5", "0.01")
    assert "/tpsl/place_order" in client.session.post.call_args.args[0]
    client.place_position_tpsl("BTCUSDT", "1", "97.5")
    assert "/tpsl/position/place_order" in client.session.post.call_args.args[0]
    with pytest.raises(ValueError, match="disabled"):
        client.place_order("BTCUSDT", "BUY", "1")
    with pytest.raises(ValueError, match="disabled"):
        client.flash_close_position("1")


def test_alert_mode_cannot_enter_legacy_tick_or_configure_account(
    tmp_path, monkeypatch
):
    from bitunix_bot.bot import BitunixBot

    monkeypatch.setenv("SIGNAL_STATE_PATH", str(tmp_path / "signals.db"))
    cfg = load("config.yaml", "/dev/null")
    cfg.mode = "live"
    cfg.trading.auto_execute_pump_fade_shorts = True
    assert not cfg.is_live
    bot = BitunixBot(cfg)
    bot.signal_scanner.refresh = MagicMock()
    bot._update_streak_state = MagicMock()
    bot._tick()
    bot.signal_scanner.refresh.assert_called_once()
    bot._update_streak_state.assert_not_called()


def test_routes_require_auth_validate_input_and_reject_legacy_actions(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("DASHBOARD_PASSWORD", "test_pass")
    scanner, _ = scanner_with_entry(tmp_path)
    cfg = load("config.yaml", "/dev/null")
    app = create_app(
        cfg, scanner.client, bot=type("Bot", (), {"signal_scanner": scanner})()
    )
    client = app.test_client()
    auth = {"Authorization": "Basic " + base64.b64encode(b"admin:test_pass").decode()}
    assert client.get("/api/signals").status_code == 401
    assert (
        client.post("/api/signals/settings", headers=auth, json=[]).status_code == 400
    )
    assert (
        client.post(
            "/api/signals/settings", headers=auth, json={"leverage": 200}
        ).status_code
        == 400
    )
    assert (
        client.post(
            "/api/admin/close-symbol", headers=auth, json={"symbol": "BTCUSDT"}
        ).status_code
        == 403
    )
    assert client.get("/api/signals", headers=auth).json["mode"] == "alerts_only"
    assert client.get("/api/momentum", headers=auth).json["strategy"] == "intraday"


def test_read_failure_does_not_serve_cached_market_as_fresh(tmp_path):
    scanner, _ = scanner_with_entry(tmp_path)
    fetch = MagicMock(return_value={"price": 100})
    with patch("time.time", return_value=NOW):
        scanner._read("quote", 1, fetch)
    fetch.side_effect = requests.Timeout("provider timeout")
    with patch("time.time", return_value=NOW + 2):
        with pytest.raises(requests.Timeout):
            scanner._read("quote", 1, fetch)
        with pytest.raises(ValueError, match="cooling down"):
            scanner._read("quote", 1, fetch)
    assert fetch.call_count == 2


def test_provider_volume_units_and_opening_gap_are_normalized():
    _, frames = market_frames()
    rows = candle_rows(frames["15m"])
    rows[-1].update(quoteVol=200, baseVol=19_740, open=99.0)
    bars = closed_candles(rows, "15m", NOW)
    assert bars[-1].volume == 200
    assert bars[-1].high == 99.0


def test_scanner_reads_current_public_schema_and_deduplicates_alerts(tmp_path):
    scanner, _decision = scanner_with_entry(tmp_path)
    _, market, frames = ready_decision()
    scanner.client.trading_pairs.return_value = [
        {
            "symbol": "BTCUSDT",
            "symbolStatus": "OPEN",
            "basePrecision": 3,
            "quotePrecision": 2,
            "minTradeVolume": 0.001,
        }
    ]
    scanner.client.tickers.return_value = [
        {"symbol": "BTCUSDT", "quoteVol": 100_000_000}
    ]
    scanner.client.klines.side_effect = lambda symbol, interval, limit: candle_rows(
        frames[interval]
    )
    scanner.client.funding_rate.return_value = {
        "lastPrice": market.price,
        "markPrice": market.mark,
        "fundingRate": 0,
        "fundingInterval": 8,
        "nextFundingTime": (NOW + 3600) * 1000,
    }
    scanner.client.depth.return_value = {
        "bids": [[market.bid, 10000]],
        "asks": [[market.ask, 10000]],
    }
    scanner.client.position_tiers.return_value = [
        {
            "startValue": 0,
            "endValue": 50000,
            "maintenanceMarginRate": 0.004,
            "leverage": 125,
        }
    ]
    with patch("time.time", return_value=NOW), patch("time.sleep"):
        scanner.refresh(force=True)
        scanner.refresh(force=True)
        snapshot = scanner.snapshot()
    assert snapshot["symbols"]["BTCUSDT"]["state"] == "ENTER_LONG"
    assert len(snapshot["history"]) == 1
    assert snapshot["history"][0]["time"] == NOW
    assert snapshot["queue"][0]["symbol"] == "BTCUSDT"
    assert snapshot["queue"][0]["state_since"] == NOW
    projection = snapshot["queue"][0]["projection"]
    assert projection["horizon_minutes"] == 5
    assert projection["low"] < projection["price"] < projection["high"]
    assert snapshot["symbols"]["BTCUSDT"]["projection"] == projection
    scanner.client.place_order.assert_not_called()
    scanner.client.pending_positions.assert_not_called()


def test_watch_alerts_are_timestamped_and_not_duplicated(tmp_path):
    scanner, decision = scanner_with_entry(tmp_path)
    decision.state = "WATCH_LONG"
    decision.signal_id = ""
    scanner._record_signal_alert(decision, NOW)
    scanner._record_signal_alert(decision, NOW + 10)
    history = scanner.store.history()
    assert len(history) == 1
    assert history[0]["state"] == "WATCH_LONG"
    assert history[0]["time"] == NOW
    assert history[0]["setup"] == "Trend pullback"
    assert history[0]["side"] == "long"


def test_queue_ranks_top_setups_and_warns_before_switch(tmp_path):
    scanner, first = scanner_with_entry(tmp_path)
    second, _, _ = ready_decision("short")
    second.symbol = "ETHUSDT"
    second.state = "WATCH_SHORT"
    first.as_of = second.as_of = NOW
    first.state_since = NOW - 30
    scanner.decisions = {first.symbol: first, second.symbol: second}
    with patch("time.time", return_value=NOW):
        snap = scanner.snapshot()
    assert [row["symbol"] for row in snap["queue"]] == ["BTCUSDT", "ETHUSDT"]
    assert snap["best_symbol"] == "BTCUSDT"
    assert snap["queue"][0]["state_since"] == NOW - 30
    first.state = "WAIT"
    second.state = "ENTER_SHORT"
    with patch("time.time", return_value=NOW + 1):
        held = scanner.snapshot()
    assert held["best_symbol"] == "BTCUSDT"
    assert held["handoff"]["to_symbol"] == "ETHUSDT"
    assert held["handoff"]["reason"] == "A higher-ranked setup is ready"
    assert held["handoff"]["seconds_remaining"] == 20
    with patch("time.time", return_value=NOW + 22):
        switched = scanner.snapshot()
    assert switched["best_symbol"] == "ETHUSDT"
    assert switched["handoff"] is None


def test_entry_expiry_warns_before_window_closes(tmp_path):
    scanner, decision = scanner_with_entry(tmp_path)
    other, _, _ = ready_decision("short")
    other.symbol = "ETHUSDT"
    other.state = "WATCH_SHORT"
    decision.as_of = other.as_of = NOW
    decision.plan.expires_at = NOW + 30
    scanner.decisions = {decision.symbol: decision, other.symbol: other}
    with patch("time.time", return_value=NOW):
        snap = scanner.snapshot()
    assert snap["handoff"]["reason"] == "Entry window ending"
    assert snap["handoff"]["to_symbol"] == "ETHUSDT"
    assert snap["handoff"]["seconds_remaining"] == 30
    assert snap["queue"][0]["seconds_remaining"] == 30


def test_trailing_stop_is_not_applied_to_an_earlier_wick():
    trade, decision, _ = new_trade()
    trade.current_stop = trade.plan.entry + 0.2
    trade.stop_updated_at = NOW + 1800
    trade.checked_at = NOW
    decision.price = trade.plan.entry + 0.5
    decision.as_of = NOW + 1900
    bar = Candle(
        NOW // 900 * 900 + 900,
        trade.plan.entry,
        trade.plan.entry + 0.6,
        trade.plan.entry - 0.1,
        trade.plan.entry + 0.5,
        100,
    )
    evaluate_exit(trade, decision, [bar], NOW + 1900, SignalsCfg())
    assert trade.state == "HOLD_LONG"


def test_evaluate_exit_stamps_the_check_time_separately_from_the_bar_cursor():
    """The card's "Checked" reads evaluated_at; checked_at stays a candle cursor."""
    from dataclasses import asdict

    from bitunix_bot.intraday import INTERVALS

    trade, decision, _ = new_trade()
    interval = INTERVALS[trade.plan.trigger_interval]
    at = NOW + 121
    decision.as_of = at
    evaluate_exit(trade, decision, [], at, SignalsCfg())
    assert trade.evaluated_at == at
    assert trade.checked_at == max(trade.opened_at, at // interval * interval)
    assert trade.checked_at != trade.evaluated_at
    assert asdict(trade)["evaluated_at"] == at
    evaluate_exit(trade, decision, [], at + 15, SignalsCfg())
    assert trade.evaluated_at == at + 15


def test_parse_open_position_accepts_bitunix_short_fields():
    parsed = parse_open_position(
        {
            "positionId": "HYPE1",
            "symbol": "HYPEUSDT",
            "qty": "36.59",
            "side": "SHORT",
            "avgOpenPrice": "80.37",
            "markPrice": "80.574",
            "unrealizedPNL": "-7.318",
            "leverage": "40",
            "ctime": NOW * 1000,
        }
    )
    assert parsed is not None
    assert parsed.side == "short"
    assert parsed.quantity == pytest.approx(36.59)
    assert parsed.entry == pytest.approx(80.37)
    assert parsed.unrealized_pnl == pytest.approx(-7.318)
    assert parsed.leverage == 40
    assert parsed.opened_at == NOW
    sell = parse_open_position(
        {
            "symbol": "HYPEUSDT",
            "size": "1",
            "positionSide": "SELL",
            "entryPrice": "10",
        }
    )
    assert sell is not None and sell.side == "short" and sell.position_id == "HYPEUSDT:short"
    assert (
        parse_open_position(
            {
                "symbol": "HYPEUSDT",
                "qty": 0,
                "side": "SHORT",
                "avgOpenPrice": 80,
            }
        )
        is None
    )


def _live_scanner(tmp_path, rows):
    scanner, decision = scanner_with_entry(tmp_path)
    scanner.client.api_key = "key"
    scanner.client.secret_key = "secret"
    scanner.client.pending_positions.return_value = rows
    return scanner, decision


def test_live_short_is_imported_and_never_sends_orders(tmp_path):
    row = {
        "positionId": "HYPE1",
        "symbol": "HYPEUSDT",
        "qty": "36.59",
        "side": "SHORT",
        "avgOpenPrice": "80.37",
        "markPrice": "80.574",
        "unrealizedPNL": "-7.318",
        "leverage": 40,
        "ctime": (NOW - 120) * 1000,
    }
    scanner, _ = _live_scanner(tmp_path, [row])
    parsed = parse_open_position(row)
    assert parsed is not None
    with patch("time.time", return_value=NOW):
        scanner._sync_exchange_positions([parsed], scanner.decisions, NOW, fetch_ok=True)
        snapshot = scanner.snapshot()
    trades = snapshot["trades"]
    assert len(trades) == 1
    trade = trades[0]
    assert trade["kind"] == "exchange"
    assert trade["symbol"] == "HYPEUSDT"
    assert trade["plan"]["side"] == "short"
    assert trade["plan"]["quantity"] == pytest.approx(36.59)
    assert trade["unrealized_pnl"] == pytest.approx(-7.318)
    assert trade["mark_price"] == pytest.approx(80.574)
    assert snapshot["positions"]["connected"] is True
    assert snapshot["positions"]["imported"] == 1
    assert trade["suggestion"] == "SET_STOP"
    assert trade["exchange_stop_confirmed"] is False
    scanner.client.place_order.assert_not_called()
    scanner.client.flash_close_position.assert_not_called()


def test_vanished_exchange_position_closes_tracking(tmp_path):
    row = {
        "positionId": "HYPE1",
        "symbol": "HYPEUSDT",
        "qty": "36.59",
        "side": "SHORT",
        "avgOpenPrice": "80.37",
        "markPrice": "80.57",
        "unrealizedPNL": "-7.3",
        "leverage": 40,
    }
    scanner, _ = _live_scanner(tmp_path, [row])
    parsed = parse_open_position(row)
    assert parsed is not None
    with patch("time.time", return_value=NOW):
        scanner._sync_exchange_positions([parsed], {}, NOW, fetch_ok=True)
        scanner._sync_exchange_positions([], {}, NOW + 15, fetch_ok=True)
    closed = scanner.store.trades()[0]
    assert closed.closed_at == NOW + 15
    assert closed.state == "CLOSED"
    assert "no longer open" in closed.reason


def test_failed_position_read_does_not_close_live_track(tmp_path):
    row = {
        "positionId": "HYPE1",
        "symbol": "HYPEUSDT",
        "qty": "36.59",
        "side": "SHORT",
        "avgOpenPrice": "80.37",
        "unrealizedPNL": "-7.3",
        "leverage": 40,
    }
    scanner, _ = _live_scanner(tmp_path, [row])
    parsed = parse_open_position(row)
    assert parsed is not None
    scanner._sync_exchange_positions([parsed], {}, NOW, fetch_ok=True)
    scanner.client.pending_positions.side_effect = RuntimeError("signed read failed")
    with patch("time.time", return_value=NOW + 30), patch("time.sleep"):
        positions, ok = scanner._load_positions()
        scanner._sync_exchange_positions(positions, {}, NOW + 30, fetch_ok=ok)
    assert ok is False
    assert scanner.store.trades(active_only=True)[0].kind == "exchange"
    assert "paused" in (scanner._positions_error or "")


def test_manual_track_is_enriched_instead_of_duplicated(tmp_path):
    scanner, decision = _live_scanner(tmp_path, [])
    with patch("time.time", return_value=NOW):
        recorded = scanner.track(
            {
                "signal_id": decision.signal_id,
                "kind": "manual",
                "entry": decision.plan.entry,
                "quantity": decision.plan.quantity,
            }
        )
        parsed = parse_open_position(
            {
                "positionId": "BTC1",
                "symbol": decision.symbol,
                "qty": str(decision.plan.quantity),
                "side": "LONG",
                "avgOpenPrice": str(decision.plan.entry),
                "markPrice": str(decision.price),
                "unrealizedPNL": "1.25",
                "leverage": 25,
            }
        )
        assert parsed is not None
        scanner._sync_exchange_positions([parsed], scanner.decisions, NOW, fetch_ok=True)
    trades = scanner.store.trades(active_only=True)
    assert len(trades) == 1
    assert trades[0].id == recorded.id
    assert trades[0].kind == "manual"
    assert trades[0].exchange_position_id == "BTC1"
    assert trades[0].unrealized_pnl == pytest.approx(1.25)


def test_imported_position_gates_same_symbol_entry(tmp_path):
    scanner, decision = _live_scanner(tmp_path, [])
    parsed = parse_open_position(
        {
            "positionId": "BTC1",
            "symbol": "BTCUSDT",
            "qty": "0.01",
            "side": "LONG",
            "avgOpenPrice": str(decision.plan.entry),
            "markPrice": str(decision.price),
            "unrealizedPNL": "0.1",
            "leverage": 25,
        }
    )
    assert parsed is not None
    with patch("time.time", return_value=NOW):
        scanner._sync_exchange_positions([parsed], scanner.decisions, NOW, fetch_ok=True)
        snapshot = scanner.snapshot()
    assert snapshot["symbols"]["BTCUSDT"]["state"] == "WATCH_LONG"
    assert any(
        check["label"] == "Tracked exposure" and not check["passed"]
        for check in snapshot["symbols"]["BTCUSDT"]["checks"]
    )


def test_snapshot_explains_missing_position_keys(tmp_path):
    scanner, _ = scanner_with_entry(tmp_path)
    with patch("time.time", return_value=NOW):
        snapshot = scanner.snapshot()
    assert snapshot["positions"]["connected"] is False
    assert snapshot["positions"]["imported"] == 0
    assert "API keys" in snapshot["positions"]["error"]
    scanner.client.pending_positions.assert_not_called()


def test_refresh_imports_open_position_outside_liquid_universe(tmp_path):
    scanner, _decision = scanner_with_entry(tmp_path)
    _, market, frames = ready_decision()
    scanner.client.api_key = "key"
    scanner.client.secret_key = "secret"
    scanner.client.pending_positions.return_value = [
        {
            "positionId": "HYPE1",
            "symbol": "HYPEUSDT",
            "qty": "36.59",
            "side": "SHORT",
            "avgOpenPrice": "80.37",
            "markPrice": "80.574",
            "unrealizedPNL": "-7.318",
            "leverage": 40,
        }
    ]
    scanner.client.trading_pairs.return_value = [
        {
            "symbol": "BTCUSDT",
            "symbolStatus": "OPEN",
            "basePrecision": 3,
            "quotePrecision": 2,
            "minTradeVolume": 0.001,
        }
    ]
    scanner.client.tickers.return_value = [
        {"symbol": "BTCUSDT", "quoteVol": 100_000_000}
    ]
    scanner.client.klines.side_effect = lambda symbol, interval, limit: candle_rows(
        frames[interval]
    )
    scanner.client.funding_rate.return_value = {
        "lastPrice": market.price,
        "markPrice": market.mark,
        "fundingRate": 0,
        "fundingInterval": 8,
        "nextFundingTime": (NOW + 3600) * 1000,
    }
    scanner.client.depth.return_value = {
        "bids": [[market.bid, 10000]],
        "asks": [[market.ask, 10000]],
    }
    scanner.client.position_tiers.return_value = [
        {
            "startValue": 0,
            "endValue": 50000,
            "maintenanceMarginRate": 0.004,
            "leverage": 125,
        }
    ]
    with patch("time.time", return_value=NOW), patch("time.sleep"):
        scanner.refresh(force=True)
        snapshot = scanner.snapshot()
    assert any(
        trade["symbol"] == "HYPEUSDT" and trade["kind"] == "exchange"
        for trade in snapshot["trades"]
    )
    scanner.client.pending_positions.assert_called()
    scanner.client.place_order.assert_not_called()
    scanner.client.flash_close_position.assert_not_called()


def test_checklist_length_is_stable_from_wait_to_entry():
    waiting, market, frames = ready_decision()
    assert [c.label for c in waiting.checks] == list(CHECKLIST_LABELS)
    mixed = evaluate_intraday(
        replace(market, symbol="ETHUSDT"),
        frames,
        None,
        SignalSettings(),
        SignalsCfg(),
        NOW,
    )
    assert [c.label for c in mixed.checks] == list(CHECKLIST_LABELS)
    assert mixed.state == "WATCH_LONG"
    assert len(waiting.checks) == len(mixed.checks) == 22
    assert [c.label for c in waiting.checks if c.group == "plan"] == [
        "Entry zone",
        "Stop size",
        "Target inside hold budget",
        "Reward after costs",
        "Funding drag",
        "Order size",
        "Execution depth",
        "Leverage ceiling",
    ]
    assert [c.label for c in waiting.checks if c.group == "market"] == [
        "Fresh market data",
        "Liquid market",
        "Spread",
        "Hold-window volatility",
        "Mark vs last",
        "Funding print window",
        "1h bias / 15m structure",
        "BTC context",
        "Not extended",
        "Blow-off guard",
        "Crowding headwind",
    ]


def test_waiting_checks_are_distinct_from_failed_checks():
    waiting = waiting_check("Entry zone", "Waiting for a completed 5m setup")
    failed = make_check(
        "Target inside hold budget",
        False,
        "No structural target that clears 1.5R inside the 120-minute travel budget (1.27%)",
    )
    assert waiting.waiting and not waiting.passed
    assert not failed.waiting and not failed.passed


def test_missing_target_still_scores_remaining_plan_gates():
    _decision, market, frames = ready_decision()
    with patch("bitunix_bot.intraday.select_targets", return_value=[]):
        blocked = evaluate_intraday(
            market, frames, None, SignalSettings(), SignalsCfg(), NOW
        )
    plan = [check for check in blocked.checks if check.group == "plan"]
    assert len(plan) == 8
    assert all(not check.waiting for check in plan)
    target = next(check for check in plan if check.label == "Target inside hold budget")
    assert not target.passed
    assert "1.5R inside the 120-minute travel budget" in target.detail
    assert any(check.label == "Funding drag" and not check.waiting for check in plan)
    assert any(check.label == "Leverage ceiling" and check.passed for check in plan)
    assert blocked.state == "WATCH_LONG"
    assert blocked.plan is None


def test_funding_print_window_blocks_entry():
    decision, market, frames = ready_decision()
    assert decision.state == "ENTER_LONG"
    for eta in (60, 300):
        blocked = evaluate_intraday(
            replace(market, next_funding=NOW + eta),
            frames,
            None,
            SignalSettings(),
            SignalsCfg(),
            NOW,
        )
        assert blocked.state == "WATCH_LONG"
        assert any(
            c.label == "Funding print window" and not c.passed for c in blocked.checks
        )
    allowed = evaluate_intraday(
        replace(market, next_funding=NOW + 301), frames, None, SignalSettings(), SignalsCfg(), NOW
    )
    assert allowed.state == "ENTER_LONG"


def test_mark_basis_blocks_entry():
    decision, market, frames = ready_decision()
    assert decision.state == "ENTER_LONG"
    for basis in (1.01, 1.002):
        blocked = evaluate_intraday(
            replace(market, mark=market.price * basis),
            frames,
            None,
            SignalSettings(),
            SignalsCfg(),
            NOW,
        )
        assert blocked.state == "WATCH_LONG"
        assert any(c.label == "Mark vs last" and not c.passed for c in blocked.checks)
    allowed = evaluate_intraday(
        replace(market, mark=market.price * 1.001), frames, None, SignalSettings(), SignalsCfg(), NOW
    )
    assert next(c for c in allowed.checks if c.label == "Mark vs last").passed


def test_portfolio_gate_does_not_change_checklist_length(tmp_path):
    scanner, decision = scanner_with_entry(tmp_path)
    before = len(decision.checks)
    with patch("time.time", return_value=NOW):
        scanner.track(
            {
                "signal_id": decision.signal_id,
                "kind": "paper",
                "entry": decision.plan.entry,
                "quantity": decision.plan.quantity,
            }
        )
        snap = scanner.snapshot()
    gated = snap["symbols"]["BTCUSDT"]
    assert gated["state"] == "WATCH_LONG"
    assert len(gated["checks"]) == before
    assert any(
        check["label"] == "Tracked exposure" and not check["passed"]
        for check in gated["checks"]
    )


def _pair_row(symbol):
    return {
        "symbol": symbol,
        "symbolStatus": "OPEN",
        "basePrecision": 3,
        "quotePrecision": 2,
        "minTradeVolume": 0.001,
    }


def _universe_scanner(tmp_path, names, **overrides):
    _, market, frames = ready_decision()
    values = {
        "max_symbols": 2,
        "universe_size": 6,
        "evaluate_batch": 2,
        "queue_size": 2,
    }
    values.update(overrides)
    cfg = SignalsCfg(**values)
    cfg.validate()
    scanner = SignalScanner(
        MagicMock(spec=BitunixClient),
        cfg,
        SignalStore(str(tmp_path / "signals.db")),
    )
    scanner.client.trading_pairs.return_value = [_pair_row(name) for name in names]
    scanner.client.tickers.return_value = [
        {"symbol": name, "quoteVol": 100_000_000 - index * 1_000_000}
        for index, name in enumerate(names)
    ]
    scanner.client.klines.side_effect = lambda symbol, interval, limit: candle_rows(
        frames[interval]
    )
    scanner.client.funding_rate.return_value = {
        "lastPrice": market.price,
        "markPrice": market.mark,
        "fundingRate": 0,
        "fundingInterval": 8,
        "nextFundingTime": (NOW + 3600) * 1000,
    }
    scanner.client.depth.return_value = {
        "bids": [[market.bid, 10000]],
        "asks": [[market.ask, 10000]],
    }
    scanner.client.position_tiers.return_value = [
        {
            "startValue": 0,
            "endValue": 50000,
            "maintenanceMarginRate": 0.004,
            "leverage": 125,
        }
    ]
    return scanner


UNIVERSE_NAMES = [
    "BTCUSDT",
    "ETHUSDT",
    "SOLUSDT",
    "XRPUSDT",
    "DOGEUSDT",
    "ADAUSDT",
]


def test_two_tier_scan_rotates_universe_and_retains_decisions(tmp_path):
    scanner = _universe_scanner(tmp_path, UNIVERSE_NAMES)
    with patch("time.time", return_value=NOW), patch("time.sleep"):
        scanner.refresh(force=True)
        first = scanner.snapshot()["scan"]
    assert first["universe"] == 6
    assert first["hot_symbols"][:2] == ["BTCUSDT", "ETHUSDT"]
    assert len(first["evaluated"]) == 4
    first_batch = [name for name in first["evaluated"] if name not in first["hot_symbols"]]
    assert first_batch == ["SOLUSDT", "XRPUSDT"]
    with patch("time.time", return_value=NOW + 15), patch("time.sleep"):
        scanner.refresh(force=True)
        second = scanner.snapshot()["scan"]
    second_batch = [
        name for name in second["evaluated"] if name not in second["hot_symbols"]
    ]
    assert second_batch == ["DOGEUSDT", "ADAUSDT"]
    assert set(scanner.decisions) == set(UNIVERSE_NAMES)
    with patch("time.time", return_value=NOW + 15):
        snapshot = scanner.snapshot()
    assert len(snapshot["queue"]) <= scanner.cfg.queue_size
    # Freshness feed: the scan clock and each coin's last checklist run.
    assert snapshot["scan"]["last_scan"] == NOW + 15
    assert snapshot["scan"]["next_scan"] == NOW + 15 + scanner.cfg.refresh_seconds
    for name, row in snapshot["symbols"].items():
        expected = NOW + 15 if name in second["evaluated"] else NOW
        assert row["evaluated_at"] == expected, name
    assert all("evaluated_at" in item for item in snapshot["queue"])


def test_universe_filters_pairs_below_planned_leverage_for_trend(tmp_path):
    scanner = _universe_scanner(tmp_path, UNIVERSE_NAMES)
    rows = scanner.client.trading_pairs.return_value
    rows[2]["maxLeverage"] = 25
    assert rows[2]["symbol"] == "SOLUSDT"
    assert scanner.store.settings().profile == "trend"
    assert scanner.store.settings().leverage == 50
    with patch("time.time", return_value=NOW), patch("time.sleep"):
        scanner.refresh(force=True)
    assert "SOLUSDT" not in scanner._universe
    assert scanner.snapshot()["scan"]["universe"] == 5
    scanner.update_settings(asdict(SignalSettings(leverage=20)))
    scanner._cache.pop("pairs", None)
    with patch("time.time", return_value=NOW + 15), patch("time.sleep"):
        scanner.refresh(force=True)
    assert "SOLUSDT" in scanner._universe


def test_watch_signal_is_promoted_to_hot_set(tmp_path):
    scanner = _universe_scanner(tmp_path, UNIVERSE_NAMES)
    with patch("time.time", return_value=NOW), patch("time.sleep"):
        scanner.refresh(force=True)
        scanner.refresh(force=True)
    for decision in scanner.decisions.values():
        decision.state = "WAIT"
        decision.side = ""
    scanner.decisions["ADAUSDT"].state = "WATCH_LONG"
    scanner.decisions["ADAUSDT"].side = "long"
    with patch("time.time", return_value=NOW + 30), patch("time.sleep"):
        scanner.refresh(force=True)
        hot = scanner.snapshot()["scan"]["hot_symbols"]
    assert "ADAUSDT" in hot


def test_scan_prunes_names_that_left_the_universe(tmp_path):
    scanner = _universe_scanner(tmp_path, UNIVERSE_NAMES)
    with patch("time.time", return_value=NOW), patch("time.sleep"):
        scanner.refresh(force=True)
        scanner.refresh(force=True)
    assert "ADAUSDT" in scanner.decisions
    scanner.decisions["ADAUSDT"].state = "WAIT"
    scanner.decisions["ADAUSDT"].side = ""
    kept = UNIVERSE_NAMES[:-1]
    scanner.client.trading_pairs.return_value = [_pair_row(name) for name in kept]
    scanner.client.tickers.return_value = [
        {"symbol": name, "quoteVol": 100_000_000 - index * 1_000_000}
        for index, name in enumerate(kept)
    ]
    scanner._cache.pop("pairs", None)
    scanner._cache.pop("tickers", None)
    with patch("time.time", return_value=NOW + 45), patch("time.sleep"):
        scanner.refresh(force=True)
    assert "ADAUSDT" not in scanner.decisions


def test_universe_config_bounds_and_shipped_yaml():
    with pytest.raises(ValueError, match="universe_size"):
        SignalsCfg(universe_size=5).validate()
    with pytest.raises(ValueError, match="universe_size"):
        SignalsCfg(max_symbols=20, universe_size=12).validate()
    with pytest.raises(ValueError, match="evaluate_batch"):
        SignalsCfg(evaluate_batch=0).validate()
    cfg = load("config.yaml", "/dev/null")
    assert cfg.signals.universe_size == 80
    assert cfg.signals.evaluate_batch == 10
    assert cfg.signals.max_symbols == 12
    assert cfg.signals.trend.trigger_interval == "5m"
    assert cfg.signals.trend.max_stop_pct == 0.60
    assert cfg.signals.trend.min_reward_risk == 1.5
    assert cfg.signals.trend.liquidation_buffer_pct == 0.25


def test_signals_cfg_matches_config_yaml():
    shipped = load("config.yaml", "/dev/null").signals
    defaults = SignalsCfg()
    for name in (f.name for f in fields(SignalsCfg)):
        if name == "enabled":
            continue
        assert getattr(shipped, name) == getattr(defaults, name), name
    assert asdict(shipped.trend) == asdict(TrendCfg())
    assert shipped.enabled is True and defaults.enabled is False
    with pytest.raises(TypeError):
        SignalsCfg(trend={"bogus": 1})
    with pytest.raises(TypeError):
        SignalsCfg(min_reward_risk=2.0)


@pytest.mark.parametrize(
    "values, match",
    [
        ({"stale_hold_fraction": 0.1}, "10 minutes"),
        ({"breakeven_at_r": 1.5}, "breakeven < trailing"),
        ({"trigger_interval": "1m"}, "trigger_interval"),
        ({"min_hourly_atr_pct_2h": 0.7}, "hourly ATR band"),
        ({"max_stop_pct": 0.15}, "stop band"),
        ({"late_hold_fraction": 0.3}, "hold fractions"),
        ({"late_hold_min_r": 1.6}, "late_hold_min_r"),
        ({"hope_exit_r": 1.0}, "hope_exit_r"),
        ({"entry_expiry_seconds": 30}, "entry_expiry_seconds"),
        ({"entry_expiry_seconds": 600.0}, "whole number"),
        ({"max_spread_pct": 0.2}, "max_spread_pct"),
        ({"entry_chase_risk_fraction": 1.5}, "entry_chase_risk_fraction"),
        ({"min_stop_atr": 0}, "finite positive"),
    ],
)
def test_trend_cfg_validation(values, match):
    with pytest.raises(ValueError, match=match):
        SignalsCfg(trend=TrendCfg(**values)).validate()
    SignalsCfg(trend={"trigger_interval": "3m"}).validate()


def test_legacy_swing_settings_rebase_to_trend():
    legacy = SignalSettings.from_dict(
        {"planning_equity": 1000, "risk_pct": 0.5, "leverage": 25, "hold_hours": 24, "profile": "swing"}
    )
    assert legacy == SignalSettings(1000, 0.5, 25, 2, "trend")
    # Leverage inside 20-100 is kept; only an out-of-band value rebases to 50x.
    kept = SignalSettings.from_dict(
        {"planning_equity": 2500, "risk_pct": 0.75, "leverage": 41, "hold_hours": 12, "profile": "swing"}
    )
    assert kept == SignalSettings(2500, 0.75, 41, 2, "trend")
    rebased = SignalSettings.from_dict(
        {"planning_equity": 2500, "risk_pct": 0.75, "leverage": 10, "hold_hours": 12, "profile": "swing"}
    )
    assert rebased == SignalSettings(2500, 0.75, 50, 2, "trend")
    profile_less = SignalSettings.from_dict(
        {"planning_equity": 1000, "risk_pct": 0.5, "leverage": 10, "hold_hours": 24}
    )
    assert profile_less == SignalSettings(1000, 0.5, 50, 2, "trend")
    assert SignalSettings.from_dict(
        {"planning_equity": 1000, "risk_pct": 0.5, "leverage": 50, "hold_hours": 2, "profile": "swing"}
    ) == SignalSettings(1000, 0.5, 50, 2, "trend")
    # Rows already saying "trend" are a client bug, not legacy data.
    with pytest.raises(ValueError, match="1 or 2 hours"):
        SignalSettings.from_dict(
            {"planning_equity": 1000, "risk_pct": 0.5, "leverage": 50, "hold_hours": 24, "profile": "trend"}
        )
    with pytest.raises(ValueError, match="from 20 to 100"):
        SignalSettings.from_dict(
            {"planning_equity": 1000, "risk_pct": 0.5, "leverage": 10, "hold_hours": 2, "profile": "trend"}
        )
    assert SignalSettings() == SignalSettings(1000.0, 0.5, 50, 2, "trend")


def test_fifty_x_liquidation_arithmetic():
    market, _ = market_frames()

    def fit(mmr, stop, leverage=50, cap=100):
        tiers = replace(market, tiers=[Tier(0, 80_000, mmr, 100)])
        return liquidation_fit(
            replace(tiers, price=100.0, mark=100.0),
            100.0,
            stop,
            943.4,
            0.18,
            0.17,
            leverage,
            0.25,
            "long",
            cap=cap,
        )

    ceiling, liquidation, tier = fit(0.005, 99.65)
    assert ceiling == 78 and tier is not None and tier.maintenance_rate == 0.005
    assert liquidation == pytest.approx(98.673, abs=1e-3)
    assert estimate_liquidation(100.0, 50, 0.18, 0.005, "long") == pytest.approx(98.673, abs=1e-3)
    assert estimate_liquidation(100.0, 50, 0.18, 0.005, "short") == pytest.approx(101.313, abs=1e-3)
    ceiling, liquidation, _ = fit(0.003, 99.80)
    assert ceiling == 100
    assert liquidation == pytest.approx(98.475, abs=1e-3)
    ceiling, liquidation, _ = fit(0.010, 99.40)
    assert ceiling == 49 and 50 > ceiling
    assert liquidation is not None and 99.40 - liquidation < 0.25
    # The profile cap bounds the search even when the pair allows more.
    assert fit(0.003, 99.80, cap=60)[0] == 60
    assert fit(0.005, 99.65, leverage=100)[1] is not None
    assert liquidation_fit(replace(market, tiers=[]), 100.0, 99.65, 943.4, 0.18, 0.17, 50, 0.25, "long") == (
        0,
        None,
        None,
    )


def test_liquidation_fit_is_shared():
    assert scalp_short.fit_plan_stop is intraday.fit_plan_stop
    assert scalp_short.leverage_stop_detail is intraday.leverage_stop_detail
    assert travel_budget(0.64, 120, 1.5) == pytest.approx(1.5 * 0.64 * math.sqrt(2))
    assert travel_budget(0.64, 60, 1.5) == pytest.approx(0.96)


def test_stop_cap_blocks_wide_stops():
    decision, _, _ = ready_decision(pullback_low=PRICE - 0.70)
    assert decision.state == "WATCH_LONG"
    stop = next(c for c in decision.checks if c.label == "Stop size")
    assert not stop.passed and "0.6%" in stop.detail and "50x limit" in stop.detail
    assert decision.metrics["hold_minutes"] == 120


def test_stop_floor_blocks_sub_noise_stops():
    decision, _, _ = ready_decision(pullback_low=PRICE - 0.11)
    assert decision.state == "WATCH_LONG"
    stop = next(c for c in decision.checks if c.label == "Stop size")
    assert not stop.passed and "(5m noise)" in stop.detail
    assert decision.plan is not None and decision.plan.stop_pct < 0.20


def test_entry_zone_is_capped_by_stop_distance():
    decision, _, frames = ready_decision(pullback_low=PRICE - 0.11)
    plan = decision.plan
    assert plan is not None
    anchor = frames["5m"][-1].close
    atr_value = decision.metrics["atr_pct"] * decision.price / 100
    assert plan.entry_high - anchor <= 0.25 * (plan.entry - plan.stop) + 1e-9
    assert plan.entry_high - anchor < 0.25 * atr_value
    assert anchor - plan.entry_low == pytest.approx(0.30 * atr_value)
    wide, _, _ = ready_decision()
    wide_atr = wide.metrics["atr_pct"] * wide.price / 100
    assert wide.plan.entry_high - anchor == pytest.approx(0.25 * wide_atr)
    assert 0.25 * wide_atr < 0.25 * (wide.plan.entry - wide.plan.stop)


def test_entry_expires_ten_minutes_after_trigger_close():
    decision, market, frames = ready_decision()
    assert decision.plan.expires_at == frames["5m"][-1].time + 900
    at_expiry = evaluate_intraday(
        replace(market, as_of=decision.plan.expires_at),
        frames,
        None,
        SignalSettings(),
        SignalsCfg(),
        decision.plan.expires_at,
    )
    assert at_expiry.state == "WATCH_LONG"
    assert at_expiry.plan is not None


def test_travel_budget_scales_with_hold():
    two, _, _ = ready_decision(settings=SignalSettings(hold_hours=2))
    one, _, _ = ready_decision(settings=SignalSettings(hold_hours=1))
    assert two.state == "ENTER_LONG", two.reasons
    assert one.state == "WATCH_LONG"
    assert one.plan is None and two.plan.target == TARGET_HIGH
    target = next(c for c in one.checks if c.label == "Target inside hold budget")
    assert not target.passed and "60-minute" in target.detail
    assert "120-minute" in next(c for c in two.checks if c.label == "Target inside hold budget").detail
    assert one.metrics["hold_minutes"] == 60 and two.metrics["hold_minutes"] == 120


def test_low_volatility_is_blocked_by_hold_window_gate():
    quiet = hourly_bars(drift=0.05, amp=0.1, span=0.12)
    two, _, _ = ready_decision(hourly=quiet)
    assert two.state != "ENTER_LONG"
    assert two.metrics["hourly_atr_pct"] < 0.35
    gate = next(c for c in two.checks if c.label == "Hold-window volatility")
    assert not gate.passed and "0.35-1.2%" in gate.detail and "120 min at 50x" in gate.detail
    one, _, _ = ready_decision(hourly=quiet, settings=SignalSettings(hold_hours=1))
    gate = next(c for c in one.checks if c.label == "Hold-window volatility")
    assert not gate.passed and "0.60-1.2%" in gate.detail
    # The 1h hold demands more volatility than the 2h hold.
    base, _, _ = ready_decision(settings=SignalSettings(hold_hours=1))
    assert next(c for c in base.checks if c.label == "Hold-window volatility").passed


def test_not_extended_and_crowding_gates():
    decision, market, frames = ready_decision()
    extension = decision.metrics["extension_atr"]
    atr_1h = decision.metrics["hourly_atr_pct"] * market.price / 100
    assert 0 < extension < 1.5

    def at(target_extension):
        price = market.price + (target_extension - extension) * atr_1h
        return replace(market, price=price, mark=price, bid=price - 0.005, ask=price + 0.005)

    def check(decision, label):
        return next(c for c in decision.checks if c.label == label)

    stretched = evaluate_intraday(at(2.5), frames, None, SignalSettings(), SignalsCfg(), NOW)
    assert not check(stretched, "Not extended").passed and stretched.state == "AVOID"
    assert stretched.metrics["extension_atr"] == pytest.approx(2.5)
    assert check(decision, "Not extended").passed
    crowded = evaluate_intraday(
        replace(market, funding_rate=0.0006), frames, None, SignalSettings(), SignalsCfg(), NOW
    )
    assert not check(crowded, "Crowding headwind").passed and crowded.state == "AVOID"
    assert crowded.avoid == check(crowded, "Crowding headwind").detail
    # Negative funding pays the long; it is never a headwind.
    paid = evaluate_intraday(
        replace(market, funding_rate=-0.0006), frames, None, SignalSettings(), SignalsCfg(), NOW
    )
    assert check(paid, "Crowding headwind").passed
    oi_crowded = evaluate_intraday(at(1.6), frames, None, SignalSettings(), SignalsCfg(), NOW, 4.0)
    assert not check(oi_crowded, "Crowding headwind").passed
    assert oi_crowded.metrics["oi_change_pct"] == 4.0
    oi_calm = evaluate_intraday(market, frames, None, SignalSettings(), SignalsCfg(), NOW, 4.0)
    assert check(oi_calm, "Crowding headwind").passed and oi_calm.state == "ENTER_LONG"
    warming = evaluate_intraday(market, frames, None, SignalSettings(), SignalsCfg(), NOW, None)
    assert check(warming, "Crowding headwind").passed and warming.state == "ENTER_LONG"


def test_session_activity_blocks_dead_hour():
    decision, market, frames = ready_decision()
    assert decision.metrics["session_volume_ratio"] >= 0.5
    dead = dict(frames)
    dead["15m"] = frames["15m"][:-4] + [replace(c, volume=0.0) for c in frames["15m"][-4:]]
    blocked = evaluate_intraday(market, dead, None, SignalSettings(), SignalsCfg(), NOW)
    liquid = next(c for c in blocked.checks if c.label == "Liquid market")
    assert not liquid.passed and "session" in liquid.detail
    assert blocked.state == "WATCH_LONG"
    assert blocked.metrics["session_volume_ratio"] == 0


def test_btc_context_not_opposed():
    _, market, frames = ready_decision()
    eth = replace(market, symbol="ETHUSDT")
    flat = [Candle(c.time, 50_000, 50_100, 49_900, 50_000, 1000.0) for c in frames["1h"]]
    assert ema_bias(flat) == "mixed"
    aligned = evaluate_intraday(eth, frames, flat, SignalSettings(), SignalsCfg(), NOW)
    assert aligned.state == "ENTER_LONG", aligned.reasons
    assert aligned.metrics["btc_trend"] == "mixed"
    assert aligned.metrics["relative_strength_pct"] > 0
    btc_outran = flat[:-1] + [replace(flat[-1], close=50_500, high=50_600)]
    lagging = evaluate_intraday(eth, frames, btc_outran, SignalSettings(), SignalsCfg(), NOW)
    context = next(c for c in lagging.checks if c.label == "BTC context")
    assert not context.passed and lagging.metrics["relative_strength_pct"] < 0
    opposed_btc = [
        Candle(c.time, 60_000 - i * 40, 60_010 - i * 40, 59_900 - i * 40, 60_000 - i * 40, 1000.0)
        for i, c in enumerate(frames["1h"])
    ]
    assert ema_bias(opposed_btc) == "short"
    opposed = evaluate_intraday(eth, frames, opposed_btc, SignalSettings(), SignalsCfg(), NOW)
    assert not next(c for c in opposed.checks if c.label == "BTC context").passed
    missing = evaluate_intraday(eth, frames, None, SignalSettings(), SignalsCfg(), NOW)
    context = next(c for c in missing.checks if c.label == "BTC context")
    assert not context.passed and context.detail == "BTC candles unavailable"
    assert missing.state == "WATCH_LONG"


def test_too_volatile_market_is_avoided_not_watched():
    decision, market, frames = ready_decision()
    assert decision.state == "ENTER_LONG"
    cfg = SignalsCfg()
    cfg.trend.max_hourly_atr_pct = 0.1
    avoided = evaluate_intraday(market, frames, None, SignalSettings(), cfg, NOW)
    assert avoided.state == "AVOID" and avoided.side == "long"
    assert avoided.actions == []
    assert "above the 0.1% ceiling" in avoided.avoid
    assert "of margin at 50x" in avoided.avoid
    assert avoided.reasons[0].startswith("Do not long at 50x:")
    # The plan (and its stop) is still computed for the card; the verdict wins.
    assert avoided.plan is not None


def test_blow_off_guard_avoids_a_pair_that_already_ran():
    _, market, frames = ready_decision()
    cfg = SignalsCfg()
    cfg.trend.max_gain_4h_pct = 0.01
    avoided = evaluate_intraday(market, frames, None, SignalSettings(), cfg, NOW)
    assert avoided.state == "AVOID"
    guard = next(c for c in avoided.checks if c.label == "Blow-off guard")
    assert not guard.passed and "over 4h in the trade direction" in guard.detail
    assert avoided.avoid == guard.detail
    assert avoided.metrics["gain_4h_pct"] > 0.01


def test_avoided_rows_rank_last_and_queue_rows_carry_the_stop(tmp_path):
    scanner, first = scanner_with_entry(tmp_path)
    second, _, _ = ready_decision("short")
    second.symbol = "ETHUSDT"
    second.state = "AVOID"
    second.avoid = "1h ATR 5.91% is above the 1.2% ceiling"
    second.actions = []
    third, _, _ = ready_decision()
    third.symbol = "SOLUSDT"
    third.state = "WAIT"
    third.plan = None
    for d in (first, second, third):
        d.as_of = NOW
    scanner.decisions = {d.symbol: d for d in (first, second, third)}
    with patch("time.time", return_value=NOW):
        snap = scanner.snapshot()
    assert [row["symbol"] for row in snap["queue"]] == ["BTCUSDT", "SOLUSDT", "ETHUSDT"]
    btc, sol, eth = snap["queue"]
    assert btc["stop"] == first.plan.stop
    assert btc["margin_loss_pct"] == first.plan.margin_loss_pct
    assert btc["liquidation_estimate"] == first.plan.liquidation_estimate
    assert btc["leverage"] == first.plan.leverage and btc["avoid"] == ""
    assert sol["stop"] is None
    assert eth["avoid"].startswith("1h ATR") and eth["state"] == "AVOID"


def test_watch_actions_name_trigger_interval_and_pullback_level():
    market, frames = market_frames()
    frames["5m"] = frames["5m"][:-1] + [
        replace(frames["5m"][-1], close=frames["5m"][-1].open - 0.01, high=frames["5m"][-1].open)
    ]
    decision = evaluate_intraday(market, frames, None, SignalSettings(), SignalsCfg(), NOW)
    assert decision.state == "WATCH_LONG" and decision.plan is None
    assert decision.actions[0]["label"] == "Needs a 5m close above"
    assert decision.actions[0]["price"] >= frames["5m"][-1].close
    assert decision.actions[1]["label"] == "or a pullback to"
    assert decision.actions[1]["price"] < frames["5m"][-1].close
    trigger = next(c for c in decision.checks if c.label == "Completed trigger candle")
    assert not trigger.passed and "completed 5m pullback reclaim" in trigger.detail
    assert all(
        c.waiting and c.detail == "Waiting for a completed 5m setup"
        for c in decision.checks
        if c.group == "plan"
    )


def projection_bars(step: float, count: int = 40, interval: int = 300) -> list[Candle]:
    start = NOW // interval * interval - count * interval
    return [
        Candle(start + k * interval, 100 + step * k, 100 + step * k + 0.3,
               100 + step * k - 0.3, 100 + step * k, 100)
        for k in range(count)
    ]


def test_next_price_projection_extrapolates_drift_inside_an_atr_band():
    from bitunix_bot.intraday import next_price_projection

    rising = next_price_projection(projection_bars(0.1), "5m", 104.0)
    assert rising["horizon_minutes"] == 5
    assert rising["price"] == pytest.approx(104.1, abs=0.02)
    assert rising["low"] < 104.0 < rising["price"] < rising["high"]
    assert rising["drift_pct"] == pytest.approx(0.1 / 104 * 100, abs=0.02)
    assert rising["high"] - rising["price"] == pytest.approx(rising["price"] - rising["low"])

    flat = next_price_projection(projection_bars(0.0), "5m", 100.0)
    assert flat["price"] == pytest.approx(100.0)
    assert flat["drift_pct"] == pytest.approx(0, abs=1e-9)
    assert flat["high"] - flat["low"] == pytest.approx(2 * 0.6)

    # A burst that outruns the smoothed ATR is clamped to the envelope; a 1m
    # trigger looks 5 bars ahead.
    start = NOW // 300 * 300 - 40 * 300
    burst = [Candle(start + k * 300, 100, 100.3, 99.7, 100, 100) for k in range(28)] + [
        Candle(start + (28 + k) * 300, 100 + 5 * k, 100.3 + 5 * k, 99.7 + 5 * k, 100 + 5 * k, 100)
        for k in range(12)
    ]
    runaway = next_price_projection(burst, "5m", 155.0)
    assert 0 < runaway["price"] - 155.0 < 5.0
    assert runaway["price"] - 155.0 == pytest.approx(runaway["high"] - runaway["price"])
    scalp = next_price_projection(projection_bars(0.02, count=90, interval=60), "1m", 101.8)
    assert scalp["horizon_minutes"] == 5
    assert scalp["price"] == pytest.approx(101.9, abs=0.02)
    assert scalp["high"] - scalp["price"] == pytest.approx(0.6 * 5 ** 0.5, abs=0.05)

    # Anchored to the live price, falling back to the last close.
    assert next_price_projection(projection_bars(0.0), "5m", None)["price"] == pytest.approx(100.0)
    assert next_price_projection(projection_bars(0.1)[:10], "5m", 100.0) is None
    assert next_price_projection(projection_bars(0.1), "2h", 100.0) is None
