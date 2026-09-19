"""Scalp-short profile: candidate gates, failed-high trigger, liquidation fit, exits, forward test."""

from __future__ import annotations

from dataclasses import replace
from unittest.mock import MagicMock, patch

import pytest

from bitunix_bot.client import BitunixClient
from bitunix_bot.config import load
from bitunix_bot.forward_test import (
    ForwardTest,
    forward_test_from_decision,
    summarize_forward_tests,
    update_forward_test,
)
from bitunix_bot.intraday import Candle, Market, Tier
from bitunix_bot.scalp_short import (
    SCALP_CHECK_LABELS,
    evaluate_scalp_short,
    find_failed_high,
    volume_delta,
)
from bitunix_bot.signal_config import ScalpCfg, SignalsCfg, SignalSettings
from bitunix_bot.signal_scanner import SignalScanner
from bitunix_bot.signal_store import SignalStore, TrackedTrade, evaluate_exit

NOW = 1_800_000_060
SETTINGS = SignalSettings(1000.0, 0.5, 50, 2, "scalp_short")


def bar(time, open_, high, low, close, volume=100.0):
    return Candle(time, open_, high, low, close, volume)


def pump_frames():
    """A coin that ran +4% into a climactic spike and printed a failed high."""
    frames = {}
    # 1h: flat at 96 then a 4-bar acceleration to 100. ATR ≈ 1.2.
    hourly = []
    for i in range(100):
        t = NOW // 3600 * 3600 - (100 - i) * 3600
        base = 96.0 if i < 96 else 96.0 + (i - 95) * 1.0
        hourly.append(bar(t, base - 0.6, base + 0.6, base - 0.6, base))
    frames["1h"] = hourly
    four = []
    for i in range(100):
        t = NOW // 14400 * 14400 - (100 - i) * 14400
        four.append(bar(t, 96.0, 97.0, 95.0, 96.0))
    frames["4h"] = four
    quarter = []
    for i in range(200):
        t = NOW // 900 * 900 - (200 - i) * 900
        if i < 184:
            close = 96.0
        else:
            close = 96.0 + (i - 183) * 0.25
        quarter.append(bar(t, close - 0.1, close + 0.2, close - 0.2, close))
    frames["15m"] = quarter
    bars = []
    for i in range(200):
        t = NOW // 60 * 60 - (200 - i) * 60
        if i < 170:
            bars.append(bar(t, 98.5, 98.55, 98.45, 98.5, 100))
        elif i < 189:
            close = 98.5 + (i - 169) * 0.09
            bars.append(bar(t, close - 0.09, close + 0.02, close - 0.1, close, 150 + (i - 169) * 10))
        elif i == 189:
            # Climax bar: highest high, 5x baseline volume, closes off the high.
            bars.append(bar(t, 100.1, 100.3, 100.05, 100.25, 600))
        else:
            # Rejection: lower highs, closes near the lows, never a new high.
            close = 100.2 - (i - 189) * 0.02
            bars.append(bar(t, close + 0.03, close + 0.06, close - 0.01, close, 120))
    frames["1m"] = bars
    price = bars[-1].close
    market = Market(
        "PUMPUSDT",
        price,
        price,
        price - 0.01,
        price + 0.01,
        5_000_000,
        5_000_000,
        50_000_000,
        0.0003,
        8,
        NOW + 3600,
        [Tier(0, 50_000, 0.004, 125)],
        0.01,
        0.01,
        NOW,
    )
    btc = [
        bar(NOW // 3600 * 3600 - (100 - i) * 3600, 50_000, 50_100, 49_900, 50_000)
        for i in range(100)
    ]
    return market, frames, btc


def cfg():
    return SignalsCfg()


def evaluate(market, frames, btc, settings=SETTINGS, oi=None, now=NOW):
    return evaluate_scalp_short(market, frames, btc, settings, cfg(), now, oi)


def test_failed_high_after_climax_enters_short_with_stop_above_spike():
    market, frames, btc = pump_frames()
    decision = evaluate(market, frames, btc)
    failed = [c for c in decision.checks if not c.passed]
    assert decision.state == "ENTER_SHORT", [f"{c.label}: {c.detail}" for c in failed]
    plan = decision.plan
    assert plan.side == "short" and plan.profile == "scalp_short"
    assert plan.trigger_interval == "1m" and plan.hold_hours == 2
    assert plan.stop > 100.3 > plan.entry > plan.target
    assert plan.stop_pct <= cfg().scalp.max_stop_pct
    assert plan.net_reward_risk >= cfg().scalp.min_reward_risk
    assert plan.leverage == 50 and plan.max_leverage >= 50
    assert plan.liquidation_estimate > plan.stop
    assert plan.expires_at == frames["1m"][-1].time + 60 + cfg().scalp.entry_expiry_seconds
    assert [c.label for c in decision.checks] == list(SCALP_CHECK_LABELS)
    assert decision.metrics["gain_4h_pct"] >= 3
    assert decision.metrics["climax_volume"] >= 3


def test_hundred_x_is_blocked_when_stop_sits_outside_liquidation():
    market, frames, btc = pump_frames()
    decision = evaluate(market, frames, btc, replace(SETTINGS, leverage=100))
    assert decision.state == "WATCH_SHORT"
    check = next(c for c in decision.checks if c.label == "Stop inside liquidation")
    assert not check.passed
    assert "would liquidate before the stop" in check.detail
    assert decision.plan is not None and decision.plan.max_leverage < 100


def test_leverage_tier_gate_blocks_pairs_below_the_planned_cap():
    market, frames, btc = pump_frames()
    capped = replace(market, tiers=[Tier(0, 50_000, 0.004, 25)])
    decision = evaluate(capped, frames, btc)
    check = next(c for c in decision.checks if c.label == "Leverage tier")
    assert not check.passed and "25x" in check.detail
    assert decision.state != "ENTER_SHORT"


def test_candidate_gates_wait_until_extended_climactic_and_crowded():
    market, frames, btc = pump_frames()
    calm = dict(frames)
    calm["1m"] = [replace(c, volume=100) for c in frames["1m"]]
    decision = evaluate(market, calm, btc)
    assert decision.state == "WAIT" and decision.side == ""
    labels = {c.label: c for c in decision.checks}
    assert not labels["Climax volume"].passed
    assert labels["Failed high trigger"].waiting
    assert labels["Entry zone"].waiting
    assert len(decision.checks) == len(SCALP_CHECK_LABELS)


def test_crowded_longs_accepts_open_interest_build_when_funding_is_flat():
    market, frames, btc = pump_frames()
    flat = replace(market, funding_rate=0.0)
    blocked = evaluate(flat, frames, btc)
    assert not next(c for c in blocked.checks if c.label == "Crowded longs").passed
    allowed = evaluate(flat, frames, btc, oi=2.5)
    assert next(c for c in allowed.checks if c.label == "Crowded longs").passed
    assert allowed.state == "ENTER_SHORT"


def test_new_high_or_close_inside_spike_body_is_not_a_failed_high():
    market, frames, btc = pump_frames()
    bars = frames["1m"]
    atr_value = 0.1
    assert find_failed_high(bars, atr_value, cfg().scalp) is not None
    new_high = bars[:-1] + [replace(bars[-1], high=100.4)]
    assert find_failed_high(new_high, atr_value, cfg().scalp) is None
    inside = bars[:-1] + [replace(bars[-1], open=100.15, close=100.12)]
    assert find_failed_high(inside, atr_value, cfg().scalp) is None
    stale_spike = bars[13:] + [replace(bars[-1], time=bars[-1].time + 60 * k) for k in range(1, 14)]
    assert find_failed_high(stale_spike, atr_value, cfg().scalp) is None


def test_flow_divergence_requires_net_selling_since_the_spike():
    market, frames, btc = pump_frames()
    bars = list(frames["1m"])
    for i in range(190, 199):
        c = bars[i]
        bars[i] = replace(c, close=c.high, open=c.low)
    frames = dict(frames, **{"1m": bars})
    decision = evaluate(market, frames, btc)
    check = next(c for c in decision.checks if c.label == "Flow divergence")
    assert not check.passed and decision.state == "WATCH_SHORT"
    assert volume_delta(Candle(0, 1, 2, 1, 2, 10)) == 10
    assert volume_delta(Candle(0, 2, 2, 1, 1, 10)) == -10


def test_alt_must_have_outrun_btc():
    market, frames, btc = pump_frames()
    pumped_btc = btc[:-1] + [replace(btc[-1], close=60_000)]
    decision = evaluate(market, frames, pumped_btc)
    assert not next(c for c in decision.checks if c.label == "BTC context").passed


def test_scalp_execution_gates_are_tighter():
    market, frames, btc = pump_frames()
    wide = replace(market, ask=market.price * 1.0005)
    decision = evaluate(wide, frames, btc)
    assert not next(c for c in decision.checks if c.label == "Spread").passed
    thin = replace(market, bid_depth_usdt=100, ask_depth_usdt=100)
    decision = evaluate(thin, frames, btc)
    assert not next(c for c in decision.checks if c.label == "Execution depth").passed


def test_target_must_fit_the_travel_budget():
    market, frames, btc = pump_frames()
    quiet_hourly = [replace(c, high=c.close + 0.05, low=c.close - 0.05) for c in frames["1h"]]
    decision = evaluate(market, dict(frames, **{"1h": quiet_hourly}), btc)
    labels = {c.label: c for c in decision.checks}
    assert not labels["Mean-reversion target"].passed
    assert decision.plan is None


@pytest.mark.parametrize(
    "values",
    [
        {"planning_equity": 1000, "risk_pct": 0.5, "leverage": 100, "hold_hours": 2},
        {"planning_equity": 1000, "risk_pct": 0.5, "leverage": 100, "hold_hours": 24, "profile": "scalp_short"},
        {"planning_equity": 1000, "risk_pct": 0.5, "leverage": 126, "hold_hours": 2, "profile": "scalp_short"},
        {"planning_equity": 1000, "risk_pct": 0.5, "leverage": 50, "hold_hours": 2, "profile": "nope"},
    ],
)
def test_profile_settings_validation(values):
    with pytest.raises(ValueError):
        SignalSettings.from_dict(values)


def test_legacy_settings_default_to_swing_and_scalp_allows_100x():
    legacy = SignalSettings.from_dict(
        {"planning_equity": 1000, "risk_pct": 0.5, "leverage": 25, "hold_hours": 24}
    )
    assert legacy.profile == "swing"
    scalp = SignalSettings.from_dict(
        {"planning_equity": 1000, "risk_pct": 0.5, "leverage": 100, "hold_hours": 1, "profile": "scalp_short"}
    )
    assert scalp.leverage == 100 and scalp.hold_hours == 1


def test_scalp_config_validation_and_shipped_yaml():
    with pytest.raises(ValueError, match="trigger_interval"):
        SignalsCfg(scalp={"trigger_interval": "2m"}).validate()
    with pytest.raises(ValueError, match="spike ages"):
        SignalsCfg(scalp=ScalpCfg(spike_min_age_bars=12, spike_max_age_bars=3)).validate()
    loaded = load("config.yaml", "/dev/null")
    assert loaded.signals.scalp.trigger_interval == "1m"
    assert loaded.signals.scalp.max_spread_pct <= 0.05
    assert loaded.signals.scalp.stale_minutes <= 20


def scalp_trade():
    market, frames, btc = pump_frames()
    decision = evaluate(market, frames, btc)
    assert decision.state == "ENTER_SHORT"
    plan = decision.plan
    return decision, TrackedTrade(
        decision.signal_id,
        market.symbol,
        "paper",
        NOW,
        plan,
        plan.stop,
        plan.entry,
    )


def test_scalp_exit_ignores_1h_trend_and_uses_minute_stale_window():
    decision, trade = scalp_trade()
    decision.metrics["trend_1h"] = "long"
    decision.metrics["trend_4h"] = "long"
    bars = decision_frames_after(decision, trade, drift=0.0, minutes=5)
    later = NOW + 5 * 60
    decision.as_of = later
    evaluate_exit(trade, decision, bars, later, cfg())
    assert trade.state == "HOLD_SHORT"
    labels = {c.label: c for c in trade.checks}
    assert labels["1h structure"].passed and labels["4h bias"].passed
    assert "20m" in labels["Progress vs review window"].detail
    stale_at = NOW + cfg().scalp.stale_minutes * 60 + 60
    decision.as_of = stale_at
    bars = decision_frames_after(decision, trade, drift=0.0, minutes=cfg().scalp.stale_minutes + 1)
    evaluate_exit(trade, decision, bars, stale_at, cfg())
    assert trade.state == "EXIT_SHORT"
    assert "failed to progress" in trade.reason


def test_scalp_hard_time_exit_at_hold_cap():
    decision, trade = scalp_trade()
    end = NOW + 2 * 3600
    decision.as_of = end
    evaluate_exit(trade, decision, [], end, cfg())
    assert trade.state == "EXIT_SHORT" and "Maximum holding time" in trade.reason


def decision_frames_after(decision, trade, drift, minutes):
    entry = trade.plan.entry
    return [
        Candle(NOW // 60 * 60 + k * 60, entry, entry + 0.01, entry - 0.01 + drift * k, entry + drift * k, 100)
        for k in range(minutes)
    ]


def test_forward_test_records_excursion_and_first_touch():
    decision, trade = scalp_trade()
    test = forward_test_from_decision(decision, NOW)
    assert test is not None and test.profile == "scalp_short" and test.interval == "1m"
    entry, stop, target = test.entry, test.stop, test.target
    risk = stop - entry
    bars = [
        Candle(NOW // 60 * 60 + 60, entry, entry + risk * 0.4, entry - risk * 0.5, entry - risk * 0.3, 100),
        Candle(NOW // 60 * 60 + 120, entry, entry + risk * 0.2, target - 0.01, target + 0.1, 100),
    ]
    update_forward_test(test, bars, NOW + 180)
    assert test.outcome == "target"
    assert test.exit_r == pytest.approx((entry - target) / risk)
    assert test.mae_r == pytest.approx(-0.4)
    assert test.mfe_r >= (entry - target) / risk
    stopped = forward_test_from_decision(decision, NOW)
    update_forward_test(
        stopped,
        [Candle(NOW // 60 * 60 + 60, entry, stop + 0.01, target - 1, target - 1, 100)],
        NOW + 120,
    )
    assert stopped.outcome == "stop" and stopped.exit_r == -1.0
    liquidated = forward_test_from_decision(decision, NOW)
    update_forward_test(
        liquidated,
        [Candle(NOW // 60 * 60 + 60, entry, liquidated.liquidation + 1, entry, entry, 100)],
        NOW + 120,
    )
    assert liquidated.outcome == "liquidation"
    expired = forward_test_from_decision(decision, NOW)
    update_forward_test(expired, [Candle(NOW // 60 * 60 + 60, entry, entry, entry, entry - risk * 0.1, 100)], NOW + 2 * 3600)
    assert expired.outcome == "expired" and expired.exit_r == pytest.approx(0.1)
    summary = summarize_forward_tests([test, stopped, liquidated, expired])
    assert summary["resolved"] == 4 and summary["outcomes"]["target"] == 1
    assert summary["liquidation_touches"] == 1 and summary["hit_rate"] == 0.25


def _pair_row(symbol, max_leverage=125):
    return {
        "symbol": symbol,
        "symbolStatus": "OPEN",
        "basePrecision": 3,
        "quotePrecision": 2,
        "minTradeVolume": 0.001,
        "maxLeverage": max_leverage,
    }


def _scalp_scanner(tmp_path, pairs):
    market, frames, btc = pump_frames()
    scanner = SignalScanner(
        MagicMock(spec=BitunixClient),
        SignalsCfg(max_symbols=3, universe_size=6, evaluate_batch=2, queue_size=2),
        SignalStore(str(tmp_path / "signals.db")),
    )
    scanner.store.save_settings(SETTINGS)
    scanner.client.trading_pairs.return_value = [_pair_row(name, lev) for name, lev in pairs]
    scanner.client.tickers.return_value = [
        {"symbol": name, "quoteVol": 100_000_000 - i * 1_000_000, "openInterest": 1_000_000 + i}
        for i, (name, _) in enumerate(pairs)
    ]

    def klines(symbol, interval, limit):
        source = btc if symbol == "BTCUSDT" and interval == "1h" else frames[interval]
        return [
            {"time": c.time * 1000, "open": c.open, "high": c.high, "low": c.low, "close": c.close, "baseVol": c.volume}
            for c in source
        ]

    scanner.client.klines.side_effect = klines
    scanner.client.funding_rate.return_value = {
        "lastPrice": market.price,
        "markPrice": market.mark,
        "fundingRate": market.funding_rate,
        "fundingInterval": 8,
        "nextFundingTime": (NOW + 3600) * 1000,
    }
    scanner.client.depth.return_value = {
        "bids": [[market.bid, 100_000]],
        "asks": [[market.ask, 100_000]],
    }
    scanner.client.position_tiers.return_value = [
        {"startValue": 0, "endValue": 50000, "maintenanceMarginRate": 0.004, "leverage": 125}
    ]
    return scanner


def test_scanner_filters_universe_by_tier_and_records_forward_tests(tmp_path):
    scanner = _scalp_scanner(
        tmp_path,
        [("BTCUSDT", 125), ("PUMPUSDT", 125), ("LOWUSDT", 25), ("ETHUSDT", 100)],
    )
    with patch("time.time", return_value=NOW), patch("time.sleep"):
        scanner.refresh(force=True)
        snapshot = scanner.snapshot()
    assert "LOWUSDT" not in scanner._universe
    assert snapshot["profile"] == "scalp_short"
    assert snapshot["settings"]["profile"] == "scalp_short"
    row = snapshot["symbols"]["PUMPUSDT"]
    assert row["state"] == "ENTER_SHORT", row["reasons"]
    assert row["metrics"]["trigger_interval"] == "1m"
    assert [c["label"] for c in row["checks"]] == list(SCALP_CHECK_LABELS)
    assert "1m" in scanner.frames["PUMPUSDT"]
    tests = scanner.store.forward_tests()
    assert "PUMPUSDT" in {t.symbol for t in tests}
    assert all(t.outcome == "open" and t.profile == "scalp_short" for t in tests)
    assert snapshot["forward_test"]["count"] >= 1
    assert snapshot["forward_test"]["outcomes"]["open"] >= 1
    # OI history: a second sample an hour later yields a delta.
    scanner.client.tickers.return_value = [
        {"symbol": name, "quoteVol": 100_000_000, "openInterest": 1_050_000}
        for name in ("BTCUSDT", "PUMPUSDT", "LOWUSDT", "ETHUSDT")
    ]
    scanner._cache.pop("tickers", None)
    with patch("time.time", return_value=NOW + 3600), patch("time.sleep"):
        scanner._record_open_interest(
            {row["symbol"]: row for row in scanner.client.tickers.return_value}, NOW + 3600
        )
    assert scanner._oi_change_pct("PUMPUSDT", NOW + 3600) == pytest.approx(5.0, rel=1e-3)


def test_switching_profile_clears_frames_and_uses_swing_checklist(tmp_path):
    scanner = _scalp_scanner(tmp_path, [("BTCUSDT", 125), ("PUMPUSDT", 125)])
    with patch("time.time", return_value=NOW), patch("time.sleep"):
        scanner.refresh(force=True)
    assert scanner.frames
    scanner.update_settings(
        {"planning_equity": 1000, "risk_pct": 0.5, "leverage": 25, "hold_hours": 24, "profile": "swing"}
    )
    assert scanner.frames == {} and scanner.decisions == {}
    assert scanner.store.settings().profile == "swing"
