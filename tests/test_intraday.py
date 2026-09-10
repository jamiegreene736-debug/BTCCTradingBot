"""Regression coverage for entry validity, risk accounting, lifecycle and read-only execution."""

from __future__ import annotations

import base64
import math
from dataclasses import asdict, replace
from unittest.mock import MagicMock, patch

import pytest
import requests

from bitunix_bot.client import BitunixClient
from bitunix_bot.config import load
from bitunix_bot.dashboard import create_app
from bitunix_bot.intraday import (
    Candle,
    Market,
    Tier,
    closed_candles,
    ema_bias,
    evaluate_intraday,
    find_setup,
    funding_cost,
    select_targets,
    trend,
)
from bitunix_bot.signal_config import SignalsCfg, SignalSettings
from bitunix_bot.signal_scanner import SignalScanner
from bitunix_bot.signal_store import SignalStore, TrackedTrade, evaluate_exit

NOW = 1_800_000_060


def market_frames(side="long"):
    frames = {}
    for interval, seconds in (("1h", 3600), ("4h", 14400)):
        values = [80 + i * 0.2 + math.sin(i * math.pi / 6) * 0.8 for i in range(100)]
        frames[interval] = [
            Candle(
                NOW // seconds * seconds - (100 - i) * seconds,
                c - 0.1,
                c + 0.3,
                c - 0.3,
                c,
                100,
            )
            for i, c in enumerate(values)
        ]
        frames[interval][6] = replace(frames[interval][6], high=110)
    bars = [
        Candle(NOW // 900 * 900 - (200 - i) * 900, 98.3, 98.7, 98.1, 98.4, 100)
        for i in range(200)
    ]
    bars[-4:] = [
        replace(bars[-4], open=98.4, high=98.8, low=98.3, close=98.6),
        replace(bars[-3], open=98.6, high=98.7, low=98.1, close=98.3),
        replace(bars[-2], open=98.3, high=98.4, low=97.95, close=98.2),
        replace(bars[-1], open=98.2, high=98.8, low=98.15, close=98.7, volume=200),
    ]
    frames["15m"] = bars
    if side == "short":
        frames = {
            key: [
                Candle(
                    c.time,
                    200 - c.open,
                    200 - c.low,
                    200 - c.high,
                    200 - c.close,
                    c.volume,
                )
                for c in values
            ]
            for key, values in frames.items()
        }
    price = frames["15m"][-1].close
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


def ready_decision(side="long"):
    market, frames = market_frames(side)
    # Keep an actual, farther structural target; yesterday's near high blocks this fixture's baseline trade.
    for i in range(96):
        frames["15m"][i] = replace(
            frames["15m"][i],
            high=102.4 if side == "long" else frames["15m"][i].high,
            low=97.6 if side == "short" else frames["15m"][i].low,
        )
    decision = evaluate_intraday(
        market, frames, None, SignalSettings(), SignalsCfg(), NOW
    )
    return decision, market, frames


@pytest.mark.parametrize("side", ["long", "short"])
def test_completed_pullback_produces_entry_with_structural_stop(side):
    decision, _, _ = ready_decision(side)
    assert decision.state == f"ENTER_{side.upper()}", decision.reasons
    assert decision.plan is not None
    assert decision.plan.net_reward_risk >= 2
    assert decision.plan.risk_usdt <= 5
    sign = 1 if side == "long" else -1
    assert sign * (decision.plan.entry - decision.plan.stop) > 0
    assert sign * (decision.plan.target - decision.plan.entry) > 0
    assert not hasattr(decision, "confidence")


def test_partial_candle_cannot_change_closed_signal():
    _, frames = market_frames()
    rows = candle_rows(frames["15m"])
    rows += [
        {
            "time": NOW // 900 * 900 * 1000,
            "open": 99,
            "high": 200,
            "low": 1,
            "close": 190,
            "baseVol": 999999,
        }
    ]
    assert closed_candles(rows, "15m", NOW) == frames["15m"]
    assert closed_candles(list(reversed(rows)), "15m", NOW) == frames["15m"]


@pytest.mark.parametrize("fault", ["gap", "nan", "stale", "duplicate", "bad_ohlc"])
def test_invalid_candles_fail_closed(fault):
    _, frames = market_frames()
    rows = candle_rows(frames["15m"])
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
        closed_candles(rows, "15m", NOW)


@pytest.mark.parametrize("side", ["long", "short"])
def test_breakout_needs_later_retest_and_volume(side):
    _, frames = market_frames()
    bars = [
        replace(c, open=99.5, close=99.5, high=100, low=99, volume=100)
        for c in frames["15m"]
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
    setup = find_setup(bars, frames["1h"], side, 1, None, SignalsCfg())
    assert setup is not None and setup.name == "Breakout & retest"
    bars[-3] = replace(bars[-3], volume=100)
    setup = find_setup(bars, frames["1h"], side, 1, None, SignalsCfg())
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
    assert funding_cost(market, "long", NOW, 12) == (12.0, 12)
    assert funding_cost(market, "short", NOW, 12) == (0, 12)


def test_exact_funding_settlement_is_charged():
    market, _ = market_frames()
    market = replace(
        market, funding_rate=0.0001, next_funding=NOW, funding_interval_hours=8
    )
    cost, payments = funding_cost(market, "long", NOW, 24)
    assert payments == 4 and cost == pytest.approx(0.04)


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


def test_higher_leverage_does_not_tighten_stop():
    _, market, frames = ready_decision()
    market = replace(market, tiers=[Tier(0, 50_000, 0.025, 125)])
    low = evaluate_intraday(
        market, frames, None, SignalSettings(leverage=10), SignalsCfg(), NOW
    )
    high = evaluate_intraday(
        market, frames, None, SignalSettings(leverage=40), SignalsCfg(), NOW
    )
    assert low.plan.stop == high.plan.stop
    assert low.state == "ENTER_LONG"
    assert high.state == "WATCH_LONG"
    assert high.plan.max_leverage < 40


def test_select_targets_skips_too_close_and_beyond_hold_budget():
    targets = select_targets(
        [100.2, 102.5, 140.0],
        100.0,
        "long",
        0.8,
        0.18,
        0.5,
        1.2,
        SignalsCfg(),
    )
    assert targets == [102.5]


def test_4h_ema_bias_can_enter_without_confirmed_4h_swings():
    _, market, frames = ready_decision()
    base = frames["4h"][0].close
    frames["4h"] = [
        replace(
            c,
            open=base + i * 0.15 - 0.04,
            close=base + i * 0.15,
            high=base + i * 0.15 + 0.02,
            low=base + i * 0.15 - 0.06,
        )
        for i, c in enumerate(frames["4h"])
    ]
    assert trend(frames["4h"]) == "mixed"
    assert ema_bias(frames["4h"]) == "long"
    decision = evaluate_intraday(
        market, frames, None, SignalSettings(), SignalsCfg(), NOW
    )
    assert decision.state == "ENTER_LONG", decision.reasons
    assert decision.metrics["trend_4h"] == "long"
    assert decision.metrics["trend_4h_structure"] == "mixed"


def test_opposite_4h_bias_blocks_entry():
    _, market, frames = ready_decision()
    frames["4h"] = [
        replace(
            c,
            open=c.close + 0.05,
            high=c.close + 0.06,
            low=c.close - 0.02,
            close=max(70, 100 - i * 0.25),
        )
        for i, c in enumerate(frames["4h"])
    ]
    assert ema_bias(frames["4h"]) != "long"
    decision = evaluate_intraday(
        market, frames, None, SignalSettings(), SignalsCfg(), NOW
    )
    assert not decision.state.startswith("ENTER")


def test_impulse_continuation_after_pullback():
    _, frames = market_frames()
    bars = [
        replace(c, open=80.4, close=80.5, high=80.7, low=80.3, volume=100)
        for c in frames["15m"]
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
    setup = find_setup(bars, frames["1h"], "long", 0.7, None, SignalsCfg())
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


@pytest.mark.parametrize("side", ["long", "short"])
@pytest.mark.parametrize("exit_type", ["stop", "target", "time", "structure", "stale"])
def test_exit_rules_are_position_specific_and_sticky(side, exit_type):
    trade, decision, frames = new_trade(side)
    now = NOW + 20
    if exit_type == "stop":
        decision.price = trade.plan.stop
    if exit_type == "target":
        decision.price = trade.plan.target
    if exit_type == "time":
        now += 24 * 3600
    if exit_type == "structure":
        decision.metrics["trend_1h"] = "short" if side == "long" else "long"
    if exit_type == "stale":
        now += 4 * 3600
    decision.as_of = now
    evaluate_exit(trade, decision, frames["15m"], now, SignalsCfg())
    assert trade.state == f"EXIT_{side.upper()}"
    decision.price = trade.plan.entry
    decision.metrics["trend_1h"] = side
    evaluate_exit(trade, decision, [], now + 1, SignalsCfg())
    assert trade.state == f"EXIT_{side.upper()}"
    assert trade.closed_at is None


def test_stale_market_causes_review_and_time_exit_still_works():
    trade, decision, _ = new_trade()
    evaluate_exit(trade, decision, [], NOW + 100, SignalsCfg())
    assert trade.state == "REVIEW"
    evaluate_exit(trade, None, [], NOW + 24 * 3600, SignalsCfg())
    assert trade.state == "EXIT_LONG"


def test_pre_entry_wick_does_not_trigger_exit():
    trade, decision, _ = new_trade()
    bar = Candle(NOW // 900 * 900, 99, 120, 10, 99, 100)
    evaluate_exit(trade, decision, [bar], NOW + 10, SignalsCfg())
    assert trade.state == "HOLD_LONG"


def test_trailing_stop_never_widens():
    trade, decision, _ = new_trade()
    original = trade.current_stop
    decision.price = trade.plan.entry + 2 * (trade.plan.entry - original)
    evaluate_exit(trade, decision, [], NOW + 10, SignalsCfg())
    tightened = trade.current_stop
    assert tightened > original
    decision.price -= 0.1
    evaluate_exit(trade, decision, [], NOW + 15, SignalsCfg())
    assert trade.current_stop >= tightened and trade.plan.stop == original


@pytest.mark.parametrize(
    "values",
    [
        {"planning_equity": float("nan")},
        {"leverage": 41},
        {"hold_hours": 48},
        {"risk_pct": 0},
        {"leverage": 25.5},
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
    scanner.update_settings(asdict(SignalSettings(leverage=40)))
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
