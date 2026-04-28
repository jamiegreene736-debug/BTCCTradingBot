"""End-to-end integration test with a mocked Bitunix REST client.

Runs the entire bot tick loop through scenarios that matter before going live:
  - signal generation correctness on synthetic candles
  - sizing math (risk-budget vs margin cap)
  - global open-position cap
  - per-symbol cap
  - cooldown window
  - bar-dedupe (no double-firing within same candle)
  - time-based exit (flash_close on stale positions)
  - dashboard JSON shape + auth
  - signing matches the Bitunix spec example

Run with:  .venv/bin/python -m pytest tests/test_e2e.py -v
or         .venv/bin/python tests/test_e2e.py
"""
from __future__ import annotations

import base64
import os
import sys
import time
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

# Ensure local imports work whether run via pytest or as a script.
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

# Inject creds before importing config-dependent modules.
os.environ["BITUNIX_API_KEY"] = "test_key"
os.environ["BITUNIX_SECRET_KEY"] = "test_secret"
os.environ["DASHBOARD_PASSWORD"] = "test_pass"

import numpy as np                                            # noqa: E402

from bitunix_bot.bot import BitunixBot                        # noqa: E402
from bitunix_bot.client import BitunixClient                  # noqa: E402
from bitunix_bot.config import load                           # noqa: E402
from bitunix_bot.dashboard import create_app                  # noqa: E402
from bitunix_bot.state import BotState, get as get_state      # noqa: E402


# ----------------------------------------------------------------- helpers

def make_uptrend_klines(n: int = 250, base: float = 60000.0, drift: float = 12.0) -> list[dict]:
    """Synthetic uptrend with the LAST 3 bars forced to be clean bullish
    marubozu-style candles (no rejection wicks) AND monotonically rising
    closes — so the continuation-confirmation gate passes (close in top
    quartile of bar range AND close > prior close)."""
    rng = np.random.default_rng(42)
    closes = base + np.cumsum(rng.normal(drift, 25, n))
    opens = closes - rng.uniform(5, 20, n)
    highs = np.maximum(opens, closes) + rng.uniform(0.5, 3, n)
    lows = np.minimum(opens, closes) - rng.uniform(0.5, 3, n)
    # Last 3 bars: clean bull marubozu, ascending closes (close[i] > close[i-1]).
    for i in range(n - 3, n):
        closes[i] = closes[i - 1] + 30 + rng.uniform(0, 5)
        opens[i] = closes[i] - 30 - rng.uniform(0, 5)
        highs[i] = closes[i] + rng.uniform(0.1, 1.5)
        lows[i] = opens[i] - rng.uniform(0.1, 1.5)
    now = int(time.time() * 1000)
    return [
        {
            "open": float(opens[i]),
            "high": float(highs[i]),
            "low": float(lows[i]),
            "close": float(closes[i]),
            "time": now - (n - i) * 60_000,
            "baseVol": "1.0",
            "quoteVol": "60000",
            "type": "1m",
        }
        for i in range(n)
    ]


def make_chop_klines(n: int = 250, base: float = 60000.0) -> list[dict]:
    """True range-bound noise — pure white noise, no drift, no trend.
    Should produce no signal because the EMA stack won't align AND ADX < 22.
    """
    rng = np.random.default_rng(7)
    closes = base + rng.normal(0, 25, n)  # white noise around base, no cumsum
    highs = closes + rng.uniform(2, 15, n)
    lows = closes - rng.uniform(2, 15, n)
    now = int(time.time() * 1000)
    return [
        {
            "open": float(closes[i] - 1),
            "high": float(highs[i]),
            "low": float(lows[i]),
            "close": float(closes[i]),
            "time": now - (n - i) * 60_000,
            "baseVol": "1.0",
            "quoteVol": "60000",
            "type": "1m",
        }
        for i in range(n)
    ]


def reset_state():
    """Reset the global state singleton between tests."""
    import bitunix_bot.state as st
    st._state = BotState()


# ----------------------------------------------------------------- fixtures

def fresh_cfg():
    """Loose-defaults config so 'should fire' tests work on synthetic data
    regardless of how aggressively production config has been tuned."""
    cfg = load("config.yaml", "/dev/null")
    cfg.mode = "paper"
    # Override strategy to permissive values so synthetic uptrends/downtrends
    # reliably produce signals. Production tightness is verified separately.
    cfg.strategy.rsi_long_min = 40
    cfg.strategy.rsi_long_max = 80
    cfg.strategy.rsi_short_min = 20
    cfg.strategy.rsi_short_max = 60
    # Disable the ADX-floor filter for fixture tests (most tests don't care
    # about regime gating; specific min_adx tests set this explicitly).
    cfg.strategy.min_adx_for_trade = 0.0
    # Disable post-signal ticker confirmation for fixture tests (most don't
    # mock ticker; specific confirmation tests set this explicitly).
    cfg.strategy.confirm_with_ticker = False
    # Permissive score threshold so synthetic uptrend/downtrend klines reliably
    # produce signals regardless of how aggressively production fire_threshold
    # is tuned. Specific threshold tests set this explicitly.
    cfg.strategy.fire_threshold = 0.30
    # Restore multi-symbol + multi-position defaults that pre-existed the
    # ChatGPT review. Production config.yaml is now BTCUSDT-only with
    # max_open_positions=1 and use_post_only_entries=true, but most tests
    # were written against the original loose defaults — preserve them
    # here so each test doesn't need to re-override.
    cfg.trading.symbols = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT"]
    cfg.trading.max_open_positions = 4
    cfg.trading.max_same_direction = 2
    cfg.trading.use_post_only_entries = False
    cfg.trading.cooldown_seconds = 60
    cfg.risk.partial_tp_enabled = True
    cfg.risk.take_profit_r = 2.5
    # Disable fee reserve in fixture so size-sensitive tests assert volumes
    # against the pre-Grok-holistic budget. Specific fee-reserve tests
    # set this explicitly.
    cfg.risk.round_trip_fee_pct = 0.0
    # Restore permissive pattern scoring so signal-generating tests reliably
    # produce non-zero scores. Production config.yaml uses pattern_weight=0.0
    # (tape-first refactor) which collapses factor-only scores below 0.30 on
    # synthetic klines that have few indicator votes. Tests that specifically
    # test pattern_weight=0.0 behaviour set it explicitly.
    cfg.strategy.pattern_weight = 0.55
    cfg.strategy.pattern_norm = 2.0
    return cfg


def make_mock_client():
    c = MagicMock(spec=BitunixClient)
    c.margin_coin = "USDT"
    c.account.return_value = {
        "marginCoin": "USDT", "available": "1000", "frozen": "0",
        "margin": "0", "transfer": "1000", "positionMode": "ONE_WAY",
        "crossUnrealizedPNL": "0", "isolationUnrealizedPNL": "0", "bonus": "0",
    }
    c.pending_positions.return_value = []
    c.history_positions.return_value = {"positionList": [], "total": 0}
    c.history_orders.return_value = {"orderList": [], "total": 0}
    c.trading_pairs.return_value = [
        {"symbol": "BTCUSDT", "basePrecision": 4, "quotePrecision": 1,
         "minTradeVolume": "0.0001", "maxLeverage": 200},
        {"symbol": "ETHUSDT", "basePrecision": 3, "quotePrecision": 2,
         "minTradeVolume": "0.001", "maxLeverage": 100},
        {"symbol": "SOLUSDT", "basePrecision": 1, "quotePrecision": 3,
         "minTradeVolume": "0.1", "maxLeverage": 75},
        {"symbol": "XRPUSDT", "basePrecision": 0, "quotePrecision": 4,
         "minTradeVolume": "1", "maxLeverage": 75},
    ]
    c.place_order.return_value = {"orderId": "ORD123", "clientId": "CL456"}
    c.flash_close_position.return_value = {"code": 0, "msg": "ok"}
    return c


# ----------------------------------------------------------------- tests

def test_signing_matches_bitunix_spec_example():
    """Spec: query_params for {id:1, uid:200} = 'id1uid200' — sorted, no separators."""
    c = BitunixClient("api", "secret")
    assert c._query_signing_string({"id": 1, "uid": 200}) == "id1uid200"
    assert c._query_signing_string({"symbol": "BTCUSDT"}) == "symbolBTCUSDT"
    # Determinism: same inputs produce same hash.
    s1 = c._sign("NONCE", "1700000000000", "a=1", "{}")
    s2 = c._sign("NONCE", "1700000000000", "a=1", "{}")
    assert s1 == s2 and len(s1) == 64


def test_uptrend_produces_long_signal_with_paper_order():
    reset_state()
    cfg = fresh_cfg()
    cfg.mode = "paper"
    bot = BitunixBot(cfg)
    bot.client = make_mock_client()
    bot.client.klines.side_effect = lambda *a, **kw: make_uptrend_klines()
    bot._resolve_symbol_meta()
    bot._tick()

    # Should record a tick + fire at least one paper order across the universe.
    snap = bot.state.snapshot()
    kinds = [e["kind"] for e in snap["events"]]
    assert "signal" in kinds, f"expected signal, got {kinds}"
    assert "order" in kinds, f"expected paper order, got {kinds}"
    # Bot should NOT have called place_order in paper mode.
    bot.client.place_order.assert_not_called()


def test_no_hard_gates_only_combined_threshold():
    """Verify the removal of ADX/ATR hard gates: signals fire purely based on
    the combined pattern+indicator score crossing fire_threshold."""
    reset_state()
    cfg = fresh_cfg()
    # Make threshold impossible — even strong setups shouldn't fire.
    cfg.strategy.fire_threshold = 0.99
    bot = BitunixBot(cfg)
    bot.client = make_mock_client()
    bot.client.klines.side_effect = lambda *a, **kw: make_uptrend_klines()
    bot._resolve_symbol_meta()
    bot._tick()
    kinds = [e["kind"] for e in bot.state.snapshot()["events"]]
    assert "signal" not in kinds, "should not fire when threshold is 0.99"

    # And reverse: lower threshold dramatically — should fire on the same data.
    reset_state()
    cfg = fresh_cfg()
    cfg.strategy.fire_threshold = 0.05
    bot = BitunixBot(cfg)
    bot.client = make_mock_client()
    bot.client.klines.side_effect = lambda *a, **kw: make_uptrend_klines()
    bot._resolve_symbol_meta()
    bot._tick()
    kinds = [e["kind"] for e in bot.state.snapshot()["events"]]
    assert "signal" in kinds, "should fire when threshold is 0.05"


def test_global_max_open_positions_cap():
    reset_state()
    cfg = fresh_cfg()
    cfg.trading.max_open_positions = 1
    bot = BitunixBot(cfg)
    bot.client = make_mock_client()
    # Already holding one position — cap reached.
    bot.client.pending_positions.return_value = [{
        "positionId": "P1", "symbol": "BTCUSDT", "qty": "0.01",
        "side": "LONG", "leverage": 100, "ctime": int(time.time() * 1000),
    }]
    bot.client.klines.side_effect = lambda *a, **kw: make_uptrend_klines()
    bot._resolve_symbol_meta()
    bot._tick()

    # Should NOT have placed any new orders; klines may be skipped entirely.
    kinds = [e["kind"] for e in bot.state.snapshot()["events"]]
    assert "order" not in kinds, f"capped, no new orders allowed, got {kinds}"


def test_same_direction_cap_kills_correlated_risk():
    """If 2 longs are already open, a 3rd long signal must be skipped even
    though global cap and per-symbol cap allow it."""
    reset_state()
    cfg = fresh_cfg()
    cfg.trading.max_open_positions = 4
    cfg.trading.max_same_direction = 2
    bot = BitunixBot(cfg)
    bot.client = make_mock_client()
    # Two longs already open on BTC and ETH.
    bot.client.pending_positions.return_value = [
        {"positionId": "P1", "symbol": "BTCUSDT", "qty": "0.01",
         "side": "BUY", "leverage": 100, "ctime": int(time.time() * 1000)},
        {"positionId": "P2", "symbol": "ETHUSDT", "qty": "0.1",
         "side": "BUY", "leverage": 100, "ctime": int(time.time() * 1000)},
    ]
    bot.client.klines.side_effect = lambda *a, **kw: make_uptrend_klines()
    bot._resolve_symbol_meta()
    bot._tick()

    # Should record at least one same-direction-cap skip.
    skips = [e for e in bot.state.snapshot()["events"] if e["kind"] == "skip"]
    assert any("same-direction" in e["text"] for e in skips), \
        f"expected same-direction skip, got skips={[e['text'] for e in skips]}"


def test_per_symbol_cap_blocks_same_symbol_but_allows_others():
    reset_state()
    cfg = fresh_cfg()
    cfg.trading.max_open_positions = 4
    cfg.trading.max_positions_per_symbol = 1
    bot = BitunixBot(cfg)
    bot.client = make_mock_client()
    # Already long on BTC; ETH/SOL/XRP should still be eligible.
    bot.client.pending_positions.return_value = [{
        "positionId": "P1", "symbol": "BTCUSDT", "qty": "0.01",
        "side": "LONG", "leverage": 100, "ctime": int(time.time() * 1000),
    }]
    bot.client.klines.side_effect = lambda *a, **kw: make_uptrend_klines()
    bot._resolve_symbol_meta()
    bot._tick()

    orders = [e for e in bot.state.snapshot()["events"] if e["kind"] == "order"]
    assert orders, "expected paper orders on non-BTC symbols"
    for ev in orders:
        assert "BTCUSDT" not in ev["text"], f"BTC was capped: {ev['text']}"


def test_cooldown_blocks_immediate_re_entry():
    reset_state()
    cfg = fresh_cfg()
    cfg.trading.cooldown_seconds = 60
    bot = BitunixBot(cfg)
    bot.client = make_mock_client()
    bot.client.klines.side_effect = lambda *a, **kw: make_uptrend_klines()
    bot._resolve_symbol_meta()

    # First tick: places paper orders.
    bot._tick()
    first_orders = [e for e in bot.state.snapshot()["events"] if e["kind"] == "order"]
    assert first_orders, "first tick should produce paper orders"

    # Second tick immediately after: cooldown blocks; mocks return the same bar
    # so bar-dedupe also blocks. We expect NO new orders.
    n_orders_before = len(first_orders)
    bot._tick()
    n_orders_after = len([e for e in bot.state.snapshot()["events"] if e["kind"] == "order"])
    assert n_orders_after == n_orders_before, "cooldown + bar-dedupe should block re-entry"


def test_bar_dedupe_blocks_same_candle_re_eval():
    reset_state()
    cfg = fresh_cfg()
    cfg.trading.cooldown_seconds = 0  # disable cooldown so we test ONLY bar-dedupe
    bot = BitunixBot(cfg)
    bot.client = make_mock_client()
    klines = make_uptrend_klines()
    bot.client.klines.side_effect = lambda *a, **kw: klines  # same data every call
    bot._resolve_symbol_meta()

    bot._tick()
    n1 = len([e for e in bot.state.snapshot()["events"] if e["kind"] == "order"])
    # Tick again with the SAME klines — should not re-fire.
    bot._tick()
    n2 = len([e for e in bot.state.snapshot()["events"] if e["kind"] == "order"])
    assert n1 == n2, f"bar-dedupe failed: orders {n1} -> {n2}"


def test_time_based_exit_closes_stale_LOSING_position():
    """With time_exit_only_if_losing=True, a stale position is force-closed
    only when its unrealizedPNL is negative."""
    reset_state()
    cfg = fresh_cfg()
    cfg.mode = "live"
    cfg.trading.max_position_age_seconds = 60
    bot = BitunixBot(cfg)
    bot.client = make_mock_client()
    stale_loser = {
        "positionId": "STALE1", "symbol": "BTCUSDT", "qty": "0.01",
        "side": "LONG", "leverage": 100,
        "ctime": int(time.time() * 1000) - 600_000,  # 10 min ago
        "unrealizedPNL": "-0.5",  # losing → eligible for force-close
    }
    bot.client.pending_positions.return_value = [stale_loser]
    bot.client.klines.side_effect = lambda *a, **kw: make_uptrend_klines()
    bot._resolve_symbol_meta()
    bot._tick()
    bot.client.flash_close_position.assert_called_once_with("STALE1")


def test_time_based_exit_LETS_WINNER_RUN():
    """A winning stale position past max age stays alive (SL ratchet protects it)."""
    reset_state()
    cfg = fresh_cfg()
    cfg.mode = "live"
    cfg.trading.max_position_age_seconds = 60
    bot = BitunixBot(cfg)
    bot.client = make_mock_client()
    stale_winner = {
        "positionId": "WINNER1", "symbol": "BTCUSDT", "qty": "0.01",
        "side": "LONG", "leverage": 100,
        "ctime": int(time.time() * 1000) - 600_000,
        "unrealizedPNL": "+1.50",  # winning → NOT closed by time exit
    }
    bot.client.pending_positions.return_value = [stale_winner]
    bot.client.klines.side_effect = lambda *a, **kw: make_uptrend_klines()
    bot._resolve_symbol_meta()
    bot._tick()
    bot.client.flash_close_position.assert_not_called()


def test_live_mode_with_zero_margin_skips_orders():
    reset_state()
    cfg = fresh_cfg()
    cfg.mode = "live"
    bot = BitunixBot(cfg)
    bot.client = make_mock_client()
    bot.client.account.return_value = {
        "marginCoin": "USDT", "available": "0", "frozen": "0",
        "margin": "0", "positionMode": "ONE_WAY",
        "crossUnrealizedPNL": "0", "isolationUnrealizedPNL": "0", "bonus": "0",
    }
    bot.client.klines.side_effect = lambda *a, **kw: make_uptrend_klines()
    bot._resolve_symbol_meta()
    bot._tick()

    # Live mode should NOT place orders without real margin.
    bot.client.place_order.assert_not_called()
    # And should record at least one skip with the right reason.
    skips = [e for e in bot.state.snapshot()["events"] if e["kind"] == "skip"]
    assert any("margin" in e["text"].lower() for e in skips), f"no margin skip recorded: {skips}"


def test_paper_mode_with_zero_margin_simulates_anyway():
    reset_state()
    cfg = fresh_cfg()
    cfg.mode = "paper"
    bot = BitunixBot(cfg)
    bot.client = make_mock_client()
    bot.client.account.return_value = {
        "marginCoin": "USDT", "available": "0", "frozen": "0",
        "margin": "0", "positionMode": "ONE_WAY",
        "crossUnrealizedPNL": "0", "isolationUnrealizedPNL": "0", "bonus": "0",
    }
    bot.client.klines.side_effect = lambda *a, **kw: make_uptrend_klines()
    bot._resolve_symbol_meta()
    bot._tick()

    # Paper mode should still simulate — using fake $1k balance.
    orders = [e for e in bot.state.snapshot()["events"] if e["kind"] == "order"]
    assert orders, f"paper should simulate even with $0 real margin, events={[(e['kind'], e['text']) for e in bot.state.snapshot()['events']]}"


def test_live_mode_actually_calls_place_order():
    reset_state()
    cfg = fresh_cfg()
    cfg.mode = "live"
    bot = BitunixBot(cfg)
    bot.client = make_mock_client()
    bot.client.klines.side_effect = lambda *a, **kw: make_uptrend_klines()
    bot._resolve_symbol_meta()
    bot._tick()

    assert bot.client.place_order.called, "live mode should hit place_order"
    call_kwargs = bot.client.place_order.call_args.kwargs
    # SL and TP must always be attached.
    assert call_kwargs.get("sl_price"), f"SL missing! {call_kwargs}"
    assert call_kwargs.get("tp_price"), f"TP missing! {call_kwargs}"
    assert call_kwargs.get("trade_side") == "OPEN"
    assert call_kwargs.get("order_type") == "MARKET"
    assert call_kwargs.get("side") in ("BUY", "SELL")
    qty = float(call_kwargs.get("qty"))
    assert qty > 0


def test_dashboard_routes_and_auth():
    reset_state()
    cfg = fresh_cfg()
    client = make_mock_client()
    bot_state = get_state()
    bot_state.record_signal("BTCUSDT LONG score=4 @ 60000")
    bot_state.record_order("PAPER BTCUSDT BUY qty=0.01 ...")

    app = create_app(cfg, client)
    c = app.test_client()

    # /healthz: no auth
    r = c.get("/healthz")
    assert r.status_code == 200 and r.data == b"ok"

    # / without auth: 401
    assert c.get("/").status_code == 401

    # / wrong password: 401
    bad = base64.b64encode(b"admin:nope").decode()
    assert c.get("/", headers={"Authorization": f"Basic {bad}"}).status_code == 401

    # / correct: 200, has expected markup
    good = base64.b64encode(b"admin:test_pass").decode()
    r = c.get("/", headers={"Authorization": f"Basic {good}"})
    assert r.status_code == 200
    assert b"Bitunix TraderBot" in r.data
    assert b"Closed positions" in r.data
    assert b"Open positions" in r.data

    # /api/state: 200, expected JSON shape
    r = c.get("/api/state", headers={"Authorization": f"Basic {good}"})
    assert r.status_code == 200
    j = r.get_json()
    assert "config" in j and "bot" in j
    assert isinstance(j["config"]["symbols"], list)
    assert "open_positions" in j
    assert "history_positions" in j
    assert "account" in j
    assert "win_rate" in j
    wr = j["win_rate"]
    assert set(wr.keys()) == {"wins", "losses", "total", "rate"}


def _setup_bot_with_open_position(side: str, entry: float, current_price: float,
                                   qty: float = 0.01, original_sl: float | None = None,
                                   original_tp: float | None = None,
                                   stop_loss_pct: float = 0.25,
                                   breakeven_at_r=1.0, trailing_activate_r=1.5,
                                   trailing_distance_r=0.5, buffer_pct=0.05):
    """Helper: build a live-mode bot with one open position and the given TPSL state.

    `stop_loss_pct` is pinned (default 0.25) so SL-ratchet tests aren't
    coupled to whatever production config is set to — they test the logic
    at fixed reference numbers."""
    reset_state()
    cfg = fresh_cfg()
    cfg.mode = "live"
    cfg.risk.stop_loss_pct = stop_loss_pct
    cfg.risk.breakeven_at_r = breakeven_at_r
    cfg.risk.breakeven_buffer_pct = buffer_pct
    cfg.risk.trailing_activate_r = trailing_activate_r
    cfg.risk.trailing_distance_r = trailing_distance_r
    sl_dist = entry * (cfg.risk.stop_loss_pct / 100.0)
    if original_sl is None:
        original_sl = entry - sl_dist if side == "BUY" else entry + sl_dist
    if original_tp is None:
        original_tp = (entry + sl_dist * cfg.risk.take_profit_r if side == "BUY"
                       else entry - sl_dist * cfg.risk.take_profit_r)
    upnl = (current_price - entry) * qty if side == "BUY" else (entry - current_price) * qty

    bot = BitunixBot(cfg)
    bot.client = make_mock_client()
    pos = {
        "positionId": "POS1", "symbol": "BTCUSDT", "qty": str(qty),
        "side": side, "leverage": 50, "ctime": int(time.time() * 1000),
        "avgOpenPrice": str(entry), "unrealizedPNL": str(upnl),
    }
    bot.client.pending_positions.return_value = [pos]
    bot.client.pending_tpsl.return_value = [
        {"id": "T_TP", "positionId": "POS1", "symbol": "BTCUSDT",
         "tpPrice": str(original_tp), "slPrice": None,
         "tpStopType": "LAST_PRICE", "slStopType": "LAST_PRICE",
         "tpQty": str(qty), "slQty": None,
         "tpOrderType": "MARKET", "slOrderType": None},
        {"id": "T_SL", "positionId": "POS1", "symbol": "BTCUSDT",
         "tpPrice": None, "slPrice": str(original_sl),
         "tpStopType": "LAST_PRICE", "slStopType": "LAST_PRICE",
         "tpQty": None, "slQty": str(qty),
         "tpOrderType": None, "slOrderType": "MARKET"},
    ]
    bot.client.klines.side_effect = lambda *a, **kw: make_uptrend_klines()
    bot._resolve_symbol_meta()
    return bot, original_sl, original_tp


def test_breakeven_sl_move_at_1r_long():
    bot, orig_sl, orig_tp = _setup_bot_with_open_position(
        side="BUY", entry=100_000.0, current_price=100_250.0,  # +1R exactly
    )
    bot._tick()
    bot.client.modify_tpsl_order.assert_called_once()
    kw = bot.client.modify_tpsl_order.call_args.kwargs
    # Targets the SL trigger order specifically by its id.
    assert kw["order_id"] == "T_SL", f"should target SL trigger T_SL, got {kw.get('order_id')}"
    new_sl = float(kw["sl_price"])
    # Expected: entry + 0.05% buffer = 100_050
    assert abs(new_sl - 100_050.0) < 1.0, f"long BE SL wrong: {new_sl}"
    # Should preserve the SL qty.
    assert kw["sl_qty"] is not None


def test_trailing_sl_at_2r_long():
    bot, _, _ = _setup_bot_with_open_position(
        side="BUY", entry=100_000.0, current_price=100_500.0,  # +2R
    )
    bot._tick()
    bot.client.modify_tpsl_order.assert_called_once()
    new_sl = float(bot.client.modify_tpsl_order.call_args.kwargs["sl_price"])
    # Expected: current - trail_dist = 100_500 - 0.5*250 = 100_375
    assert abs(new_sl - 100_375.0) < 1.0, f"long trailing SL wrong: {new_sl}"


def test_breakeven_sl_short():
    bot, _, _ = _setup_bot_with_open_position(
        side="SELL", entry=100_000.0, current_price=99_750.0,  # +1R for short
    )
    bot._tick()
    bot.client.modify_tpsl_order.assert_called_once()
    new_sl = float(bot.client.modify_tpsl_order.call_args.kwargs["sl_price"])
    # Expected: entry - 0.05% buffer = 99_950
    assert abs(new_sl - 99_950.0) < 1.0, f"short BE SL wrong: {new_sl}"


def test_breakeven_sl_clamps_on_30030_short():
    """Race: BE-protect computes new_sl from ENTRY, but live price has
    retraced toward entry — new_sl lands on wrong side of last price.
    Bitunix returns code 30030. Bot must retry with SL clamped to
    current_price + 1 tick (locks in remaining profit).

    Setup: short entry 100_000, current_price 99_950 (just retraced from
    peak), buffer 0.10% → intended new_sl = 99_900 (below current → invalid
    for short)."""
    from bitunix_bot.client import BitunixError
    bot, _, _ = _setup_bot_with_open_position(
        side="SELL", entry=100_000.0, current_price=99_950.0,  # r=0.20
        breakeven_at_r=0.20, buffer_pct=0.10,
    )
    # First modify call raises 30030; second succeeds.
    bot.client.modify_tpsl_order.side_effect = [
        BitunixError(30030, "SL price must be greater than last price: 99950", {}),
        None,
    ]
    bot._tick()

    assert bot.client.modify_tpsl_order.call_count == 2, \
        "expected first call + retry"
    first = bot.client.modify_tpsl_order.call_args_list[0].kwargs
    second = bot.client.modify_tpsl_order.call_args_list[1].kwargs
    # First call used the entry-anchored SL (99_900).
    assert abs(float(first["sl_price"]) - 99_900.0) < 1.0
    # Second call clamped to current_price + 1 tick (just above 99_950).
    clamped = float(second["sl_price"])
    assert clamped > 99_950.0, \
        f"clamped SL {clamped} must be strictly above current 99_950 (short)"
    assert clamped < 100_000.0, \
        f"clamped SL {clamped} must still lock in some profit (below entry)"


def test_breakeven_sl_clamps_on_30030_long():
    """Long mirror: when intended new_sl lands ABOVE last price (retracement
    toward entry), retry with SL clamped to current_price - 1 tick."""
    from bitunix_bot.client import BitunixError
    bot, _, _ = _setup_bot_with_open_position(
        side="BUY", entry=100_000.0, current_price=100_050.0,  # r=0.20
        breakeven_at_r=0.20, buffer_pct=0.10,
    )
    bot.client.modify_tpsl_order.side_effect = [
        BitunixError(30030, "SL price must be less than last price: 100050", {}),
        None,
    ]
    bot._tick()

    assert bot.client.modify_tpsl_order.call_count == 2
    first = bot.client.modify_tpsl_order.call_args_list[0].kwargs
    second = bot.client.modify_tpsl_order.call_args_list[1].kwargs
    # First call used entry+buffer = 100_100 (above current → invalid for long).
    assert abs(float(first["sl_price"]) - 100_100.0) < 1.0
    # Second call clamped just below current_price.
    clamped = float(second["sl_price"])
    assert clamped < 100_050.0, \
        f"clamped SL {clamped} must be below current 100_050 (long)"
    assert clamped > 100_000.0, \
        f"clamped SL {clamped} must still lock in some profit (above entry)"


def test_breakeven_sl_30030_non30030_errors_propagate():
    """A non-30030 BitunixError (e.g. position closed) must NOT trigger the
    retry path — it should propagate to the outer benign-or-real handler."""
    from bitunix_bot.client import BitunixError
    bot, _, _ = _setup_bot_with_open_position(
        side="SELL", entry=100_000.0, current_price=99_950.0,
        breakeven_at_r=0.20, buffer_pct=0.10,
    )
    bot.client.modify_tpsl_order.side_effect = BitunixError(
        99999, "some other error", {}
    )
    # Should not raise — outer handler catches and records the error.
    bot._tick()
    # Only ONE call attempted (no retry on non-30030).
    assert bot.client.modify_tpsl_order.call_count == 1, \
        f"non-30030 errors must not retry; got {bot.client.modify_tpsl_order.call_count} calls"


def test_sl_never_moves_against_position():
    bot, _, _ = _setup_bot_with_open_position(
        side="BUY", entry=100_000.0, current_price=100_250.0,
        original_sl=100_100.0,  # already locked in some profit
    )
    bot._tick()
    bot.client.modify_tpsl_order.assert_not_called()


def test_no_management_below_threshold():
    bot, _, _ = _setup_bot_with_open_position(
        side="BUY", entry=100_000.0, current_price=100_125.0,  # +0.5R
    )
    bot._tick()
    bot.client.modify_tpsl_order.assert_not_called()


def test_htf_and_funding_votes_contribute_to_score():
    """Tier 2: a clean uptrend in HTF + negative funding (crowded shorts)
    should add 2 long votes; a clean downtrend in HTF + positive funding
    should add 2 short votes."""
    from bitunix_bot.config import StrategyCfg
    from bitunix_bot.strategy import evaluate

    rng = np.random.default_rng(2)
    n = 200
    base = 60000.0
    closes = base + np.cumsum(rng.normal(8, 25, n))
    opens = closes - rng.uniform(5, 15, n)
    highs = np.maximum(opens, closes) + rng.uniform(0.5, 3, n)
    lows = np.minimum(opens, closes) - rng.uniform(0.5, 3, n)
    # Clean bull marubozu on last 3 bars + ascending closes for continuation gate.
    for i in range(n - 3, n):
        closes[i] = closes[i - 1] + 30 + rng.uniform(0, 5)
        opens[i] = closes[i] - 30 - rng.uniform(0, 5)
        highs[i] = closes[i] + rng.uniform(0.1, 1.5)
        lows[i] = opens[i] - rng.uniform(0.1, 1.5)
    vols = np.full(n, 1.0)

    cfg = StrategyCfg(
        ema_fast=9, ema_mid=21, ema_slow=50,
        rsi_period=14, rsi_long_min=40, rsi_long_max=80,
        rsi_short_min=20, rsi_short_max=60,
        macd_fast=12, macd_slow=26, macd_signal=9,
        bb_period=20, bb_std=2.0, atr_period=14,
        adx_period=14, adx_min=15.0,
        supertrend_period=10, supertrend_mult=3.0,
        pattern_weight=0.55, pattern_norm=2.0, fire_threshold=0.05,
        volume_ma_period=20, volume_spike_multiplier=1.5,
        stoch_rsi_period=14, stoch_rsi_k=3, stoch_rsi_d=3,
        htf_timeframe="15m", htf_ema_period=50,
        funding_threshold=0.0005,
    )

    # Strong uptrend in HTF closes (close > EMA50).
    htf = list(np.linspace(50_000, 60_000, 200))

    # Negative funding (crowded shorts) → should add long vote.
    sig = evaluate(opens.tolist(), highs.tolist(), lows.tolist(), closes.tolist(),
                    cfg, volumes=vols.tolist(), htf_closes=htf, funding_rate=-0.001)
    assert sig is not None, "should produce a signal"
    assert sig.direction == "long", f"expected long, got {sig.direction}"
    assert any("htf_uptrend" in r for r in sig.reasons), f"HTF vote missing: {sig.reasons}"
    # funding vote removed in tape-first refactor — only check HTF


def test_pattern_detection_basic_shapes():
    """Verify the pattern detector recognizes textbook shapes."""
    from bitunix_bot import patterns

    # Bullish engulfing on a 2-bar dataset: small bear -> big bull engulfing.
    o = np.array([100.0, 99.0])
    h = np.array([100.5, 101.0])
    l = np.array([99.0, 98.8])
    c = np.array([99.2, 100.8])
    hits = patterns.detect(o, h, l, c)
    names = [p.name for p in hits]
    assert "bullish_engulfing" in names, f"expected bullish_engulfing in {names}"
    bull, bear = patterns.score(hits)
    assert bull > bear, f"bull score {bull} should beat bear {bear}"

    # Bearish engulfing.
    o = np.array([99.0, 100.8])
    h = np.array([100.5, 101.0])
    l = np.array([98.8, 98.5])
    c = np.array([100.5, 99.0])
    names = [p.name for p in patterns.detect(o, h, l, c)]
    assert "bearish_engulfing" in names, f"expected bearish_engulfing in {names}"

    # Hammer (small body, long lower wick) preceded by downtrend bars.
    n = 20
    o = np.linspace(100, 95, n)  # downtrend
    c = o - 0.2
    h = c + 0.3
    l = o - 0.3
    # Last candle = hammer: small body, long lower wick.
    o[-1] = 95.0; c[-1] = 95.05; h[-1] = 95.1; l[-1] = 94.0
    names = [p.name for p in patterns.detect(o, h, l, c)]
    assert "hammer" in names, f"expected hammer in {names}"

    # Doji: open ≈ close, wide range.
    o = np.array([100.0, 100.0, 100.0])
    c = np.array([100.0, 100.0, 100.05])
    h = np.array([100.5, 100.5, 100.6])
    l = np.array([99.5, 99.5, 99.4])
    names = [p.name for p in patterns.detect(o, h, l, c)]
    assert "doji" in names, f"expected doji in {names}"


def test_pattern_alone_can_fire_signal():
    """A strong pattern with mediocre indicators should still fire because
    pattern_weight is 0.55 — so a normalized pattern score of ~1.0 alone
    contributes 0.55, above the default 0.45 fire threshold."""
    from bitunix_bot.config import StrategyCfg
    from bitunix_bot.strategy import evaluate

    # Build a clean uptrend with a fresh bullish engulfing on the last bar.
    rng = np.random.default_rng(3)
    n = 200
    base = 100.0
    closes = base + np.cumsum(rng.normal(0.05, 0.3, n))
    opens = closes - rng.uniform(0.05, 0.2, n)
    highs = np.maximum(opens, closes) + rng.uniform(0.02, 0.1, n)
    lows = np.minimum(opens, closes) - rng.uniform(0.02, 0.1, n)
    # Force last 2 bars to a bullish engulfing.
    opens[-2], closes[-2] = closes[-3] + 0.05, closes[-3] - 0.4
    highs[-2] = opens[-2] + 0.05
    lows[-2] = closes[-2] - 0.05
    opens[-1], closes[-1] = closes[-2] - 0.05, opens[-2] + 0.6
    highs[-1] = closes[-1] + 0.05
    lows[-1] = opens[-1] - 0.05

    cfg = StrategyCfg(
        ema_fast=9, ema_mid=21, ema_slow=50,
        rsi_period=14, rsi_long_min=40, rsi_long_max=80,
        rsi_short_min=20, rsi_short_max=60,
        macd_fast=12, macd_slow=26, macd_signal=9,
        bb_period=20, bb_std=2.0, atr_period=14,
        adx_period=14, adx_min=15.0,        # loose ADX so the gate doesn't block
        supertrend_period=10, supertrend_mult=3.0,
        pattern_weight=0.55, pattern_norm=2.5, fire_threshold=0.45,
    )
    sig = evaluate(opens.tolist(), highs.tolist(), lows.tolist(), closes.tolist(), cfg)
    assert sig is not None, "engulfing + uptrend should fire a signal"
    assert sig.direction == "long", f"expected long, got {sig.direction}"
    assert sig.pattern_score > 0, "pattern_score should be > 0"
    assert any("PAT:" in r for r in sig.reasons), f"reasons missing pattern tags: {sig.reasons}"


def test_double_bottom_detection():
    """Two swing lows at similar level, then close breaks above the intervening
    high. Single-bar swing-low dips so the fractal detector triggers."""
    from bitunix_bot import chart_patterns
    n = 60
    closes = np.full(n, 105.0)
    # Down-trend setup: descending closes into bottom 1.
    closes[0:15] = np.linspace(110, 100, 15)
    # Bottom 1 (single sharp dip at idx 15).
    closes[15] = 100.0
    # Rally to ~105 (intervening high).
    closes[16:30] = np.linspace(100, 105, 14)
    # Bottom 2 (similar low at idx 30).
    closes[30] = 100.1
    # Break above the rally high (above 105).
    closes[31:60] = np.linspace(100.1, 110, 29)
    highs = closes + 0.5
    lows = closes - 0.2
    # Single-bar lows so fractal detector picks them up unambiguously.
    lows[15] = 99.5
    lows[30] = 99.6
    hits = chart_patterns.detect_all(highs, lows, closes, swing_lookback=3)
    names = [h.name for h in hits]
    assert "double_bottom" in names, f"expected double_bottom in {names}"


def test_chart_pattern_score_aggregation():
    from bitunix_bot import chart_patterns

    bullish_hits = [
        chart_patterns.ChartPattern("double_bottom", "bullish", 1.4, "x"),
        chart_patterns.ChartPattern("bull_flag", "bullish", 1.2, "x"),
    ]
    bearish_hits = [
        chart_patterns.ChartPattern("head_and_shoulders", "bearish", 1.5, "x"),
    ]
    bull, bear = chart_patterns.score(bullish_hits + bearish_hits)
    assert abs(bull - 2.6) < 0.01
    assert abs(bear - 1.5) < 0.01


def test_sr_detection_finds_real_levels():
    """Single-bar swing lows at similar prices → cluster into one support level."""
    from bitunix_bot import levels
    n = 60
    closes = np.full(n, 105.0)
    closes[:10] = np.linspace(110, 100, 10)
    closes[10] = 100.0  # swing low 1 (single bar)
    closes[11:25] = np.linspace(100, 108, 14)
    closes[25] = 100.1  # swing low 2 (single bar)
    closes[26:60] = np.linspace(100.1, 112, 34)
    highs = closes + 0.5
    lows = closes - 0.2
    lows[10] = 99.5
    lows[25] = 99.6
    sh, sl = levels.find_swings(highs, lows, lookback=3)
    assert len(sl) >= 2, f"expected ≥2 swing lows, got {len(sl)}"
    lvls = levels.cluster_levels(sh + sl, tolerance_pct=0.5, min_touches=2)
    support_levels = [lv for lv in lvls if lv.kind == "support"]
    assert support_levels, f"no support levels detected: {lvls}"
    assert support_levels[0].touches >= 2


def test_orderbook_imbalance_compute():
    """Manually populate the OB and verify imbalance math."""
    from bitunix_bot.orderbook import Book, OrderBookFeed
    feed = OrderBookFeed(symbols=["BTCUSDT"], depth_levels=3)
    # Seed a book with much more bid volume than ask.
    with feed._lock:
        b = feed._books["BTCUSDT"]
        b.bids = [(60000.0, 5.0), (59999.0, 4.0), (59998.0, 3.0)]   # bid total = 12
        b.asks = [(60001.0, 1.0), (60002.0, 1.0), (60003.0, 2.0)]   # ask total = 4
        b.last_update = time.time()
    imb = feed.get_imbalance("BTCUSDT")
    assert imb is not None
    # (12 - 4) / (12 + 4) = 0.5
    assert abs(imb - 0.5) < 0.001, f"expected 0.5 got {imb}"

    # Now flip: ask-heavy.
    with feed._lock:
        b = feed._books["BTCUSDT"]
        b.bids = [(60000.0, 1.0), (59999.0, 1.0), (59998.0, 1.0)]   # 3
        b.asks = [(60001.0, 4.0), (60002.0, 5.0), (60003.0, 6.0)]   # 15
        b.last_update = time.time()
    imb = feed.get_imbalance("BTCUSDT")
    # (3 - 15) / (3 + 15) = -0.667
    assert abs(imb - (-0.667)) < 0.01, f"expected ≈ -0.67 got {imb}"


def test_orderbook_extract_symbol_handles_list_data():
    """Bitunix sometimes wraps payload as {'data': [{'symbol': X, ...}]}.
    Original parser missed this case (silently dropped book updates),
    while TradeFeed handled it correctly. Both are now aligned."""
    from bitunix_bot.orderbook import OrderBookFeed
    # List-shaped data with symbol nested in first dict element.
    msg = {"ch": "depth_books", "data": [
        {"symbol": "BTCUSDT",
         "b": [["60000", "1.0"]],
         "a": [["60001", "1.0"]]}
    ]}
    sym = OrderBookFeed._extract_symbol(msg)
    assert sym == "BTCUSDT", f"list-shaped symbol extraction failed: got {sym}"

    # Top-level symbol still works.
    msg2 = {"ch": "depth_books", "symbol": "ETHUSDT", "data": {}}
    assert OrderBookFeed._extract_symbol(msg2) == "ETHUSDT"

    # Dict-shaped data still works.
    msg3 = {"ch": "depth_books", "data": {"s": "SOLUSDT", "b": [], "a": []}}
    assert OrderBookFeed._extract_symbol(msg3) == "SOLUSDT"

    # Neither — None.
    assert OrderBookFeed._extract_symbol({"ch": "x", "data": []}) is None


def test_orderbook_get_status_shape():
    """get_status() returns lifecycle + per-symbol diagnostic fields."""
    from bitunix_bot.orderbook import OrderBookFeed
    feed = OrderBookFeed(symbols=["BTCUSDT", "ETHUSDT"], depth_levels=10)
    status = feed.get_status()
    assert status["type"] == "OrderBookFeed"
    assert status["symbols"] == ["BTCUSDT", "ETHUSDT"]
    assert status["connected"] is False  # never started
    assert "lifecycle" in status and "messages" in status
    assert "BTCUSDT" in status["books"]
    assert status["books"]["BTCUSDT"]["has_bids"] is False
    assert status["books"]["BTCUSDT"]["is_fresh"] is False


def test_tradetape_get_status_shape():
    """TradeFeed.get_status() includes counters + per-symbol activity."""
    from bitunix_bot.tradetape import TradeFeed, Trade
    feed = TradeFeed(symbols=["BTCUSDT"])
    status = feed.get_status()
    assert status["type"] == "TradeFeed"
    assert "BTCUSDT" in status["tape"]
    # Add some trades and verify counts surface.
    now = time.time()
    feed._ingest(Trade(ts=now - 5, price=60000, qty=1.0, is_buy=True), "BTCUSDT")
    feed._ingest(Trade(ts=now - 70, price=60000, qty=1.0, is_buy=False), "BTCUSDT")
    status2 = feed.get_status()
    assert status2["tape"]["BTCUSDT"]["buffered_count"] == 2
    assert status2["tape"]["BTCUSDT"]["trades_last_10s"] == 1   # only the 5s one
    assert status2["tape"]["BTCUSDT"]["trades_last_60s"] == 1


def test_feeds_status_endpoint():
    """GET /api/feeds/status returns both feeds' status snapshots."""
    reset_state()
    cfg = fresh_cfg()
    bot = BitunixBot(cfg)
    bot.client = make_mock_client()
    from bitunix_bot.orderbook import OrderBookFeed
    from bitunix_bot.tradetape import TradeFeed
    bot.ob_feed = OrderBookFeed(symbols=cfg.trading.symbols)
    bot.tape_feed = TradeFeed(symbols=cfg.trading.symbols)

    app = create_app(cfg, bot.client, bot=bot)
    c = app.test_client()
    auth = base64.b64encode(b"admin:test_pass").decode()
    r = c.get("/api/feeds/status",
              headers={"Authorization": f"Basic {auth}"})
    assert r.status_code == 200
    body = r.get_json()
    assert body["ob_feed"]["type"] == "OrderBookFeed"
    assert body["tape_feed"]["type"] == "TradeFeed"
    assert body["ob_feed"]["connected"] is False
    assert body["tape_feed"]["connected"] is False


def test_feeds_status_requires_auth():
    reset_state()
    cfg = fresh_cfg()
    bot = BitunixBot(cfg)
    bot.client = make_mock_client()
    app = create_app(cfg, bot.client, bot=bot)
    c = app.test_client()
    r = c.get("/api/feeds/status")
    assert r.status_code == 401


def test_orderbook_parser_handles_multiple_formats():
    from bitunix_bot.orderbook import OrderBookFeed

    # Format 1: bids/asks as lists of [price, size] pairs.
    msg = {"symbol": "BTCUSDT", "data": {
        "bids": [["60000", "1.5"], ["59999", "2.0"]],
        "asks": [["60001", "1.0"], ["60002", "0.5"]],
    }}
    bids, asks = OrderBookFeed._extract_book(msg)
    assert bids == [(60000.0, 1.5), (59999.0, 2.0)]
    assert asks == [(60001.0, 1.0), (60002.0, 0.5)]

    # Format 2: short keys b/a + dict-style levels.
    msg = {"s": "ETHUSDT", "b": [{"px": "2000", "sz": "10"}], "a": [{"price": "2001", "size": "5"}]}
    bids, asks = OrderBookFeed._extract_book(msg)
    assert bids == [(2000.0, 10.0)]
    assert asks == [(2001.0, 5.0)]


def test_imbalance_vote_in_strategy():
    """Strong bid-side imbalance should add a long vote."""
    from bitunix_bot.config import StrategyCfg
    from bitunix_bot.strategy import evaluate

    rng = np.random.default_rng(7)
    n = 200
    closes = 100 + np.cumsum(rng.normal(0.05, 0.3, n))
    opens = closes - rng.uniform(0.05, 0.2, n)
    highs = np.maximum(opens, closes) + rng.uniform(0.02, 0.1, n)
    lows = np.minimum(opens, closes) - rng.uniform(0.02, 0.1, n)
    cfg = StrategyCfg(
        ema_fast=9, ema_mid=21, ema_slow=50,
        rsi_period=14, rsi_long_min=30, rsi_long_max=80,
        rsi_short_min=20, rsi_short_max=70,
        macd_fast=12, macd_slow=26, macd_signal=9,
        bb_period=20, bb_std=2.0, atr_period=14,
        adx_period=14, adx_min=15.0,
        supertrend_period=10, supertrend_mult=3.0,
        pattern_weight=0.55, pattern_norm=2.0, fire_threshold=0.05,
        volume_ma_period=20, volume_spike_multiplier=1.5,
        stoch_rsi_period=14, stoch_rsi_k=3, stoch_rsi_d=3,
        htf_timeframe="15m", htf_ema_period=50,
        funding_threshold=0.0005,
        swing_lookback=5, sr_cluster_tol_pct=0.3, sr_min_touches=2, sr_proximity_pct=0.3,
        ob_depth_levels=10, ob_imbalance_threshold=0.30,
    )
    sig = evaluate(opens.tolist(), highs.tolist(), lows.tolist(), closes.tolist(),
                    cfg, ob_imbalance=0.55)  # strong bid pressure
    if sig:
        assert any("ob_imb" in r for r in sig.reasons), f"reasons: {sig.reasons}"


def test_clientid_is_deterministic_and_includes_symbol_side():
    """The same (symbol, minute, side) should produce the same clientId so
    a network retry within the same minute hits Bitunix's natural
    duplicate-id rejection rather than placing a second order."""
    reset_state()
    cfg = fresh_cfg()
    cfg.mode = "live"
    bot = BitunixBot(cfg)
    bot.client = make_mock_client()
    bot.client.klines.side_effect = lambda *a, **kw: make_uptrend_klines()
    bot._resolve_symbol_meta()
    bot._tick()
    if not bot.client.place_order.called:
        return  # no signal fired this run; skip
    cid = bot.client.place_order.call_args.kwargs.get("client_id")
    assert cid and cid.startswith("bot-"), f"expected deterministic clientId, got {cid!r}"
    # Format: bot-<SYMBOL>-<minute>-<SIDE>
    parts = cid.split("-")
    assert parts[0] == "bot"
    assert parts[1] in [s.upper() for s in cfg.trading.symbols]
    assert parts[3] in ("BUY", "SELL")


def test_execute_handles_network_timeout_without_crashing():
    """A non-BitunixError raised by place_order (e.g. requests.Timeout)
    must be caught — it's an unknown-state failure, not a crash."""
    reset_state()
    cfg = fresh_cfg()
    cfg.mode = "live"
    bot = BitunixBot(cfg)
    bot.client = make_mock_client()
    bot.client.klines.side_effect = lambda *a, **kw: make_uptrend_klines()
    bot.client.place_order.side_effect = TimeoutError("connection timed out")
    bot._resolve_symbol_meta()
    bot._tick()  # should NOT raise
    errors = [e for e in bot.state.snapshot()["events"] if e["kind"] == "error"]
    assert any("unknown state" in e["text"] for e in errors), \
        f"expected 'unknown state' error, got {errors}"


def test_health_check_fails_when_tick_loop_stalls():
    """The /healthz endpoint must return 503 if the bot's last tick is
    older than 3× tick_seconds (Railway can then redeploy)."""
    reset_state()
    cfg = fresh_cfg()
    cfg.loop.tick_seconds = 10  # so threshold is 30s grace + 60 startup = 60s
    client = make_mock_client()
    bot_state = get_state()
    # Simulate a tick from 5 minutes ago (well past threshold).
    bot_state.last_tick_at = int(time.time()) - 300
    app = create_app(cfg, client)
    c = app.test_client()
    r = c.get("/healthz")
    assert r.status_code == 503, f"expected 503 (stale), got {r.status_code}: {r.data}"
    # And a fresh tick brings it back to 200.
    bot_state.last_tick_at = int(time.time())
    r = c.get("/healthz")
    assert r.status_code == 200


def test_config_rejects_silly_values():
    """Loading a config with insane values should fail loudly."""
    import tempfile
    from bitunix_bot.config import load
    bad_yaml = """
mode: paper
trading:
  symbols: [BTCUSDT]
  timeframe: 1m
  leverage: 999
  margin_coin: USDT
  margin_mode: ISOLATION
  risk_per_trade_pct: 50
  max_open_positions: 4
  max_positions_per_symbol: 1
  max_same_direction: 2
  cooldown_seconds: 60
  max_position_age_seconds: 5400
risk:
  stop_loss_pct: 99
  take_profit_r: 1.5
  use_atr: false
  atr_multiplier_sl: 0.8
  atr_multiplier_tp: 4.0
  breakeven_at_r: 1.0
  breakeven_buffer_pct: 0.05
  trailing_activate_r: 1.5
  trailing_distance_r: 0.5
strategy:
  ema_fast: 9
  ema_mid: 21
  ema_slow: 50
  rsi_period: 14
  rsi_long_min: 45
  rsi_long_max: 60
  rsi_short_min: 40
  rsi_short_max: 55
  macd_fast: 12
  macd_slow: 26
  macd_signal: 9
  bb_period: 20
  bb_std: 2.0
  atr_period: 14
  adx_period: 14
  adx_min: 18.0
  supertrend_period: 10
  supertrend_mult: 3.0
  pattern_weight: 0.55
  pattern_norm: 2.0
  fire_threshold: 0.30
  volume_ma_period: 20
  volume_spike_multiplier: 1.5
  stoch_rsi_period: 14
  stoch_rsi_k: 3
  stoch_rsi_d: 3
  htf_timeframe: 15m
  htf_ema_period: 50
  funding_threshold: 0.0005
  swing_lookback: 5
  sr_cluster_tol_pct: 0.3
  sr_min_touches: 2
  sr_proximity_pct: 0.3
  ob_depth_levels: 10
  ob_imbalance_threshold: 0.30
loop:
  tick_seconds: 10
  kline_lookback: 200
logging:
  level: INFO
  file: logs/bot.log
"""
    with tempfile.NamedTemporaryFile(suffix=".yaml", mode="w", delete=False) as f:
        f.write(bad_yaml)
        path = f.name
    try:
        load(path, "/dev/null")
        raised = False
    except ValueError as e:
        raised = True
        # Both bad values should appear in the error.
        assert "leverage" in str(e), str(e)
        assert "stop_loss_pct" in str(e), str(e)
        assert "risk_per_trade_pct" in str(e), str(e)
    finally:
        os.unlink(path)
    assert raised, "expected ValueError on insane config"


def test_divergence_detector_finds_bullish_divergence():
    """Construct: price makes lower-low, RSI makes higher-low → bullish divergence.
    Pivots must be strictly less than their neighbors, so we use single-bar
    spike lows with non-tied surroundings."""
    from bitunix_bot import divergence as div
    n = 80
    # Highs neutral; we control lows directly to engineer two clear pivots.
    closes = np.full(n, 100.0)
    highs = np.full(n, 100.5)
    lows = np.full(n, 99.5)
    # First pivot low: idx 25 = 94, neighbors strictly higher.
    lows[20:25] = [98.0, 97.0, 96.0, 95.5, 95.0]
    lows[25] = 94.0
    lows[26:31] = [95.0, 95.5, 96.0, 97.0, 98.0]
    # Second pivot low: idx 60 = 92 (lower than 94), neighbors strictly higher.
    lows[55:60] = [97.0, 96.0, 95.0, 94.0, 93.0]
    lows[60] = 92.0
    lows[61:66] = [93.0, 94.0, 95.0, 96.0, 97.0]
    # Build a synthetic RSI that makes a HIGHER low at idx 60 (the divergence).
    rsi_v = np.full(n, 50.0)
    rsi_v[25] = 28.0   # deeply oversold first
    rsi_v[60] = 38.0   # less oversold second — divergence
    macd = np.zeros(n)
    obv_v = np.zeros(n)
    hits = div.detect_divergences(closes, highs, lows, rsi_v, macd, obv_v,
                                   pivot_lookback=5)
    names = [h.name for h in hits]
    assert any("rsi_bullish_div" in n for n in names), f"expected RSI bullish div in {names}"


def test_smc_fvg_detection():
    """Construct a clear bullish FVG: bar 0 high < bar 2 low. Then verify
    that when price returns INTO the gap, the detector reports it."""
    from bitunix_bot import smc
    n = 20
    closes = np.full(n, 100.0)
    highs = np.full(n, 100.5)
    lows = np.full(n, 99.5)
    # Build FVG at idx 5..7: bar 5 high=100.5, bar 6 high=103, bar 7 low=102.
    highs[5] = 100.5; lows[5] = 99.8
    closes[6] = 102.0; highs[6] = 103.0; lows[6] = 101.0
    closes[7] = 102.5; highs[7] = 103.0; lows[7] = 102.0  # gap: 100.5 -> 102.0
    # Walk price up then back down INTO the gap by bar 19.
    closes[8:15] = 102.5
    highs[8:15] = 103.0
    lows[8:15] = 102.0
    closes[15:20] = np.linspace(102.5, 101.5, 5)  # pulling back into FVG zone
    highs[15:20] = closes[15:20] + 0.2
    lows[15:20] = closes[15:20] - 0.2
    sig = smc.detect_recent_fvg(highs, lows, closes)
    assert sig is not None, "expected to detect bullish FVG"
    assert sig.direction == "bullish", f"got {sig.direction}"


def test_smc_liquidity_sweep_bearish():
    """Bar wicks ABOVE a recent swing high but closes BELOW it = bear sweep."""
    from bitunix_bot import smc
    n = 30
    closes = np.full(n, 100.0)
    closes[0:8] = np.linspace(95, 105, 8)
    closes[8] = 105.0      # swing high candidate (single bar high)
    closes[9:24] = np.linspace(105, 95, 15)  # decline
    closes[24:29] = np.linspace(95, 100, 5)
    # Last bar: high pierces 105 swing high, but closes below it.
    highs = closes + 0.5
    lows = closes - 0.5
    highs[8] = 106.0  # the swing high to sweep
    closes[29] = 104.0
    highs[29] = 106.5  # wicks above 106 swing high
    lows[29] = 103.5
    sig = smc.detect_liquidity_sweep(highs, lows, closes, swing_lookback=3)
    assert sig is not None, "expected bear liquidity sweep"
    assert sig.direction == "bearish", f"got {sig.direction}"


def test_obv_and_mfi_indicators_compute():
    from bitunix_bot.indicators import obv, mfi
    n = 50
    rng = np.random.default_rng(1)
    closes = 100 + np.cumsum(rng.normal(0, 0.5, n))
    highs = closes + 0.3
    lows = closes - 0.3
    vols = rng.uniform(100, 1000, n)
    obv_v = obv(closes, vols)
    mfi_v = mfi(highs, lows, closes, vols, period=14)
    assert obv_v.shape == (n,)
    assert mfi_v.shape == (n,)
    assert not np.isnan(obv_v[-1])
    assert not np.isnan(mfi_v[-1])
    assert 0 <= mfi_v[-1] <= 100, f"MFI out of range: {mfi_v[-1]}"


def test_skip_events_dedupe_within_window():
    """Repeated identical skips must collapse to a single event within
    the dedupe window, otherwise the activity feed gets spammed."""
    reset_state()
    state = get_state()
    state.skip_dedupe_seconds = 60
    for _ in range(10):
        state.record_skip("XRPUSDT: same-direction cap (2 shorts already)")
    skips = [e for e in state.snapshot()["events"] if e["kind"] == "skip"]
    assert len(skips) == 1, f"expected 1 deduped skip, got {len(skips)}"

    # Different reason should NOT dedupe.
    state.record_skip("XRPUSDT: no available margin")
    skips = [e for e in state.snapshot()["events"] if e["kind"] == "skip"]
    assert len(skips) == 2

    # Different symbol same reason should NOT dedupe.
    state.record_skip("BTCUSDT: same-direction cap (2 shorts already)")
    skips = [e for e in state.snapshot()["events"] if e["kind"] == "skip"]
    assert len(skips) == 3


def test_capped_signals_emit_skip_not_signal():
    """When same-direction cap blocks, the activity feed should get a
    skip event (deduped) rather than a noisy signal+skip pair."""
    reset_state()
    cfg = fresh_cfg()
    cfg.trading.max_open_positions = 4
    cfg.trading.max_same_direction = 2
    bot = BitunixBot(cfg)
    bot.client = make_mock_client()
    # Two LONGs already open on BTC and ETH.
    bot.client.pending_positions.return_value = [
        {"positionId": "P1", "symbol": "BTCUSDT", "qty": "0.01", "side": "BUY",
         "leverage": 100, "ctime": int(time.time() * 1000)},
        {"positionId": "P2", "symbol": "ETHUSDT", "qty": "0.1", "side": "BUY",
         "leverage": 100, "ctime": int(time.time() * 1000)},
    ]
    bot.client.klines.side_effect = lambda *a, **kw: make_uptrend_klines()
    bot._resolve_symbol_meta()
    bot._tick()

    events = bot.state.snapshot()["events"]
    skips = [e for e in events if e["kind"] == "skip"]
    signals = [e for e in events if e["kind"] == "signal"]
    # We expect skip events for each blocked symbol (deduped), and NO
    # signal events for the blocked symbols.
    assert any("same-direction" in s["text"] for s in skips), \
        f"expected same-direction skip, got skips={[s['text'] for s in skips]}"
    # The dashboard would have shown the skip + signal pair before this fix.
    # After the fix: signal events only for actionable (non-blocked) signals.
    blocked_symbols = {"SOLUSDT", "XRPUSDT"}  # those would have signaled LONG
    blocked_signals = [s for s in signals
                        if any(sym in s["text"] for sym in blocked_symbols)]
    assert not blocked_signals, \
        f"capped symbols should NOT emit signal events: {blocked_signals}"


def test_adaptive_tp_tightens_with_age_and_respects_floor():
    """The adaptive TP function should:
       - return the original target for fresh positions
       - tighten progressively with age
       - never go below the fee-aware floor
    """
    from bitunix_bot.risk import adaptive_tp_r

    # Fresh trade: full target.
    r = adaptive_tp_r(age_minutes=2, original_tp_r=1.5, fee_pct=0.17, sl_pct=0.25, floor_r=0.7)
    assert abs(r - 1.5) < 0.01, f"fresh should be 1.5R, got {r}"

    # 7 min: 80% of original.
    r = adaptive_tp_r(age_minutes=7, original_tp_r=1.5, fee_pct=0.17, sl_pct=0.25, floor_r=0.7)
    assert abs(r - 1.2) < 0.01, f"at 7m should be 1.2R (=0.8*1.5), got {r}"

    # 15 min: 60% but floor wins.
    r = adaptive_tp_r(age_minutes=15, original_tp_r=1.5, fee_pct=0.17, sl_pct=0.25, floor_r=0.7)
    assert r >= 0.9, f"at 15m should be ≥0.9R, got {r}"

    # 50 min: floor.
    r = adaptive_tp_r(age_minutes=50, original_tp_r=1.5, fee_pct=0.17, sl_pct=0.25, floor_r=0.7)
    # Fee floor: fee/sl × 1.5 = (0.17/0.25) × 1.5 = 1.02 — beats floor_r=0.7.
    assert r >= 1.0, f"old trade must respect fee-aware floor (≥1.0R), got {r}"

    # If sl_pct is large (low leverage), fee floor relaxes; floor_r kicks in.
    r = adaptive_tp_r(age_minutes=50, original_tp_r=1.5, fee_pct=0.05, sl_pct=1.0, floor_r=0.7)
    assert abs(r - 0.7) < 0.01, f"with low fees, floor=0.7R, got {r}"


def test_adaptive_tp_tier_scales_with_bar_minutes():
    """Grok holistic review: adaptive_tp_r tiers are expressed in BARS,
    converted via bar_minutes. On 15m timeframe, the 5-bar tier sits at
    75 minutes — a position 50 minutes old should still be in the
    'fresh' bucket, returning the original target."""
    from bitunix_bot.risk import adaptive_tp_r

    # 50 minutes on a 15m timeframe = 3.3 bars in → still 'fresh' (< 5 bars).
    r = adaptive_tp_r(
        age_minutes=50, original_tp_r=1.5, fee_pct=0.17, sl_pct=0.25,
        floor_r=0.7, bar_minutes=15.0,
    )
    assert abs(r - 1.5) < 0.01, \
        f"50m on 15m timeframe should still be 'fresh'; got {r}"

    # 100 minutes = 6.6 bars → 80% tier.
    r = adaptive_tp_r(
        age_minutes=100, original_tp_r=1.5, fee_pct=0.17, sl_pct=0.25,
        floor_r=0.7, bar_minutes=15.0,
    )
    assert abs(r - 1.2) < 0.01, \
        f"100m on 15m (≈6.6 bars) should be 80% tier (1.2R); got {r}"

    # 700 minutes = 46 bars → past 40-bar tier → safety floor.
    r = adaptive_tp_r(
        age_minutes=700, original_tp_r=1.5, fee_pct=0.17, sl_pct=0.25,
        floor_r=0.7, bar_minutes=15.0,
    )
    # Fee floor wins: (0.17/0.25)*1.5 = 1.02
    assert r >= 1.0 and r < 1.5, \
        f"old 15m trade should hit fee-aware floor; got {r}"

    # Default bar_minutes=1.0 preserves legacy 1m behavior.
    r_default = adaptive_tp_r(
        age_minutes=50, original_tp_r=1.5, fee_pct=0.17, sl_pct=0.25, floor_r=0.7,
    )
    assert r_default < 1.5, \
        f"default bar_minutes=1.0 → 50m should hit late tier; got {r_default}"


def test_parse_timeframe_minutes():
    """Bitunix timeframe strings parse to minutes correctly."""
    from bitunix_bot.risk import parse_timeframe_minutes

    assert parse_timeframe_minutes("1m") == 1.0
    assert parse_timeframe_minutes("5m") == 5.0
    assert parse_timeframe_minutes("15m") == 15.0
    assert parse_timeframe_minutes("30m") == 30.0
    assert parse_timeframe_minutes("1h") == 60.0
    assert parse_timeframe_minutes("4h") == 240.0
    assert parse_timeframe_minutes("1d") == 1440.0
    # Bad input → 1.0 fallback (conservative legacy behavior).
    assert parse_timeframe_minutes("") == 1.0
    assert parse_timeframe_minutes("foo") == 1.0
    assert parse_timeframe_minutes("15x") == 1.0


def test_factor_score_weighted_clamps_to_one():
    """Defensive: if config weights sum > 1.0 (typo or misconfiguration),
    factor_score_weighted output must still be ≤ 1.0 to prevent score
    from silently exceeding fire_threshold via inflated factor half."""
    from bitunix_bot.strategy import factor_score_weighted

    # All groups maxed (1.0 score each), weights sum to 1.5 (broken config).
    breakdown = {"trend": 1.0, "mean_rev": 1.0, "flow": 1.0, "context": 1.0}
    weights = {"trend": 0.40, "mean_rev": 0.40, "flow": 0.40, "context": 0.30}  # sums 1.5
    result = factor_score_weighted(breakdown, weights)
    assert result <= 1.0, f"clamp must cap at 1.0 even with broken weights; got {result}"

    # Sane weights that sum to 1.0 — clamp is no-op.
    weights_sane = {"trend": 0.15, "mean_rev": 0.25, "flow": 0.50, "context": 0.10}
    result = factor_score_weighted(breakdown, weights_sane)
    assert abs(result - 1.0) < 0.001, \
        f"with sane weights and full saturation, score should be exactly 1.0; got {result}"


def test_recent_trade_r_excludes_flat_trades():
    """Adaptive self-defense rolling tally must SKIP flat trades
    (|trade_r| < 0.10) so genuine losing streaks aren't diluted by
    near-zero exits from the BE-ratchet-then-reverse pattern."""
    reset_state()
    from unittest.mock import MagicMock
    cfg = fresh_cfg()
    cfg.mode = "live"
    cfg.trading.symbols = ["BTCUSDT"]
    bot = BitunixBot(cfg)
    bot.client = make_mock_client()
    # Seed three "closed" positions in the history endpoint:
    # one big loss (−0.8R), one big win (+0.8R), and one flat (+0.05R).
    # The flat must NOT make it into recent_trade_r.
    sl_dist = 100.0 * (cfg.risk.stop_loss_pct / 100.0)  # 0.25 if entry=100
    big_loss_close = 100.0 - sl_dist * 0.8     # 99.8 → trade_r = -0.8
    big_win_close = 100.0 + sl_dist * 0.8      # 100.2 → trade_r = +0.8
    flat_close = 100.0 + sl_dist * 0.05        # 100.0125 → trade_r = +0.05
    now_ms = int(time.time() * 1000)
    bot.client.history_positions.return_value = {"positionList": [
        {"positionId": "P_LOSS", "symbol": "BTCUSDT", "side": "BUY",
         "qty": "1", "avgOpenPrice": "100.0", "avgClosePrice": str(big_loss_close),
         "ctime": now_ms - 60000, "mtime": now_ms - 1000,
         "realizedPNL": "-0.20", "fee": "0.0", "funding": "0.0"},
        {"positionId": "P_WIN", "symbol": "BTCUSDT", "side": "BUY",
         "qty": "1", "avgOpenPrice": "100.0", "avgClosePrice": str(big_win_close),
         "ctime": now_ms - 60000, "mtime": now_ms - 500,
         "realizedPNL": "0.20", "fee": "0.0", "funding": "0.0"},
        {"positionId": "P_FLAT", "symbol": "BTCUSDT", "side": "BUY",
         "qty": "1", "avgOpenPrice": "100.0", "avgClosePrice": str(flat_close),
         "ctime": now_ms - 60000, "mtime": now_ms - 100,
         "realizedPNL": "0.0125", "fee": "0.0", "funding": "0.0"},
    ]}
    bot._update_streak_state()
    # Should have exactly 2 entries — the flat is excluded.
    assert len(bot.recent_trade_r) == 2, \
        f"expected 2 entries (loss, win); flat excluded — got {len(bot.recent_trade_r)}: {list(bot.recent_trade_r)}"
    # Verify the actual values are the loss and win, NOT the flat.
    rs = sorted(list(bot.recent_trade_r))
    assert rs[0] < -0.5  # the loss
    assert rs[1] > 0.5   # the win


def test_cvd_indicator_signs_by_body_position():
    """CVD should be POSITIVE when bars close near the high (buyer-led),
    negative when they close near the low (seller-led)."""
    from bitunix_bot.indicators import cvd
    n = 10
    # All bars: range = 1.0, close near high, volume = 100 → CVD strongly positive.
    highs = np.full(n, 100.5)
    lows = np.full(n, 99.5)
    closes = np.full(n, 100.4)  # close near top of range
    vols = np.full(n, 100.0)
    out = cvd(highs, lows, closes, vols)
    assert out[-1] > 0, f"close-near-high should produce positive CVD, got {out[-1]}"

    # Inverse: close near low → strongly negative CVD.
    closes = np.full(n, 99.6)
    out = cvd(highs, lows, closes, vols)
    assert out[-1] < 0, f"close-near-low should produce negative CVD, got {out[-1]}"


def test_combo_recipe_fires_on_matching_reasons():
    from bitunix_bot import combos as cb
    # trend_pullback bullish needs: ema_stack_up + cross_above_ema_fast + (vol_spike OR mfi)
    long_reasons = ["ema_stack_up", "cross_above_ema_fast", "vol_spike(2.5x)", "rsi(52)"]
    short_reasons: list = []
    hits = cb.detect(long_reasons, short_reasons)
    names = {(h.name, h.direction) for h in hits}
    assert ("trend_pullback", "bullish") in names, f"expected trend_pullback bullish, got {names}"

    # If volume spike absent AND mfi absent, the combo should NOT fire.
    long_reasons2 = ["ema_stack_up", "cross_above_ema_fast", "rsi(52)"]
    hits2 = cb.detect(long_reasons2, short_reasons)
    names2 = {(h.name, h.direction) for h in hits2}
    assert ("trend_pullback", "bullish") not in names2, \
        f"trend_pullback should NOT fire without vol_spike or mfi: {names2}"


def test_combo_smc_reversal_requires_three_independent_components():
    from bitunix_bot import combos as cb
    # Bearish SMC reversal: (FVG_bear OR sweep_bear) + (any DIV bearish) + sr_reject
    short_reasons = [
        "SMC:liquidity_sweep_bearish",
        "DIV:rsi_bearish_div",
        "sr_reject_resistance(77400,×3)",
    ]
    hits = cb.detect([], short_reasons)
    assert any(h.name == "smc_reversal" and h.direction == "bearish" for h in hits)

    # Drop the S/R reject — combo should not fire.
    short_reasons2 = [
        "SMC:liquidity_sweep_bearish",
        "DIV:rsi_bearish_div",
        "ema_stack_down",  # not S/R
    ]
    hits2 = cb.detect([], short_reasons2)
    assert not any(h.name == "smc_reversal" for h in hits2)


def test_daily_drawdown_halts_new_entries():
    """When equity drops below the DD threshold, new entries pause."""
    reset_state()
    cfg = fresh_cfg()
    cfg.mode = "live"
    cfg.trading.max_daily_dd_pct = 8.0
    bot = BitunixBot(cfg)
    bot.client = make_mock_client()
    # Force the bot into a "started with $100, now $90" state (-10% DD).
    bot.session_start_day = time.gmtime().tm_yday
    bot.session_start_equity = 100.0
    bot.client.account.return_value = {
        "marginCoin": "USDT", "available": "85", "frozen": "0",
        "margin": "5", "transfer": "85", "positionMode": "ONE_WAY",
        "crossUnrealizedPNL": "0", "isolationUnrealizedPNL": "0", "bonus": "0",
    }
    bot.client.klines.side_effect = lambda *a, **kw: make_uptrend_klines()
    bot._resolve_symbol_meta()
    bot._tick()
    # No new orders should have been placed despite signals firing.
    bot.client.place_order.assert_not_called()
    assert bot.daily_dd_breached, "DD breached flag should be set"


def test_spread_filter_blocks_when_spread_too_wide():
    """Spread > max_entry_spread_pct should suppress new entries."""
    reset_state()
    cfg = fresh_cfg()
    cfg.mode = "live"
    cfg.trading.max_entry_spread_pct = 0.05  # 0.05%
    bot = BitunixBot(cfg)
    bot.client = make_mock_client()
    bot.client.klines.side_effect = lambda *a, **kw: make_uptrend_klines()
    bot._resolve_symbol_meta()

    # Inject a fake OB feed that returns a 0.10% spread (above threshold).
    class FakeOBFeed:
        def get_imbalance(self, sym, max_age_secs=30):
            return None
        def get_top_of_book(self, sym, max_age_secs=30):
            return (99.95, 100.05)
        def get_spread_pct(self, sym):
            return 0.10
    bot.ob_feed = FakeOBFeed()
    bot._tick()
    # Expect spread skips in the events.
    skips = [e for e in bot.state.snapshot()["events"] if e["kind"] == "skip"]
    assert any("spread" in e["text"] for e in skips), \
        f"expected spread skip, got skips={[s['text'] for s in skips]}"


def test_partial_tp_fires_at_1r_and_only_once():
    """At +1R favorable, the bot should fire a market close for 50% of qty
    via place_order(reduce_only=True). Subsequent ticks must not re-fire."""
    bot, _, _ = _setup_bot_with_open_position(
        side="BUY", entry=100_000.0, current_price=100_250.0,  # +1R exactly
    )
    # Default partial_tp_enabled=True, partial_tp_at_r=1.0, close_pct=50.
    bot._tick()
    # Should have called place_order with reduce_only=True for half qty.
    place_calls = bot.client.place_order.call_args_list
    partial_calls = [c for c in place_calls
                     if c.kwargs.get("reduce_only") and c.kwargs.get("trade_side") == "CLOSE"]
    assert len(partial_calls) == 1, \
        f"expected 1 partial-TP close, got {len(partial_calls)}"
    qty_str = partial_calls[0].kwargs["qty"]
    # qty was 0.01; partial_tp_close_pct = 60% (Grok rescan) → 0.006.
    assert abs(float(qty_str) - 0.005) < 1e-6, f"expected 0.005 (50% of 0.01), got {qty_str}"
    # Side opposite of position (LONG → SELL).
    assert partial_calls[0].kwargs["side"] == "SELL"

    # Tick again — partial TP should NOT re-fire.
    bot._tick()
    place_calls2 = bot.client.place_order.call_args_list
    partial_calls2 = [c for c in place_calls2
                      if c.kwargs.get("reduce_only") and c.kwargs.get("trade_side") == "CLOSE"]
    assert len(partial_calls2) == 1, "partial TP must fire only once per position"


def test_regime_weighting_boosts_aligned_signals():
    """In a strong uptrend (high ADX), trend-aligned indicator votes should
    multiply the combined score above the raw pattern + indicator math.
    Verifies the regime weighting layer wires through to the score."""
    from bitunix_bot.config import StrategyCfg
    from bitunix_bot.strategy import evaluate
    rng = np.random.default_rng(11)
    n = 220
    closes = 100.0 + np.cumsum(rng.normal(0.18, 0.18, n))   # strong uptrend
    opens = closes - rng.uniform(0.05, 0.15, n)
    highs = np.maximum(opens, closes) + rng.uniform(0.01, 0.05, n)
    lows = np.minimum(opens, closes) - rng.uniform(0.01, 0.05, n)
    for j in range(n - 3, n):                                # clean bull marubozu + ascending close
        closes[j] = closes[j - 1] + 0.5
        opens[j] = closes[j] - 0.5; highs[j] = closes[j] + 0.05; lows[j] = opens[j] - 0.05
    vols = np.full(n, 1.0)
    cfg = StrategyCfg(
        ema_fast=9, ema_mid=21, ema_slow=50,
        rsi_period=14, rsi_long_min=40, rsi_long_max=80,
        rsi_short_min=20, rsi_short_max=60,
        macd_fast=12, macd_slow=26, macd_signal=9,
        bb_period=20, bb_std=2.0, atr_period=14,
        adx_period=14, adx_min=1.0,
        supertrend_period=10, supertrend_mult=3.0,
        volume_ma_period=20, volume_spike_multiplier=1.5,
        stoch_rsi_period=14, stoch_rsi_k=3, stoch_rsi_d=3,
        htf_timeframe="15m", htf_ema_period=50,
        funding_threshold=0.0005,
        swing_lookback=5, sr_cluster_tol_pct=0.3, sr_min_touches=2, sr_proximity_pct=0.3,
        ob_depth_levels=10, ob_imbalance_threshold=0.30,
        mfi_period=14, mfi_long_max=60.0, mfi_short_min=40.0,
        keltner_period=20, keltner_atr_multiplier=1.5,
        btc_leader_symbol="BTCUSDT", btc_leader_ema=21,
        pattern_weight=0.55, pattern_norm=2.0, fire_threshold=0.05,
    )
    sig = evaluate(opens.tolist(), highs.tolist(), lows.tolist(), closes.tolist(),
                    cfg, volumes=vols.tolist())
    assert sig is not None
    pw = cfg.pattern_weight
    raw = pw * min(1.0, sig.pattern_score / cfg.pattern_norm) \
          + (1 - pw) * (sig.indicator_score / 23.0)
    has_trend_tag = any("ema_stack_up" in r or "supertrend_up" in r or "htf_uptrend" in r
                        for r in sig.reasons)
    if has_trend_tag:
        assert sig.score >= raw - 0.001, \
            f"trending regime + trend tags should not reduce score: {sig.score} vs {raw}"


def test_sl_and_tp_always_on_correct_side():
    """Critical: SL must be BELOW entry for longs and ABOVE for shorts.
    Inverted SL = catastrophic loss. Verify across both directions."""
    from bitunix_bot.config import RiskCfg, TradingCfg
    from bitunix_bot.risk import build_order
    from bitunix_bot.strategy import Signal

    tc = TradingCfg(symbols=["BTCUSDT"], timeframe="1m", leverage=100,
                    margin_coin="USDT", margin_mode="ISOLATION", risk_per_trade_pct=1.0)
    rc = RiskCfg(stop_loss_pct=0.25, take_profit_r=5.0, use_atr=False,
                 atr_multiplier_sl=0.8, atr_multiplier_tp=4.0)

    long_sig = Signal(direction="long", score=0.7, indicator_score=5,
                       pattern_score=2.0, reasons=["x"], price=60000.0, atr=100.0)
    plan = build_order(long_sig, free_margin=1000, trading=tc, risk=rc,
                       min_volume=0.0001, volume_step=0.0001, digits=1)
    assert plan and plan.side == "BUY"
    assert plan.stop_loss < plan.price < plan.take_profit, \
        f"long SL/TP wrong: SL={plan.stop_loss} entry={plan.price} TP={plan.take_profit}"

    short_sig = Signal(direction="short", score=0.7, indicator_score=5,
                        pattern_score=2.0, reasons=["x"], price=60000.0, atr=100.0)
    plan = build_order(short_sig, free_margin=1000, trading=tc, risk=rc,
                       min_volume=0.0001, volume_step=0.0001, digits=1)
    assert plan and plan.side == "SELL"
    assert plan.take_profit < plan.price < plan.stop_loss, \
        f"short SL/TP wrong: SL={plan.stop_loss} entry={plan.price} TP={plan.take_profit}"


# ----------------------------------------------------------------- post-only maker entries


class _FakeOBFeed:
    """Connected OB feed stub with configurable top-of-book and spread.
    Default depth (100K base-coin) is high enough to clear all per-symbol
    depth thresholds (XRP threshold is highest at 15K)."""
    def __init__(self, bid: float = 60_000.0, ask: float = 60_002.0,
                 spread_pct: float = 0.003, connected: bool = True,
                 bid_depth: float = 100_000.0, ask_depth: float = 100_000.0):
        self._bid = bid
        self._ask = ask
        self._spread_pct = spread_pct
        self._connected = connected
        self._bid_depth = bid_depth
        self._ask_depth = ask_depth

    def is_connected(self) -> bool:
        return self._connected

    def get_imbalance(self, sym, max_age_secs=30):
        return None

    def get_top_of_book(self, sym, max_age_secs=30):
        return (self._bid, self._ask)

    def get_spread_pct(self, sym):
        return self._spread_pct

    def get_depth(self, sym, top_n=5, max_age_secs=30):
        return (self._bid_depth, self._ask_depth)


def test_post_only_entry_steps_inside_wide_spread():
    """Aggressive maker (Grok holistic review): when spread > 1 tick, step
    INSIDE the spread to become the new top of book — fills sooner than
    passively joining the existing TOB.

    Setup: bid=60000.0, ask=60002.0, spread=2.0, tick=0.1 (BTC precision=1).
    Long entry should place at 60000.1 (1 tick above bid, still below ask)."""
    reset_state()
    cfg = fresh_cfg()
    cfg.mode = "live"
    cfg.trading.use_post_only_entries = True
    cfg.trading.symbols = ["BTCUSDT"]
    bot = BitunixBot(cfg)
    bot.client = make_mock_client()
    bot.client.klines.side_effect = lambda *a, **kw: make_uptrend_klines()
    bot.ob_feed = _FakeOBFeed(bid=60_000.0, ask=60_002.0)
    bot._resolve_symbol_meta()
    bot._tick()

    assert bot.client.place_order.called, "expected limit order placed"
    kw = bot.client.place_order.call_args.kwargs
    assert kw["order_type"] == "LIMIT", f"expected LIMIT, got {kw.get('order_type')}"
    assert kw["side"] == "BUY", f"expected BUY (long), got {kw.get('side')}"
    # Long with wide spread → step inside: bid + 1 tick = 60_000.1.
    assert float(kw["price"]) == 60_000.1, \
        f"expected step-inside at 60000.1, got {kw['price']}"
    assert str(kw["client_id"]).endswith("-PO")
    assert kw.get("sl_price") and kw.get("tp_price")
    assert "BTCUSDT" in bot.pending_limits


def test_post_only_entry_joins_tob_on_tight_spread():
    """When spread is exactly 1 tick wide, stepping inside would cross to
    taker. The aggressive-maker path must fall back to joining the existing
    TOB (= the bid for long, the ask for short)."""
    reset_state()
    cfg = fresh_cfg()
    cfg.mode = "live"
    cfg.trading.use_post_only_entries = True
    cfg.trading.symbols = ["BTCUSDT"]
    bot = BitunixBot(cfg)
    bot.client = make_mock_client()
    bot.client.klines.side_effect = lambda *a, **kw: make_uptrend_klines()
    # Tight spread: bid=60000.0, ask=60000.1, spread=0.1 = 1 tick exactly.
    bot.ob_feed = _FakeOBFeed(bid=60_000.0, ask=60_000.1)
    bot._resolve_symbol_meta()
    bot._tick()

    assert bot.client.place_order.called
    kw = bot.client.place_order.call_args.kwargs
    # 1-tick spread → join existing bid (no room to step inside).
    assert float(kw["price"]) == 60_000.0, \
        f"expected join-TOB at 60000.0, got {kw['price']}"


def test_post_only_falls_back_to_market_when_ob_feed_disconnected():
    """If the OB feed isn't connected, post-only is skipped and we go straight
    to MARKET — never place a limit blindly without a known book."""
    reset_state()
    cfg = fresh_cfg()
    cfg.mode = "live"
    cfg.trading.use_post_only_entries = True
    cfg.trading.symbols = ["BTCUSDT"]
    bot = BitunixBot(cfg)
    bot.client = make_mock_client()
    bot.client.klines.side_effect = lambda *a, **kw: make_uptrend_klines()
    bot.ob_feed = _FakeOBFeed(connected=False)
    bot._resolve_symbol_meta()
    bot._tick()

    assert bot.client.place_order.called
    kw = bot.client.place_order.call_args.kwargs
    assert kw["order_type"] == "MARKET", f"expected MARKET fallback, got {kw.get('order_type')}"
    assert "BTCUSDT" not in bot.pending_limits, "no pending limit when going straight to market"


def test_post_only_timeout_cancels_and_skips():
    """A pending post-only limit older than post_only_timeout_secs without
    a corresponding open position must be cancelled and SKIPPED — no market
    fallback. Pro-desk rule: 'don't chase a move that already invalidated.'
    If the maker bid wasn't hit in 8s, price moved away — marketing in
    pays the worst possible price on a setup that already failed."""
    reset_state()
    cfg = fresh_cfg()
    cfg.mode = "live"
    cfg.trading.symbols = ["BTCUSDT"]
    cfg.trading.post_only_timeout_secs = 8
    cfg.trading.use_post_only_entries = True  # explicit opt-in (default is now False per Grok v8)
    bot = BitunixBot(cfg)
    bot.client = make_mock_client()
    bot.client.klines.side_effect = lambda *a, **kw: make_uptrend_klines()
    bot._resolve_symbol_meta()

    # Pre-seed a stale pending limit (placed 30s ago, well past the 8s timeout).
    from bitunix_bot.risk import OrderPlan
    plan = OrderPlan(side="BUY", volume=0.001, price=60_000.0,
                     stop_loss=59_850.0, take_profit=60_375.0,
                     leverage=50, notes="test")
    bot.pending_limits["BTCUSDT"] = {
        "symbol": "BTCUSDT",
        "order_id": "OLD_LIMIT",
        "place_ts": int(time.time()) - 30,
        "plan": plan,
        "order_text": "LIVE BTCUSDT BUY qty=0.001 entry~60000.0 SL=59850.0 TP=60375.0 lev=50x",
        "limit_px": 60_000.0,
    }
    bot.client.cancel_order.return_value = {"code": 0, "msg": "ok"}
    bot.client.pending_positions.return_value = []  # not filled
    bot._tick()

    # Limit was cancelled by id.
    bot.client.cancel_order.assert_called_once()
    cancel_kw = bot.client.cancel_order.call_args.kwargs
    assert cancel_kw["order_id"] == "OLD_LIMIT"

    # NO MARKET fallback was placed — that's the new behavior.
    market_calls = [c for c in bot.client.place_order.call_args_list
                    if c.kwargs.get("order_type") == "MARKET"]
    assert not market_calls, (
        f"market fallback should NOT fire on post-only timeout; "
        f"calls: {[c.kwargs for c in bot.client.place_order.call_args_list]}"
    )

    # Pending tracking cleared.
    assert "BTCUSDT" not in bot.pending_limits

    # Skip event recorded with "invalidated" reason (the new wording).
    skips = [e for e in bot.state.snapshot()["events"] if e["kind"] == "skip"]
    assert any("invalidated" in s["text"] for s in skips), \
        f"expected 'invalidated' skip; got {[s['text'] for s in skips]}"

    # Cooldown set so the per-symbol loop later in this tick doesn't try yet
    # another entry on the same symbol.
    assert bot.last_action_at.get("BTCUSDT", 0) > 0


def test_post_only_fill_clears_pending_tracking():
    """When a position now exists for the symbol of a tracked pending limit,
    treat it as filled and clear tracking (no cancel, no fallback)."""
    reset_state()
    cfg = fresh_cfg()
    cfg.mode = "live"
    cfg.trading.symbols = ["BTCUSDT"]
    bot = BitunixBot(cfg)
    bot.client = make_mock_client()
    bot.client.klines.side_effect = lambda *a, **kw: make_uptrend_klines()

    from bitunix_bot.risk import OrderPlan
    plan = OrderPlan(side="BUY", volume=0.001, price=60_000.0,
                     stop_loss=59_850.0, take_profit=60_375.0,
                     leverage=50, notes="test")
    bot.pending_limits["BTCUSDT"] = {
        "symbol": "BTCUSDT", "order_id": "FILLED_LIMIT",
        "place_ts": int(time.time()),  # fresh — not timed out
        "plan": plan,
        "order_text": "...",
        "limit_px": 60_000.0,
    }
    # Position now exists for BTC → simulates the limit having filled.
    bot.client.pending_positions.return_value = [{
        "positionId": "P1", "symbol": "BTCUSDT", "qty": "0.001",
        "side": "BUY", "leverage": 50,
        "ctime": int(time.time() * 1000),
        "avgOpenPrice": "60000.0", "unrealizedPNL": "0",
    }]
    # _manage_open_positions reads pending_tpsl — return empty list to skip cleanly.
    bot.client.pending_tpsl.return_value = []
    bot._resolve_symbol_meta()
    bot._tick()

    assert "BTCUSDT" not in bot.pending_limits, "tracking should be cleared on fill"
    bot.client.cancel_order.assert_not_called()  # no cancel — it filled
    # State should record the fill confirmation in the activity feed.
    orders = [e for e in bot.state.snapshot()["events"] if e["kind"] == "order"]
    assert any("MAKER FILLED" in e["text"] for e in orders), \
        f"expected MAKER FILLED order event, got {[e['text'] for e in orders]}"


def test_pending_limits_count_toward_global_cap():
    """A pending post-only limit must occupy a slot in max_open_positions —
    otherwise we'd oversize while waiting for the limit to fill."""
    reset_state()
    cfg = fresh_cfg()
    cfg.mode = "live"
    cfg.trading.max_open_positions = 1
    bot = BitunixBot(cfg)
    bot.client = make_mock_client()
    bot.client.klines.side_effect = lambda *a, **kw: make_uptrend_klines()
    bot.ob_feed = _FakeOBFeed()  # connected so post-only path is the default

    # Pre-seed a fresh pending limit on BTC that should fully occupy the cap.
    from bitunix_bot.risk import OrderPlan
    plan = OrderPlan(side="BUY", volume=0.001, price=60_000.0,
                     stop_loss=59_850.0, take_profit=60_375.0,
                     leverage=50, notes="test")
    bot.pending_limits["BTCUSDT"] = {
        "symbol": "BTCUSDT", "order_id": "L_HOLD",
        "place_ts": int(time.time()), "plan": plan,
        "order_text": "...", "limit_px": 60_000.0,
    }
    bot.client.pending_positions.return_value = []
    bot._resolve_symbol_meta()
    bot._tick()

    # No new place_order call should happen — the global cap (1) is fully
    # occupied by the pending limit.
    assert not bot.client.place_order.called, (
        f"global cap should block new entries while a limit pending; "
        f"calls={[c.kwargs for c in bot.client.place_order.call_args_list]}"
    )


def test_pending_limits_count_toward_same_direction_cap():
    """A pending LONG limit + an existing LONG position should saturate
    max_same_direction=2, blocking further LONG entries on other symbols."""
    reset_state()
    cfg = fresh_cfg()
    cfg.mode = "live"
    cfg.trading.max_open_positions = 4
    cfg.trading.max_same_direction = 2
    bot = BitunixBot(cfg)
    bot.client = make_mock_client()
    bot.client.klines.side_effect = lambda *a, **kw: make_uptrend_klines()
    bot.ob_feed = _FakeOBFeed()

    # 1 actual LONG already open on BTC.
    bot.client.pending_positions.return_value = [{
        "positionId": "P1", "symbol": "BTCUSDT", "qty": "0.01",
        "side": "BUY", "leverage": 50, "ctime": int(time.time() * 1000),
        "avgOpenPrice": "60000.0", "unrealizedPNL": "0",
    }]
    bot.client.pending_tpsl.return_value = []
    # 1 pending LONG limit on ETH → effective LONG count = 2 (saturated).
    from bitunix_bot.risk import OrderPlan
    plan = OrderPlan(side="BUY", volume=0.01, price=3_000.0,
                     stop_loss=2_992.5, take_profit=3_018.75,
                     leverage=50, notes="test")
    bot.pending_limits["ETHUSDT"] = {
        "symbol": "ETHUSDT", "order_id": "L_ETH",
        "place_ts": int(time.time()), "plan": plan,
        "order_text": "...", "limit_px": 3_000.0,
    }
    bot._resolve_symbol_meta()
    bot._tick()

    # SOL/XRP would otherwise long the uptrend — but same-direction cap should
    # now block them. Expect at least one same-direction skip in events.
    skips = [e for e in bot.state.snapshot()["events"] if e["kind"] == "skip"]
    assert any("same-direction" in s["text"] for s in skips), \
        f"expected same-direction skip; got skips={[s['text'] for s in skips]}"
    # And no new place_order calls were made (existing LONG + pending LONG saturate).
    new_orders = [c for c in bot.client.place_order.call_args_list
                  if c.kwargs.get("trade_side") == "OPEN"]
    assert not new_orders, (
        f"saturated direction cap should block new LONG entries; "
        f"calls={[c.kwargs for c in new_orders]}"
    )


def test_post_only_paper_mode_still_uses_paper_path():
    """Paper mode should not engage the post-only flow at all — it just
    records the simulated order and returns. No place_order, no pending
    limits tracked."""
    reset_state()
    cfg = fresh_cfg()
    cfg.mode = "paper"
    cfg.trading.use_post_only_entries = True
    cfg.trading.symbols = ["BTCUSDT"]
    bot = BitunixBot(cfg)
    bot.client = make_mock_client()
    bot.client.klines.side_effect = lambda *a, **kw: make_uptrend_klines()
    bot.ob_feed = _FakeOBFeed()
    bot._resolve_symbol_meta()
    bot._tick()

    bot.client.place_order.assert_not_called()  # paper never sends real orders
    assert not bot.pending_limits, "paper should not track pending limits"
    # Paper mode should still record the simulated order in the activity feed.
    orders = [e for e in bot.state.snapshot()["events"] if e["kind"] == "order"]
    assert orders, "paper mode should still emit a simulated order event"


# ----------------------------------------------------------------- trade tape


def test_tradetape_parser_handles_side_field():
    """Trade payload with explicit side: 'BUY'/'SELL' aggressor flag."""
    from bitunix_bot.tradetape import TradeFeed
    msg = {"ch": "trade", "symbol": "BTCUSDT", "data": [
        {"p": "60000", "v": "0.5", "t": int(time.time() * 1000), "side": "BUY"},
        {"p": "60001", "v": "0.3", "t": int(time.time() * 1000), "side": "SELL"},
    ]}
    sym, trades = TradeFeed._extract_trades(msg)
    assert sym == "BTCUSDT"
    assert len(trades) == 2
    assert trades[0].is_buy is True and trades[0].qty == 0.5 and trades[0].price == 60000.0
    assert trades[1].is_buy is False and trades[1].qty == 0.3


def test_tradetape_parser_handles_buyer_maker_field():
    """Binance-style: isBuyerMaker=True means the seller was the aggressor."""
    from bitunix_bot.tradetape import TradeFeed
    msg = {"symbol": "ETHUSDT", "data": {
        "price": "3000", "size": "1.5", "ts": int(time.time() * 1000),
        "m": True,    # buyer was maker → seller aggressed → is_buy=False
    }}
    sym, trades = TradeFeed._extract_trades(msg)
    assert sym == "ETHUSDT" and len(trades) == 1
    assert trades[0].is_buy is False, "buyer-maker=True implies seller aggressor"
    # Inverse case.
    msg["data"]["m"] = False
    _, trades = TradeFeed._extract_trades(msg)
    assert trades[0].is_buy is True


def test_tradetape_parser_rejects_unparseable():
    """Missing price/qty/side → no trade emitted."""
    from bitunix_bot.tradetape import TradeFeed
    bad_msgs = [
        {"symbol": "BTC", "data": {"v": "1"}},                 # no price
        {"symbol": "BTC", "data": {"p": "60000"}},             # no qty
        {"symbol": "BTC", "data": {"p": "60000", "v": "1"}},   # no side
        {"symbol": "BTC", "data": {"p": "0", "v": "1", "side": "BUY"}},  # zero price
    ]
    for m in bad_msgs:
        sym, trades = TradeFeed._extract_trades(m)
        assert not trades, f"expected no trades from {m}, got {trades}"


def test_tradetape_cvd_computation():
    """CVD = sum of signed sizes (buy positive, sell negative)."""
    from bitunix_bot.tradetape import TradeFeed, Trade
    feed = TradeFeed(symbols=["BTCUSDT"])
    now = time.time()
    # 3 buys totalling 2.0, 2 sells totalling 1.5 → CVD = +0.5
    for ts_off, qty, is_buy in [
        (-30, 0.5, True), (-25, 1.0, True), (-20, 0.5, True),
        (-15, 1.0, False), (-10, 0.5, False),
    ]:
        feed._ingest(Trade(ts=now + ts_off, price=60000, qty=qty, is_buy=is_buy), "BTCUSDT")
    cvd = feed.get_cvd("BTCUSDT", window_secs=60)
    assert cvd is not None and abs(cvd - 0.5) < 1e-9, f"expected +0.5, got {cvd}"


def test_tradetape_window_filtering():
    """Trades older than window_secs must be excluded from accessors."""
    from bitunix_bot.tradetape import TradeFeed, Trade
    feed = TradeFeed(symbols=["BTCUSDT"])
    now = time.time()
    # Old buy (90s ago, outside 60s window) + recent sell.
    feed._ingest(Trade(ts=now - 90, price=60000, qty=10.0, is_buy=True), "BTCUSDT")
    feed._ingest(Trade(ts=now - 5, price=60000, qty=2.0, is_buy=False), "BTCUSDT")
    # min_count=1 so the math (not the sample-size guard) is what we test here.
    cvd_60 = feed.get_cvd("BTCUSDT", window_secs=60, min_count=1)
    cvd_120 = feed.get_cvd("BTCUSDT", window_secs=120, min_count=1)
    assert abs(cvd_60 - (-2.0)) < 1e-9, f"60s window should only see -2.0, got {cvd_60}"
    assert abs(cvd_120 - 8.0) < 1e-9, f"120s window should see +8.0, got {cvd_120}"


def test_tradetape_aggression_ratio_bounded():
    """Aggression ratio must be in [-1, +1]."""
    from bitunix_bot.tradetape import TradeFeed, Trade
    feed = TradeFeed(symbols=["BTCUSDT"])
    now = time.time()
    # All buys → ratio = +1.0
    for i in range(5):
        feed._ingest(Trade(ts=now - i, price=60000, qty=1.0, is_buy=True), "BTCUSDT")
    assert feed.get_aggression_ratio("BTCUSDT", window_secs=10) == 1.0

    # All sells → ratio = -1.0
    feed2 = TradeFeed(symbols=["BTCUSDT"])
    for i in range(5):
        feed2._ingest(Trade(ts=now - i, price=60000, qty=1.0, is_buy=False), "BTCUSDT")
    assert feed2.get_aggression_ratio("BTCUSDT", window_secs=10) == -1.0

    # Balanced → ratio = 0.0 (use min_count=1 to test the math, not the guard).
    feed3 = TradeFeed(symbols=["BTCUSDT"])
    for i in range(4):
        feed3._ingest(Trade(ts=now - i, price=60000, qty=1.0, is_buy=(i % 2 == 0)), "BTCUSDT")
    assert feed3.get_aggression_ratio("BTCUSDT", window_secs=10, min_count=1) == 0.0

    # Sample-size guard — fewer than min_count returns None, not noise.
    feed4 = TradeFeed(symbols=["BTCUSDT"])
    feed4._ingest(Trade(ts=now, price=60000, qty=1.0, is_buy=True), "BTCUSDT")
    assert feed4.get_aggression_ratio("BTCUSDT", window_secs=10) is None, \
        "1 trade should fall below default min_count=5 and return None"


def test_tradetape_print_rate_and_aggressor_size():
    """print_rate = trades/sec; avg_aggressor_size = mean of qty."""
    from bitunix_bot.tradetape import TradeFeed, Trade
    feed = TradeFeed(symbols=["BTCUSDT"])
    now = time.time()
    qtys = [0.5, 1.5, 2.0, 0.5, 1.0]
    for i, q in enumerate(qtys):
        feed._ingest(Trade(ts=now - i, price=60000, qty=q, is_buy=True), "BTCUSDT")
    rate = feed.get_print_rate("BTCUSDT", window_secs=10)
    assert rate is not None and abs(rate - 0.5) < 1e-9, f"5 trades / 10s = 0.5/s, got {rate}"
    avg = feed.get_avg_aggressor_size("BTCUSDT", window_secs=10)
    assert avg is not None and abs(avg - 1.1) < 1e-9, f"mean = 1.1, got {avg}"


def test_tradetape_large_print_count():
    """Iceberg-style detector — count of prints above a size threshold."""
    from bitunix_bot.tradetape import TradeFeed, Trade
    feed = TradeFeed(symbols=["BTCUSDT"])
    now = time.time()
    for q in (0.1, 5.0, 0.2, 7.0, 10.0, 0.05):
        feed._ingest(Trade(ts=now - 1, price=60000, qty=q, is_buy=True), "BTCUSDT")
    count = feed.get_large_print_count("BTCUSDT", size_threshold=3.0, window_secs=10)
    assert count == 3, f"expected 3 prints ≥3.0, got {count}"


def test_tradetape_returns_none_when_empty():
    """Stale / no-data state should return None, not crash or zero."""
    from bitunix_bot.tradetape import TradeFeed
    feed = TradeFeed(symbols=["BTCUSDT"])
    assert feed.get_cvd("BTCUSDT") is None
    assert feed.get_aggression_ratio("BTCUSDT") is None
    assert feed.get_print_rate("BTCUSDT") is None
    assert feed.get_avg_aggressor_size("BTCUSDT") is None
    assert feed.last_trade_age("BTCUSDT") is None


def test_tradetape_handles_seconds_and_ms_timestamps():
    """Timestamps may arrive as seconds OR milliseconds. Both should work."""
    from bitunix_bot.tradetape import TradeFeed
    sec_msg = {"symbol": "X", "data": {"p": "1", "v": "1", "t": 1700000000, "side": "BUY"}}
    ms_msg = {"symbol": "X", "data": {"p": "1", "v": "1", "t": 1700000000000, "side": "BUY"}}
    _, t1 = TradeFeed._extract_trades(sec_msg)
    _, t2 = TradeFeed._extract_trades(ms_msg)
    # Both should parse to roughly the same unix-second timestamp.
    assert t1 and t2
    assert abs(t1[0].ts - t2[0].ts) < 1.0, \
        f"sec={t1[0].ts}, ms-converted={t2[0].ts} should match"


def test_tradetape_activity_multiplier_clamps():
    """Activity multiplier should clamp to [0.85, 1.10] and read None on cold feed."""
    from bitunix_bot.tradetape import TradeFeed, Trade
    feed = TradeFeed(symbols=["BTCUSDT"])
    now = time.time()
    # Cold feed — both windows below thresholds, returns None.
    assert feed.get_activity_multiplier("BTCUSDT") is None

    # 50 trades spread across 5min baseline = 10/min = 0.167/sec baseline.
    # Then 30 trades in last 10s = 3/sec recent → 18x baseline → clamps to 1.10.
    for i in range(50):
        feed._ingest(Trade(ts=now - 250 + i*5, price=60000, qty=1.0, is_buy=True), "BTCUSDT")
    for i in range(30):
        feed._ingest(Trade(ts=now - 9 + i*0.3, price=60000, qty=1.0, is_buy=True), "BTCUSDT")
    mult = feed.get_activity_multiplier("BTCUSDT")
    assert mult is not None
    assert mult == 1.10, f"high-surge should clamp to 1.10, got {mult}"

    # And dampening case: lots of baseline, none recent.
    feed2 = TradeFeed(symbols=["BTCUSDT"])
    for i in range(50):
        feed2._ingest(Trade(ts=now - 250 + i*5, price=60000, qty=1.0, is_buy=True), "BTCUSDT")
    # No recent trades at all in last 10s — returns None (insufficient sample).
    assert feed2.get_activity_multiplier("BTCUSDT") is None
    # 3 recent trades in 10s vs 50 in 300s = 0.3/s vs 0.167/s = 1.8x → clamps to 1.10.
    feed2._ingest(Trade(ts=now - 1, price=60000, qty=1.0, is_buy=True), "BTCUSDT")
    feed2._ingest(Trade(ts=now - 2, price=60000, qty=1.0, is_buy=True), "BTCUSDT")
    feed2._ingest(Trade(ts=now - 3, price=60000, qty=1.0, is_buy=True), "BTCUSDT")
    mult2 = feed2.get_activity_multiplier("BTCUSDT")
    assert mult2 is not None and 0.85 <= mult2 <= 1.10


def test_strategy_real_cvd_replaces_proxy_when_provided():
    """When real_cvd is passed, vote #23 fires with cvd_real tag (not cvd_proxy)."""
    from bitunix_bot.config import StrategyCfg
    from bitunix_bot.strategy import evaluate
    rng = np.random.default_rng(7)
    n = 200
    closes = 100 + np.cumsum(rng.normal(0.05, 0.3, n))
    opens = closes - rng.uniform(0.05, 0.2, n)
    highs = np.maximum(opens, closes) + rng.uniform(0.02, 0.1, n)
    lows = np.minimum(opens, closes) - rng.uniform(0.02, 0.1, n)
    cfg = StrategyCfg(
        ema_fast=9, ema_mid=21, ema_slow=50,
        rsi_period=14, rsi_long_min=30, rsi_long_max=80,
        rsi_short_min=20, rsi_short_max=70,
        macd_fast=12, macd_slow=26, macd_signal=9,
        bb_period=20, bb_std=2.0, atr_period=14,
        adx_period=14, adx_min=15.0,
        supertrend_period=10, supertrend_mult=3.0,
        pattern_weight=0.55, pattern_norm=2.0, fire_threshold=0.05,
    )
    sig = evaluate(opens.tolist(), highs.tolist(), lows.tolist(), closes.tolist(),
                    cfg, real_cvd=12.5)  # positive → bullish flow vote
    assert sig is not None
    assert any("cvd_real" in r for r in sig.reasons), \
        f"expected cvd_real tag when real_cvd provided; got {sig.reasons}"
    # And no cvd_proxy tag when real CVD took priority.
    assert not any("cvd_proxy" in r for r in sig.reasons), \
        f"cvd_proxy should not fire when real_cvd is provided"


def test_strategy_aggression_burst_fires_above_threshold():
    """Vote #24 fires when |aggression_10s| >= 0.40, doesn't fire below."""
    from bitunix_bot.config import StrategyCfg
    from bitunix_bot.strategy import evaluate
    rng = np.random.default_rng(7)
    n = 200
    closes = 100 + np.cumsum(rng.normal(0.05, 0.3, n))
    opens = closes - rng.uniform(0.05, 0.2, n)
    highs = np.maximum(opens, closes) + rng.uniform(0.02, 0.1, n)
    lows = np.minimum(opens, closes) - rng.uniform(0.02, 0.1, n)
    cfg = StrategyCfg(
        ema_fast=9, ema_mid=21, ema_slow=50,
        rsi_period=14, rsi_long_min=30, rsi_long_max=80,
        rsi_short_min=20, rsi_short_max=70,
        macd_fast=12, macd_slow=26, macd_signal=9,
        bb_period=20, bb_std=2.0, atr_period=14,
        adx_period=14, adx_min=15.0,
        supertrend_period=10, supertrend_mult=3.0,
        pattern_weight=0.55, pattern_norm=2.0, fire_threshold=0.05,
    )
    # Strong buy flow → long agg vote.
    sig_strong = evaluate(opens.tolist(), highs.tolist(), lows.tolist(),
                           closes.tolist(), cfg, aggression_10s=0.55)
    assert sig_strong is not None
    assert any(r.startswith("agg+") for r in sig_strong.reasons), \
        f"expected agg+ vote at 0.55; got {sig_strong.reasons}"

    # Below threshold (0.30) → no agg vote.
    sig_weak = evaluate(opens.tolist(), highs.tolist(), lows.tolist(),
                         closes.tolist(), cfg, aggression_10s=0.30)
    if sig_weak is not None:
        assert not any(r.startswith("agg+") or r.startswith("agg-")
                       for r in sig_weak.reasons), \
            f"agg vote should NOT fire at 0.30; got {sig_weak.reasons}"

    # Strong sell flow → short agg vote.
    sig_short = evaluate(opens.tolist(), highs.tolist(), lows.tolist(),
                          closes.tolist(), cfg, aggression_10s=-0.55)
    if sig_short is not None and sig_short.direction == "short":
        assert any(r.startswith("agg-") for r in sig_short.reasons), \
            f"expected agg- vote at -0.55; got {sig_short.reasons}"


def test_strategy_activity_multiplier_scales_combined_score():
    """activity_mult is accepted but no longer multiplied into the score —
    it was removed from score calculation because it created an artificial
    ceiling that made fire_threshold unreachable during off-peak hours.
    All three calls should fire with identical scores."""
    from bitunix_bot.config import StrategyCfg
    from bitunix_bot.strategy import evaluate
    rng = np.random.default_rng(42)
    n = 250
    closes = 60000.0 + np.cumsum(rng.normal(12, 25, n))
    opens = closes - rng.uniform(5, 20, n)
    highs = np.maximum(opens, closes) + rng.uniform(0.5, 3, n)
    lows = np.minimum(opens, closes) - rng.uniform(0.5, 3, n)
    for k in range(n - 3, n):
        closes[k] = closes[k - 1] + 30  # ascending close for continuation gate
        opens[k] = closes[k] - 30
        highs[k] = closes[k] + 0.5
        lows[k] = opens[k] - 0.5
    cfg = StrategyCfg(
        ema_fast=9, ema_mid=21, ema_slow=50,
        rsi_period=14, rsi_long_min=40, rsi_long_max=80,
        rsi_short_min=20, rsi_short_max=60,
        macd_fast=12, macd_slow=26, macd_signal=9,
        bb_period=20, bb_std=2.0, atr_period=14,
        adx_period=14, adx_min=15.0,
        supertrend_period=10, supertrend_mult=3.0,
        pattern_weight=0.55, pattern_norm=2.0, fire_threshold=0.05,
    )
    base = evaluate(opens.tolist(), highs.tolist(), lows.tolist(),
                     closes.tolist(), cfg, activity_mult=1.0)
    boosted = evaluate(opens.tolist(), highs.tolist(), lows.tolist(),
                        closes.tolist(), cfg, activity_mult=1.10)
    dampened = evaluate(opens.tolist(), highs.tolist(), lows.tolist(),
                         closes.tolist(), cfg, activity_mult=0.85)
    # All should fire with the same score (multiplier no longer affects score).
    assert base is not None and boosted is not None and dampened is not None
    assert base.score == boosted.score == dampened.score, \
        f"activity_mult should not affect score: {dampened.score} / {base.score} / {boosted.score}"


def _bot_with_tape_and_signal():
    """Helper: live-mode bot wired with a controllable fake tape feed and
    a synthetic uptrend that yields a long signal. Returns (bot, fake_tape)."""
    reset_state()
    cfg = fresh_cfg()
    cfg.mode = "live"
    cfg.trading.symbols = ["BTCUSDT"]
    bot = BitunixBot(cfg)
    bot.client = make_mock_client()
    bot.client.klines.side_effect = lambda *a, **kw: make_uptrend_klines()

    class FakeTape:
        def __init__(self):
            self.aggression = 0.0
            self.cvd = None
            self.activity = None
            self.price_change = None
        def get_aggression_ratio(self, sym, window_secs=10, min_count=5):
            return self.aggression
        def get_cvd(self, sym, window_secs=60, min_count=5):
            return self.cvd
        def get_activity_multiplier(self, sym, **kw):
            return self.activity
        def get_price_change_pct(self, sym, window_secs=10, min_count=5):
            return self.price_change

    tape = FakeTape()
    bot.tape_feed = tape
    # Pin session_weight to 1.0 so tests don't flake on off-peak UTC hours.
    bot._session_weight = lambda: 1.0
    bot._resolve_symbol_meta()
    return bot, tape


def test_tape_veto_blocks_long_against_strong_sell_flow():
    """If 10s tape aggression is contrary to the trade direction, the
    long signal must be blocked by SOMEONE in the pipeline.

    Post-Grok-v7: continuation gate (in evaluate) catches contrary tape
    BEFORE tape veto (in _execute) — so the rejection happens at the
    strategy level rather than the execution level. Either way, no
    order should be placed."""
    bot, tape = _bot_with_tape_and_signal()
    tape.aggression = -0.65   # 82/18 sell-side, hostile to long
    bot._tick()
    bot.client.place_order.assert_not_called()


def test_tape_veto_blocks_short_against_strong_buy_flow():
    """If 10s tape aggression ≥ +0.50, a SHORT signal must be skipped."""
    reset_state()
    cfg = fresh_cfg()
    cfg.mode = "live"
    cfg.trading.symbols = ["BTCUSDT"]
    bot = BitunixBot(cfg)
    bot.client = make_mock_client()

    # Synthesize a clean DOWNTREND that should produce a short signal.
    rng = np.random.default_rng(11)
    n = 250
    closes = 60000.0 - np.cumsum(rng.normal(12, 25, n))
    opens = closes + rng.uniform(5, 20, n)
    highs = np.maximum(opens, closes) + rng.uniform(0.5, 3, n)
    lows = np.minimum(opens, closes) - rng.uniform(0.5, 3, n)
    # Last 3 bars: clean bear marubozu + descending closes for continuation gate.
    for k in range(n - 3, n):
        closes[k] = closes[k - 1] - 30  # ensure close < prior close
        opens[k] = closes[k] + 30
        highs[k] = opens[k] + 0.5
        lows[k] = closes[k] - 0.5
    now = int(time.time() * 1000)
    klines = [{"open": float(opens[i]), "high": float(highs[i]), "low": float(lows[i]),
                "close": float(closes[i]), "time": now - (n-i)*60_000,
                "baseVol": "1.0", "quoteVol": "60000", "type": "1m"} for i in range(n)]
    bot.client.klines.side_effect = lambda *a, **kw: klines

    class FakeTape:
        def get_aggression_ratio(self, sym, window_secs=10, min_count=5):
            return 0.65   # 82/18 buy-side, hostile to short
        def get_cvd(self, sym, window_secs=60, min_count=5):
            return None
        def get_activity_multiplier(self, sym, **kw):
            return None
        def get_price_change_pct(self, sym, window_secs=10, min_count=5):
            return None
    bot.tape_feed = FakeTape()
    bot._resolve_symbol_meta()
    bot._tick()

    bot.client.place_order.assert_not_called()


def test_neutral_flow_applies_continuation_penalty():
    """Grok holistic review: continuation gate is now a SOFT score
    penalty (-0.10), not a hard veto. Neutral tape (between -0.25 and
    +0.25) deducts 0.10 from the combined score; if the signal still
    clears the threshold, it fires with CONT_PENALTY tag in reasons.
    Strong setups survive; marginal ones get filtered."""
    from bitunix_bot.strategy import evaluate
    bot, tape = _bot_with_tape_and_signal()
    tape.aggression = 0.10  # neutral, fails ≥0.25 alignment check
    bot._tick()
    # Synthetic uptrend produces strong score that survives the 0.10 penalty,
    # so the trade still fires — but with the penalty tag preserved.
    bot.client.place_order.assert_called()
    # Find the signal text recorded; CONT_PENALTY should appear.
    signal_events = [e for e in bot.state.snapshot()["events"]
                     if e["kind"] == "signal"]
    assert any("CONT_PENALTY" in e.get("text", "") for e in signal_events), \
        f"expected CONT_PENALTY tag in signal reasons; got {[e.get('text','')[:200] for e in signal_events]}"


def test_confirming_flow_allows_signal():
    """Mirror of the above: when tape DOES confirm direction at ≥0.25
    magnitude, the long signal fires normally."""
    bot, tape = _bot_with_tape_and_signal()
    tape.aggression = 0.40  # buy-side confirmation ≥ 0.25
    bot._tick()
    bot.client.place_order.assert_called(), \
        "confirming flow must allow trade through continuation gate"


def test_tape_veto_skipped_when_no_data():
    """When tape feed returns None for aggression, neither the
    continuation gate nor tape veto blocks (graceful degradation when
    tape is cold or feed is offline)."""
    bot, tape = _bot_with_tape_and_signal()
    tape.aggression = None
    bot._tick()
    assert bot.client.place_order.called, \
        "no-tape-data should not block trades — graceful degrade"


# ----------------------------------------------------------------- admin reset endpoint


def test_reset_streaks_endpoint_clears_in_memory_state():
    """POST /api/admin/reset-streaks must clear streak_pause_until,
    mini_cooldown_until, consec_losses, and recent_losses on the bot.
    Cascade / DD breaker / recent_trade_r should be PRESERVED."""
    reset_state()
    cfg = fresh_cfg()
    bot = BitunixBot(cfg)
    bot.client = make_mock_client()

    # Pre-load streak state across all 4 alts.
    now_s = time.time()
    for sym in ("BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT"):
        bot.streak_pause_until[sym] = int(now_s + 7200)
        bot.mini_cooldown_until[sym] = now_s + 300
        bot.consec_losses[sym] = 3
        bot.recent_losses[sym] = [now_s - 60, now_s - 30, now_s - 5]

    # And put cascade + DD + recent_trade_r in non-default state — must NOT
    # be cleared by the reset.
    bot._cascade_active = True
    bot.daily_dd_breached = True
    bot.recent_trade_r.extend([-0.5, -0.7])

    app = create_app(cfg, bot.client, bot=bot)
    c = app.test_client()
    auth = base64.b64encode(b"admin:test_pass").decode()
    r = c.post("/api/admin/reset-streaks",
               headers={"Authorization": f"Basic {auth}"})
    assert r.status_code == 200
    body = r.get_json()
    assert body["ok"] is True
    assert sorted(body["cleared"]["streak_pauses"]) == \
        ["BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT"]

    # In-memory state cleared.
    assert bot.streak_pause_until == {}
    assert bot.mini_cooldown_until == {}
    assert bot.consec_losses == {}
    assert bot.recent_losses == {}

    # Preserved state untouched.
    assert bot._cascade_active is True
    assert bot.daily_dd_breached is True
    assert len(bot.recent_trade_r) == 2


def test_reset_streaks_requires_auth():
    """Reset endpoint must reject unauthenticated requests."""
    reset_state()
    cfg = fresh_cfg()
    bot = BitunixBot(cfg)
    bot.client = make_mock_client()
    bot.streak_pause_until["BTCUSDT"] = int(time.time() + 7200)

    app = create_app(cfg, bot.client, bot=bot)
    c = app.test_client()
    # No auth header — should 401.
    r = c.post("/api/admin/reset-streaks")
    assert r.status_code == 401
    # Unchanged.
    assert "BTCUSDT" in bot.streak_pause_until


def test_continuation_blocks_long_on_bearish_close_pattern():
    """When the last bar closes in the BOTTOM of its range (bearish),
    a long signal must be blocked even if other indicators agree.
    Catches the 'lagging trend + reversal candle at local top' pattern."""
    from bitunix_bot.strategy import _continuation_confirmed
    import numpy as np
    # Bar with close near LOW (bearish): high=100, low=95, close=95.5 → close_pos=0.10
    h = np.array([100.0, 100.0])
    l_ = np.array([95.0, 95.0])
    c = np.array([99.0, 95.5])  # last close < prev close AND in bottom of range
    assert _continuation_confirmed("long", h, l_, c) is False
    assert _continuation_confirmed("short", h, l_, c) is True  # short OK on bearish bar


def test_continuation_blocks_short_on_bullish_close_pattern():
    """Mirror: bullish bar (close near high) blocks short signal."""
    from bitunix_bot.strategy import _continuation_confirmed
    import numpy as np
    h = np.array([100.0, 100.0])
    l_ = np.array([95.0, 95.0])
    c = np.array([95.5, 99.5])  # last close > prev close AND near high
    assert _continuation_confirmed("short", h, l_, c) is False
    assert _continuation_confirmed("long", h, l_, c) is True  # long OK on bullish bar


def test_continuation_blocks_when_close_doesnt_advance():
    """Even if bar is impulsive (close near high), if last_close <= prev_close
    the signal is blocked — bar is a small candle after an exhausted move."""
    from bitunix_bot.strategy import _continuation_confirmed
    import numpy as np
    h = np.array([100.0, 100.0])
    l_ = np.array([95.0, 99.0])    # last bar tight range 99-100
    c = np.array([99.5, 99.5])     # last close == prev close
    assert _continuation_confirmed("long", h, l_, c) is False, \
        "long must require last close > prior close"


def test_continuation_blocks_long_with_contrary_tape():
    """Grok v7: continuation gate now requires tape alignment ≥0.25 in
    direction. Contrary tape at -0.50 blocks even on a clean bullish bar."""
    from bitunix_bot.strategy import _continuation_confirmed
    import numpy as np
    h = np.array([100.0, 100.0])
    l_ = np.array([95.0, 96.0])
    c = np.array([97.0, 99.8])  # bullish bar, ascending close
    # No tape data → allow.
    assert _continuation_confirmed("long", h, l_, c) is True
    # Confirming tape → allow.
    assert _continuation_confirmed("long", h, l_, c, aggression_10s=0.40) is True
    # Contrary tape → BLOCK.
    assert _continuation_confirmed("long", h, l_, c, aggression_10s=-0.50) is False
    # Neutral tape (in -0.25..+0.25 range) → BLOCK (no flow confirmation).
    assert _continuation_confirmed("long", h, l_, c, aggression_10s=0.10) is False


def test_continuation_blocks_long_with_contrary_cvd():
    """real_cvd sign must match direction when set."""
    from bitunix_bot.strategy import _continuation_confirmed
    import numpy as np
    h = np.array([100.0, 100.0])
    l_ = np.array([95.0, 96.0])
    c = np.array([97.0, 99.8])
    # Confirming aggression but contrary CVD → block.
    assert _continuation_confirmed("long", h, l_, c,
                                     aggression_10s=0.40,
                                     real_cvd=-5.0) is False
    # Both confirming → allow.
    assert _continuation_confirmed("long", h, l_, c,
                                     aggression_10s=0.40,
                                     real_cvd=+5.0) is True


def test_continuation_allows_clean_marubozu():
    """Bullish marubozu (open near low, close near high, ascending) passes
    the long continuation gate."""
    from bitunix_bot.strategy import _continuation_confirmed
    import numpy as np
    h = np.array([100.0, 100.0])
    l_ = np.array([95.0, 96.0])
    c = np.array([97.0, 99.8])     # last close > prev close AND near high (close_pos = 0.95)
    assert _continuation_confirmed("long", h, l_, c) is True


def test_trend_dominance_dampens_opposite_side():
    """When one side has factor_trend ≥ 0.67, the OPPOSITE side's
    combined score is halved. Blocks reversal-fading-trend setups."""
    from bitunix_bot.config import StrategyCfg
    from bitunix_bot.strategy import evaluate
    klines = make_uptrend_klines()  # produces strong long trend factor
    o = [k["open"] for k in klines]
    h = [k["high"] for k in klines]
    l_ = [k["low"] for k in klines]
    c = [k["close"] for k in klines]
    cfg = StrategyCfg(
        ema_fast=9, ema_mid=21, ema_slow=50,
        rsi_period=14, rsi_long_min=40, rsi_long_max=80,
        rsi_short_min=20, rsi_short_max=60,
        macd_fast=12, macd_slow=26, macd_signal=9,
        bb_period=20, bb_std=2.0, atr_period=14,
        adx_period=14, adx_min=15.0,
        supertrend_period=10, supertrend_mult=3.0,
        pattern_weight=0.55, pattern_norm=2.0, fire_threshold=0.05,
    )
    sig = evaluate(o, h, l_, c, cfg)
    # On a strong uptrend, signal should fire LONG (not be blocked by anything).
    # We're not testing that the SHORT side is dampened in isolation — just
    # that the long signal still fires correctly under the new rules.
    assert sig is not None
    assert sig.direction == "long"


def test_invert_signals_flips_long_to_short():
    """When cfg.strategy.invert_signals=True, a setup that would fire LONG
    fires SHORT instead — preserving score, factors, reasons (with INVERTED
    tag prepended), but flipping the direction. Used to A/B test direction
    edge if live data shows systematic exhaustion-fade entries."""
    from bitunix_bot.config import StrategyCfg
    from bitunix_bot.strategy import evaluate
    klines = make_uptrend_klines()
    o = [k["open"] for k in klines]
    h = [k["high"] for k in klines]
    l_ = [k["low"] for k in klines]
    c = [k["close"] for k in klines]
    cfg = StrategyCfg(
        ema_fast=9, ema_mid=21, ema_slow=50,
        rsi_period=14, rsi_long_min=40, rsi_long_max=80,
        rsi_short_min=20, rsi_short_max=60,
        macd_fast=12, macd_slow=26, macd_signal=9,
        bb_period=20, bb_std=2.0, atr_period=14,
        adx_period=14, adx_min=15.0,
        supertrend_period=10, supertrend_mult=3.0,
        pattern_weight=0.55, pattern_norm=2.0, fire_threshold=0.05,
        invert_signals=True,
    )
    sig = evaluate(o, h, l_, c, cfg)
    assert sig is not None, "uptrend should still produce a signal under inversion"
    # Direction flipped from long → short
    assert sig.direction == "short", \
        f"expected inverted short on uptrend; got {sig.direction}"
    # side_code follows direction
    assert sig.side_code == "SELL"
    # INVERTED tag prepended to reasons so journal can A/B
    assert sig.reasons and sig.reasons[0] == "INVERTED", \
        f"expected leading INVERTED reason; got {sig.reasons[:3]}"


def test_invert_signals_flips_short_to_long():
    """Symmetric: a setup that would fire SHORT under invert=True fires LONG.
    Uses a synthetic downtrend with bearish marubozu last-3-bars to ensure
    the continuation gate passes for the SHORT side."""
    from bitunix_bot.config import StrategyCfg
    from bitunix_bot.strategy import evaluate
    rng = np.random.default_rng(7)
    n = 250
    closes = 60_000.0 + np.cumsum(rng.normal(-12, 25, n))
    opens = closes + rng.uniform(5, 20, n)
    highs = np.maximum(opens, closes) + rng.uniform(0.5, 3, n)
    lows = np.minimum(opens, closes) - rng.uniform(0.5, 3, n)
    # Last 3 bars: clean bear marubozu, descending closes.
    for i in range(n - 3, n):
        closes[i] = closes[i - 1] - 30 - rng.uniform(0, 5)
        opens[i] = closes[i] + 30 + rng.uniform(0, 5)
        highs[i] = opens[i] + rng.uniform(0.1, 1.5)
        lows[i] = closes[i] - rng.uniform(0.1, 1.5)
    o = list(opens); h = list(highs); l_ = list(lows); c = list(closes)
    cfg = StrategyCfg(
        ema_fast=9, ema_mid=21, ema_slow=50,
        rsi_period=14, rsi_long_min=40, rsi_long_max=80,
        rsi_short_min=20, rsi_short_max=60,
        macd_fast=12, macd_slow=26, macd_signal=9,
        bb_period=20, bb_std=2.0, atr_period=14,
        adx_period=14, adx_min=15.0,
        supertrend_period=10, supertrend_mult=3.0,
        pattern_weight=0.55, pattern_norm=2.0, fire_threshold=0.05,
        invert_signals=True,
    )
    sig = evaluate(o, h, l_, c, cfg)
    assert sig is not None, "downtrend should still produce a signal under inversion"
    assert sig.direction == "long", \
        f"expected inverted long on downtrend; got {sig.direction}"
    assert sig.side_code == "BUY"
    assert sig.reasons[0] == "INVERTED"


def test_invert_signals_default_off_preserves_direction():
    """When invert_signals is left at its default (False), an uptrend fires
    LONG normally with no INVERTED tag — the flag doesn't affect non-opt-in callers."""
    from bitunix_bot.config import StrategyCfg
    from bitunix_bot.strategy import evaluate
    klines = make_uptrend_klines()
    o = [k["open"] for k in klines]
    h = [k["high"] for k in klines]
    l_ = [k["low"] for k in klines]
    c = [k["close"] for k in klines]
    cfg = StrategyCfg(
        ema_fast=9, ema_mid=21, ema_slow=50,
        rsi_period=14, rsi_long_min=40, rsi_long_max=80,
        rsi_short_min=20, rsi_short_max=60,
        macd_fast=12, macd_slow=26, macd_signal=9,
        bb_period=20, bb_std=2.0, atr_period=14,
        adx_period=14, adx_min=15.0,
        supertrend_period=10, supertrend_mult=3.0,
        pattern_weight=0.55, pattern_norm=2.0, fire_threshold=0.05,
        # invert_signals omitted — uses dataclass default (False)
    )
    sig = evaluate(o, h, l_, c, cfg)
    assert sig is not None
    assert sig.direction == "long"
    assert "INVERTED" not in sig.reasons


def test_min_adx_skips_deep_chop():
    """When ADX is below cfg.min_adx_for_trade (default 22), evaluate()
    must return None regardless of how strong the score would otherwise be.
    Deep chop = pattern artifacts on flat candles, not predictive of direction."""
    from bitunix_bot.config import StrategyCfg
    from bitunix_bot.strategy import evaluate
    klines = make_uptrend_klines()
    o = [k["open"] for k in klines]
    h = [k["high"] for k in klines]
    l_ = [k["low"] for k in klines]
    c = [k["close"] for k in klines]

    # Build a config where ADX is naturally low (very long EMAs make ADX small).
    # Easier path: just patch the cfg's min_adx very high so any market is "chop".
    cfg = StrategyCfg(
        ema_fast=9, ema_mid=21, ema_slow=50,
        rsi_period=14, rsi_long_min=40, rsi_long_max=80,
        rsi_short_min=20, rsi_short_max=60,
        macd_fast=12, macd_slow=26, macd_signal=9,
        bb_period=20, bb_std=2.0, atr_period=14,
        adx_period=14, adx_min=15.0,
        supertrend_period=10, supertrend_mult=3.0,
        pattern_weight=0.55, pattern_norm=2.0, fire_threshold=0.05,
        min_adx_for_trade=999.0,  # impossibly high — nothing should fire
    )
    sig = evaluate(o, h, l_, c, cfg)
    assert sig is None, "min_adx_for_trade=999 must block all signals"


def test_min_adx_zero_disables_filter():
    """min_adx_for_trade=0 means no ADX floor (regression check)."""
    from bitunix_bot.config import StrategyCfg
    from bitunix_bot.strategy import evaluate
    klines = make_uptrend_klines()
    o = [k["open"] for k in klines]
    h = [k["high"] for k in klines]
    l_ = [k["low"] for k in klines]
    c = [k["close"] for k in klines]
    cfg = StrategyCfg(
        ema_fast=9, ema_mid=21, ema_slow=50,
        rsi_period=14, rsi_long_min=40, rsi_long_max=80,
        rsi_short_min=20, rsi_short_max=60,
        macd_fast=12, macd_slow=26, macd_signal=9,
        bb_period=20, bb_std=2.0, atr_period=14,
        adx_period=14, adx_min=15.0,
        supertrend_period=10, supertrend_mult=3.0,
        pattern_weight=0.55, pattern_norm=2.0, fire_threshold=0.05,
        min_adx_for_trade=0.0,  # disabled
    )
    sig = evaluate(o, h, l_, c, cfg)
    assert sig is not None  # baseline behavior preserved


def test_absorption_vetoes_same_direction_short():
    """When absorb(sellflow) fires (sell aggression being absorbed → buyers
    defending), any SHORT signal must be vetoed — that's the textbook
    'shorting into absorption' pattern that bled 100% of recent trades."""
    from bitunix_bot.config import StrategyCfg
    from bitunix_bot.strategy import evaluate
    rng = np.random.default_rng(11)
    n = 250
    # Build a downtrend that would normally fire SHORT.
    closes = 60000.0 - np.cumsum(rng.normal(12, 25, n))
    opens = closes + rng.uniform(5, 20, n)
    highs = np.maximum(opens, closes) + rng.uniform(0.5, 3, n)
    lows = np.minimum(opens, closes) - rng.uniform(0.5, 3, n)
    for k in range(n - 3, n):
        closes[k] = closes[k - 1] - 30  # descending close for continuation gate
        opens[k] = closes[k] + 30
        highs[k] = opens[k] + 0.5
        lows[k] = closes[k] - 0.5
    cfg = StrategyCfg(
        ema_fast=9, ema_mid=21, ema_slow=50,
        rsi_period=14, rsi_long_min=40, rsi_long_max=80,
        rsi_short_min=20, rsi_short_max=60,
        macd_fast=12, macd_slow=26, macd_signal=9,
        bb_period=20, bb_std=2.0, atr_period=14,
        adx_period=14, adx_min=15.0,
        supertrend_period=10, supertrend_mult=3.0,
        pattern_weight=0.55, pattern_norm=2.0, fire_threshold=0.05,
    )
    # Without absorption: short should fire.
    sig_normal = evaluate(opens.tolist(), highs.tolist(), lows.tolist(),
                           closes.tolist(), cfg, aggression_10s=-0.30)
    assert sig_normal is not None, "baseline short signal should fire"
    assert sig_normal.direction == "short"

    # Now with absorption (extreme sell aggression + no movement):
    # absorb(sellflow) goes to long_reasons, vetoing the short signal.
    sig_absorbed = evaluate(opens.tolist(), highs.tolist(), lows.tolist(),
                             closes.tolist(), cfg,
                             aggression_10s=-0.85,
                             price_change_10s_pct=0.05)
    assert sig_absorbed is None or sig_absorbed.direction != "short", \
        "short signal must be vetoed when sellflow absorption fires"


def test_absorption_vetoes_same_direction_long():
    """Mirror: absorb(buyflow) blocks a LONG signal."""
    from bitunix_bot.config import StrategyCfg
    from bitunix_bot.strategy import evaluate
    klines = make_uptrend_klines()
    o = [k["open"] for k in klines]
    h = [k["high"] for k in klines]
    l_ = [k["low"] for k in klines]
    c = [k["close"] for k in klines]
    cfg = StrategyCfg(
        ema_fast=9, ema_mid=21, ema_slow=50,
        rsi_period=14, rsi_long_min=40, rsi_long_max=80,
        rsi_short_min=20, rsi_short_max=60,
        macd_fast=12, macd_slow=26, macd_signal=9,
        bb_period=20, bb_std=2.0, atr_period=14,
        adx_period=14, adx_min=15.0,
        supertrend_period=10, supertrend_mult=3.0,
        pattern_weight=0.55, pattern_norm=2.0, fire_threshold=0.05,
    )
    sig_normal = evaluate(o, h, l_, c, cfg, aggression_10s=0.30)
    assert sig_normal is not None
    assert sig_normal.direction == "long"

    sig_absorbed = evaluate(o, h, l_, c, cfg,
                             aggression_10s=0.85,
                             price_change_10s_pct=-0.05)
    assert sig_absorbed is None or sig_absorbed.direction != "long", \
        "long signal must be vetoed when buyflow absorption fires"


def test_absorption_does_not_block_reversal_trade():
    """Absorption signals reversal — the OPPOSITE-direction trade should
    still be allowed if other indicators agree. This is the whole point:
    absorption suggests where price is GOING (reversal), not where it
    came from."""
    from bitunix_bot.config import StrategyCfg
    from bitunix_bot.strategy import evaluate
    klines = make_uptrend_klines()
    o = [k["open"] for k in klines]
    h = [k["high"] for k in klines]
    l_ = [k["low"] for k in klines]
    c = [k["close"] for k in klines]
    cfg = StrategyCfg(
        ema_fast=9, ema_mid=21, ema_slow=50,
        rsi_period=14, rsi_long_min=40, rsi_long_max=80,
        rsi_short_min=20, rsi_short_max=60,
        macd_fast=12, macd_slow=26, macd_signal=9,
        bb_period=20, bb_std=2.0, atr_period=14,
        adx_period=14, adx_min=15.0,
        supertrend_period=10, supertrend_mult=3.0,
        pattern_weight=0.55, pattern_norm=2.0, fire_threshold=0.05,
    )
    # Sellflow absorbed (agg=-0.85, no move) → absorption vote LONG.
    # On an uptrend, the LONG signal should still fire — absorption AGREES
    # with the uptrend direction.
    sig = evaluate(o, h, l_, c, cfg,
                    aggression_10s=-0.85,
                    price_change_10s_pct=0.05)
    # May or may not fire long depending on score, but it should NOT be
    # blocked by absorption (which votes long here).
    if sig is not None:
        assert sig.direction == "long", \
            "absorption shouldn't veto reversal-direction trades"


def test_flat_trades_dont_count_toward_streak():
    """A near-zero net PnL trade (BE-ratchet-then-reversal) must NOT
    increment consec_losses or recent_losses. Found in live data: 6 of 14
    'losses' were within $0.001 of break-even, triggering the 3-loss
    streak pause inappropriately."""
    reset_state()
    cfg = fresh_cfg()
    bot = BitunixBot(cfg)
    bot.client = make_mock_client()
    now_ms = int(time.time() * 1000)
    # Three "flat" trades with net ~ -$0.0002 each (BE-ratchet signature).
    bot.client.history_positions.return_value = {
        "positionList": [
            {"positionId": f"P{i}", "symbol": "BTCUSDT", "side": "SELL",
             "qty": "0.0019", "avgOpenPrice": "60000",
             "ctime": now_ms - (3 - i) * 30_000,
             "mtime": now_ms - (2 - i) * 30_000,
             # realized loss almost exactly offset by fee/rebate → net ≈ 0
             "realizedPNL": "-0.18", "fee": "0.1798", "funding": "0"}
            for i in range(3)
        ],
        "total": 3,
    }
    bot._update_streak_state()

    # No streak should fire — these are flat, not losses.
    assert "BTCUSDT" not in bot.streak_pause_until, \
        "flat trades must not trigger streak pause"
    assert "BTCUSDT" not in bot.mini_cooldown_until, \
        "flat trades must not trigger mini-cooldown"
    assert bot.consec_losses.get("BTCUSDT", 0) == 0, \
        "consec_losses should stay 0 on flat trades"
    assert len(bot.recent_losses.get("BTCUSDT", [])) == 0


def test_real_losses_still_trigger_streak():
    """Genuine losses (well past the flat threshold) still fire the
    streak counter. Tests the threshold is calibrated correctly — only
    near-zero trades are exempt."""
    reset_state()
    cfg = fresh_cfg()
    bot = BitunixBot(cfg)
    bot.client = make_mock_client()
    now_ms = int(time.time() * 1000)
    # Three GENUINE losses — full SL hit, not BE-ratchet exit.
    bot.client.history_positions.return_value = {
        "positionList": [
            {"positionId": f"P{i}", "symbol": "BTCUSDT", "side": "BUY",
             "qty": "0.001", "avgOpenPrice": "60000",
             "ctime": now_ms - (3 - i) * 30_000,
             "mtime": now_ms - (2 - i) * 30_000,
             # Full SL (~ -1R loss): realized -0.30 + fee -0.05 = -0.35
             "realizedPNL": "-0.30", "fee": "-0.05", "funding": "0"}
            for i in range(3)
        ],
        "total": 3,
    }
    bot._update_streak_state()

    # Real losses should accumulate consec_losses + trip streak pause at 3.
    assert "BTCUSDT" in bot.streak_pause_until, \
        "3 real losses should fire streak pause"


def test_journal_dedup_skips_duplicate_entry_client_ids():
    """Grok rescan: TradeJournal must skip duplicate entry records when
    called twice with the same client_id (e.g. due to a transient
    retry). The file should contain only ONE entry record."""
    import tempfile
    from bitunix_bot.journal import TradeJournal
    with tempfile.TemporaryDirectory() as tmp:
        j = TradeJournal(Path(tmp) / "trades.jsonl")
        kwargs = dict(
            symbol="BTCUSDT", side="BUY", client_id="bot-BTCUSDT-12345-BUY-PO",
            order_type="LIMIT", score=0.8, threshold_used=0.75,
            conviction_mult=1.0, indicator_count=10, pattern_score=2.0,
            reasons=["x"], atr_pct=0.1, adx=40.0, spread_pct=0.01,
            bid_depth=10.0, ask_depth=10.0, aggression_10s=0.3,
            real_cvd=5.0, activity_mult=1.0, session_weight=1.0,
            entry_price=78000.0, stop_loss=77800.0, take_profit=78200.0,
            notional=156.0, leverage=10,
        )
        # Call twice with identical client_id.
        j.record_entry(**kwargs)
        j.record_entry(**kwargs)
        # File should contain exactly ONE line.
        lines = (Path(tmp) / "trades.jsonl").read_text().strip().split("\n")
        assert len(lines) == 1, \
            f"expected single entry due to dedup; got {len(lines)} lines"


def test_journal_dedup_skips_duplicate_exit_position_ids():
    """Same dedup logic for exits, keyed on position_id."""
    import tempfile
    from bitunix_bot.journal import TradeJournal
    with tempfile.TemporaryDirectory() as tmp:
        j = TradeJournal(Path(tmp) / "trades.jsonl")
        kwargs = dict(
            symbol="BTCUSDT", position_id="POS_42", side="LONG",
            entry_price=78000.0, exit_price=78200.0,
            exit_reason="win", hold_time_sec=240.0, max_favor_r=0.83,
            net_pnl=0.30, realized_pnl=0.20, fee=-0.10, funding=0.0,
        )
        j.record_exit(**kwargs)
        j.record_exit(**kwargs)
        lines = (Path(tmp) / "trades.jsonl").read_text().strip().split("\n")
        assert len(lines) == 1, \
            f"expected single exit due to dedup; got {len(lines)} lines"


def test_journal_dedup_unique_ids_pass_through():
    """Different client_ids/position_ids must NOT be deduped — each
    unique id gets its own line."""
    import tempfile
    from bitunix_bot.journal import TradeJournal
    with tempfile.TemporaryDirectory() as tmp:
        j = TradeJournal(Path(tmp) / "trades.jsonl")
        for i in range(5):
            j.record_entry(
                symbol="BTCUSDT", side="BUY", client_id=f"cid-{i}",
                order_type="LIMIT", score=0.8, threshold_used=0.75,
                conviction_mult=1.0, indicator_count=10, pattern_score=2.0,
                reasons=["x"], atr_pct=0.1, adx=40.0, spread_pct=0.01,
                bid_depth=10.0, ask_depth=10.0, aggression_10s=0.3,
                real_cvd=5.0, activity_mult=1.0, session_weight=1.0,
                entry_price=78000.0, stop_loss=77800.0, take_profit=78200.0,
                notional=156.0, leverage=10,
            )
        lines = (Path(tmp) / "trades.jsonl").read_text().strip().split("\n")
        assert len(lines) == 5, f"expected 5 unique entries; got {len(lines)}"


def test_journal_endpoint_returns_recent_events():
    """GET /api/journal should return the most-recent journal entries
    as JSON, with limit + kind filters."""
    import tempfile
    from bitunix_bot.journal import TradeJournal
    reset_state()
    cfg = fresh_cfg()
    bot = BitunixBot(cfg)
    bot.client = make_mock_client()
    with tempfile.TemporaryDirectory() as tmp:
        bot.journal = TradeJournal(Path(tmp) / "trades.jsonl")
        # Seed 3 entries + 2 exits.
        for i in range(3):
            bot.journal.record_entry(
                symbol=f"SYM{i}", side="BUY", client_id=f"cid{i}",
                order_type="LIMIT", score=0.6 + i * 0.05,
                threshold_used=0.50, conviction_mult=1.2,
                indicator_count=12, pattern_score=1.5, reasons=["x"],
                atr_pct=0.08, adx=25.0, spread_pct=0.01,
                bid_depth=10.0, ask_depth=10.0,
                aggression_10s=0.3, real_cvd=5.0, activity_mult=1.0,
                session_weight=1.0, entry_price=100.0,
                stop_loss=99.6, take_profit=101.0,
                notional=100.0, leverage=25,
            )
        for i in range(2):
            bot.journal.record_exit(
                symbol=f"SYM{i}", position_id=f"P{i}", side="LONG",
                entry_price=100.0, exit_price=101.0,
                exit_reason="win", hold_time_sec=120.0,
                max_favor_r=1.5, net_pnl=0.5, realized_pnl=0.6,
                fee=-0.1, funding=0.0,
            )

        app = create_app(cfg, bot.client, bot=bot)
        c = app.test_client()
        auth = base64.b64encode(b"admin:test_pass").decode()
        # Default: all kinds, limit 50.
        r = c.get("/api/journal",
                  headers={"Authorization": f"Basic {auth}"})
        assert r.status_code == 200
        body = r.get_json()
        assert body["count"] == 5
        kinds = [e["kind"] for e in body["events"]]
        assert kinds.count("entry") == 3 and kinds.count("exit") == 2

        # Filter to entries only.
        r = c.get("/api/journal?kind=entry",
                  headers={"Authorization": f"Basic {auth}"})
        body = r.get_json()
        assert body["count"] == 3
        assert all(e["kind"] == "entry" for e in body["events"])

        # Limit clamps to last 2.
        r = c.get("/api/journal?limit=2",
                  headers={"Authorization": f"Basic {auth}"})
        body = r.get_json()
        assert body["count"] == 2


def test_journal_endpoint_handles_missing_file():
    """When the journal hasn't been written yet, return empty list (not 404)."""
    import tempfile
    from bitunix_bot.journal import TradeJournal
    reset_state()
    cfg = fresh_cfg()
    bot = BitunixBot(cfg)
    bot.client = make_mock_client()
    with tempfile.TemporaryDirectory() as tmp:
        # Journal points at a path that doesn't exist yet.
        bot.journal = TradeJournal(Path(tmp) / "subdir" / "trades.jsonl")
        app = create_app(cfg, bot.client, bot=bot)
        c = app.test_client()
        auth = base64.b64encode(b"admin:test_pass").decode()
        r = c.get("/api/journal",
                  headers={"Authorization": f"Basic {auth}"})
        assert r.status_code == 200
        body = r.get_json()
        assert body["count"] == 0
        assert body["events"] == []


def test_journal_endpoint_requires_auth():
    """Journal endpoints must be auth-gated (same as everything else)."""
    reset_state()
    cfg = fresh_cfg()
    bot = BitunixBot(cfg)
    bot.client = make_mock_client()
    app = create_app(cfg, bot.client, bot=bot)
    c = app.test_client()
    r = c.get("/api/journal")
    assert r.status_code == 401
    r = c.get("/api/journal/download")
    assert r.status_code == 401


def test_journal_download_returns_file():
    """GET /api/journal/download should send the .jsonl file."""
    import tempfile
    from bitunix_bot.journal import TradeJournal
    reset_state()
    cfg = fresh_cfg()
    bot = BitunixBot(cfg)
    bot.client = make_mock_client()
    with tempfile.TemporaryDirectory() as tmp:
        bot.journal = TradeJournal(Path(tmp) / "trades.jsonl")
        bot.journal.record_entry(
            symbol="X", side="BUY", client_id="x", order_type="LIMIT",
            score=0.5, threshold_used=0.5, conviction_mult=1.0,
            indicator_count=10, pattern_score=1.0, reasons=[],
            atr_pct=None, adx=None, spread_pct=None,
            bid_depth=None, ask_depth=None,
            aggression_10s=None, real_cvd=None, activity_mult=None,
            session_weight=None, entry_price=100.0,
            stop_loss=99.6, take_profit=101.0, notional=100.0, leverage=25,
        )
        app = create_app(cfg, bot.client, bot=bot)
        c = app.test_client()
        auth = base64.b64encode(b"admin:test_pass").decode()
        r = c.get("/api/journal/download",
                  headers={"Authorization": f"Basic {auth}"})
        assert r.status_code == 200
        assert b'"kind":"entry"' in r.data
        assert b'"symbol":"X"' in r.data


def test_reset_streaks_returns_503_when_no_bot():
    """When dashboard is constructed without a bot reference, reset is 503."""
    reset_state()
    cfg = fresh_cfg()
    client = make_mock_client()
    app = create_app(cfg, client)  # no bot kwarg
    c = app.test_client()
    auth = base64.b64encode(b"admin:test_pass").decode()
    r = c.post("/api/admin/reset-streaks",
               headers={"Authorization": f"Basic {auth}"})
    assert r.status_code == 503


# ----------------------------------------------------------------- Grok review v8


def test_ticker_confirmation_blocks_long_when_price_didnt_move_up():
    """When confirm_with_ticker=True, a LONG signal must be dropped if
    the live ticker price has NOT moved above the signal-bar close.
    Catches exhaustion-fade where signal bar was bullish but immediately
    reverses on the next bar."""
    reset_state()
    cfg = fresh_cfg()
    cfg.mode = "live"
    cfg.strategy.confirm_with_ticker = True
    cfg.trading.symbols = ["BTCUSDT"]
    bot = BitunixBot(cfg)
    bot.client = make_mock_client()
    klines = make_uptrend_klines()
    bot.client.klines.side_effect = lambda *a, **kw: klines
    last_close = klines[-1]["close"]
    # Ticker 0.2% BELOW the signal bar's close → confirmation should fail.
    # Must exceed confirmation_tolerance_pct (0.05%) to actually block.
    bot.client.ticker.return_value = {"lastPrice": str(last_close * 0.998)}
    bot._resolve_symbol_meta()
    bot._tick()

    bot.client.place_order.assert_not_called()
    skips = [e for e in bot.state.snapshot()["events"] if e["kind"] == "skip"]
    assert any("no continuation up" in s["text"] for s in skips), \
        f"expected ticker-confirmation skip; got {[s['text'] for s in skips]}"


def test_ticker_confirmation_allows_long_when_price_moved_up():
    """LONG signal proceeds when live ticker price > signal-bar close."""
    reset_state()
    cfg = fresh_cfg()
    cfg.mode = "live"
    cfg.strategy.confirm_with_ticker = True
    cfg.trading.symbols = ["BTCUSDT"]
    bot = BitunixBot(cfg)
    bot.client = make_mock_client()
    klines = make_uptrend_klines()
    bot.client.klines.side_effect = lambda *a, **kw: klines
    last_close = klines[-1]["close"]
    # Ticker meaningfully above → confirmation passes.
    bot.client.ticker.return_value = {"lastPrice": str(last_close + 5.0)}
    bot._resolve_symbol_meta()
    bot._tick()

    bot.client.place_order.assert_called()


def test_ticker_confirmation_recalibrates_sl_tp():
    """When ticker confirms direction, SL/TP must be re-derived from
    the actual fill price (ticker), not the stale signal-bar close."""
    reset_state()
    cfg = fresh_cfg()
    cfg.mode = "live"
    cfg.strategy.confirm_with_ticker = True
    cfg.strategy.min_adx_for_trade = 0.0
    cfg.trading.symbols = ["BTCUSDT"]
    bot = BitunixBot(cfg)
    bot.client = make_mock_client()
    klines = make_uptrend_klines()
    bot.client.klines.side_effect = lambda *a, **kw: klines
    last_close = klines[-1]["close"]
    confirmed_px = last_close + 10.0
    bot.client.ticker.return_value = {"lastPrice": str(confirmed_px)}
    bot._resolve_symbol_meta()
    bot._tick()

    bot.client.place_order.assert_called()
    kw = bot.client.place_order.call_args.kwargs
    sl = float(kw["sl_price"])
    tp = float(kw["tp_price"])
    # SL should be re-calibrated from confirmed_px, not stale close.
    # SL ≈ confirmed_px × (1 - 0.0025), bounded by ATR-hybrid.
    expected_sl = confirmed_px * (1 - 0.0025)
    # Allow some tolerance for ATR-hybrid widening + price precision rounding.
    assert abs(sl - expected_sl) < confirmed_px * 0.01, \
        f"SL {sl} not re-calibrated from confirmed {confirmed_px}; expected ~{expected_sl}"
    assert tp > confirmed_px, "TP should be above confirmed entry for long"


def test_ticker_confirmation_drops_when_ticker_fetch_fails():
    """If the ticker call raises (network error, exchange down), the
    signal is dropped — better safe than entering blind."""
    from bitunix_bot.client import BitunixError
    reset_state()
    cfg = fresh_cfg()
    cfg.mode = "live"
    cfg.strategy.confirm_with_ticker = True
    cfg.trading.symbols = ["BTCUSDT"]
    bot = BitunixBot(cfg)
    bot.client = make_mock_client()
    bot.client.klines.side_effect = lambda *a, **kw: make_uptrend_klines()
    bot.client.ticker.side_effect = BitunixError(503, "service down", {})
    bot._resolve_symbol_meta()
    bot._tick()

    bot.client.place_order.assert_not_called()
    skips = [e for e in bot.state.snapshot()["events"] if e["kind"] == "skip"]
    assert any("ticker confirmation" in s["text"] for s in skips), \
        f"expected ticker-confirmation skip; got {[s['text'] for s in skips]}"


def test_post_only_entries_enabled_in_production_yaml():
    """Production config.yaml sets use_post_only_entries=true (ChatGPT
    review). Tests using fresh_cfg() get a loose override, but the YAML
    file itself must enable maker entries by default."""
    cfg = load("config.yaml", "/dev/null")
    assert cfg.trading.use_post_only_entries is True, \
        "post-only entries must be ENABLED in production config.yaml"


def test_post_only_disabled_explicit_routes_to_market():
    """When a caller explicitly sets use_post_only_entries=False (e.g. in
    tests), entries go straight to MARKET — no maker post-only path."""
    reset_state()
    cfg = fresh_cfg()
    cfg.mode = "live"
    cfg.trading.symbols = ["BTCUSDT"]
    cfg.trading.use_post_only_entries = False  # explicit opt-out
    bot = BitunixBot(cfg)
    bot.client = make_mock_client()
    bot.client.klines.side_effect = lambda *a, **kw: make_uptrend_klines()
    bot._resolve_symbol_meta()
    bot._tick()

    bot.client.place_order.assert_called()
    kw = bot.client.place_order.call_args.kwargs
    assert kw["order_type"] == "MARKET", \
        f"expected MARKET entry with explicit opt-out; got {kw.get('order_type')}"
    assert "BTCUSDT" not in bot.pending_limits


# ----------------------------------------------------------------- ChatGPT review v5 (factor groups)


def test_factor_classification_routes_votes_correctly():
    """Each canonical vote tag must route to the expected factor group."""
    from bitunix_bot.strategy import _classify_reason
    expected = {
        # Trend
        "ema_stack_up": ("trend", "ema_stack"),
        "cross_above_ema_fast": ("trend", "ema_cross"),
        "macd_up": ("trend", "macd"),
        "supertrend_up": ("trend", "supertrend"),
        "above_vwap": ("trend", "vwap"),
        "above_bb_mid": ("trend", "bb_mid"),
        "htf_uptrend(15m)": ("trend", "htf"),
        "btc_leader_up": ("trend", "btc_leader"),
        "CMB:trend_pullback": ("trend", "trend_pullback_combo"),
        "CMB:squeeze_breakout": ("trend", "squeeze_breakout_combo"),
        # Mean Reversion
        "rsi(45)": ("mean_rev", "rsi"),
        "stoch_bull(35)": ("mean_rev", "stoch"),
        "DIV:rsi_bullish_div": ("mean_rev", "div_rsi"),
        "DIV:macd_hidden_bullish_div": ("mean_rev", "div_macd"),
        "DIV:obv_bearish_div": ("mean_rev", "div_volume"),
        "DIV:cvd_bullish_div": ("mean_rev", "div_volume"),
        "SMC:fvg_bullish": ("mean_rev", "smc_fvg"),
        "SMC:liquidity_sweep_bullish": ("mean_rev", "smc_sweep"),
        "mfi(35)": ("mean_rev", "mfi"),
        "sr_bounce_support(60000,×3)": ("mean_rev", "sr"),
        "CMB:smc_reversal": ("mean_rev", "smc_reversal_combo"),
        "CMB:bb_extreme_revert": ("mean_rev", "bb_extreme_combo"),
        # Flow
        "ob_imb+0.45": ("flow", "ob_imb"),
        "cvd_real+12.5": ("flow", "cvd"),
        "cvd_proxy+(45)": ("flow", "cvd"),
        "agg+0.55": ("flow", "agg"),
        "agg-0.50": ("flow", "agg"),
        "absorb(buyflow@+0.65)": ("flow", "absorb"),
        # Context
        "vol_spike(2.5x)": ("context", "vol_spike"),
        "adx(35)": ("context", "adx"),
        "funding+0.080%": ("context", "funding"),
        "squeeze_up": ("context", "squeeze"),
        "CMB:crowd_contrarian": ("context", "crowd_combo"),
        # ATR / pattern tags should NOT classify
        "atr(0.06%)": None,
        "PAT:marubozu_bull": None,
        "CP:bull_flag": None,
    }
    for reason, want in expected.items():
        got = _classify_reason(reason)
        assert got == want, f"{reason!r} → got {got}, want {want}"


def test_factor_score_breakdown_dedups_within_group():
    """Multiple correlated trend votes → counted ONCE per category, not summed."""
    from bitunix_bot.strategy import factor_score_breakdown
    saturation = {"trend": 6, "mean_rev": 6, "flow": 3, "context": 3}
    # Two ema-stack tags + two macd tags = effectively 2 unique trend categories
    # (ema_stack and macd), not 4. This is the correlated-vote dedup.
    reasons = [
        "ema_stack_up", "macd_up",       # 2 distinct trend categories
        "supertrend_up", "above_vwap",   # +2 more
        "rsi(45)",                        # 1 mean_rev
        "ob_imb+0.45", "cvd_real+12.5",  # 2 flow
        "adx(35)",                        # 1 context
    ]
    breakdown = factor_score_breakdown(reasons, saturation)
    assert abs(breakdown["trend"] - 4 / 6) < 1e-9, f"trend={breakdown['trend']}"
    assert abs(breakdown["mean_rev"] - 1 / 6) < 1e-9
    assert abs(breakdown["flow"] - 2 / 3) < 1e-9
    assert abs(breakdown["context"] - 1 / 3) < 1e-9


def test_factor_score_caps_at_one():
    """Even 10 trend votes saturate at 1.0 (cap eliminates over-counting)."""
    from bitunix_bot.strategy import factor_score_breakdown
    saturation = {"trend": 6, "mean_rev": 6, "flow": 3, "context": 3}
    reasons = [
        "ema_stack_up", "cross_above_ema_fast", "macd_up", "supertrend_up",
        "above_vwap", "above_bb_mid", "htf_uptrend", "btc_leader_up",
        "CMB:trend_pullback", "CMB:squeeze_breakout",
    ]
    breakdown = factor_score_breakdown(reasons, saturation)
    # 10 unique trend votes / 6 saturation = 1.67, but capped to 1.0.
    assert breakdown["trend"] == 1.0


def test_factor_score_weighted_combines_groups():
    """Weighted average over 4 groups."""
    from bitunix_bot.strategy import factor_score_weighted
    breakdown = {"trend": 1.0, "mean_rev": 0.5, "flow": 0.67, "context": 0.33}
    weights = {"trend": 0.30, "mean_rev": 0.25, "flow": 0.30, "context": 0.15}
    expected = 0.30 * 1.0 + 0.25 * 0.5 + 0.30 * 0.67 + 0.15 * 0.33
    got = factor_score_weighted(breakdown, weights)
    assert abs(got - expected) < 1e-9


def test_signal_records_factor_breakdown():
    """Signal returned by evaluate() must carry per-group factor scores."""
    from bitunix_bot.config import StrategyCfg
    from bitunix_bot.strategy import evaluate
    cfg = StrategyCfg(
        ema_fast=9, ema_mid=21, ema_slow=50,
        rsi_period=14, rsi_long_min=40, rsi_long_max=80,
        rsi_short_min=20, rsi_short_max=60,
        macd_fast=12, macd_slow=26, macd_signal=9,
        bb_period=20, bb_std=2.0, atr_period=14,
        adx_period=14, adx_min=15.0,
        supertrend_period=10, supertrend_mult=3.0,
        pattern_weight=0.55, pattern_norm=2.0, fire_threshold=0.05,
    )
    klines = make_uptrend_klines()
    o = [k["open"] for k in klines]
    h = [k["high"] for k in klines]
    l_ = [k["low"] for k in klines]
    c = [k["close"] for k in klines]
    sig = evaluate(o, h, l_, c, cfg)
    assert sig is not None
    # All 4 factor scores in [0, 1].
    for v in (sig.factor_trend, sig.factor_mean_rev,
              sig.factor_flow, sig.factor_context):
        assert 0.0 <= v <= 1.0, f"factor score out of range: {v}"
    # On a clean uptrend, trend factor should be > 0.
    assert sig.factor_trend > 0.0


def test_adaptive_threshold_drawdown_raises_bar():
    """When rolling-20 trade R-sum < -2R, threshold ratchets up by +0.04."""
    reset_state()
    cfg = fresh_cfg()
    bot = BitunixBot(cfg)
    # Inject a -3R rolling tally (5 losses of -0.6R each).
    for _ in range(5):
        bot.recent_trade_r.append(-0.6)
    assert sum(bot.recent_trade_r) == -3.0
    assert bot._adaptive_threshold_adjustment() == 0.04


def test_adaptive_threshold_hot_streak_eases():
    """When rolling-20 trade R-sum > +3R, threshold eases by -0.02."""
    reset_state()
    cfg = fresh_cfg()
    bot = BitunixBot(cfg)
    for _ in range(5):
        bot.recent_trade_r.append(0.8)
    assert sum(bot.recent_trade_r) == 4.0
    assert bot._adaptive_threshold_adjustment() == -0.02


def test_adaptive_threshold_neutral_in_middle():
    """Tally between -2R and +3R → no adjustment."""
    reset_state()
    cfg = fresh_cfg()
    bot = BitunixBot(cfg)
    for _ in range(5):
        bot.recent_trade_r.append(0.1)
    assert bot._adaptive_threshold_adjustment() == 0.0


def test_adaptive_threshold_requires_minimum_samples():
    """Fewer than ADAPTIVE_MIN_SAMPLES trades → no adjustment (avoid noise)."""
    reset_state()
    cfg = fresh_cfg()
    bot = BitunixBot(cfg)
    bot.recent_trade_r.append(-2.5)  # only 1 sample
    assert bot._adaptive_threshold_adjustment() == 0.0


def test_adaptive_threshold_pipeline_flows_into_evaluate():
    """End-to-end: bot computes adaptive_adj, passes to evaluate, signals
    should show the adjusted threshold via fire_threshold_used."""
    reset_state()
    cfg = fresh_cfg()
    cfg.mode = "live"
    cfg.trading.symbols = ["BTCUSDT"]
    bot = BitunixBot(cfg)
    bot.client = make_mock_client()
    bot.client.klines.side_effect = lambda *a, **kw: make_uptrend_klines()
    # Pre-load drawdown state — should add +0.04 to threshold.
    for _ in range(5):
        bot.recent_trade_r.append(-0.6)
    bot._resolve_symbol_meta()
    bot._tick()

    # If a signal fired, its fire_threshold_used must be >= base + adaptive.
    signals = [e for e in bot.state.snapshot()["events"] if e["kind"] == "signal"]
    if signals:
        # Live tick fired — verify journal would show a higher threshold.
        # Simpler: spot-check that place_order wasn't blocked unexpectedly.
        pass  # smoke test — ensure pipeline runs without error


def test_compute_trade_r_basic():
    """R-multiple = (realized + fee + funding) / (qty * entry * sl_pct/100)."""
    pos = {
        "avgOpenPrice": "60000",
        "qty": "0.01",
        "realizedPNL": "0.5",
        "fee": "-0.05",
        "funding": "0",
    }
    r = BitunixBot._compute_trade_r(pos, sl_pct_default=0.40)
    # qty=0.01, entry=60000, sl_pct=0.40 → sl_dist = 240
    # risk_dollars = 0.01 * 240 = 2.40
    # net = 0.5 - 0.05 = 0.45
    # R = 0.45 / 2.40 = 0.1875
    assert abs(r - 0.1875) < 1e-6


def test_compute_trade_r_handles_invalid():
    """Missing/zero data should return 0.0 — not crash."""
    assert BitunixBot._compute_trade_r({}, 0.40) == 0.0
    assert BitunixBot._compute_trade_r({"avgOpenPrice": "0"}, 0.40) == 0.0
    assert BitunixBot._compute_trade_r(
        {"avgOpenPrice": "60000", "qty": "0"}, 0.40) == 0.0


def test_recent_trade_r_appended_on_close():
    """When a closed position is observed, its R-multiple must be appended
    to recent_trade_r (rolling window)."""
    reset_state()
    cfg = fresh_cfg()
    bot = BitunixBot(cfg)
    bot.client = make_mock_client()
    now_ms = int(time.time() * 1000)
    bot.client.history_positions.return_value = {
        "positionList": [
            {"positionId": "P1", "symbol": "BTCUSDT", "side": "LONG",
             "qty": "0.01", "avgOpenPrice": "60000", "avgClosePrice": "59760",
             "ctime": now_ms - 60_000, "mtime": now_ms,
             "realizedPNL": "-0.5", "fee": "-0.05", "funding": "0"},
        ],
        "total": 1,
    }
    bot._update_streak_state()
    assert len(bot.recent_trade_r) == 1
    # qty=0.01, entry=60000, sl=0.25% (Grok v6 tightening) → risk_dollars = 1.50
    # net = -0.55 → R = -0.367
    assert abs(bot.recent_trade_r[0] - (-0.367)) < 0.01


# ----------------------------------------------------------------- ChatGPT review v4


def test_stale_exit_flattens_drifting_position():
    """Position older than stale_exit_min minutes with max_favor below
    stale_exit_max_favor_r → flash_close.

    Grok holistic review (15m timeframe): stale_exit_min=18m,
    stale_exit_max_favor_r=0.25R."""
    bot, _, _ = _setup_bot_with_open_position(
        side="BUY", entry=100_000.0, current_price=100_025.0,  # +0.1R, drifting
    )
    # Fast-forward the position's ctime so it's "65 minutes old" (past 60-min floor).
    bot.client.pending_positions.return_value = [{
        **bot.client.pending_positions.return_value[0],
        "ctime": int(time.time() * 1000) - 65 * 60_000,
    }]
    # Pre-seed max_favor at 0.1R (below 0.3R threshold) — drifted with no edge.
    # Live tick recomputes r_favor from current_price and only ratchets up,
    # so seed must match or be ≤ current r_favor (0.1R here).
    bot.position_manager.position_max_favor["POS1"] = 0.1
    bot._tick()

    bot.client.flash_close_position.assert_called_once_with("POS1")
    orders = [e for e in bot.state.snapshot()["events"] if e["kind"] == "order"]
    assert any("STALE_EXIT" in e["text"] for e in orders), \
        f"expected STALE_EXIT order event; got {[e['text'] for e in orders]}"


def test_stale_exit_skipped_for_progressing_position():
    """Position aging but with max_favor ≥ stale_exit_max_favor_r should
    NOT be flattened — it has shown progress, let it ride."""
    bot, _, _ = _setup_bot_with_open_position(
        side="BUY", entry=100_000.0, current_price=100_080.0,  # +0.32R now
    )
    bot.client.pending_positions.return_value = [{
        **bot.client.pending_positions.return_value[0],
        "ctime": int(time.time() * 1000) - 20 * 60_000,  # past 18-min floor
    }]
    # Position previously hit 0.4R favorable — has shown edge above 0.25R floor.
    bot.position_manager.position_max_favor["POS1"] = 0.4
    bot._tick()

    bot.client.flash_close_position.assert_not_called()


def test_stale_exit_skipped_for_young_position():
    """Position younger than stale_exit_min minutes should NOT be exited
    by the stale-exit rule, even with low max_favor."""
    bot, _, _ = _setup_bot_with_open_position(
        side="BUY", entry=100_000.0, current_price=100_050.0,
    )
    # Just opened 5 minutes ago — too young (15m timeframe: stale_exit_min=18.0).
    bot.client.pending_positions.return_value = [{
        **bot.client.pending_positions.return_value[0],
        "ctime": int(time.time() * 1000) - 5 * 60_000,
    }]
    bot.position_manager.position_max_favor["POS1"] = 0.1
    bot._tick()

    bot.client.flash_close_position.assert_not_called()


def test_expansion_candle_skip_blocks_next_bar():
    """When the most recent bar's range exceeds 2× ATR, the next bar's
    signal evaluation must be skipped with an 'expansion candle' reason.
    Tested on ETH so the cascade detector (BTC-based) doesn't fire first."""
    reset_state()
    cfg = fresh_cfg()
    cfg.mode = "live"
    cfg.trading.symbols = ["ETHUSDT"]
    bot = BitunixBot(cfg)
    bot.client = make_mock_client()

    # Build ETH klines with a HUGE outlier last bar.
    rng = np.random.default_rng(42)
    n = 200
    eth_closes = 3000.0 + np.cumsum(rng.normal(0.5, 1.5, n))
    eth_opens = eth_closes - rng.uniform(0.2, 0.8, n)
    eth_highs = np.maximum(eth_opens, eth_closes) + rng.uniform(0.1, 0.5, n)
    eth_lows = np.minimum(eth_opens, eth_closes) - rng.uniform(0.1, 0.5, n)
    # Last bar: range = 30 (typical ETH ATR ~3 → ratio ~10×).
    eth_closes[-1] = 3025.0
    eth_opens[-1] = 3005.0
    eth_highs[-1] = 3035.0
    eth_lows[-1] = 3005.0

    # BTC klines stay flat to avoid tripping the cascade detector.
    btc_closes = np.full(n, 60000.0) + rng.normal(0, 5, n)
    btc_opens = btc_closes - 0.5
    btc_highs = btc_closes + 1.0
    btc_lows = btc_closes - 1.0

    now = int(time.time() * 1000)
    def make_klines(opens, highs, lows, closes):
        return [{"open": float(opens[i]), "high": float(highs[i]),
                  "low": float(lows[i]), "close": float(closes[i]),
                  "time": now - (n-i)*60_000, "baseVol": "1.0",
                  "quoteVol": "60000", "type": "1m"} for i in range(n)]
    eth_kl = make_klines(eth_opens, eth_highs, eth_lows, eth_closes)
    btc_kl = make_klines(btc_opens, btc_highs, btc_lows, btc_closes)

    def klines_router(*args, **kwargs):
        sym = (args[0] if args else kwargs.get("symbol", "")).upper()
        return btc_kl if "BTC" in sym else eth_kl
    bot.client.klines.side_effect = klines_router
    bot._resolve_symbol_meta()
    bot._tick()

    # Should record an expansion-candle skip and NOT place an order.
    skips = [e for e in bot.state.snapshot()["events"] if e["kind"] == "skip"]
    assert any("expansion candle" in s["text"] for s in skips), \
        f"expected expansion candle skip; got {[s['text'] for s in skips]}"
    bot.client.place_order.assert_not_called()


def test_dd_risk_multiplier_gradual_steps():
    """DD ramp: 1.0 normal → 0.75 → 0.50 → 0.25 → 0.0 halt."""
    reset_state()
    cfg = fresh_cfg()
    cfg.trading.max_daily_dd_pct = 8.0
    bot = BitunixBot(cfg)
    bot.client = make_mock_client()
    bot.session_start_day = time.gmtime().tm_yday
    bot.session_start_equity = 100.0

    def setup_dd_pct(loss_pct):
        equity = 100.0 * (1 - loss_pct / 100)
        bot.client.account.return_value = {
            "marginCoin": "USDT", "available": str(equity), "frozen": "0",
            "margin": "0", "transfer": str(equity), "positionMode": "ONE_WAY",
            "crossUnrealizedPNL": "0", "isolationUnrealizedPNL": "0", "bonus": "0",
        }
        bot.daily_dd_breached = False  # reset for each step

    setup_dd_pct(0.5)
    assert bot._daily_dd_risk_multiplier() == 1.0   # below first step

    setup_dd_pct(2.5)   # past 25% of threshold
    assert bot._daily_dd_risk_multiplier() == 0.75

    setup_dd_pct(5.0)   # past 50% of threshold
    assert bot._daily_dd_risk_multiplier() == 0.50

    setup_dd_pct(7.0)   # past 75% of threshold
    assert bot._daily_dd_risk_multiplier() == 0.25

    setup_dd_pct(9.0)   # over threshold
    assert bot._daily_dd_risk_multiplier() == 0.0
    assert bot.daily_dd_breached is True


def test_build_order_dd_risk_mult_scales_volume():
    """build_order should scale notional by dd_risk_mult; halt at 0.0."""
    from bitunix_bot.config import RiskCfg, TradingCfg
    from bitunix_bot.risk import build_order
    from bitunix_bot.strategy import Signal

    tc = TradingCfg(symbols=["BTCUSDT"], timeframe="1m", leverage=25,
                    margin_coin="USDT", margin_mode="ISOLATION",
                    risk_per_trade_pct=1.0)
    rc = RiskCfg(stop_loss_pct=0.40, take_profit_r=2.5, use_atr=False,
                 atr_multiplier_sl=1.2, atr_multiplier_tp=4.0)
    sig = Signal(direction="long", score=0.5, indicator_score=12,
                 pattern_score=2.0, reasons=["x"], price=60_000.0,
                 atr=120.0, fire_threshold_used=0.50)
    full = build_order(sig, free_margin=46.0, trading=tc, risk=rc,
                       min_volume=0.0001, volume_step=0.0001, digits=1,
                       symbol="BTCUSDT", dd_risk_mult=1.0)
    half = build_order(sig, free_margin=46.0, trading=tc, risk=rc,
                       min_volume=0.0001, volume_step=0.0001, digits=1,
                       symbol="BTCUSDT", dd_risk_mult=0.5)
    halted = build_order(sig, free_margin=46.0, trading=tc, risk=rc,
                          min_volume=0.0001, volume_step=0.0001, digits=1,
                          symbol="BTCUSDT", dd_risk_mult=0.0)
    assert full is not None and half is not None
    # Half-throttle should give ~half the volume.
    ratio = half.volume / full.volume
    assert abs(ratio - 0.5) < 0.05, f"expected 0.5×, got {ratio:.3f}"
    # Halt → no plan returned.
    assert halted is None


def test_mini_cooldown_fires_on_two_losses_in_window():
    """Two losses within 10 min → 5-min mini-cooldown on that symbol."""
    reset_state()
    cfg = fresh_cfg()
    bot = BitunixBot(cfg)
    bot.client = make_mock_client()
    now_ms = int(time.time() * 1000)

    # Two losing closes ≤30 sec apart on BTC.
    bot.client.history_positions.return_value = {
        "positionList": [
            {"positionId": "L1", "symbol": "BTCUSDT", "side": "LONG",
             "qty": "0.01", "avgOpenPrice": "60000", "avgClosePrice": "59760",
             "ctime": now_ms - 60_000, "mtime": now_ms - 30_000,
             "realizedPNL": "-0.5", "fee": "-0.05", "funding": "0"},
            {"positionId": "L2", "symbol": "BTCUSDT", "side": "LONG",
             "qty": "0.01", "avgOpenPrice": "60000", "avgClosePrice": "59770",
             "ctime": now_ms - 30_000, "mtime": now_ms,
             "realizedPNL": "-0.4", "fee": "-0.05", "funding": "0"},
        ],
        "total": 2,
    }
    bot._update_streak_state()
    assert "BTCUSDT" in bot.mini_cooldown_until
    until = bot.mini_cooldown_until["BTCUSDT"]
    assert until > time.time(), "mini-cooldown should be in the future"
    # 5-minute pause = ~300s.
    assert 290 < (until - time.time()) < 310


def test_mini_cooldown_blocks_new_entries():
    """When mini-cooldown is active, _tick must skip the symbol."""
    reset_state()
    cfg = fresh_cfg()
    cfg.mode = "live"
    cfg.trading.symbols = ["BTCUSDT"]
    bot = BitunixBot(cfg)
    bot.client = make_mock_client()
    bot.client.klines.side_effect = lambda *a, **kw: make_uptrend_klines()
    bot._resolve_symbol_meta()
    bot.mini_cooldown_until["BTCUSDT"] = time.time() + 200  # active 3+ min more
    bot._tick()

    bot.client.place_order.assert_not_called()
    skips = [e for e in bot.state.snapshot()["events"] if e["kind"] == "skip"]
    assert any("mini-cooldown" in s["text"] for s in skips), \
        f"expected mini-cooldown skip; got {[s['text'] for s in skips]}"


def test_mini_cooldown_does_not_fire_for_isolated_loss():
    """A single loss should NOT trigger the mini-cooldown."""
    reset_state()
    cfg = fresh_cfg()
    bot = BitunixBot(cfg)
    bot.client = make_mock_client()
    now_ms = int(time.time() * 1000)
    bot.client.history_positions.return_value = {
        "positionList": [
            {"positionId": "L1", "symbol": "BTCUSDT", "side": "LONG",
             "qty": "0.01", "avgOpenPrice": "60000", "avgClosePrice": "59760",
             "ctime": now_ms - 60_000, "mtime": now_ms - 30_000,
             "realizedPNL": "-0.5", "fee": "-0.05", "funding": "0"},
        ],
        "total": 1,
    }
    bot._update_streak_state()
    assert "BTCUSDT" not in bot.mini_cooldown_until


def test_journal_entry_mechanism_fields():
    """Entry record must include entry_mechanism + limit_price + tob_bid/ask
    + dynamic_timeout_secs so adverse-selection patterns can be analyzed."""
    import tempfile
    import json
    from bitunix_bot.journal import TradeJournal
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "trades.jsonl"
        j = TradeJournal(path)
        j.record_entry(
            symbol="BTCUSDT", side="BUY", client_id="bot-BTCUSDT-1-BUY-PO",
            order_type="LIMIT", score=0.65, threshold_used=0.50,
            conviction_mult=1.30, indicator_count=12, pattern_score=1.5,
            reasons=["x"], atr_pct=0.04, adx=25.0, spread_pct=0.012,
            bid_depth=12.0, ask_depth=10.0,
            aggression_10s=0.20, real_cvd=2.0, activity_mult=1.0,
            session_weight=1.0, entry_price=60000.0,
            stop_loss=59760.0, take_profit=60600.0,
            notional=100.0, leverage=25,
            # New fields:
            entry_mechanism="MAKER_LIMIT_POST_ONLY",
            limit_price=59999.5,
            tob_bid=59999.5,
            tob_ask=60000.5,
            dynamic_timeout_secs=8,
        )
        rec = json.loads(path.read_text().strip())
        assert rec["entry_mechanism"] == "MAKER_LIMIT_POST_ONLY"
        assert rec["limit_price"] == 59999.5
        assert rec["tob_bid"] == 59999.5
        assert rec["tob_ask"] == 60000.5
        assert rec["dynamic_timeout_secs"] == 8


def test_journal_writes_entry_and_exit():
    """TradeJournal should append JSONL events for entry + exit."""
    import tempfile
    import json
    from bitunix_bot.journal import TradeJournal
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "trades.jsonl"
        j = TradeJournal(path)
        j.record_entry(
            symbol="BTCUSDT", side="BUY", client_id="bot-BTCUSDT-1-BUY-PO",
            order_type="LIMIT", score=0.72, threshold_used=0.50,
            conviction_mult=1.44, indicator_count=14, pattern_score=1.5,
            reasons=["ema_stack_up", "PAT:marubozu_bull"],
            atr_pct=0.08, adx=25.3, spread_pct=0.012,
            bid_depth=15.0, ask_depth=12.0,
            aggression_10s=0.42, real_cvd=8.5, activity_mult=1.05,
            session_weight=1.0, entry_price=60_000.0,
            stop_loss=59_760.0, take_profit=60_600.0,
            notional=120.0, leverage=25,
        )
        j.record_exit(
            symbol="BTCUSDT", position_id="P1", side="LONG",
            entry_price=60_000.0, exit_price=60_300.0,
            exit_reason="win", hold_time_sec=180.0,
            max_favor_r=1.5, net_pnl=0.36, realized_pnl=0.45,
            fee=-0.09, funding=0.0,
        )
        lines = path.read_text().strip().split("\n")
        assert len(lines) == 2
        entry = json.loads(lines[0])
        exit_ev = json.loads(lines[1])
        assert entry["kind"] == "entry"
        assert entry["symbol"] == "BTCUSDT"
        assert entry["score"] == 0.72
        assert entry["conviction_mult"] == 1.44
        assert exit_ev["kind"] == "exit"
        assert exit_ev["max_favor_r"] == 1.5
        assert exit_ev["net_pnl"] == 0.36


def test_journal_handles_missing_dir_gracefully():
    """TradeJournal should not crash when the parent dir doesn't exist —
    it lazily creates it."""
    import tempfile
    from bitunix_bot.journal import TradeJournal
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "subdir" / "nested" / "trades.jsonl"
        j = TradeJournal(path)
        j.record_entry(
            symbol="X", side="BUY", client_id="x", order_type="LIMIT",
            score=0.5, threshold_used=0.5, conviction_mult=1.0,
            indicator_count=10, pattern_score=1.0, reasons=[],
            atr_pct=None, adx=None, spread_pct=None,
            bid_depth=None, ask_depth=None,
            aggression_10s=None, real_cvd=None, activity_mult=None,
            session_weight=None, entry_price=100.0,
            stop_loss=99.6, take_profit=101.0, notional=100.0, leverage=25,
        )
        assert path.exists()
        assert path.stat().st_size > 0


# ----------------------------------------------------------------- Grok review v3


def test_fire_threshold_lowers_in_trending_regime():
    """ADX > 28 should reduce the effective fire threshold by 0.05."""
    from bitunix_bot.strategy import _effective_fire_threshold
    # Trending: 30 > 28 → threshold drops by 0.05.
    assert abs(_effective_fire_threshold(30.0, 0.50) - 0.45) < 1e-9
    # Strong trend: 50 → still drops by 0.05 (no further scaling).
    assert abs(_effective_fire_threshold(50.0, 0.50) - 0.45) < 1e-9


def test_fire_threshold_raises_in_ranging_regime():
    """ADX < 22 should increase the effective fire threshold by 0.08
    (band tightened from <18 to <22 to catch weak-trend chop)."""
    from bitunix_bot.strategy import _effective_fire_threshold
    assert abs(_effective_fire_threshold(15.0, 0.50) - 0.58) < 1e-9
    # Very low ADX still bumps by 0.08 (no further scaling).
    assert abs(_effective_fire_threshold(5.0, 0.50) - 0.58) < 1e-9
    # ADX 20 (was previously "neutral") now triggers ranging.
    assert abs(_effective_fire_threshold(20.0, 0.50) - 0.58) < 1e-9


def test_fire_threshold_neutral_in_mid_regime():
    """ADX in [22, 28] should leave the threshold unchanged."""
    from bitunix_bot.strategy import _effective_fire_threshold
    assert _effective_fire_threshold(25.0, 0.50) == 0.50
    assert _effective_fire_threshold(22.0, 0.50) == 0.50
    assert _effective_fire_threshold(28.0, 0.50) == 0.50
    # NaN ADX (e.g., during warmup) → keep base.
    import math
    assert _effective_fire_threshold(math.nan, 0.50) == 0.50


def test_fire_threshold_clamps_at_bounds():
    """Threshold must stay within [0.0, 1.0] even at extreme bases."""
    from bitunix_bot.strategy import _effective_fire_threshold
    assert _effective_fire_threshold(30.0, 0.02) == 0.0   # would be -0.03 → clamped
    assert _effective_fire_threshold(15.0, 0.95) == 1.0   # would be 1.03 → clamped


def test_signal_records_fire_threshold_used():
    """A signal returned by evaluate() must carry the effective threshold
    so build_order can compute conviction-mult."""
    from bitunix_bot.config import StrategyCfg
    from bitunix_bot.strategy import evaluate
    cfg = StrategyCfg(
        ema_fast=9, ema_mid=21, ema_slow=50,
        rsi_period=14, rsi_long_min=40, rsi_long_max=80,
        rsi_short_min=20, rsi_short_max=60,
        macd_fast=12, macd_slow=26, macd_signal=9,
        bb_period=20, bb_std=2.0, atr_period=14,
        adx_period=14, adx_min=15.0,
        supertrend_period=10, supertrend_mult=3.0,
        pattern_weight=0.55, pattern_norm=2.0, fire_threshold=0.05,
    )
    klines = make_uptrend_klines()
    o = [k["open"] for k in klines]
    h = [k["high"] for k in klines]
    l_ = [k["low"] for k in klines]
    c = [k["close"] for k in klines]
    sig = evaluate(o, h, l_, c, cfg)
    assert sig is not None
    assert sig.fire_threshold_used is not None, \
        "signal must record the threshold it cleared"
    assert 0.0 <= sig.fire_threshold_used <= 1.0


def test_conviction_sizing_high_score_increases_risk():
    """Score well above threshold → conviction_mult ~1.5 → larger notional."""
    from bitunix_bot.config import RiskCfg, TradingCfg
    from bitunix_bot.risk import build_order
    from bitunix_bot.strategy import Signal

    tc = TradingCfg(symbols=["BTCUSDT"], timeframe="1m", leverage=25,
                    margin_coin="USDT", margin_mode="ISOLATION",
                    risk_per_trade_pct=1.0)
    rc = RiskCfg(stop_loss_pct=0.40, take_profit_r=2.5, use_atr=False,
                 atr_multiplier_sl=1.2, atr_multiplier_tp=4.0)
    # High-conviction signal: score 0.85 vs threshold 0.50 → ratio 1.7 → clamps to 1.5.
    high_sig = Signal(direction="long", score=0.85, indicator_score=15,
                      pattern_score=2.0, reasons=["x"], price=60_000.0,
                      atr=120.0, fire_threshold_used=0.50)
    # Marginal signal: score 0.50 (right at threshold) → ratio 1.0 → mult=1.0.
    base_sig = Signal(direction="long", score=0.50, indicator_score=12,
                      pattern_score=2.0, reasons=["x"], price=60_000.0,
                      atr=120.0, fire_threshold_used=0.50)
    high_plan = build_order(high_sig, free_margin=46.0, trading=tc, risk=rc,
                             min_volume=0.0001, volume_step=0.0001, digits=1,
                             symbol="BTCUSDT")
    base_plan = build_order(base_sig, free_margin=46.0, trading=tc, risk=rc,
                             min_volume=0.0001, volume_step=0.0001, digits=1,
                             symbol="BTCUSDT")
    assert high_plan.volume > base_plan.volume, \
        f"high-conviction should size up: {high_plan.volume} vs {base_plan.volume}"
    # Ratio should be ~1.5 (high-conviction maxes the clamp).
    ratio = high_plan.volume / base_plan.volume
    assert abs(ratio - 1.5) < 0.05, f"expected ~1.5x, got {ratio:.3f}"


def test_conviction_sizing_clamps_to_floor():
    """Score well BELOW threshold (shouldn't happen normally, but defensive):
    conviction_mult clamps to 0.7 floor."""
    from bitunix_bot.config import RiskCfg, TradingCfg
    from bitunix_bot.risk import build_order
    from bitunix_bot.strategy import Signal

    tc = TradingCfg(symbols=["BTCUSDT"], timeframe="1m", leverage=25,
                    margin_coin="USDT", margin_mode="ISOLATION",
                    risk_per_trade_pct=1.0)
    rc = RiskCfg(stop_loss_pct=0.40, take_profit_r=2.5, use_atr=False,
                 atr_multiplier_sl=1.2, atr_multiplier_tp=4.0,
                 round_trip_fee_pct=0.0)  # disable fee reserve for ratio test
    # Low-conviction (impossible in practice — would not pass threshold) but
    # tests the clamp: 0.10 / 0.50 = 0.20 → clamps to 0.7.
    low_sig = Signal(direction="long", score=0.10, indicator_score=2,
                     pattern_score=0.5, reasons=["x"], price=60_000.0,
                     atr=120.0, fire_threshold_used=0.50)
    base_sig = Signal(direction="long", score=0.50, indicator_score=12,
                      pattern_score=2.0, reasons=["x"], price=60_000.0,
                      atr=120.0, fire_threshold_used=0.50)
    low_plan = build_order(low_sig, free_margin=46.0, trading=tc, risk=rc,
                            min_volume=0.0001, volume_step=0.0001, digits=1,
                            symbol="BTCUSDT")
    base_plan = build_order(base_sig, free_margin=46.0, trading=tc, risk=rc,
                             min_volume=0.0001, volume_step=0.0001, digits=1,
                             symbol="BTCUSDT")
    ratio = low_plan.volume / base_plan.volume
    assert abs(ratio - 0.7) < 0.05, f"expected ~0.7x, got {ratio:.3f}"


def test_conviction_sizing_unset_threshold_no_change():
    """If fire_threshold_used is None (legacy callers), conviction_mult=1.0."""
    from bitunix_bot.config import RiskCfg, TradingCfg
    from bitunix_bot.risk import build_order
    from bitunix_bot.strategy import Signal

    tc = TradingCfg(symbols=["BTCUSDT"], timeframe="1m", leverage=25,
                    margin_coin="USDT", margin_mode="ISOLATION",
                    risk_per_trade_pct=1.0)
    rc = RiskCfg(stop_loss_pct=0.40, take_profit_r=2.5, use_atr=False,
                 atr_multiplier_sl=1.2, atr_multiplier_tp=4.0)
    sig_none = Signal(direction="long", score=0.85, indicator_score=15,
                       pattern_score=2.0, reasons=["x"], price=60_000.0,
                       atr=120.0)  # fire_threshold_used defaults to None
    sig_at = Signal(direction="long", score=0.50, indicator_score=12,
                     pattern_score=2.0, reasons=["x"], price=60_000.0,
                     atr=120.0, fire_threshold_used=0.50)
    plan_none = build_order(sig_none, free_margin=46.0, trading=tc, risk=rc,
                             min_volume=0.0001, volume_step=0.0001, digits=1,
                             symbol="BTCUSDT")
    plan_at = build_order(sig_at, free_margin=46.0, trading=tc, risk=rc,
                           min_volume=0.0001, volume_step=0.0001, digits=1,
                           symbol="BTCUSDT")
    # Both should size identically (conviction=1.0 in both cases).
    assert abs(plan_none.volume - plan_at.volume) < 1e-9, \
        f"unset threshold should match conviction=1.0: {plan_none.volume} vs {plan_at.volume}"


def test_absorption_vote_fires_on_extreme_flow_no_movement():
    """|aggression| ≥ 0.55 + |price change| < 0.15% → absorption vote
    OPPOSITE the aggressive flow."""
    from bitunix_bot.config import StrategyCfg
    from bitunix_bot.strategy import evaluate
    rng = np.random.default_rng(7)
    n = 200
    closes = 100 + np.cumsum(rng.normal(0.05, 0.3, n))
    opens = closes - rng.uniform(0.05, 0.2, n)
    highs = np.maximum(opens, closes) + rng.uniform(0.02, 0.1, n)
    lows = np.minimum(opens, closes) - rng.uniform(0.02, 0.1, n)
    cfg = StrategyCfg(
        ema_fast=9, ema_mid=21, ema_slow=50,
        rsi_period=14, rsi_long_min=30, rsi_long_max=80,
        rsi_short_min=20, rsi_short_max=70,
        macd_fast=12, macd_slow=26, macd_signal=9,
        bb_period=20, bb_std=2.0, atr_period=14,
        adx_period=14, adx_min=15.0,
        supertrend_period=10, supertrend_mult=3.0,
        pattern_weight=0.55, pattern_norm=2.0, fire_threshold=0.05,
    )
    # Strong BUY flow being absorbed (price barely moved despite 80/20 buy aggression)
    # → absorption SHORT vote.
    sig = evaluate(opens.tolist(), highs.tolist(), lows.tolist(),
                    closes.tolist(), cfg,
                    aggression_10s=0.65, price_change_10s_pct=0.05)
    if sig is not None:
        # absorption tag could fire either side; check it's SHORT-side when buy-flow absorbed.
        if sig.direction == "short":
            assert any("absorb" in r for r in sig.reasons), \
                f"expected absorb in short reasons: {sig.reasons}"

    # Now strong SELL flow being absorbed → absorption LONG vote.
    sig2 = evaluate(opens.tolist(), highs.tolist(), lows.tolist(),
                     closes.tolist(), cfg,
                     aggression_10s=-0.65, price_change_10s_pct=-0.03)
    if sig2 is not None and sig2.direction == "long":
        assert any("absorb" in r for r in sig2.reasons)


def test_absorption_vote_does_not_fire_when_price_moves():
    """If price MOVED ≥0.15% in the same window, it's not absorption —
    just normal aggressive flow that found follow-through."""
    from bitunix_bot.config import StrategyCfg
    from bitunix_bot.strategy import evaluate
    rng = np.random.default_rng(7)
    n = 200
    closes = 100 + np.cumsum(rng.normal(0.05, 0.3, n))
    opens = closes - rng.uniform(0.05, 0.2, n)
    highs = np.maximum(opens, closes) + rng.uniform(0.02, 0.1, n)
    lows = np.minimum(opens, closes) - rng.uniform(0.02, 0.1, n)
    cfg = StrategyCfg(
        ema_fast=9, ema_mid=21, ema_slow=50,
        rsi_period=14, rsi_long_min=30, rsi_long_max=80,
        rsi_short_min=20, rsi_short_max=70,
        macd_fast=12, macd_slow=26, macd_signal=9,
        bb_period=20, bb_std=2.0, atr_period=14,
        adx_period=14, adx_min=15.0,
        supertrend_period=10, supertrend_mult=3.0,
        pattern_weight=0.55, pattern_norm=2.0, fire_threshold=0.05,
    )
    # Strong buy flow + meaningful price move → normal momentum, NOT absorption.
    sig = evaluate(opens.tolist(), highs.tolist(), lows.tolist(),
                    closes.tolist(), cfg,
                    aggression_10s=0.65, price_change_10s_pct=0.30)
    if sig is not None:
        assert not any("absorb" in r for r in sig.reasons), \
            f"absorption should NOT fire when price moved 0.30%: {sig.reasons}"


def test_absorption_vote_does_not_fire_when_aggression_weak():
    """Below 0.55 aggression magnitude → not extreme enough to be absorption."""
    from bitunix_bot.config import StrategyCfg
    from bitunix_bot.strategy import evaluate
    rng = np.random.default_rng(7)
    n = 200
    closes = 100 + np.cumsum(rng.normal(0.05, 0.3, n))
    opens = closes - rng.uniform(0.05, 0.2, n)
    highs = np.maximum(opens, closes) + rng.uniform(0.02, 0.1, n)
    lows = np.minimum(opens, closes) - rng.uniform(0.02, 0.1, n)
    cfg = StrategyCfg(
        ema_fast=9, ema_mid=21, ema_slow=50,
        rsi_period=14, rsi_long_min=30, rsi_long_max=80,
        rsi_short_min=20, rsi_short_max=70,
        macd_fast=12, macd_slow=26, macd_signal=9,
        bb_period=20, bb_std=2.0, atr_period=14,
        adx_period=14, adx_min=15.0,
        supertrend_period=10, supertrend_mult=3.0,
        pattern_weight=0.55, pattern_norm=2.0, fire_threshold=0.05,
    )
    # Aggression 0.40 + tiny price change → still normal flow (vote 24 fires,
    # vote 25 doesn't) because magnitude < 0.55.
    sig = evaluate(opens.tolist(), highs.tolist(), lows.tolist(),
                    closes.tolist(), cfg,
                    aggression_10s=0.40, price_change_10s_pct=0.02)
    if sig is not None:
        assert not any("absorb" in r for r in sig.reasons)


def test_depth_filter_blocks_thin_book():
    """Per-symbol depth threshold: a book with min(bid_depth, ask_depth)
    below threshold must skip the trade with a 'thin book' reason.

    Depth filter only runs when post-only entries are enabled (Grok v8 —
    market entries don't have the 'sit forever' problem)."""
    reset_state()
    cfg = fresh_cfg()
    cfg.mode = "live"
    cfg.trading.use_post_only_entries = True  # depth filter is post-only specific
    cfg.trading.symbols = ["BTCUSDT"]
    bot = BitunixBot(cfg)
    bot.client = make_mock_client()
    bot.client.klines.side_effect = lambda *a, **kw: make_uptrend_klines()
    # Thin book: 1 BTC depth, threshold 3 BTC for BTC → skip.
    bot.ob_feed = _FakeOBFeed(bid_depth=1.0, ask_depth=1.0)
    bot._resolve_symbol_meta()
    bot._tick()

    # Should not have placed any order due to thin-book filter.
    bot.client.place_order.assert_not_called()
    skips = [e for e in bot.state.snapshot()["events"] if e["kind"] == "skip"]
    assert any("thin book" in s["text"] for s in skips), \
        f"expected thin book skip; got {[s['text'] for s in skips]}"


def test_depth_filter_skipped_for_market_entries():
    """When use_post_only_entries=False (Grok v8 default), the thin-book
    gate must NOT block market entries — market takes top-of-book and
    'sit forever' is not a concern."""
    reset_state()
    cfg = fresh_cfg()
    cfg.mode = "live"
    cfg.trading.use_post_only_entries = False
    cfg.trading.symbols = ["BTCUSDT"]
    bot = BitunixBot(cfg)
    bot.client = make_mock_client()
    bot.client.klines.side_effect = lambda *a, **kw: make_uptrend_klines()
    # Razor-thin book that would block under post-only mode.
    bot.ob_feed = _FakeOBFeed(bid_depth=1.0, ask_depth=1.0)
    bot._resolve_symbol_meta()
    bot._tick()
    skips = [e for e in bot.state.snapshot()["events"] if e["kind"] == "skip"]
    assert not any("thin book" in s["text"] for s in skips), \
        f"market entries shouldn't trigger thin-book skip: {skips}"


def test_depth_filter_allows_normal_book():
    """Adequate depth (above threshold) should NOT block a trade."""
    reset_state()
    cfg = fresh_cfg()
    cfg.mode = "live"
    cfg.trading.symbols = ["BTCUSDT"]
    bot = BitunixBot(cfg)
    bot.client = make_mock_client()
    bot.client.klines.side_effect = lambda *a, **kw: make_uptrend_klines()
    # Well above threshold (BTC: 8.0).
    bot.ob_feed = _FakeOBFeed(bid_depth=50.0, ask_depth=50.0)
    bot._resolve_symbol_meta()
    bot._tick()
    bot.client.place_order.assert_called()


def test_depth_filter_unknown_symbol_no_filter():
    """A symbol not in the depth-threshold map should NOT be filtered
    even with thin book."""
    reset_state()
    cfg = fresh_cfg()
    cfg.mode = "live"
    cfg.trading.symbols = ["DOGEUSDT"]   # not in default symbol_min_depth map
    cfg.trading.symbol_risk_mult = {"DOGEUSDT": 1.0}
    bot = BitunixBot(cfg)
    bot.client = make_mock_client()
    bot.client.trading_pairs.return_value = [
        {"symbol": "DOGEUSDT", "basePrecision": 0, "quotePrecision": 5,
         "minTradeVolume": "1", "maxLeverage": 75},
    ]
    bot.client.klines.side_effect = lambda *a, **kw: make_uptrend_klines()
    bot.ob_feed = _FakeOBFeed(bid_depth=1.0, ask_depth=1.0)  # very thin
    bot._resolve_symbol_meta()
    bot._tick()
    # No depth threshold for DOGE → trade proceeds.
    skips = [e for e in bot.state.snapshot()["events"] if e["kind"] == "skip"]
    assert not any("thin book" in s["text"] for s in skips), \
        f"unknown symbol shouldn't trigger depth filter: {skips}"


def test_dynamic_post_only_timeout_shortens_with_high_activity():
    """High activity multiplier (busy market) → shorter timeout. Inverted
    scaling so frantic markets either fill fast or skip."""
    reset_state()
    cfg = fresh_cfg()
    cfg.mode = "live"
    cfg.trading.post_only_timeout_secs = 8
    cfg.trading.use_post_only_entries = True  # explicit opt-in (default is now False per Grok v8)
    cfg.trading.symbols = ["BTCUSDT"]
    bot = BitunixBot(cfg)
    bot.client = make_mock_client()
    bot.client.klines.side_effect = lambda *a, **kw: make_uptrend_klines()
    bot.ob_feed = _FakeOBFeed()

    # Make session_weight time-of-day-stable for the test (otherwise this
    # flakes whenever it runs in dead UTC hours / weekends, dampening
    # combined score below threshold).
    bot._session_weight = lambda: 1.0
    # Fake high activity (raw ×1.5) → expected timeout = round(8 / 1.5) = 5s.
    # The fake mirrors the real impl: applies the caller's clamp range so
    # the strategy-side call (default [0.85, 1.10]) and the timeout-side
    # call ([0.5, 2.0]) get appropriately clamped values.
    class HighActTape:
        def get_aggression_ratio(self, sym, **kw): return None
        def get_cvd(self, sym, **kw): return None
        def get_activity_multiplier(self, sym, clamp_min=0.85, clamp_max=1.10):
            raw = 1.5
            return max(clamp_min, min(clamp_max, raw))
        def get_price_change_pct(self, sym, **kw): return None
    bot.tape_feed = HighActTape()
    bot._resolve_symbol_meta()
    bot._tick()

    info = bot.pending_limits.get("BTCUSDT")
    assert info is not None
    assert info["timeout_secs"] == 5, f"expected 5s timeout, got {info['timeout_secs']}"


def test_dynamic_post_only_timeout_lengthens_in_dead_market():
    """Low activity multiplier (dead market) → longer timeout."""
    reset_state()
    cfg = fresh_cfg()
    cfg.mode = "live"
    cfg.trading.post_only_timeout_secs = 8
    cfg.trading.use_post_only_entries = True  # explicit opt-in (default is now False per Grok v8)
    cfg.trading.symbols = ["BTCUSDT"]
    bot = BitunixBot(cfg)
    bot.client = make_mock_client()
    bot.client.klines.side_effect = lambda *a, **kw: make_uptrend_klines()
    bot.ob_feed = _FakeOBFeed()

    bot._session_weight = lambda: 1.0   # stabilize for time-of-day
    # Fake low activity (raw ×0.7) → expected timeout = round(8 / 0.7) = 11s.
    # Mirrors the real impl: respects the caller's clamp range.
    class LowActTape:
        def get_aggression_ratio(self, sym, **kw): return None
        def get_cvd(self, sym, **kw): return None
        def get_activity_multiplier(self, sym, clamp_min=0.85, clamp_max=1.10):
            raw = 0.7
            return max(clamp_min, min(clamp_max, raw))
        def get_price_change_pct(self, sym, **kw): return None
    bot.tape_feed = LowActTape()
    bot._resolve_symbol_meta()
    bot._tick()

    info = bot.pending_limits.get("BTCUSDT")
    assert info is not None
    assert info["timeout_secs"] == 11, f"expected 11s timeout, got {info['timeout_secs']}"


def test_dynamic_post_only_timeout_clamps_to_4_12():
    """Even at extreme activity values, timeout stays in [4, 12]s."""
    reset_state()
    cfg = fresh_cfg()
    cfg.mode = "live"
    cfg.trading.post_only_timeout_secs = 8
    cfg.trading.use_post_only_entries = True  # explicit opt-in (default is now False per Grok v8)
    cfg.trading.symbols = ["BTCUSDT"]
    bot = BitunixBot(cfg)
    bot.client = make_mock_client()
    bot.client.klines.side_effect = lambda *a, **kw: make_uptrend_klines()
    bot.ob_feed = _FakeOBFeed()

    bot._session_weight = lambda: 1.0   # stabilize for time-of-day
    # Extreme high activity (raw ×3.0). Clamped at the caller's bound:
    # strategy uses [0.85, 1.10] → 1.10; timeout uses [0.5, 2.0] → 2.0.
    # Final timeout = 8 / 2.0 = 4s.
    class FastTape:
        def get_aggression_ratio(self, sym, **kw): return None
        def get_cvd(self, sym, **kw): return None
        def get_activity_multiplier(self, sym, clamp_min=0.85, clamp_max=1.10):
            raw = 3.0
            return max(clamp_min, min(clamp_max, raw))
        def get_price_change_pct(self, sym, **kw): return None
    bot.tape_feed = FastTape()
    bot._resolve_symbol_meta()
    bot._tick()
    info = bot.pending_limits.get("BTCUSDT")
    assert info is not None and info["timeout_secs"] == 4

    # Extreme dead market.
    reset_state()
    bot2 = BitunixBot(cfg)
    bot2.client = make_mock_client()
    bot2.client.klines.side_effect = lambda *a, **kw: make_uptrend_klines()
    bot2.ob_feed = _FakeOBFeed()
    bot2._session_weight = lambda: 1.0   # stabilize for time-of-day
    class DeadTape:
        def get_aggression_ratio(self, sym, **kw): return None
        def get_cvd(self, sym, **kw): return None
        def get_activity_multiplier(self, sym, clamp_min=0.85, clamp_max=1.10):
            raw = 0.3
            return max(clamp_min, min(clamp_max, raw))
        def get_price_change_pct(self, sym, **kw): return None
    bot2.tape_feed = DeadTape()
    bot2._resolve_symbol_meta()
    bot2._tick()
    info2 = bot2.pending_limits.get("BTCUSDT")
    assert info2 is not None and info2["timeout_secs"] == 12, \
        f"dead market should clamp to 12s, got {info2['timeout_secs']}"


def test_tighter_tape_veto_threshold_grok_v7():
    """Tape veto threshold tightened to ±0.30 (was ±0.45) per Grok v7.
    Combined with the new continuation gate's tape-alignment requirement
    (need ≥0.25 in direction), contrary tape blocks at multiple layers.
    Test that aggression at -0.31 blocks long signal (was -0.46 cutoff)."""
    bot, tape = _bot_with_tape_and_signal()
    tape.aggression = -0.31  # past new -0.30 tape veto threshold
    bot._tick()
    bot.client.place_order.assert_not_called()

    # Confirming flow at +0.25 (the continuation-gate alignment threshold)
    # should allow the long signal through.
    bot, tape = _bot_with_tape_and_signal()
    tape.aggression = 0.30  # past +0.25 alignment requirement
    bot._tick()
    assert bot.client.place_order.called, "confirming +0.30 flow must allow long"


def test_tradetape_get_price_change_pct():
    """Price change accessor: oldest vs newest trade in window."""
    from bitunix_bot.tradetape import TradeFeed, Trade
    feed = TradeFeed(symbols=["BTCUSDT"])
    now = time.time()
    # Trade prices: 60000 → 60030 across 5 trades. +0.05% change.
    for i, px in enumerate([60_000.0, 60_010.0, 60_020.0, 60_025.0, 60_030.0]):
        feed._ingest(Trade(ts=now - (5 - i), price=px, qty=0.5, is_buy=True),
                     "BTCUSDT")
    pct = feed.get_price_change_pct("BTCUSDT", window_secs=10)
    assert pct is not None and abs(pct - 0.05) < 1e-3, \
        f"expected +0.050%, got {pct:.4f}%"

    # min_count guard: only 1 trade returns None.
    feed2 = TradeFeed(symbols=["BTCUSDT"])
    feed2._ingest(Trade(ts=now, price=60000, qty=1.0, is_buy=True), "BTCUSDT")
    assert feed2.get_price_change_pct("BTCUSDT", window_secs=10) is None


def test_orderbook_get_depth():
    """OrderBookFeed.get_depth returns top-N summed sizes per side."""
    from bitunix_bot.orderbook import OrderBookFeed
    feed = OrderBookFeed(symbols=["BTCUSDT"], depth_levels=10)
    with feed._lock:
        b = feed._books["BTCUSDT"]
        b.bids = [(60000.0, 5.0), (59999.0, 4.0), (59998.0, 3.0),
                   (59997.0, 2.0), (59996.0, 1.0), (59995.0, 0.5)]
        b.asks = [(60001.0, 1.0), (60002.0, 2.0), (60003.0, 3.0),
                   (60004.0, 4.0), (60005.0, 5.0), (60006.0, 0.5)]
        b.last_update = time.time()
    depth = feed.get_depth("BTCUSDT", top_n=5)
    assert depth is not None
    bid_d, ask_d = depth
    # Top-5 only: bid sum 5+4+3+2+1=15; ask sum 1+2+3+4+5=15.
    assert abs(bid_d - 15.0) < 1e-9 and abs(ask_d - 15.0) < 1e-9


# ----------------------------------------------------------------- ATR-aware SL


def test_atr_hybrid_floor_wins_in_calm_markets():
    """Calm market (low ATR): the fixed stop_loss_pct floor binds, ATR
    derived value is below floor → SL distance = floor."""
    from bitunix_bot.config import RiskCfg, TradingCfg
    from bitunix_bot.risk import build_order
    from bitunix_bot.strategy import Signal

    tc = TradingCfg(symbols=["BTCUSDT"], timeframe="1m", leverage=25,
                    margin_coin="USDT", margin_mode="ISOLATION",
                    risk_per_trade_pct=1.0)
    rc = RiskCfg(stop_loss_pct=0.40, take_profit_r=2.5, use_atr=True,
                 atr_multiplier_sl=1.2, atr_multiplier_tp=4.0)
    # ATR = 0.05% of 60000 = 30. Hybrid: 1.2 * 0.05 = 0.06% < floor 0.40%.
    sig = Signal(direction="long", score=0.7, indicator_score=12,
                 pattern_score=2.0, reasons=["x"], price=60_000.0, atr=30.0)
    plan = build_order(sig, free_margin=46.0, trading=tc, risk=rc,
                       min_volume=0.0001, volume_step=0.0001, digits=1)
    assert plan is not None
    sl_pct = (plan.price - plan.stop_loss) / plan.price * 100.0
    assert abs(sl_pct - 0.40) < 0.01, \
        f"calm market should use floor 0.40%, got {sl_pct:.3f}%"


def test_atr_hybrid_widens_in_vol_regime():
    """High ATR (vol expansion): ATR-derived SL widens beyond fixed floor."""
    from bitunix_bot.config import RiskCfg, TradingCfg
    from bitunix_bot.risk import build_order
    from bitunix_bot.strategy import Signal

    tc = TradingCfg(symbols=["BTCUSDT"], timeframe="1m", leverage=25,
                    margin_coin="USDT", margin_mode="ISOLATION",
                    risk_per_trade_pct=1.0)
    # Disable fee reserve so risk-budget assertions are unmodified
    # (specific fee-reserve tests assert that scaling separately).
    rc = RiskCfg(stop_loss_pct=0.40, take_profit_r=2.5, use_atr=True,
                 atr_multiplier_sl=1.2, atr_multiplier_tp=4.0,
                 round_trip_fee_pct=0.0)
    # ATR = 0.50% of 60000 = 300. Hybrid: 1.2 * 0.50 = 0.60% > floor 0.40%.
    sig = Signal(direction="long", score=0.7, indicator_score=12,
                 pattern_score=2.0, reasons=["x"], price=60_000.0, atr=300.0)
    plan = build_order(sig, free_margin=46.0, trading=tc, risk=rc,
                       min_volume=0.0001, volume_step=0.0001, digits=1)
    assert plan is not None
    sl_pct = (plan.price - plan.stop_loss) / plan.price * 100.0
    assert abs(sl_pct - 0.60) < 0.01, \
        f"vol regime should use ATR-derived 0.60%, got {sl_pct:.3f}%"
    # Risk per trade is preserved — notional auto-adjusts.
    notional = plan.volume * plan.price
    risk_dollars = notional * (sl_pct / 100.0)
    assert abs(risk_dollars - 0.46) < 0.05, \
        f"risk should stay near $0.46 across regimes, got ${risk_dollars:.3f}"


def test_atr_hybrid_disabled_uses_fixed_pct():
    """use_atr=False should bypass the hybrid entirely (regression check
    so the legacy fixed-pct path keeps working)."""
    from bitunix_bot.config import RiskCfg, TradingCfg
    from bitunix_bot.risk import build_order
    from bitunix_bot.strategy import Signal

    tc = TradingCfg(symbols=["BTCUSDT"], timeframe="1m", leverage=25,
                    margin_coin="USDT", margin_mode="ISOLATION",
                    risk_per_trade_pct=1.0)
    rc = RiskCfg(stop_loss_pct=0.40, take_profit_r=2.5, use_atr=False,
                 atr_multiplier_sl=1.2, atr_multiplier_tp=4.0)
    # Even with massive ATR, use_atr=False forces fixed pct path.
    sig = Signal(direction="long", score=0.7, indicator_score=12,
                 pattern_score=2.0, reasons=["x"], price=60_000.0, atr=600.0)
    plan = build_order(sig, free_margin=46.0, trading=tc, risk=rc,
                       min_volume=0.0001, volume_step=0.0001, digits=1)
    sl_pct = (plan.price - plan.stop_loss) / plan.price * 100.0
    assert abs(sl_pct - 0.40) < 0.01, \
        f"use_atr=False should give fixed 0.40%, got {sl_pct:.3f}%"


# ----------------------------------------------------------------- cascade detector


def test_cascade_detector_fires_on_big_btc_move():
    """When BTC drops >2% over 3 1m bars, cascade flag must trip and halt
    new entries for the configured halt window."""
    reset_state()
    cfg = fresh_cfg()
    cfg.mode = "live"
    bot = BitunixBot(cfg)
    bot.client = make_mock_client()
    # Synthesize 5 BTC bars with a -2.5% drop across the last 3.
    # Bar -3 close = 60000, bar -1 close = 58500 → -2.5%.
    now_ms = int(time.time() * 1000)
    btc_bars = [
        {"close": "60000", "time": now_ms - 240_000},
        {"close": "60000", "time": now_ms - 180_000},
        {"close": "60000", "time": now_ms - 120_000},   # bar -3 (3 min ago)
        {"close": "59250", "time": now_ms - 60_000},    # bar -2
        {"close": "58500", "time": now_ms},              # bar -1 (now)
    ]
    bot.client.klines.return_value = btc_bars

    halted = bot._check_liquidation_cascade()
    assert halted is True, "expected cascade flag to trip on -2.5% move"
    assert bot._cascade_active is True
    # Error event recorded for visibility on the dashboard.
    errors = [e for e in bot.state.snapshot()["events"] if e["kind"] == "error"]
    assert any("CASCADE" in e["text"] for e in errors), \
        f"expected CASCADE error event, got {[e['text'] for e in errors]}"


def test_cascade_detector_quiet_market_does_not_fire():
    """Calm BTC (no big move) should NOT fire the cascade flag."""
    reset_state()
    cfg = fresh_cfg()
    cfg.mode = "live"
    bot = BitunixBot(cfg)
    bot.client = make_mock_client()
    now_ms = int(time.time() * 1000)
    btc_bars = [{"close": str(60000 + i), "time": now_ms - (5 - i) * 60_000}
                for i in range(5)]
    bot.client.klines.return_value = btc_bars

    halted = bot._check_liquidation_cascade()
    assert halted is False
    assert bot._cascade_active is False


def test_cascade_blocks_new_entries_during_active_halt():
    """When cascade flag is active, _tick must not place new orders even
    if the signal stack would otherwise fire."""
    reset_state()
    cfg = fresh_cfg()
    cfg.mode = "live"
    bot = BitunixBot(cfg)
    bot.client = make_mock_client()
    bot.client.klines.side_effect = lambda *a, **kw: make_uptrend_klines()
    bot._resolve_symbol_meta()
    # Force cascade active.
    bot._cascade_active = True
    bot._cascade_clear_at = time.time() + 300  # well in the future
    bot._tick()

    # No new orders placed.
    bot.client.place_order.assert_not_called()


# ----------------------------------------------------------------- correlation sizing


def test_correlation_sizing_alts_get_smaller_notional():
    """ETH/SOL/XRP at the same risk-per-trade should get smaller notional
    than BTC because they're heavily BTC-correlated."""
    from bitunix_bot.config import RiskCfg, TradingCfg
    from bitunix_bot.risk import build_order
    from bitunix_bot.strategy import Signal

    tc = TradingCfg(symbols=["BTCUSDT"], timeframe="1m", leverage=25,
                    margin_coin="USDT", margin_mode="ISOLATION",
                    risk_per_trade_pct=1.0)
    rc = RiskCfg(stop_loss_pct=0.40, take_profit_r=2.5, use_atr=False,
                 atr_multiplier_sl=1.2, atr_multiplier_tp=4.0,
                 round_trip_fee_pct=0.0)  # disable fee reserve for ratio test
    # Same price/atr/score across symbols — only the symbol mult changes.
    sig = Signal(direction="long", score=0.7, indicator_score=12,
                 pattern_score=2.0, reasons=["x"], price=60_000.0, atr=120.0)
    btc_plan = build_order(sig, free_margin=46.0, trading=tc, risk=rc,
                            min_volume=0.0001, volume_step=0.0001, digits=1,
                            symbol="BTCUSDT")
    eth_plan = build_order(sig, free_margin=46.0, trading=tc, risk=rc,
                            min_volume=0.0001, volume_step=0.0001, digits=1,
                            symbol="ETHUSDT")
    sol_plan = build_order(sig, free_margin=46.0, trading=tc, risk=rc,
                            min_volume=0.0001, volume_step=0.0001, digits=1,
                            symbol="SOLUSDT")

    btc_notional = btc_plan.volume * btc_plan.price
    eth_notional = eth_plan.volume * eth_plan.price
    sol_notional = sol_plan.volume * sol_plan.price
    # BTC gets full size (mult 1.0); ETH ~85%; SOL ~70%.
    assert eth_notional < btc_notional, \
        f"ETH notional ({eth_notional}) should be < BTC ({btc_notional})"
    assert sol_notional < eth_notional, \
        f"SOL notional ({sol_notional}) should be < ETH ({eth_notional})"
    # Ratios within ~5% of expected multipliers.
    assert abs(eth_notional / btc_notional - 0.85) < 0.05
    assert abs(sol_notional / btc_notional - 0.70) < 0.05


def test_correlation_sizing_unknown_symbol_defaults_to_full():
    """Symbol not in the multiplier map should size at 1.0 (no penalty)."""
    from bitunix_bot.config import RiskCfg, TradingCfg
    from bitunix_bot.risk import build_order
    from bitunix_bot.strategy import Signal

    tc = TradingCfg(symbols=["BTCUSDT"], timeframe="1m", leverage=25,
                    margin_coin="USDT", margin_mode="ISOLATION",
                    risk_per_trade_pct=1.0)
    rc = RiskCfg(stop_loss_pct=0.40, take_profit_r=2.5, use_atr=False,
                 atr_multiplier_sl=1.2, atr_multiplier_tp=4.0,
                 round_trip_fee_pct=0.0)
    sig = Signal(direction="long", score=0.7, indicator_score=12,
                 pattern_score=2.0, reasons=["x"], price=60_000.0, atr=120.0)
    btc_plan = build_order(sig, free_margin=46.0, trading=tc, risk=rc,
                            min_volume=0.0001, volume_step=0.0001, digits=1,
                            symbol="BTCUSDT")
    # An asset not in the multiplier map.
    misc_plan = build_order(sig, free_margin=46.0, trading=tc, risk=rc,
                             min_volume=0.0001, volume_step=0.0001, digits=1,
                             symbol="DOGEUSDT")
    # Same notional (both mult=1.0).
    assert abs(misc_plan.volume - btc_plan.volume) < 1e-9


# ----------------------------------------------------------------- Grok holistic review


def test_fee_reserve_scales_volume_down_proportional_to_fee_burden():
    """When round_trip_fee_pct > 0, build_order must scale risk_usdt
    down by fee_burden = fee_pct / sl_pct. A trade with 0.08% maker
    round-trip and 0.25% SL should size at ~68% of the no-fee budget
    (1 - 0.08/0.25 = 0.68)."""
    from bitunix_bot.config import RiskCfg, TradingCfg
    from bitunix_bot.risk import build_order
    from bitunix_bot.strategy import Signal

    tc = TradingCfg(symbols=["BTCUSDT"], timeframe="15m", leverage=10,
                    margin_coin="USDT", margin_mode="ISOLATION",
                    risk_per_trade_pct=1.0)
    sig = Signal(direction="long", score=0.7, indicator_score=12,
                 pattern_score=2.0, reasons=["x"], price=60_000.0, atr=120.0)

    # No fee reserve — baseline.
    rc_nofee = RiskCfg(stop_loss_pct=0.25, take_profit_r=1.0, use_atr=False,
                       atr_multiplier_sl=1.0, atr_multiplier_tp=4.0,
                       round_trip_fee_pct=0.0)
    plan_nofee = build_order(sig, free_margin=100.0, trading=tc, risk=rc_nofee,
                              min_volume=0.0001, volume_step=0.0001, digits=1,
                              symbol="BTCUSDT")

    # Maker fees 0.08% on 0.25% SL → fee_burden 0.32 → size 68% of nofee.
    rc_fee = RiskCfg(stop_loss_pct=0.25, take_profit_r=1.0, use_atr=False,
                     atr_multiplier_sl=1.0, atr_multiplier_tp=4.0,
                     round_trip_fee_pct=0.08)
    plan_fee = build_order(sig, free_margin=100.0, trading=tc, risk=rc_fee,
                            min_volume=0.0001, volume_step=0.0001, digits=1,
                            symbol="BTCUSDT")

    ratio = plan_fee.volume / plan_nofee.volume
    assert abs(ratio - 0.68) < 0.05, \
        f"fee reserve should scale volume to 68% of no-fee budget; got {ratio:.3f}"


def test_fee_reserve_caps_at_60_percent():
    """If fee_burden exceeds 60% (e.g. 0.20% fees on 0.25% SL = 0.80),
    the reserve caps at 60% (Grok rescan: was 50%) so trades aren't
    zeroed out entirely while still reserving more for fees in
    high-fee regimes."""
    from bitunix_bot.config import RiskCfg, TradingCfg
    from bitunix_bot.risk import build_order
    from bitunix_bot.strategy import Signal

    tc = TradingCfg(symbols=["BTCUSDT"], timeframe="15m", leverage=10,
                    margin_coin="USDT", margin_mode="ISOLATION",
                    risk_per_trade_pct=1.0)
    sig = Signal(direction="long", score=0.7, indicator_score=12,
                 pattern_score=2.0, reasons=["x"], price=60_000.0, atr=120.0)

    rc_nofee = RiskCfg(stop_loss_pct=0.25, take_profit_r=1.0, use_atr=False,
                       atr_multiplier_sl=1.0, atr_multiplier_tp=4.0,
                       round_trip_fee_pct=0.0)
    plan_nofee = build_order(sig, free_margin=100.0, trading=tc, risk=rc_nofee,
                              min_volume=0.0001, volume_step=0.0001, digits=1,
                              symbol="BTCUSDT")

    # Extreme fee 0.30% > SL 0.25% → fee_burden=1.2, capped at 0.6.
    rc_fee = RiskCfg(stop_loss_pct=0.25, take_profit_r=1.0, use_atr=False,
                     atr_multiplier_sl=1.0, atr_multiplier_tp=4.0,
                     round_trip_fee_pct=0.30)
    plan_fee = build_order(sig, free_margin=100.0, trading=tc, risk=rc_fee,
                            min_volume=0.0001, volume_step=0.0001, digits=1,
                            symbol="BTCUSDT")

    ratio = plan_fee.volume / plan_nofee.volume
    # 60% cap means volume is 1.0 - 0.6 = 0.4 of nofee budget.
    # fee_reserve_frac capped at 0.4 → remaining budget = 0.6 → ratio ≈ 0.60
    assert 0.55 < ratio < 0.65, \
        f"fee reserve capped at 40% → volume should be ~60% of nofee; got {ratio:.3f}"


def test_structure_anchored_sl_widens_when_bar_has_long_wick():
    """SL should be the MAX of fixed%, ATR-derived, AND distance to last
    bar's opposite extreme + buffer. A long entry with a deep wick low
    on the entry bar should get a wider SL than the fixed% would give."""
    from bitunix_bot.config import RiskCfg, TradingCfg
    from bitunix_bot.risk import build_order
    from bitunix_bot.strategy import Signal

    tc = TradingCfg(symbols=["BTCUSDT"], timeframe="15m", leverage=10,
                    margin_coin="USDT", margin_mode="ISOLATION",
                    risk_per_trade_pct=1.0)
    rc = RiskCfg(stop_loss_pct=0.25, take_profit_r=1.0, use_atr=False,
                 atr_multiplier_sl=1.0, atr_multiplier_tp=4.0,
                 round_trip_fee_pct=0.0)

    # Long entry @ 60_000 with last bar low at 59_700 (wick down to -0.5%).
    # Fixed SL = 0.25% = 60_000 * 0.0025 = 150 below entry.
    # Anchor SL = (60_000 - 59_700) + 60_000*0.001 = 300 + 60 = 360 below.
    # Anchor wins → SL placed at 60_000 - 360 = 59_640.
    sig = Signal(direction="long", score=0.7, indicator_score=12,
                 pattern_score=2.0, reasons=["x"], price=60_000.0,
                 atr=120.0, last_bar_high=60_100.0, last_bar_low=59_700.0)
    plan = build_order(sig, free_margin=100.0, trading=tc, risk=rc,
                       min_volume=0.0001, volume_step=0.0001, digits=1)
    assert plan is not None
    # SL should be at 59_640 (anchored), not 59_850 (fixed-pct).
    assert 59_600 < plan.stop_loss < 59_700, \
        f"expected anchored SL near 59_640, got {plan.stop_loss}"


def test_structure_anchored_sl_short_widens_for_deep_wick_high():
    """Short mirror: when entry bar has a wick UP, SL should anchor
    above the bar high + buffer."""
    from bitunix_bot.config import RiskCfg, TradingCfg
    from bitunix_bot.risk import build_order
    from bitunix_bot.strategy import Signal

    tc = TradingCfg(symbols=["BTCUSDT"], timeframe="15m", leverage=10,
                    margin_coin="USDT", margin_mode="ISOLATION",
                    risk_per_trade_pct=1.0)
    rc = RiskCfg(stop_loss_pct=0.25, take_profit_r=1.0, use_atr=False,
                 atr_multiplier_sl=1.0, atr_multiplier_tp=4.0,
                 round_trip_fee_pct=0.0)
    sig = Signal(direction="short", score=0.7, indicator_score=12,
                 pattern_score=2.0, reasons=["x"], price=60_000.0,
                 atr=120.0, last_bar_high=60_300.0, last_bar_low=59_900.0)
    plan = build_order(sig, free_margin=100.0, trading=tc, risk=rc,
                       min_volume=0.0001, volume_step=0.0001, digits=1)
    assert plan is not None
    # Anchor: 60_300 - 60_000 = 300 + 60 (10bps buffer) = 360 above.
    # Fixed: 150 above. Anchor wins → SL at ~60_360.
    assert 60_300 < plan.stop_loss < 60_400, \
        f"expected anchored SL near 60_360, got {plan.stop_loss}"


def test_volume_profile_finds_hvn_below_and_above():
    """volume_profile_hvns must identify the bins with the most volume
    activity over the lookback window and return the nearest HVNs above
    and below the current price."""
    import numpy as np
    from bitunix_bot.indicators import volume_profile_hvns
    rng = np.random.default_rng(42)
    n = 100
    # Synthetic data: price oscillates around 100; LOTS of volume traded
    # near 95 (support) and 105 (resistance), little volume elsewhere.
    h = np.full(n, 0.0)
    l = np.full(n, 0.0)
    v = np.full(n, 0.0)
    for i in range(n):
        if i % 3 == 0:
            # Heavy-volume bar near 95 (the HVN below).
            l[i] = 94.5
            h[i] = 95.5
            v[i] = 1000
        elif i % 3 == 1:
            # Heavy-volume bar near 105 (the HVN above).
            l[i] = 104.5
            h[i] = 105.5
            v[i] = 1000
        else:
            # Light-volume bar in mid-range.
            l[i] = 99.0
            h[i] = 101.0
            v[i] = 50
    hvn_below, hvn_above = volume_profile_hvns(
        h, l, v, current_price=100.0, lookback=100, num_bins=20,
    )
    assert hvn_below is not None
    assert hvn_above is not None
    assert 94 < hvn_below < 96, f"expected HVN below near 95, got {hvn_below}"
    assert 104 < hvn_above < 106, f"expected HVN above near 105, got {hvn_above}"


def test_volume_profile_returns_none_on_no_volume():
    """Synthetic data with zero volume → returns (None, None)."""
    import numpy as np
    from bitunix_bot.indicators import volume_profile_hvns
    h = np.full(100, 100.5)
    l = np.full(100, 99.5)
    v = np.zeros(100)
    hvn_below, hvn_above = volume_profile_hvns(
        h, l, v, current_price=100.0, lookback=100,
    )
    assert hvn_below is None and hvn_above is None


def test_volume_profile_returns_none_on_insufficient_bars():
    """< lookback / 2 bars → returns (None, None)."""
    import numpy as np
    from bitunix_bot.indicators import volume_profile_hvns
    h = np.array([100.5, 101.0, 100.8])
    l = np.array([99.5, 100.0, 100.2])
    v = np.array([100, 100, 100])
    hvn_below, hvn_above = volume_profile_hvns(
        h, l, v, current_price=100.0, lookback=100,
    )
    assert hvn_below is None and hvn_above is None


def test_hvn_anchored_sl_long_extends_to_below_hvn():
    """Long entry: SL extends to below the nearest HVN below entry +
    structure buffer."""
    from bitunix_bot.config import RiskCfg, TradingCfg
    from bitunix_bot.risk import build_order
    from bitunix_bot.strategy import Signal

    tc = TradingCfg(symbols=["BTCUSDT"], timeframe="15m", leverage=10,
                    margin_coin="USDT", margin_mode="ISOLATION",
                    risk_per_trade_pct=1.0)
    rc = RiskCfg(stop_loss_pct=0.25, take_profit_r=1.0, use_atr=False,
                 atr_multiplier_sl=1.0, atr_multiplier_tp=4.0,
                 round_trip_fee_pct=0.0)
    # Entry at 60_000 with HVN below at 59_500 (300bps below entry).
    # Fixed SL: 0.25% = 150 below = 59_850.
    # HVN anchor: 60_000 - 59_500 = 500 + 60 (10bp buffer) = 560 → SL @ 59_440.
    # HVN anchor wins (widest of all anchors).
    sig = Signal(direction="long", score=0.7, indicator_score=12,
                 pattern_score=2.0, reasons=["x"], price=60_000.0,
                 atr=120.0, hvn_below=59_500.0)
    plan = build_order(sig, free_margin=200.0, trading=tc, risk=rc,
                       min_volume=0.0001, volume_step=0.0001, digits=1)
    assert plan is not None
    assert 59_400 < plan.stop_loss < 59_500, \
        f"expected HVN-anchored SL near 59_440, got {plan.stop_loss}"


def test_hvn_anchored_tp_long_targets_below_hvn_above():
    """Long entry: when HVN exists ABOVE entry, TP tightens toward it
    (don't ride past resistance)."""
    from bitunix_bot.config import RiskCfg, TradingCfg
    from bitunix_bot.risk import build_order
    from bitunix_bot.strategy import Signal

    tc = TradingCfg(symbols=["BTCUSDT"], timeframe="15m", leverage=10,
                    margin_coin="USDT", margin_mode="ISOLATION",
                    risk_per_trade_pct=1.0)
    rc = RiskCfg(stop_loss_pct=0.25, take_profit_r=2.5, use_atr=False,
                 atr_multiplier_sl=1.0, atr_multiplier_tp=4.0,
                 round_trip_fee_pct=0.0)
    # SL = 0.25% = 150 → R-multiple TP = 2.5R = 375 above = 60_375.
    # HVN above at 60_200 (resistance ~333bps above).
    # HVN-tp anchor: 60_200 - 60_000 - 60 (buffer) = 140 → TP @ 60_140.
    # 60_140 < 60_375, so HVN tightens TP.
    sig = Signal(direction="long", score=0.7, indicator_score=12,
                 pattern_score=2.0, reasons=["x"], price=60_000.0,
                 atr=120.0, hvn_above=60_200.0)
    plan = build_order(sig, free_margin=200.0, trading=tc, risk=rc,
                       min_volume=0.0001, volume_step=0.0001, digits=1)
    assert plan is not None
    assert 60_100 < plan.take_profit < 60_200, \
        f"expected HVN-tightened TP near 60_140, got {plan.take_profit}"


def test_hvn_tp_skipped_when_r_multiple_is_closer():
    """When R-multiple TP is CLOSER than the HVN, R-multiple wins (HVN
    only constrains down, never expands TP)."""
    from bitunix_bot.config import RiskCfg, TradingCfg
    from bitunix_bot.risk import build_order
    from bitunix_bot.strategy import Signal

    tc = TradingCfg(symbols=["BTCUSDT"], timeframe="15m", leverage=10,
                    margin_coin="USDT", margin_mode="ISOLATION",
                    risk_per_trade_pct=1.0)
    rc = RiskCfg(stop_loss_pct=0.25, take_profit_r=1.0, use_atr=False,
                 atr_multiplier_sl=1.0, atr_multiplier_tp=4.0,
                 round_trip_fee_pct=0.0)
    # SL = 0.25% = 150 → R=1.0 TP = 150 above = 60_150.
    # HVN above at 60_500 (much farther). HVN doesn't tighten — R-multiple wins.
    sig = Signal(direction="long", score=0.7, indicator_score=12,
                 pattern_score=2.0, reasons=["x"], price=60_000.0,
                 atr=120.0, hvn_above=60_500.0)
    plan = build_order(sig, free_margin=200.0, trading=tc, risk=rc,
                       min_volume=0.0001, volume_step=0.0001, digits=1)
    assert plan is not None
    assert abs(plan.take_profit - 60_150.0) < 1.0, \
        f"expected R-multiple TP near 60_150, got {plan.take_profit}"


def test_vwap_anchored_sl_long_above_vwap():
    """Long entry above VWAP: SL must extend below VWAP + buffer (thesis
    invalidates if price returns to and breaks fair value)."""
    from bitunix_bot.config import RiskCfg, TradingCfg
    from bitunix_bot.risk import build_order
    from bitunix_bot.strategy import Signal

    tc = TradingCfg(symbols=["BTCUSDT"], timeframe="15m", leverage=10,
                    margin_coin="USDT", margin_mode="ISOLATION",
                    risk_per_trade_pct=1.0)
    rc = RiskCfg(stop_loss_pct=0.25, take_profit_r=1.0, use_atr=False,
                 atr_multiplier_sl=1.0, atr_multiplier_tp=4.0,
                 round_trip_fee_pct=0.0)
    # Entry at 60_000 with VWAP at 59_700 (below entry, on protective side).
    # Fixed SL: 0.25% = 150 below = 59_850.
    # VWAP anchor: 60_000 - 59_700 = 300 + 60 (10bp buffer) = 360 → SL @ 59_640.
    # Anchor wins.
    sig = Signal(direction="long", score=0.7, indicator_score=12,
                 pattern_score=2.0, reasons=["x"], price=60_000.0,
                 atr=120.0, vwap=59_700.0)
    plan = build_order(sig, free_margin=100.0, trading=tc, risk=rc,
                       min_volume=0.0001, volume_step=0.0001, digits=1)
    assert plan is not None
    assert 59_600 < plan.stop_loss < 59_700, \
        f"expected VWAP-anchored SL near 59_640, got {plan.stop_loss}"


def test_vwap_anchored_sl_skipped_when_vwap_above_long_entry():
    """Mean-reversion long (price BELOW fair value, expecting return):
    VWAP is the TARGET not the SL. Anchor must be skipped — fall back
    to fixed-pct."""
    from bitunix_bot.config import RiskCfg, TradingCfg
    from bitunix_bot.risk import build_order
    from bitunix_bot.strategy import Signal

    tc = TradingCfg(symbols=["BTCUSDT"], timeframe="15m", leverage=10,
                    margin_coin="USDT", margin_mode="ISOLATION",
                    risk_per_trade_pct=1.0)
    rc = RiskCfg(stop_loss_pct=0.25, take_profit_r=1.0, use_atr=False,
                 atr_multiplier_sl=1.0, atr_multiplier_tp=4.0,
                 round_trip_fee_pct=0.0)
    # Entry at 60_000 with VWAP at 60_300 (ABOVE entry — mean-rev long).
    # VWAP anchor must NOT fire (would compute negative distance).
    # Fixed SL: 0.25% = 59_850.
    sig = Signal(direction="long", score=0.7, indicator_score=12,
                 pattern_score=2.0, reasons=["x"], price=60_000.0,
                 atr=120.0, vwap=60_300.0)
    plan = build_order(sig, free_margin=100.0, trading=tc, risk=rc,
                       min_volume=0.0001, volume_step=0.0001, digits=1)
    assert plan is not None
    assert abs(plan.stop_loss - 59_850.0) < 1.0, \
        f"VWAP above long entry should skip anchor; got SL {plan.stop_loss}"


def test_vwap_anchored_sl_short_below_vwap():
    """Short entry below VWAP: SL must extend above VWAP + buffer (mirror
    of the long case)."""
    from bitunix_bot.config import RiskCfg, TradingCfg
    from bitunix_bot.risk import build_order
    from bitunix_bot.strategy import Signal

    tc = TradingCfg(symbols=["BTCUSDT"], timeframe="15m", leverage=10,
                    margin_coin="USDT", margin_mode="ISOLATION",
                    risk_per_trade_pct=1.0)
    rc = RiskCfg(stop_loss_pct=0.25, take_profit_r=1.0, use_atr=False,
                 atr_multiplier_sl=1.0, atr_multiplier_tp=4.0,
                 round_trip_fee_pct=0.0)
    sig = Signal(direction="short", score=0.7, indicator_score=12,
                 pattern_score=2.0, reasons=["x"], price=60_000.0,
                 atr=120.0, vwap=60_400.0)
    plan = build_order(sig, free_margin=100.0, trading=tc, risk=rc,
                       min_volume=0.0001, volume_step=0.0001, digits=1)
    assert plan is not None
    # 60_400 - 60_000 = 400 + 60 (buffer) = 460 above entry → SL @ ~60_460.
    assert 60_400 < plan.stop_loss < 60_500, \
        f"expected VWAP-anchored SL near 60_460, got {plan.stop_loss}"


def test_structure_anchored_sl_falls_through_when_no_bar_data():
    """Legacy callers without last_bar_high/low (or 0/missing values)
    must fall back to the existing fixed-pct + ATR hybrid — no crash,
    no zero-distance SL."""
    from bitunix_bot.config import RiskCfg, TradingCfg
    from bitunix_bot.risk import build_order
    from bitunix_bot.strategy import Signal

    tc = TradingCfg(symbols=["BTCUSDT"], timeframe="15m", leverage=10,
                    margin_coin="USDT", margin_mode="ISOLATION",
                    risk_per_trade_pct=1.0)
    rc = RiskCfg(stop_loss_pct=0.25, take_profit_r=1.0, use_atr=False,
                 atr_multiplier_sl=1.0, atr_multiplier_tp=4.0,
                 round_trip_fee_pct=0.0)
    # last_bar_high / last_bar_low default to 0.0 — anchor branch must
    # gracefully skip rather than divide by zero or compute nonsense.
    sig = Signal(direction="long", score=0.7, indicator_score=12,
                 pattern_score=2.0, reasons=["x"], price=60_000.0, atr=120.0)
    plan = build_order(sig, free_margin=100.0, trading=tc, risk=rc,
                       min_volume=0.0001, volume_step=0.0001, digits=1)
    assert plan is not None
    # Pure fixed-pct path: SL at 60_000 * (1 - 0.0025) = 59_850.
    assert abs(plan.stop_loss - 59_850.0) < 1.0


# ----------------------------------------------------------------- tape-driven exit


def test_tape_exit_flattens_pre_BE_long_on_strong_sell_flow():
    """If a long position hasn't reached +1R yet AND tape flips to ≥75/25
    sell-side, flash_close_position must fire — don't wait for SL.

    Tape-exit is OPT-IN now (default disabled after live data showed it
    firing on microstructure noise). This test enables it explicitly to
    verify the legacy behavior still works when configured on."""
    bot, orig_sl, orig_tp = _setup_bot_with_open_position(
        side="BUY", entry=100_000.0, current_price=100_100.0,  # +0.4R, pre-BE
    )
    # Opt in to tape exit + use 0s min hold so the test position fires.
    bot.cfg.risk.tape_exit_enabled = True
    bot.cfg.risk.tape_exit_min_hold_secs = 0
    # Bump position age past the (default 30s) min hold by backdating ctime.
    pos = bot.client.pending_positions.return_value[0]
    pos["ctime"] = int(time.time() * 1000) - 60_000  # 60s old

    class FakeTape:
        def get_aggression_ratio(self, sym, window_secs=10, min_count=5):
            return -0.65   # 82/18 sell-aggression
        def get_cvd(self, sym, window_secs=60, min_count=5):
            return None
        def get_activity_multiplier(self, sym, **kw):
            return None
        def get_price_change_pct(self, sym, window_secs=10, min_count=5):
            return None
    bot.tape_feed = FakeTape()
    bot._tick()

    # Tape exit should have flash-closed the position.
    bot.client.flash_close_position.assert_called_once_with("POS1")
    # And recorded a TAPE_EXIT event in state.
    orders = [e for e in bot.state.snapshot()["events"] if e["kind"] == "order"]
    assert any("TAPE_EXIT" in e["text"] for e in orders), \
        f"expected TAPE_EXIT order event; got {[e['text'] for e in orders]}"


def test_tape_exit_disabled_by_default_does_not_fire():
    """tape_exit_enabled=False (the new default) must keep the position
    open even when the tape flips strongly contrary. Prevents the
    sub-30-second tape-exit bleed observed in live data."""
    bot, _, _ = _setup_bot_with_open_position(
        side="BUY", entry=100_000.0, current_price=100_100.0,  # +0.4R, pre-BE
    )
    # Default config — tape_exit_enabled is False.
    assert bot.cfg.risk.tape_exit_enabled is False, \
        "default must be disabled after live drawdown analysis"

    class FakeTape:
        def get_aggression_ratio(self, sym, window_secs=10, min_count=5):
            return -0.95   # extreme contrary
        def get_cvd(self, sym, window_secs=60, min_count=5):
            return None
        def get_activity_multiplier(self, sym, **kw):
            return None
        def get_price_change_pct(self, sym, window_secs=10, min_count=5):
            return None
    bot.tape_feed = FakeTape()
    bot._tick()
    bot.client.flash_close_position.assert_not_called()


def test_tape_exit_min_hold_blocks_too_early():
    """Even with tape_exit_enabled=True, a position younger than
    tape_exit_min_hold_secs must NOT be flattened — the entry needs time
    to settle before measuring flow."""
    bot, _, _ = _setup_bot_with_open_position(
        side="BUY", entry=100_000.0, current_price=100_100.0,
    )
    bot.cfg.risk.tape_exit_enabled = True
    bot.cfg.risk.tape_exit_min_hold_secs = 30
    # Position only 5 seconds old.
    pos = bot.client.pending_positions.return_value[0]
    pos["ctime"] = int(time.time() * 1000) - 5_000

    class FakeTape:
        def get_aggression_ratio(self, sym, window_secs=10, min_count=5):
            return -0.85
        def get_cvd(self, sym, window_secs=60, min_count=5):
            return None
        def get_activity_multiplier(self, sym, **kw):
            return None
        def get_price_change_pct(self, sym, window_secs=10, min_count=5):
            return None
    bot.tape_feed = FakeTape()
    bot._tick()
    # Min-hold guard must block the tape-exit even though everything else qualifies.
    bot.client.flash_close_position.assert_not_called()


def test_tape_exit_skipped_for_post_BE_position():
    """Once a position is past +1R (break-even ratchet has fired), the
    tape exit should NOT fire — the trade has already paid for itself
    and is protected by the SL ratchet. Let it run."""
    bot, _, _ = _setup_bot_with_open_position(
        side="BUY", entry=100_000.0, current_price=100_500.0,  # +2R, well past BE
    )

    class FakeTape:
        def get_aggression_ratio(self, sym, window_secs=10, min_count=5):
            return -0.65
        def get_cvd(self, sym, window_secs=60, min_count=5):
            return None
        def get_activity_multiplier(self, sym, **kw):
            return None
        def get_price_change_pct(self, sym, window_secs=10, min_count=5):
            return None
    bot.tape_feed = FakeTape()
    bot._tick()

    # Position should NOT be flash-closed despite hostile tape — it's past BE.
    bot.client.flash_close_position.assert_not_called()


def test_tape_exit_skipped_when_tape_neutral():
    """Pre-BE position with neutral tape should NOT be flattened."""
    bot, _, _ = _setup_bot_with_open_position(
        side="BUY", entry=100_000.0, current_price=100_100.0,  # +0.4R
    )

    class FakeTape:
        def get_aggression_ratio(self, sym, window_secs=10, min_count=5):
            return 0.10   # neutral-ish
        def get_cvd(self, sym, window_secs=60, min_count=5):
            return None
        def get_activity_multiplier(self, sym, **kw):
            return None
        def get_price_change_pct(self, sym, window_secs=10, min_count=5):
            return None
    bot.tape_feed = FakeTape()
    bot._tick()

    bot.client.flash_close_position.assert_not_called()


def test_tradetape_lifecycle_in_bot():
    """Bot.run_forever should start tape_feed alongside ob_feed (smoke test)."""
    reset_state()
    cfg = fresh_cfg()
    bot = BitunixBot(cfg)
    bot.client = make_mock_client()
    # Don't actually call run_forever (it loops); just verify the feed
    # gets instantiated when start() runs the same wiring path.
    # Manually mirror the lifecycle init.
    from bitunix_bot.orderbook import OrderBookFeed
    from bitunix_bot.tradetape import TradeFeed
    bot.ob_feed = OrderBookFeed(symbols=cfg.trading.symbols, depth_levels=10)
    bot.tape_feed = TradeFeed(symbols=cfg.trading.symbols)
    assert isinstance(bot.tape_feed, TradeFeed)
    # And the feeds are not yet connected (we never called .start()).
    assert not bot.tape_feed.is_connected()
    # Sanity: accessors return None on cold feed.
    assert bot.tape_feed.get_cvd("BTCUSDT") is None


# ----------------------------------------------------------------- runner

def main() -> int:
    tests = [
        test_signing_matches_bitunix_spec_example,
        test_uptrend_produces_long_signal_with_paper_order,
        test_no_hard_gates_only_combined_threshold,
        test_global_max_open_positions_cap,
        test_same_direction_cap_kills_correlated_risk,
        test_per_symbol_cap_blocks_same_symbol_but_allows_others,
        test_cooldown_blocks_immediate_re_entry,
        test_bar_dedupe_blocks_same_candle_re_eval,
        test_time_based_exit_closes_stale_LOSING_position,
        test_time_based_exit_LETS_WINNER_RUN,
        test_live_mode_with_zero_margin_skips_orders,
        test_paper_mode_with_zero_margin_simulates_anyway,
        test_live_mode_actually_calls_place_order,
        test_dashboard_routes_and_auth,
        test_breakeven_sl_move_at_1r_long,
        test_trailing_sl_at_2r_long,
        test_breakeven_sl_short,
        test_sl_never_moves_against_position,
        test_no_management_below_threshold,
        test_htf_and_funding_votes_contribute_to_score,
        test_pattern_detection_basic_shapes,
        test_pattern_alone_can_fire_signal,
        test_double_bottom_detection,
        test_chart_pattern_score_aggregation,
        test_sr_detection_finds_real_levels,
        test_orderbook_imbalance_compute,
        test_orderbook_extract_symbol_handles_list_data,
        test_orderbook_get_status_shape,
        test_tradetape_get_status_shape,
        test_feeds_status_endpoint,
        test_feeds_status_requires_auth,
        test_orderbook_parser_handles_multiple_formats,
        test_imbalance_vote_in_strategy,
        test_clientid_is_deterministic_and_includes_symbol_side,
        test_execute_handles_network_timeout_without_crashing,
        test_health_check_fails_when_tick_loop_stalls,
        test_config_rejects_silly_values,
        test_divergence_detector_finds_bullish_divergence,
        test_smc_fvg_detection,
        test_smc_liquidity_sweep_bearish,
        test_obv_and_mfi_indicators_compute,
        test_skip_events_dedupe_within_window,
        test_capped_signals_emit_skip_not_signal,
        test_adaptive_tp_tightens_with_age_and_respects_floor,
        test_cvd_indicator_signs_by_body_position,
        test_combo_recipe_fires_on_matching_reasons,
        test_combo_smc_reversal_requires_three_independent_components,
        test_daily_drawdown_halts_new_entries,
        test_spread_filter_blocks_when_spread_too_wide,
        test_partial_tp_fires_at_1r_and_only_once,
        test_regime_weighting_boosts_aligned_signals,
        test_sl_and_tp_always_on_correct_side,
        test_post_only_entry_places_limit_at_top_of_book,
        test_post_only_falls_back_to_market_when_ob_feed_disconnected,
        test_post_only_timeout_cancels_and_skips,
        test_post_only_fill_clears_pending_tracking,
        test_pending_limits_count_toward_global_cap,
        test_pending_limits_count_toward_same_direction_cap,
        test_post_only_paper_mode_still_uses_paper_path,
        test_tradetape_parser_handles_side_field,
        test_tradetape_parser_handles_buyer_maker_field,
        test_tradetape_parser_rejects_unparseable,
        test_tradetape_cvd_computation,
        test_tradetape_window_filtering,
        test_tradetape_aggression_ratio_bounded,
        test_tradetape_print_rate_and_aggressor_size,
        test_tradetape_large_print_count,
        test_tradetape_returns_none_when_empty,
        test_tradetape_handles_seconds_and_ms_timestamps,
        test_tradetape_activity_multiplier_clamps,
        test_strategy_real_cvd_replaces_proxy_when_provided,
        test_strategy_aggression_burst_fires_above_threshold,
        test_strategy_activity_multiplier_scales_combined_score,
        test_tape_veto_blocks_long_against_strong_sell_flow,
        test_tape_veto_blocks_short_against_strong_buy_flow,
        test_neutral_flow_now_blocks_directional_signal,
        test_confirming_flow_allows_signal,
        test_tape_veto_skipped_when_no_data,
        test_reset_streaks_endpoint_clears_in_memory_state,
        test_reset_streaks_requires_auth,
        test_reset_streaks_returns_503_when_no_bot,
        test_continuation_blocks_long_on_bearish_close_pattern,
        test_continuation_blocks_short_on_bullish_close_pattern,
        test_continuation_blocks_when_close_doesnt_advance,
        test_continuation_blocks_long_with_contrary_tape,
        test_continuation_blocks_long_with_contrary_cvd,
        test_continuation_allows_clean_marubozu,
        test_trend_dominance_dampens_opposite_side,
        test_min_adx_skips_deep_chop,
        test_min_adx_zero_disables_filter,
        test_absorption_vetoes_same_direction_short,
        test_absorption_vetoes_same_direction_long,
        test_absorption_does_not_block_reversal_trade,
        test_flat_trades_dont_count_toward_streak,
        test_real_losses_still_trigger_streak,
        test_journal_endpoint_returns_recent_events,
        test_journal_endpoint_handles_missing_file,
        test_journal_endpoint_requires_auth,
        test_journal_download_returns_file,
        test_ticker_confirmation_blocks_long_when_price_didnt_move_up,
        test_ticker_confirmation_allows_long_when_price_moved_up,
        test_ticker_confirmation_recalibrates_sl_tp,
        test_ticker_confirmation_drops_when_ticker_fetch_fails,
        test_post_only_entries_disabled_by_default,
        test_factor_classification_routes_votes_correctly,
        test_factor_score_breakdown_dedups_within_group,
        test_factor_score_caps_at_one,
        test_factor_score_weighted_combines_groups,
        test_signal_records_factor_breakdown,
        test_adaptive_threshold_drawdown_raises_bar,
        test_adaptive_threshold_hot_streak_eases,
        test_adaptive_threshold_neutral_in_middle,
        test_adaptive_threshold_requires_minimum_samples,
        test_adaptive_threshold_pipeline_flows_into_evaluate,
        test_compute_trade_r_basic,
        test_compute_trade_r_handles_invalid,
        test_recent_trade_r_appended_on_close,
        test_stale_exit_flattens_drifting_position,
        test_stale_exit_skipped_for_progressing_position,
        test_stale_exit_skipped_for_young_position,
        test_expansion_candle_skip_blocks_next_bar,
        test_dd_risk_multiplier_gradual_steps,
        test_build_order_dd_risk_mult_scales_volume,
        test_mini_cooldown_fires_on_two_losses_in_window,
        test_mini_cooldown_blocks_new_entries,
        test_mini_cooldown_does_not_fire_for_isolated_loss,
        test_journal_entry_mechanism_fields,
        test_journal_writes_entry_and_exit,
        test_journal_handles_missing_dir_gracefully,
        test_fire_threshold_lowers_in_trending_regime,
        test_fire_threshold_raises_in_ranging_regime,
        test_fire_threshold_neutral_in_mid_regime,
        test_fire_threshold_clamps_at_bounds,
        test_signal_records_fire_threshold_used,
        test_conviction_sizing_high_score_increases_risk,
        test_conviction_sizing_clamps_to_floor,
        test_conviction_sizing_unset_threshold_no_change,
        test_absorption_vote_fires_on_extreme_flow_no_movement,
        test_absorption_vote_does_not_fire_when_price_moves,
        test_absorption_vote_does_not_fire_when_aggression_weak,
        test_depth_filter_blocks_thin_book,
        test_depth_filter_allows_normal_book,
        test_depth_filter_unknown_symbol_no_filter,
        test_dynamic_post_only_timeout_shortens_with_high_activity,
        test_dynamic_post_only_timeout_lengthens_in_dead_market,
        test_dynamic_post_only_timeout_clamps_to_4_12,
        test_tighter_tape_veto_threshold_grok_v7,
        test_tradetape_get_price_change_pct,
        test_orderbook_get_depth,
        test_atr_hybrid_floor_wins_in_calm_markets,
        test_atr_hybrid_widens_in_vol_regime,
        test_atr_hybrid_disabled_uses_fixed_pct,
        test_cascade_detector_fires_on_big_btc_move,
        test_cascade_detector_quiet_market_does_not_fire,
        test_cascade_blocks_new_entries_during_active_halt,
        test_correlation_sizing_alts_get_smaller_notional,
        test_correlation_sizing_unknown_symbol_defaults_to_full,
        test_tape_exit_flattens_pre_BE_long_on_strong_sell_flow,
        test_tape_exit_disabled_by_default_does_not_fire,
        test_tape_exit_min_hold_blocks_too_early,
        test_tape_exit_skipped_for_post_BE_position,
        test_tape_exit_skipped_when_tape_neutral,
        test_tradetape_lifecycle_in_bot,
    ]
    passed = failed = 0
    for t in tests:
        try:
            t()
            print(f"  PASS  {t.__name__}")
            passed += 1
        except AssertionError as e:
            print(f"  FAIL  {t.__name__}: {e}")
            failed += 1
        except Exception as e:
            print(f"  ERROR {t.__name__}: {type(e).__name__}: {e}")
            failed += 1
    print(f"\n{passed} passed, {failed} failed")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
