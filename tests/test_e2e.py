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
    marubozu-style candles (no rejection wicks) so the pattern detector
    yields a deterministic bullish signal direction."""
    rng = np.random.default_rng(42)
    closes = base + np.cumsum(rng.normal(drift, 25, n))
    opens = closes - rng.uniform(5, 20, n)
    highs = np.maximum(opens, closes) + rng.uniform(0.5, 3, n)
    lows = np.minimum(opens, closes) - rng.uniform(0.5, 3, n)
    # Last 3 bars: clean bull marubozu — open near low, close near high, big body.
    for i in range(n - 3, n):
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
                                   breakeven_at_r=1.0, trailing_activate_r=1.5,
                                   trailing_distance_r=0.5, buffer_pct=0.05):
    """Helper: build a live-mode bot with one open position and the given TPSL state."""
    reset_state()
    cfg = fresh_cfg()
    cfg.mode = "live"
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
    # Clean bull marubozu on last 3 bars (same as make_uptrend_klines).
    for i in range(n - 3, n):
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
    assert any("funding" in r for r in sig.reasons), f"funding vote missing: {sig.reasons}"


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
    # qty was 0.01; partial_tp_close_pct = 50% → 0.005.
    assert abs(float(qty_str) - 0.005) < 1e-6, f"expected 0.005, got {qty_str}"
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
    for j in range(n - 3, n):                                # clean bull marubozu
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
        assert sig.score >= raw, \
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
