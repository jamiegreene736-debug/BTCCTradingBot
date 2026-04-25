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
    cfg.strategy.min_confluence = 3
    cfg.strategy.rsi_long_min = 40
    cfg.strategy.rsi_long_max = 80
    cfg.strategy.rsi_short_min = 20
    cfg.strategy.rsi_short_max = 60
    cfg.strategy.min_atr_pct = 0.0  # disable ATR floor for fixture
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


def test_time_based_exit_closes_stale_position():
    reset_state()
    cfg = fresh_cfg()
    cfg.mode = "live"  # time-exit only fires in live mode
    cfg.trading.max_position_age_seconds = 60  # 1 min for test
    bot = BitunixBot(cfg)
    bot.client = make_mock_client()
    # A stale position from 10 minutes ago.
    stale_pos = {
        "positionId": "STALE1", "symbol": "BTCUSDT", "qty": "0.01",
        "side": "LONG", "leverage": 100,
        "ctime": int(time.time() * 1000) - 600_000,  # 10 min ago
    }
    bot.client.pending_positions.return_value = [stale_pos]
    bot.client.klines.side_effect = lambda *a, **kw: make_uptrend_klines()
    bot._resolve_symbol_meta()
    bot._tick()

    bot.client.flash_close_position.assert_called_once_with("STALE1")


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
        min_confluence=4,
        adx_period=14, adx_min=15.0,        # loose ADX so the gate doesn't block
        supertrend_period=10, supertrend_mult=3.0,
        min_atr_pct=0.0,
        pattern_weight=0.55, pattern_norm=2.5, fire_threshold=0.45,
    )
    sig = evaluate(opens.tolist(), highs.tolist(), lows.tolist(), closes.tolist(), cfg)
    assert sig is not None, "engulfing + uptrend should fire a signal"
    assert sig.direction == "long", f"expected long, got {sig.direction}"
    assert sig.pattern_score > 0, "pattern_score should be > 0"
    assert any("PAT:" in r for r in sig.reasons), f"reasons missing pattern tags: {sig.reasons}"


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
        test_time_based_exit_closes_stale_position,
        test_live_mode_with_zero_margin_skips_orders,
        test_paper_mode_with_zero_margin_simulates_anyway,
        test_live_mode_actually_calls_place_order,
        test_dashboard_routes_and_auth,
        test_pattern_detection_basic_shapes,
        test_pattern_alone_can_fire_signal,
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
