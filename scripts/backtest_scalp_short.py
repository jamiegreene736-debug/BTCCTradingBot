"""Replay the scalp-short profile over recent 1m candles; never places orders.

Structural sensitivity check for the parabolic-exhaustion fade. Every ENTER
signal becomes a forward test measured on the following trigger-interval
candles: first touch of target, stop or estimated liquidation, else expiry at
the hold cap. Book depth, spread, funding and open-interest history are not
reconstructed from candles, so those gates use the declared assumptions below
and the "Crowded longs" gate is satisfied by the assumed funding rate.

    python3 scripts/backtest_scalp_short.py --hours 48 --leverage 100 \
        --symbols auto --top 15 --output /tmp/scalp-short-replay.json
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
import time
from collections import Counter
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bitunix_bot.client import BitunixClient
from bitunix_bot.forward_test import (
    ForwardTest,
    forward_test_from_decision,
    summarize_forward_tests,
    update_forward_test,
)
from bitunix_bot.intraday import INTERVALS, Candle, Market, Tier, closed_candles
from bitunix_bot.scalp_short import evaluate_scalp_short
from bitunix_bot.signal_config import SignalsCfg, SignalSettings
from bitunix_bot.symbol_meta import (
    row_is_tradeable_usdt_perp,
    row_max_leverage,
    row_quote_volume_usdt,
    row_symbol,
)


def history(
    client: BitunixClient, symbol: str, interval: str, start: int, now: int
) -> list[Candle]:
    rows: dict[int, dict[str, object]] = {}
    cursor = now * 1000
    cutoff = start - 200 * INTERVALS[interval]
    while cursor > cutoff * 1000:
        batch = client.klines(symbol, interval, limit=200, end_time=cursor)
        if not batch:
            raise ValueError(f"Incomplete historical candles: {symbol} {interval}")
        oldest = min(int(row["time"]) for row in batch)
        if oldest >= cursor:
            raise ValueError("Provider pagination did not advance")
        rows.update({int(row["time"]): row for row in batch})
        cursor = oldest - 1
        time.sleep(0.15)
    return closed_candles(list(rows.values()), interval, now, allow_gaps=True)


def complete_frames(frames: dict[str, list[Candle]], now: int) -> bool:
    return all(
        len(bars) >= 64
        and bars[-1].time == (now // INTERVALS[interval] - 1) * INTERVALS[interval]
        and all(
            b.time - a.time == INTERVALS[interval] for a, b in itertools.pairwise(bars)
        )
        for interval, bars in frames.items()
    )


def pick_symbols(client: BitunixClient, top: int, min_leverage: int) -> list[str]:
    """Top 24h gainers among liquid USDT perps whose cap allows the leverage."""
    pairs = {row_symbol(p): p for p in client.trading_pairs() if row_is_tradeable_usdt_perp(p)}
    rows = [t for t in client.tickers() if row_symbol(t) in pairs]
    cfg = SignalsCfg()

    def change(row: dict[str, object]) -> float:
        for key in ("priceChangePercent", "change24h", "changePercent", "priceChange"):
            value = row.get(key)
            if value not in (None, ""):
                try:
                    return float(value)
                except (TypeError, ValueError):
                    continue
        return 0.0

    liquid = [
        t
        for t in rows
        if row_quote_volume_usdt(t) >= cfg.min_quote_volume
        and row_max_leverage(pairs[row_symbol(t)]) >= min_leverage
    ]
    liquid.sort(key=change, reverse=True)
    chosen = [row_symbol(t) for t in liquid[:top]]
    if "BTCUSDT" not in chosen:
        chosen.append("BTCUSDT")
    return chosen


def replay(
    histories: dict[str, dict[str, list[Candle]]],
    start: int,
    end: int,
    settings: SignalSettings,
    cfg: SignalsCfg,
    spread_pct: float,
    funding_rate: float,
    maintenance_rate: float,
    tier_leverage: int,
) -> dict[str, object]:
    interval = cfg.scalp.trigger_interval
    step = INTERVALS[interval]
    states: Counter[str] = Counter()
    blockers: Counter[str] = Counter()
    tests: dict[str, ForwardTest] = {}
    for now in range(start // step * step, end // step * step, step):
        slices = {
            symbol: {
                name: [c for c in bars if c.time + INTERVALS[name] <= now][-200:]
                for name, bars in frames.items()
            }
            for symbol, frames in histories.items()
        }
        btc_hourly = slices.get("BTCUSDT", {}).get("1h")
        for symbol, frames in slices.items():
            if symbol == "BTCUSDT":
                continue
            current = next((c for c in histories[symbol][interval] if c.time == now), None)
            if current is None or not complete_frames(frames, now):
                states["DATA_UNAVAILABLE"] += 1
                continue
            for test in tests.values():
                if test.symbol == symbol and test.outcome == "open":
                    update_forward_test(test, frames[interval], now)
            price = current.open
            market = Market(
                symbol,
                price,
                price,
                price * (1 - spread_pct / 200),
                price * (1 + spread_pct / 200),
                2_000_000,
                2_000_000,
                sum(c.volume * c.close for c in frames["15m"][-96:]),
                funding_rate,
                8,
                (now // 28800 + 1) * 28800,
                [Tier(0, 50_000_000, maintenance_rate, tier_leverage)],
                0.000001,
                0.000001,
                now,
            )
            decision = evaluate_scalp_short(
                market, frames, btc_hourly, settings, cfg, now, None
            )
            states[decision.state] += 1
            if decision.state == "WATCH_SHORT":
                for check in decision.checks:
                    if not check.passed and not check.waiting:
                        blockers[check.label] += 1
            if any(t.symbol == symbol and t.outcome == "open" for t in tests.values()):
                continue
            test = forward_test_from_decision(decision, now)
            if test is not None and test.id not in tests:
                tests[test.id] = test
    for test in tests.values():
        if test.outcome == "open":
            update_forward_test(test, histories[test.symbol][interval], end)
    results = list(tests.values())
    return {
        "start": start,
        "end": end,
        "settings": asdict(settings),
        "scalp": asdict(cfg.scalp),
        "evaluations": dict(states),
        "watch_blockers": dict(blockers),
        "summary": summarize_forward_tests(results),
        "signals": [asdict(t) for t in results],
        "limitations": (
            "Candle replay only, not validated profitability. Assumed spread, depth, "
            "funding and maintenance rate; no order-book, open-interest or mark-basis "
            "history. Stops count first on ambiguous candles. Exit R ignores fees "
            "beyond the plan's cost estimate."
        ),
        "assumptions": {
            "spread_pct": spread_pct,
            "funding_rate_per_interval": funding_rate,
            "maintenance_rate": maintenance_rate,
            "tier_leverage": tier_leverage,
            "depth_usdt_per_side": 2_000_000,
            "round_trip_fees_pct": cfg.round_trip_fee_pct,
            "slippage_pct": cfg.slippage_pct,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hours", type=int, default=48)
    parser.add_argument("--symbols", default="auto", help="comma list or 'auto' for top 24h gainers")
    parser.add_argument("--top", type=int, default=15)
    parser.add_argument("--leverage", type=int, default=100)
    parser.add_argument("--hold-hours", type=int, default=2, choices=(1, 2))
    parser.add_argument("--equity", type=float, default=1000.0)
    parser.add_argument("--risk-pct", type=float, default=0.5)
    parser.add_argument("--spread-pct", type=float, default=0.02)
    parser.add_argument("--funding-rate", type=float, default=0.0002)
    parser.add_argument("--maintenance-rate", type=float, default=0.004)
    parser.add_argument("--tier-leverage", type=int, default=125)
    parser.add_argument("--output", default="")
    args = parser.parse_args()
    settings = SignalSettings(
        args.equity, args.risk_pct, args.leverage, args.hold_hours, "scalp_short"
    )
    settings.validate()
    cfg = SignalsCfg()
    client = BitunixClient("", "", read_only=True)
    symbols = (
        pick_symbols(client, args.top, args.leverage)
        if args.symbols == "auto"
        else [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    )
    if "BTCUSDT" not in symbols:
        symbols.append("BTCUSDT")
    now = int(time.time()) // 60 * 60
    start = now - args.hours * 3600
    intervals = (cfg.scalp.trigger_interval, "15m", "1h", "4h")
    histories: dict[str, dict[str, list[Candle]]] = {}
    for symbol in symbols:
        try:
            histories[symbol] = {
                interval: history(client, symbol, interval, start, now)
                for interval in intervals
            }
        except (ValueError, KeyError) as exc:
            print(f"Skipping {symbol}: {exc}", file=sys.stderr)
    if "BTCUSDT" not in histories:
        print("BTCUSDT history is required for the BTC context gate", file=sys.stderr)
        return 1
    report = replay(
        histories,
        start,
        now,
        settings,
        cfg,
        args.spread_pct,
        args.funding_rate,
        args.maintenance_rate,
        args.tier_leverage,
    )
    report["symbols"] = sorted(histories)
    text = json.dumps(report, indent=2)
    if args.output:
        Path(args.output).write_text(text)
    summary = report["summary"]
    print(
        f"{len(histories)} symbols, {args.hours}h at {args.leverage}x: "
        f"{summary['count']} signals, hit rate {summary['hit_rate']}, "
        f"avg R {summary['average_r']}, liquidation touches {summary['liquidation_touches']}"
    )
    print("WATCH blockers:", report["watch_blockers"])
    if args.output:
        print(f"Report written to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
