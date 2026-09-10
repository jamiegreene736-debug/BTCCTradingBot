"""Chronological candle replay with declared execution assumptions, never live orders.

This is a structural sensitivity check. Historical book, mark-price basis, risk tiers,
funding changes and actual stop fills are not reconstructed from candles.
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
from bitunix_bot.intraday import (
    INTERVALS,
    Candle,
    Market,
    Tier,
    closed_candles,
    evaluate_intraday,
    number,
)
from bitunix_bot.signal_config import SignalsCfg, SignalSettings
from bitunix_bot.signal_store import TrackedTrade, evaluate_exit


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


def replay(
    histories: dict[str, dict[str, list[Candle]]],
    start: int,
    end: int,
    settings: SignalSettings,
    spread_pct: float,
    funding_rate: float,
) -> dict[str, object]:
    cfg = SignalsCfg()
    active: dict[str, TrackedTrade] = {}
    closed: list[dict[str, object]] = []
    states: Counter[str] = Counter()
    adverse: dict[str, float] = {}
    unresolved_gaps = 0
    for now in range(start // 900 * 900, end // 900 * 900, 900):
        slices = {
            symbol: {
                interval: [c for c in bars if c.time + INTERVALS[interval] <= now][
                    -200:
                ]
                for interval, bars in frames.items()
            }
            for symbol, frames in histories.items()
        }
        for symbol, frames in slices.items():
            current = next((c for c in histories[symbol]["15m"] if c.time == now), None)
            if (
                current is None
                or not complete_frames(frames, now)
                or not complete_frames(slices["BTCUSDT"], now)
            ):
                states["DATA_UNAVAILABLE"] += 1
                if active.pop(symbol, None):
                    unresolved_gaps += 1
                continue
            price = current.open
            market = Market(
                symbol,
                price,
                price,
                price * (1 - spread_pct / 200),
                price * (1 + spread_pct / 200),
                500_000,
                500_000,
                sum(c.volume * c.close for c in frames["15m"][-96:]),
                funding_rate,
                8,
                (now // 28800 + 1) * 28800,
                [Tier(0, 50_000_000, 0.005, 40)],
                0.000001,
                0.000001,
                now,
            )
            decision = evaluate_intraday(
                market, frames, slices.get("BTCUSDT", {}).get("1h"), settings, cfg, now
            )
            states[decision.state] += 1
            trade = active.get(symbol)
            if trade:
                sign = 1 if trade.plan.side == "long" else -1
                last = frames["15m"][-1]
                worst = last.low if sign == 1 else last.high
                adverse[trade.id] = max(
                    adverse.get(trade.id, 0),
                    sign * (trade.plan.entry - worst) / trade.plan.entry * 100,
                )
                old_stop = trade.current_stop
                evaluate_exit(trade, decision, frames["15m"], now, cfg)
                if trade.state.startswith("EXIT_"):
                    if "Stop level" in trade.reason:
                        # Assume the worse outcome if both levels are touched in one candle.
                        fill = (
                            min(old_stop, last.open)
                            if sign == 1
                            else max(old_stop, last.open)
                        )
                    elif "profit target" in trade.reason:
                        fill = trade.plan.target
                    else:
                        fill = market.bid if sign == 1 else market.ask
                    net = (
                        sign * (fill - trade.plan.entry) * trade.plan.quantity
                        - trade.plan.notional * trade.plan.cost_pct / 100
                    )
                    closed.append(
                        {
                            "symbol": symbol,
                            "side": trade.plan.side,
                            "opened_at": trade.opened_at,
                            "closed_at": now,
                            "reason": trade.reason,
                            "estimated_net_usdt": net,
                            "net_r": net / trade.plan.risk_usdt,
                            "max_adverse_pct": adverse.get(trade.id, 0),
                        }
                    )
                    del active[symbol]
                continue
            if decision.state.startswith("ENTER_") and decision.plan:
                total_risk = (
                    sum(t.plan.risk_usdt for t in active.values())
                    + decision.plan.risk_usdt
                )
                same_side = sum(t.plan.side == decision.side for t in active.values())
                if (
                    total_risk
                    <= settings.planning_equity * cfg.max_total_risk_pct / 100
                    and same_side < cfg.max_same_direction
                ):
                    active[symbol] = TrackedTrade(
                        decision.signal_id,
                        symbol,
                        "paper",
                        now,
                        decision.plan,
                        decision.plan.stop,
                        decision.plan.entry,
                        checked_at=now,
                    )
    return {
        "start": start,
        "end": end,
        "settings": asdict(settings),
        "evaluations": dict(states),
        "closed_trades": closed,
        "closed_count": len(closed),
        "open_at_end": len(active),
        "unresolved_trades_due_to_gaps": unresolved_gaps,
        "estimated_net_usdt": sum(number(t["estimated_net_usdt"]) for t in closed),
        "limitations": "Candle replay only, not validated profitability. Fixed assumed funding, spread, maintenance margin and depth; no historical mark basis, queue position or intra-candle path. Costs charged for maximum hold, stops first on ambiguous candles. Open trades excluded from realized total.",
        "assumptions": {
            "spread_pct": spread_pct,
            "funding_rate_per_8h": funding_rate,
            "maintenance_rate": 0.005,
            "depth_usdt_per_side": 500_000,
            "round_trip_fees_pct": cfg.round_trip_fee_pct,
            "slippage_pct": cfg.slippage_pct,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--symbols", default="BTCUSDT,ETHUSDT")
    parser.add_argument("--days", type=int, default=7)
    parser.add_argument("--leverage", type=int, default=25)
    parser.add_argument("--spread-pct", type=float, default=0.02)
    parser.add_argument("--funding-rate", type=float, default=0.0001)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if (
        not 1 <= args.days <= 90
        or not 0 <= args.spread_pct <= 1
        or not -0.01 <= args.funding_rate <= 0.01
    ):
        parser.error("Use 1–90 days and finite, plausible spread/funding assumptions")
    settings = SignalSettings(leverage=args.leverage)
    settings.validate()
    symbols = list(dict.fromkeys(["BTCUSDT"] + args.symbols.upper().split(",")))
    if len(symbols) > 10 or any(
        not s.isalnum() or not s.endswith("USDT") for s in symbols
    ):
        parser.error("Provide at most ten USDT symbols")
    now = int(time.time())
    start = now // 900 * 900 - args.days * 86400
    client = BitunixClient("", "", read_only=True, timeout=10)
    histories = {
        s: {
            interval: history(client, s, interval, start, now) for interval in INTERVALS
        }
        for s in symbols
    }
    report = replay(histories, start, now, settings, args.spread_pct, args.funding_rate)
    Path(args.output).write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(
        json.dumps(
            {key: value for key, value in report.items() if key != "closed_trades"},
            indent=2,
        )
    )
    print(f"Closed trades: {report['closed_count']}")


if __name__ == "__main__":
    main()
