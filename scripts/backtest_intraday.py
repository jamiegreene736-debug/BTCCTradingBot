"""Replay the trend profile over recent candles; never places orders.

Structural sensitivity check for the 50x-class, 60/120-minute continuation
profile. Every ENTER signal is assumed filled at the plan entry on the signal
bar and is then driven by ``evaluate_exit`` on completed trigger-interval
candles, exactly as the overlay would drive a tracked trade. Historical order
books, mark basis, risk tiers, funding changes and actual stop fills are not
reconstructed from candles; those gates read the declared assumptions below.

    python3 scripts/backtest_intraday.py --days 14 \\
        --symbols ETHUSDT,SOLUSDT,DOGEUSDT,XRPUSDT,SUIUSDT \\
        --leverage 50 --hold-hours 2 --output /tmp/trend-replay-2h.json
"""

from __future__ import annotations

import argparse
import itertools
import json
import statistics
import sys
import time
from bisect import bisect_right
from collections import Counter
from collections.abc import Sequence
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bitunix_bot.client import BitunixClient
from bitunix_bot.intraday import (
    INTERVALS,
    PROFILE,
    Candle,
    Market,
    Tier,
    closed_candles,
    evaluate_intraday,
    number,
)
from bitunix_bot.signal_config import SignalsCfg, SignalSettings
from bitunix_bot.signal_store import TrackedTrade, evaluate_exit

DEPTH_USDT = 500_000
SLICE_BARS = 200
# Exit reason substrings (signal_store.evaluate_exit) -> histogram class. A stop
# hit after the stop was ratcheted (breakeven or trailing) is reported as "trail".
EXIT_CLASSES: tuple[tuple[str, str], ...] = (
    ("Stop level", "stop"),
    ("profit target", "target"),
    ("Maximum holding time", "time"),
    ("review window", "stale"),
    ("Hold window closing", "late"),
    ("structure reversed", "structure"),
    ("liquidation buffer", "liquidation_buffer"),
    ("Do not wait for a reversal", "hope"),
    ("Unprotected", "unprotected"),
)


def exit_class(reason: str, stop_advanced: bool) -> str:
    for needle, label in EXIT_CLASSES:
        if needle in reason:
            return "trail" if label == "stop" and stop_advanced else label
    return "other"


def intervals_for(cfg: SignalsCfg) -> tuple[str, str, str]:
    """Trigger, structure and bias frames; the 4h frame is not fetched."""
    return (cfg.trend.trigger_interval, "15m", "1h")


def history(
    client: BitunixClient, symbol: str, interval: str, start: int, now: int
) -> list[Candle]:
    rows: dict[int, dict[str, object]] = {}
    seconds_ms = INTERVALS[interval] * 1000
    cursor = now * 1000
    cutoff = start - SLICE_BARS * INTERVALS[interval]
    previous_oldest: int | None = None
    while cursor > cutoff * 1000:
        batch = client.klines(symbol, interval, limit=200, end_time=cursor)
        if not batch:
            raise ValueError(f"Incomplete historical candles: {symbol} {interval}")
        oldest = min(int(row["time"]) for row in batch)
        if previous_oldest is not None and oldest >= previous_oldest:
            raise ValueError("Provider pagination did not advance")
        rows.update({int(row["time"]): row for row in batch})
        # endTime is exclusive and pages can drop candles near their edges;
        # overlap the next page by 30 bars and deduplicate by timestamp.
        previous_oldest = oldest
        cursor = oldest + 30 * seconds_ms
        time.sleep(0.15)
    # Sub-15m feeds omit bars when nothing traded; fill those flat so a quiet
    # five minutes does not drop an open trade as an unresolved gap.
    return closed_candles(
        list(rows.values()),
        interval,
        now,
        allow_gaps=True,
        fill_gaps=INTERVALS[interval] < 900,
    )


def complete_frames(frames: dict[str, list[Candle]], now: int) -> bool:
    return all(
        len(bars) >= 64
        and bars[-1].time == (now // INTERVALS[interval] - 1) * INTERVALS[interval]
        and all(
            b.time - a.time == INTERVALS[interval] for a, b in itertools.pairwise(bars)
        )
        for interval, bars in frames.items()
    )


def completed_slice(
    bars: list[Candle], times: list[int], interval: str, now: int
) -> list[Candle]:
    """The last SLICE_BARS candles that have closed by ``now``."""
    index = bisect_right(times, now - INTERVALS[interval])
    return bars[max(0, index - SLICE_BARS) : index]


def liquidation_distance_pct(trade: TrackedTrade) -> float | None:
    plan = trade.plan
    if plan.liquidation_estimate is None or plan.entry <= 0:
        return None
    sign = 1 if plan.side == "long" else -1
    return sign * (plan.entry - plan.liquidation_estimate) / plan.entry * 100


def trade_record(
    trade: TrackedTrade,
    closed_at: int,
    fill: float,
    net: float,
    reason: str,
    max_adverse_pct: float,
    liquidation_touched: bool,
) -> dict[str, object]:
    plan = trade.plan
    distance = liquidation_distance_pct(trade)
    return {
        "symbol": trade.symbol,
        "side": plan.side,
        "opened_at": trade.opened_at,
        "closed_at": closed_at,
        "minutes_held": (closed_at - trade.opened_at) / 60,
        "reason": trade.reason,
        "exit_reason": reason,
        "fill": fill,
        "estimated_net_usdt": net,
        "net_r": net / plan.risk_usdt if plan.risk_usdt else None,
        "stop_pct": plan.stop_pct,
        "leverage": plan.leverage,
        "max_leverage": plan.max_leverage,
        "liquidation_distance_pct": distance,
        "max_adverse_pct": max_adverse_pct,
        "adverse_share_of_liquidation": (
            max_adverse_pct / distance if distance is not None and distance > 0 else None
        ),
        "liquidation_touched": liquidation_touched,
    }


def replay(
    histories: dict[str, dict[str, list[Candle]]],
    start: int,
    end: int,
    settings: SignalSettings,
    spread_pct: float,
    funding_rate: float,
    maintenance_rate: float = 0.005,
    tier_leverage: int = 100,
    cfg: SignalsCfg | None = None,
) -> dict[str, object]:
    cfg = cfg or SignalsCfg()
    interval = cfg.trend.trigger_interval
    step = INTERVALS[interval]
    hold_minutes = settings.hold_hours * 60
    times = {
        symbol: {name: [c.time for c in bars] for name, bars in frames.items()}
        for symbol, frames in histories.items()
    }
    by_time = {
        symbol: {c.time: c for c in frames.get(interval, [])}
        for symbol, frames in histories.items()
    }
    active: dict[str, TrackedTrade] = {}
    closed: list[dict[str, object]] = []
    states: Counter[str] = Counter()
    blockers: Counter[str] = Counter()
    exit_reasons: Counter[str] = Counter()
    adverse: dict[str, float] = {}
    distances: dict[str, float | None] = {}
    touched: set[str] = set()
    entries = 0
    plans = 0
    plans_ceiling_ge_50 = 0
    unresolved_gaps = 0
    for now in range(start // step * step, end // step * step, step):
        slices = {
            symbol: {
                name: completed_slice(bars, times[symbol][name], name, now)
                for name, bars in frames.items()
            }
            for symbol, frames in histories.items()
        }
        btc = slices.get("BTCUSDT")
        btc_ok = btc is not None and complete_frames(btc, now)
        for symbol, frames in slices.items():
            current = by_time[symbol].get(now)
            if (
                current is None
                or btc is None
                or not btc_ok
                or not complete_frames(frames, now)
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
                DEPTH_USDT,
                DEPTH_USDT,
                # Quote units so the session-activity term compares like with like.
                sum(c.volume for c in frames["15m"][-96:]),
                funding_rate,
                8,
                (now // 28800 + 1) * 28800,
                [Tier(0, 50_000_000, maintenance_rate, tier_leverage)],
                0.000001,
                0.000001,
                now,
            )
            decision = evaluate_intraday(
                market, frames, btc["1h"], settings, cfg, now, None
            )
            states[decision.state] += 1
            if decision.plan:
                plans += 1
                if decision.plan.max_leverage >= 50:
                    plans_ceiling_ge_50 += 1
            if not decision.state.startswith("ENTER_"):
                for check in decision.checks:
                    if not check.passed and not check.waiting:
                        blockers[check.label] += 1
            trade = active.get(symbol)
            if trade:
                plan = trade.plan
                sign = 1 if plan.side == "long" else -1
                last = frames[interval][-1]
                worst = last.low if sign == 1 else last.high
                adverse[trade.id] = max(
                    adverse.get(trade.id, 0.0),
                    sign * (plan.entry - worst) / plan.entry * 100,
                )
                if (
                    plan.liquidation_estimate is not None
                    and sign * (worst - plan.liquidation_estimate) <= 0
                ):
                    touched.add(trade.id)
                old_stop = trade.current_stop
                evaluate_exit(trade, decision, frames[interval], now, cfg)
                if trade.state.startswith("EXIT_"):
                    if "Stop level" in trade.reason:
                        # Assume the worse outcome if both levels are touched in one candle.
                        fill = (
                            min(old_stop, last.open)
                            if sign == 1
                            else max(old_stop, last.open)
                        )
                    elif "profit target" in trade.reason:
                        fill = plan.target
                    else:
                        fill = market.bid if sign == 1 else market.ask
                    net = (
                        sign * (fill - plan.entry) * plan.quantity
                        - plan.notional * plan.cost_pct / 100
                    )
                    reason = exit_class(trade.reason, sign * (old_stop - plan.stop) > 0)
                    exit_reasons[reason] += 1
                    closed.append(
                        trade_record(
                            trade,
                            now,
                            fill,
                            net,
                            reason,
                            adverse.get(trade.id, 0.0),
                            trade.id in touched,
                        )
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
                    trade = TrackedTrade(
                        decision.signal_id,
                        symbol,
                        "paper",
                        now,
                        decision.plan,
                        decision.plan.stop,
                        decision.plan.entry,
                        checked_at=now,
                    )
                    active[symbol] = trade
                    distances[trade.id] = liquidation_distance_pct(trade)
                    entries += 1
    closed_count = len(closed)
    days = (end - start) / 86400
    held = [number(t["minutes_held"]) for t in closed]
    net_rs = [number(t["net_r"]) for t in closed if t["net_r"] is not None]
    hits = sum(
        1
        for t in closed
        if t["exit_reason"] == "target"
        or (
            t["exit_reason"] in ("trail", "late")
            and number(t["estimated_net_usdt"]) > 0
        )
    )
    adverse_shares = [
        adverse[trade_id] / distance
        for trade_id, distance in distances.items()
        if distance is not None and distance > 0 and trade_id in adverse
    ]
    return {
        "start": start,
        "end": end,
        "settings": asdict(settings),
        "trend": asdict(cfg.trend),
        "hold_minutes": hold_minutes,
        "evaluations": dict(states),
        "watch_blockers": dict(blockers),
        "closed_trades": closed,
        "closed_count": closed_count,
        "entries": entries,
        "entries_per_symbol_day": (
            entries / (len(histories) * days) if histories and days > 0 else None
        ),
        "open_at_end": len(active),
        "unresolved_trades_due_to_gaps": unresolved_gaps,
        "exit_reasons": dict(exit_reasons),
        "median_minutes_held": statistics.median(held) if held else None,
        "share_time_stop": (
            exit_reasons["time"] / closed_count if closed_count else None
        ),
        "liquidation_touches": len(touched),
        "worst_adverse_share_of_liquidation": (
            max(adverse_shares) if adverse_shares else None
        ),
        "hit_rate": hits / closed_count if closed_count else None,
        "average_net_r": statistics.fmean(net_rs) if net_rs else None,
        "estimated_net_usdt": sum(number(t["estimated_net_usdt"]) for t in closed),
        "plans": plans,
        "share_plans_ceiling_ge_50": plans_ceiling_ge_50 / plans if plans else None,
        "limitations": (
            "Candle replay only, not validated profitability. Fixed assumed funding, "
            "spread, maintenance margin, tier cap and depth; no historical mark "
            "basis, open-interest history, queue position or intra-candle path. "
            "Entries fill at the plan entry on the signal bar; stops count first on "
            "ambiguous candles; costs are charged for the maximum hold. Open trades "
            "are excluded from the realized total."
        ),
        "assumptions": {
            "spread_pct": spread_pct,
            "funding_rate_per_interval": funding_rate,
            "maintenance_rate": maintenance_rate,
            "tier_leverage": tier_leverage,
            "depth_usdt_per_side": DEPTH_USDT,
            "round_trip_fees_pct": cfg.round_trip_fee_pct,
            "slippage_pct": cfg.slippage_pct,
        },
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--symbols", default="BTCUSDT,ETHUSDT,SOLUSDT,DOGEUSDT,XRPUSDT")
    parser.add_argument("--days", type=int, default=7, help="1-30")
    parser.add_argument("--leverage", type=int, default=50, help="20-100")
    parser.add_argument("--hold-hours", type=int, default=2, choices=(1, 2))
    parser.add_argument("--equity", type=float, default=1000.0)
    parser.add_argument("--risk-pct", type=float, default=0.5)
    parser.add_argument("--spread-pct", type=float, default=0.03)
    parser.add_argument("--funding-rate", type=float, default=0.0001)
    parser.add_argument("--maintenance-rate", type=float, default=0.005)
    parser.add_argument("--tier-leverage", type=int, default=100)
    parser.add_argument("--output", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if (
        not 1 <= args.days <= 30
        or not 0 <= args.spread_pct <= 1
        or not -0.01 <= args.funding_rate <= 0.01
        or not 0 < args.maintenance_rate <= 0.05
        or not 1 <= args.tier_leverage <= 125
    ):
        parser.error(
            "Use 1-30 days and finite, plausible spread, funding, maintenance and tier assumptions"
        )
    settings = SignalSettings(
        args.equity, args.risk_pct, args.leverage, args.hold_hours, PROFILE
    )
    try:
        settings.validate()
    except ValueError as exc:
        parser.error(str(exc))
    symbols = list(
        dict.fromkeys(
            ["BTCUSDT"]
            + [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
        )
    )
    if len(symbols) > 10 or any(
        not s.isalnum() or not s.endswith("USDT") for s in symbols
    ):
        parser.error("Provide at most ten USDT symbols")
    cfg = SignalsCfg()
    intervals = intervals_for(cfg)
    step = INTERVALS[intervals[0]]
    now = int(time.time()) // 60 * 60
    start = now // step * step - args.days * 86400
    client = BitunixClient("", "", read_only=True, timeout=10)
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
        args.spread_pct,
        args.funding_rate,
        args.maintenance_rate,
        args.tier_leverage,
        cfg,
    )
    report["symbols"] = sorted(histories)
    Path(args.output).write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(
        json.dumps(
            {
                key: value
                for key, value in report.items()
                if key not in ("closed_trades", "trend")
            },
            indent=2,
        )
    )
    print(
        f"{len(histories)} symbols, {args.days}d at {args.leverage}x / "
        f"{args.hold_hours}h: {report['closed_count']} closed trades, "
        f"hit rate {report['hit_rate']}, avg net R {report['average_net_r']}, "
        f"liquidation touches {report['liquidation_touches']}"
    )
    print("WATCH blockers:", report["watch_blockers"])
    print(f"Report written to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
