"""Forward test of every ENTER alert: excursion in R and first-touch outcome.

No orders. Each ENTER alert becomes a record measured against the completed
trigger-interval candles that follow it, until the target, stop or estimated
liquidation is touched or the hold horizon expires. Stops and targets touched
inside the same bar count as a stop.
"""

from __future__ import annotations

from dataclasses import dataclass

from .intraday import INTERVALS, Candle, Decision


@dataclass
class ForwardTest:
    id: str
    symbol: str
    side: str
    profile: str
    setup: str
    interval: str
    opened_at: int
    horizon_seconds: int
    entry: float
    stop: float
    target: float
    liquidation: float | None
    mfe_r: float = 0.0
    mae_r: float = 0.0
    outcome: str = "open"
    exit_r: float | None = None
    resolved_at: int | None = None
    checked_at: int = 0
    bars_seen: int = 0


def forward_test_from_decision(decision: Decision, now: int) -> ForwardTest | None:
    plan = decision.plan
    if not plan or not decision.signal_id or not decision.state.startswith("ENTER_"):
        return None
    return ForwardTest(
        f"ft:{decision.signal_id}",
        decision.symbol,
        plan.side,
        plan.profile,
        decision.setup,
        plan.trigger_interval,
        now,
        plan.hold_hours * 3600,
        plan.entry,
        plan.stop,
        plan.target,
        plan.liquidation_estimate,
    )


def update_forward_test(test: ForwardTest, candles: list[Candle], now: int) -> ForwardTest:
    if test.outcome != "open":
        return test
    sign = 1 if test.side == "long" else -1
    risk = abs(test.entry - test.stop)
    if risk <= 0:
        test.outcome, test.resolved_at = "invalid", now
        return test
    seconds = INTERVALS.get(test.interval, 60)
    reward_r = sign * (test.target - test.entry) / risk
    fresh = [
        c
        for c in candles
        if c.time >= test.opened_at // seconds * seconds and c.time > test.checked_at
    ]
    last_close: float | None = None
    for candle in fresh:
        favorable = candle.high if sign == 1 else candle.low
        adverse = candle.low if sign == 1 else candle.high
        test.mfe_r = max(test.mfe_r, sign * (favorable - test.entry) / risk)
        test.mae_r = min(test.mae_r, sign * (adverse - test.entry) / risk)
        test.bars_seen += 1
        test.checked_at = candle.time
        last_close = candle.close
        liquidated = (
            test.liquidation is not None
            and sign * (adverse - test.liquidation) <= 0
        )
        if liquidated:
            test.outcome, test.exit_r = "liquidation", sign * (test.liquidation - test.entry) / risk
        elif sign * (adverse - test.stop) <= 0:
            test.outcome, test.exit_r = "stop", -1.0
        elif sign * (favorable - test.target) >= 0:
            test.outcome, test.exit_r = "target", reward_r
        if test.outcome != "open":
            test.resolved_at = candle.time + seconds
            return test
    if now >= test.opened_at + test.horizon_seconds:
        test.outcome = "expired"
        test.resolved_at = now
        if last_close is None and candles:
            last_close = candles[-1].close
        test.exit_r = (
            sign * (last_close - test.entry) / risk if last_close is not None else None
        )
    return test


def summarize_forward_tests(tests: list[ForwardTest]) -> dict[str, object]:
    resolved = [t for t in tests if t.outcome not in ("open", "invalid")]
    with_exit = [t for t in resolved if t.exit_r is not None]
    counts = {
        outcome: sum(1 for t in tests if t.outcome == outcome)
        for outcome in ("open", "target", "stop", "liquidation", "expired")
    }
    return {
        "count": len(tests),
        "resolved": len(resolved),
        "outcomes": counts,
        "hit_rate": (
            round(counts["target"] / len(resolved), 3) if resolved else None
        ),
        "average_r": (
            round(sum(t.exit_r for t in with_exit) / len(with_exit), 3)
            if with_exit
            else None
        ),
        "average_mfe_r": (
            round(sum(t.mfe_r for t in resolved) / len(resolved), 3) if resolved else None
        ),
        "average_mae_r": (
            round(sum(t.mae_r for t in resolved) / len(resolved), 3) if resolved else None
        ),
        "liquidation_touches": counts["liquidation"],
    }
