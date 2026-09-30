"""Durable alert, recorded, and imported live-position state; never places orders."""

from __future__ import annotations

import json
import math
import sqlite3
import threading
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any

from .forward_test import ForwardTest
from .intraday import INTERVALS, Candle, Check, Decision, TradePlan
from .signal_config import SCALP_PROFILES, SignalsCfg, SignalSettings


@dataclass
class TrackedTrade:
    id: str
    symbol: str
    kind: str
    opened_at: int
    plan: TradePlan
    current_stop: float
    best_price: float
    state: str = "HOLD"
    reason: str = "Monitoring the original trade plan"
    # Bar cursor: start of the trigger candle the next evaluation resumes from.
    checked_at: int = 0
    # Wall-clock time of the last hold/close evaluation; the overlay's "Checked".
    evaluated_at: int = 0
    closed_at: int | None = None
    exit_price: float | None = None
    estimated_net_pnl: float | None = None
    stop_updated_at: int = 0
    mark_price: float | None = None
    unrealized_pnl: float | None = None
    exchange_position_id: str = ""
    suggestion: str = ""
    hold_confidence: int | None = None
    checks: list[Check] = field(default_factory=list)
    exchange_stop_confirmed: bool = False


HOLD_CHECK_GROUPS: dict[str, str] = {
    "Fresh market data": "risk",
    "Stop not reached": "risk",
    "Target still open": "risk",
    "Hold time remaining": "risk",
    "Drawdown contained": "risk",
    "Room to stop": "risk",
    "Liquidation buffer": "risk",
    "Exchange protective stop": "risk",
    "Structure intact": "structure",
    "Bias intact": "structure",
    "Progress vs review window": "structure",
    "Session VWAP": "tape",
    "Funding carry": "cost",
}
HOLD_CHECK_LABELS: tuple[str, ...] = tuple(HOLD_CHECK_GROUPS)


def hold_check(label: str, passed: bool, detail: str) -> Check:
    return Check(label, passed, detail, HOLD_CHECK_GROUPS[label])


def order_hold_checks(checks: list[Check]) -> list[Check]:
    by_label = {item.label: item for item in checks}
    return [
        by_label.get(label)
        or hold_check(label, False, "Waiting for a fresh hold evaluation")
        for label in HOLD_CHECK_LABELS
    ]


def _finite(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def exchange_stop_ok(trade: TrackedTrade) -> bool:
    return trade.kind == "paper" or trade.exchange_stop_confirmed


@dataclass(frozen=True)
class ExitRules:
    """Per-profile exit thresholds read once per evaluation."""

    stale_after_s: int
    stale_need_r: float
    stale_window: str
    breakeven_at_r: float
    trailing_activate_r: float
    liquidation_buffer_pct: float
    hope_exit_r: float
    structure_key: str | None
    late_after_s: int | None
    late_min_r: float | None


def exit_rules(plan: TradePlan, cfg: SignalsCfg) -> ExitRules:
    """Exit thresholds for a plan: scalp reads cfg.scalp, everything else cfg.trend.

    Trend windows are fractions of the planned hold in minutes; the stale
    review never fires before 10 minutes. A fade scalp is entered against the
    trend by design, so it carries no structure-reversal exit and no late-hold
    exit.
    """
    if plan.profile in SCALP_PROFILES:
        s = cfg.scalp
        return ExitRules(
            s.stale_minutes * 60,
            s.stale_progress_r,
            f"{s.stale_minutes}m",
            s.breakeven_at_r,
            s.trailing_activate_r,
            s.liquidation_buffer_pct,
            0.75,
            None,
            None,
            None,
        )
    t = cfg.trend
    hold_minutes = plan.hold_hours * 60
    stale_after_s = max(600, round(t.stale_hold_fraction * hold_minutes) * 60)
    return ExitRules(
        stale_after_s,
        t.stale_progress_r,
        f"{stale_after_s // 60}m",
        t.breakeven_at_r,
        t.trailing_activate_r,
        t.liquidation_buffer_pct,
        t.hope_exit_r,
        "trend_15m",
        round(t.late_hold_fraction * hold_minutes * 60),
        t.late_hold_min_r,
    )


def _apply_live_suggestion(trade: TrackedTrade) -> None:
    side = trade.plan.side.upper()
    passed = sum(1 for item in trade.checks if item.passed)
    total = len(trade.checks)
    trade.hold_confidence = round(100 * passed / total) if total else 0
    if trade.state.startswith("EXIT_"):
        trade.suggestion = f"CLOSE_{side}"
        trade.hold_confidence = min(trade.hold_confidence, 10)
    elif not exchange_stop_ok(trade):
        trade.suggestion = "SET_STOP"
        trade.hold_confidence = min(trade.hold_confidence, 20)
        trade.reason = (
            f"Set the Bitunix stop at {trade.current_stop:.5g} now. "
            "The overlay cannot prevent liquidation. Do not wait for a reversal."
        )
    elif trade.state == "REVIEW":
        trade.suggestion = "REVIEW"
    elif trade.hold_confidence >= 70:
        trade.suggestion = f"HOLD_{side}"
    else:
        trade.suggestion = "CONSIDER_CLOSE"


def evaluate_exit(
    trade: TrackedTrade,
    decision: Decision | None,
    candles: list[Candle],
    now: int,
    cfg: SignalsCfg,
) -> TrackedTrade:
    if trade.closed_at:
        return trade
    side = trade.plan.side
    sign = 1 if side == "long" else -1
    opposite = "short" if side == "long" else "long"
    scalp = trade.plan.profile in SCALP_PROFILES
    rules = exit_rules(trade.plan, cfg)
    interval_seconds = INTERVALS.get(trade.plan.trigger_interval, 900)
    age = now - trade.opened_at
    hold_minutes = trade.plan.hold_hours * 60
    minutes_left = hold_minutes - age / 60
    # Plans held for at most two hours read in minutes; older stored plans
    # (pre-1.8 swing rows) keep their hour strings.
    in_minutes = trade.plan.hold_hours <= 2
    time_left = (
        f"{max(0.0, minutes_left):.0f} min left"
        if in_minutes
        else f"{max(0.0, minutes_left / 60):.1f}h left"
    )
    max_hold = age >= trade.plan.hold_hours * 3600
    data_fresh = (
        decision is not None
        and decision.price is not None
        and 0 <= now - decision.as_of <= cfg.max_data_age_seconds
    )
    last = _finite(getattr(decision, "price", None)) if decision else None
    mark = _finite(trade.mark_price) or last
    metrics = decision.metrics if decision else {}
    trend_1h = metrics.get("trend_1h")
    trend_15m = metrics.get("trend_15m")
    vwap = _finite(metrics.get("vwap"))
    funding_pct = _finite(metrics.get("funding_rate_pct"))
    initial_risk = abs(trade.plan.entry - trade.plan.stop)
    live = mark if mark is not None else last
    current_r = (
        sign * (live - trade.plan.entry) / initial_risk
        if live is not None and initial_risk > 0
        else None
    )
    room_r = (
        sign * (live - trade.current_stop) / initial_risk
        if live is not None and initial_risk > 0
        else None
    )
    # A fade scalp is entered against the trend by design; only trend trades
    # exit on a completed structure reversal (15m for trend).
    structure_reversed = (
        rules.structure_key is not None
        and metrics.get(rules.structure_key) == opposite
    )
    stop_hit = False
    target_hit = False
    stale = False
    progress_r = None
    candidate_best = trade.best_price
    if data_fresh and last is not None:
        new_bars = [
            candle
            for candle in candles
            if candle.time >= max(trade.opened_at, trade.checked_at)
        ]
        adverse = [last] + [
            candle.low if side == "long" else candle.high for candle in new_bars
        ]
        favorable = [last] + [
            candle.high if side == "long" else candle.low for candle in new_bars
        ]
        stop_bars = [candle for candle in new_bars if candle.time >= trade.stop_updated_at]
        trailing_adverse = [last] + [
            candle.low if side == "long" else candle.high for candle in stop_bars
        ]
        stop_hit = any(
            sign * (price - trade.current_stop) <= 0 for price in trailing_adverse
        ) or any(sign * (price - trade.plan.stop) <= 0 for price in adverse)
        target_hit = any(sign * (price - trade.plan.target) >= 0 for price in favorable)
        candidate_best = max(
            [trade.best_price] + favorable, key=lambda price: sign * price
        )
        progress_r = (
            sign * (candidate_best - trade.plan.entry) / initial_risk
            if initial_risk > 0
            else None
        )
        stale = (
            progress_r is not None
            and age >= rules.stale_after_s
            and progress_r < rules.stale_need_r
        )
    late = (
        rules.late_after_s is not None
        and rules.late_min_r is not None
        and current_r is not None
        and age >= rules.late_after_s
        and current_r < rules.late_min_r
    )

    liq = _finite(trade.plan.liquidation_estimate)
    liq_room_pct = (
        sign * (live - liq) / trade.plan.entry * 100
        if live is not None and liq is not None and trade.plan.entry
        else None
    )
    paying = (
        funding_pct is not None
        and (
            (side == "long" and funding_pct > 0)
            or (side == "short" and funding_pct < 0)
        )
    )
    trade.checks = order_hold_checks(
        [
            hold_check(
                "Fresh market data",
                data_fresh,
                "Live suggestion needs a fresh market snapshot"
                if data_fresh
                else "Market data unavailable; check Bitunix and the exchange-side stop",
            ),
            hold_check(
                "Stop not reached",
                data_fresh and not stop_hit,
                "Stop still intact"
                if data_fresh and not stop_hit
                else (
                    "Stop level reached; verify your exchange fill"
                    if stop_hit
                    else "Cannot confirm the stop without a fresh price"
                ),
            ),
            hold_check(
                "Target still open",
                data_fresh and not target_hit,
                "Structural target has not printed"
                if data_fresh and not target_hit
                else (
                    "Structural profit target reached"
                    if target_hit
                    else "Cannot confirm the target without a fresh price"
                ),
            ),
            hold_check(
                "Hold time remaining",
                not max_hold,
                (
                    f"{max(0.0, minutes_left):.0f} of {hold_minutes} min hold left"
                    if in_minutes
                    else f"{max(0.0, minutes_left / 60):.1f}h of {trade.plan.hold_hours}h hold left"
                )
                if not max_hold
                else "Maximum holding time reached",
            ),
            hold_check(
                "Drawdown contained",
                current_r is not None and current_r > -0.5,
                (
                    f"{current_r:+.2f}R from entry; review a close below −0.5R"
                    if current_r is not None
                    else "Unrealized R is unavailable"
                ),
            ),
            hold_check(
                "Room to stop",
                room_r is not None and room_r > 0.25,
                (
                    f"{room_r:.2f}R remaining to the working stop; under 0.25R is a close review"
                    if room_r is not None
                    else "Distance to stop is unavailable"
                ),
            ),
            hold_check(
                "Liquidation buffer",
                liq_room_pct is not None and liq_room_pct >= rules.liquidation_buffer_pct,
                (
                    f"{liq_room_pct:.2f}% to estimated isolated liquidation; need ≥{rules.liquidation_buffer_pct:g}%"
                    if liq_room_pct is not None
                    else "Estimated liquidation unavailable; verify the price on Bitunix"
                ),
            ),
            hold_check(
                "Exchange protective stop",
                exchange_stop_ok(trade),
                (
                    "Paper track; no Bitunix stop required"
                    if trade.kind == "paper"
                    else (
                        f"Confirmed Bitunix stop at {trade.current_stop:.5g}"
                        if trade.exchange_stop_confirmed
                        else (
                            f"Bitunix stop at {trade.current_stop:.5g} is not confirmed. "
                            "A reversal will not save an unprotected position"
                        )
                    )
                ),
            ),
            hold_check(
                "Structure intact",
                scalp or trend_15m == side,
                (
                    "Mean-reversion scalp; the trend structure is faded by design"
                    if scalp
                    else f"15m structure {trend_15m}"
                    if trend_15m
                    else "15m structure unavailable"
                ),
            ),
            hold_check(
                "Bias intact",
                scalp or trend_1h == side,
                (
                    "Mean-reversion scalp; the 1h bias is not required"
                    if scalp
                    else f"1h bias {trend_1h}"
                    if trend_1h
                    else "1h bias unavailable"
                ),
            ),
            hold_check(
                "Progress vs review window",
                data_fresh and not stale,
                (
                    f"{progress_r:.2f}R best progress; need {rules.stale_need_r:g}R within {rules.stale_window}"
                    if progress_r is not None
                    else "Progress cannot be measured without a fresh price"
                ),
            ),
            hold_check(
                "Session VWAP",
                (
                    live is not None and vwap is not None
                    if scalp
                    else live is not None and vwap is not None and sign * (live - vwap) >= 0
                ),
                (
                    (
                        f"Price {live:.5g} vs session VWAP {vwap:.5g}; VWAP is the mean-reversion target"
                        if scalp
                        else f"Price {live:.5g} vs session VWAP {vwap:.5g}; hold wants the {side} side"
                    )
                    if live is not None and vwap is not None
                    else "Session VWAP unavailable"
                ),
            ),
            hold_check(
                "Funding carry",
                funding_pct is not None and (not paying or abs(funding_pct) < 0.005),
                (
                    f"Funding {funding_pct:.4f}% / interval; "
                    + ("paying against the position" if paying else "receiving or flat")
                    if funding_pct is not None
                    else "Funding rate unavailable"
                ),
            ),
        ]
    )

    if trade.state.startswith("EXIT_"):
        _apply_live_suggestion(trade)
        return trade
    if max_hold:
        trade.state, trade.reason = (
            f"EXIT_{side.upper()}",
            "Maximum holding time reached",
        )
        _apply_live_suggestion(trade)
        return trade
    if not data_fresh:
        trade.state, trade.reason = (
            "REVIEW",
            "Market data unavailable; check the position and exchange-side stop",
        )
        _apply_live_suggestion(trade)
        return trade

    stop_advanced = False
    if stop_hit:
        trade.state, trade.reason = (
            f"EXIT_{side.upper()}",
            "Stop level reached; verify your exchange fill",
        )
    elif target_hit:
        trade.state, trade.reason = (
            f"EXIT_{side.upper()}",
            "Structural profit target reached",
        )
    elif structure_reversed:
        trade.state, trade.reason = (
            f"EXIT_{side.upper()}",
            "Completed 15m structure reversed against the trade",
        )
    elif (
        liq_room_pct is not None
        and liq_room_pct < rules.liquidation_buffer_pct
    ):
        trade.state, trade.reason = (
            f"EXIT_{side.upper()}",
            "Estimated liquidation buffer is gone. Close on Bitunix now. "
            "A reversal will not beat liquidation.",
        )
    elif current_r is not None and current_r <= -rules.hope_exit_r:
        trade.state, trade.reason = (
            f"EXIT_{side.upper()}",
            f"Do not wait for a reversal. Live {current_r:+.2f}R; the structural "
            "stop is next. Close on Bitunix now.",
        )
    elif (
        not exchange_stop_ok(trade)
        and current_r is not None
        and current_r <= -0.5
    ):
        trade.state, trade.reason = (
            f"EXIT_{side.upper()}",
            "Unprotected position is already losing. Do not wait for a bounce; "
            "close on Bitunix or the exchange will liquidate you.",
        )
    else:
        trade.best_price = candidate_best
        if stale:
            trade.state, trade.reason = (
                f"EXIT_{side.upper()}",
                "Trade failed to progress within the review window",
            )
        elif late:
            trade.state, trade.reason = (
                f"EXIT_{side.upper()}",
                "Hold window closing without progress; close on Bitunix",
            )
        else:
            trade.state = f"HOLD_{side.upper()}"
            proposed = trade.current_stop
            if progress_r is not None and progress_r >= rules.breakeven_at_r:
                covered = (
                    trade.plan.entry
                    + sign * trade.plan.entry * trade.plan.cost_pct / 100
                )
                if sign * (covered - proposed) > 0:
                    proposed = covered
            if (
                progress_r is not None
                and progress_r >= rules.trailing_activate_r
                and last is not None
            ):
                # Ratchet on observed closes; keep at least the original risk distance.
                trail = last - sign * initial_risk
                if sign * (trail - proposed) > 0:
                    proposed = trail
            if sign * (proposed - trade.current_stop) > 0:
                trade.current_stop = proposed
                trade.stop_updated_at = now
                trade.reason = (
                    "Protective stop advanced; update the exchange-side stop manually"
                )
                stop_advanced = True
            elif current_r is not None:
                structure_text = (
                    f"1h {trend_1h or '—'}" if scalp else f"15m {trend_15m or '—'}"
                )
                trade.reason = (
                    f"Live: {live:.5g} · {current_r:+.2f}R · {structure_text} · "
                    f"{time_left}"
                )
            else:
                trade.reason = "Original setup remains active"
    _apply_live_suggestion(trade)
    if trade.suggestion == "CONSIDER_CLOSE" and not stop_advanced:
        failed = next((item for item in trade.checks if not item.passed), None)
        if failed:
            trade.reason = failed.detail
    # checked_at keeps the start of the current trigger candle so its full range
    # is examined once it closes. It is a bar cursor, not the check time: the
    # overlay reads evaluated_at, which advances on every evaluation.
    trade.checked_at = max(trade.opened_at, now // interval_seconds * interval_seconds)
    trade.evaluated_at = now
    return trade


def _load_trade(value: dict[str, Any]) -> TrackedTrade:
    payload = dict(value)
    plan = payload.get("plan")
    if isinstance(plan, dict):
        plan_fields = {item.name for item in fields(TradePlan)}
        payload["plan"] = TradePlan(
            **{key: plan[key] for key in plan if key in plan_fields}
        )
    raw_checks = payload.get("checks") or []
    check_fields = {item.name for item in fields(Check)}
    payload["checks"] = [
        Check(**{key: item[key] for key in item if key in check_fields})
        if isinstance(item, dict)
        else item
        for item in raw_checks
    ]
    allowed = {item.name for item in fields(TrackedTrade)}
    return TrackedTrade(**{key: payload[key] for key in payload if key in allowed})


class SignalStore:
    def __init__(self, path: str):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(path, check_same_thread=False)
        self.connection.execute("PRAGMA journal_mode=WAL")
        self.lock = threading.RLock()
        with self.connection:
            self.connection.execute(
                "CREATE TABLE IF NOT EXISTS settings (id INTEGER PRIMARY KEY, payload TEXT NOT NULL)"
            )
            self.connection.execute(
                "CREATE TABLE IF NOT EXISTS trades (id TEXT PRIMARY KEY, payload TEXT NOT NULL)"
            )
            self.connection.execute(
                "CREATE TABLE IF NOT EXISTS alerts (id TEXT PRIMARY KEY, time INTEGER NOT NULL, payload TEXT NOT NULL)"
            )
            self.connection.execute(
                "CREATE TABLE IF NOT EXISTS forward_tests (id TEXT PRIMARY KEY, time INTEGER NOT NULL, payload TEXT NOT NULL)"
            )

    def settings(self) -> SignalSettings:
        with self.lock:
            row = self.connection.execute(
                "SELECT payload FROM settings WHERE id=1"
            ).fetchone()
        return SignalSettings.from_dict(json.loads(row[0])) if row else SignalSettings()

    def save_settings(self, settings: SignalSettings) -> None:
        settings.validate()
        with self.lock, self.connection:
            self.connection.execute(
                "INSERT OR REPLACE INTO settings VALUES (1, ?)",
                (json.dumps(asdict(settings)),),
            )

    def trades(self, active_only: bool = False) -> list[TrackedTrade]:
        with self.lock:
            rows = self.connection.execute(
                "SELECT payload FROM trades ORDER BY rowid DESC"
            ).fetchall()
        result = []
        for row in rows:
            trade = _load_trade(json.loads(row[0]))
            if not active_only or trade.closed_at is None:
                result.append(trade)
        return result

    def save_trade(self, trade: TrackedTrade) -> None:
        with self.lock, self.connection:
            self.connection.execute(
                "INSERT OR REPLACE INTO trades VALUES (?, ?)",
                (trade.id, json.dumps(asdict(trade), allow_nan=False)),
            )

    def record_alert(
        self,
        key: str,
        state: str,
        symbol: str,
        reason: str,
        now: int,
        *,
        setup: str = "",
        side: str = "",
    ) -> None:
        payload = {
            "id": key,
            "state": state,
            "symbol": symbol,
            "reason": reason,
            "time": now,
            "setup": setup,
            "side": side,
        }
        with self.lock, self.connection:
            self.connection.execute(
                "INSERT OR IGNORE INTO alerts VALUES (?, ?, ?)",
                (key, now, json.dumps(payload)),
            )
            self.connection.execute(
                "DELETE FROM alerts WHERE id NOT IN (SELECT id FROM alerts ORDER BY time DESC, rowid DESC LIMIT 500)"
            )

    def history(self) -> list[dict[str, object]]:
        with self.lock:
            rows = self.connection.execute(
                "SELECT payload FROM alerts ORDER BY time DESC, rowid DESC LIMIT 50"
            ).fetchall()
        return [json.loads(row[0]) for row in rows]

    def record_forward_test(self, test: ForwardTest) -> bool:
        with self.lock, self.connection:
            cursor = self.connection.execute(
                "INSERT OR IGNORE INTO forward_tests VALUES (?, ?, ?)",
                (test.id, test.opened_at, json.dumps(asdict(test), allow_nan=False)),
            )
            self.connection.execute(
                "DELETE FROM forward_tests WHERE id NOT IN (SELECT id FROM forward_tests ORDER BY time DESC, rowid DESC LIMIT 1000)"
            )
            return cursor.rowcount > 0

    def save_forward_test(self, test: ForwardTest) -> None:
        with self.lock, self.connection:
            self.connection.execute(
                "INSERT OR REPLACE INTO forward_tests VALUES (?, ?, ?)",
                (test.id, test.opened_at, json.dumps(asdict(test), allow_nan=False)),
            )

    def forward_tests(self, open_only: bool = False, limit: int = 1000) -> list[ForwardTest]:
        with self.lock:
            rows = self.connection.execute(
                "SELECT payload FROM forward_tests ORDER BY time DESC, rowid DESC LIMIT ?",
                (limit,),
            ).fetchall()
        allowed = {item.name for item in fields(ForwardTest)}
        result = []
        for row in rows:
            payload = json.loads(row[0])
            test = ForwardTest(**{key: payload[key] for key in payload if key in allowed})
            if not open_only or test.outcome == "open":
                result.append(test)
        return result
