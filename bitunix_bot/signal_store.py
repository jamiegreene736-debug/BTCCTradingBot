"""Durable alert and manually recorded trade state; never places exchange orders."""

from __future__ import annotations

import json
import sqlite3
import threading
from dataclasses import asdict, dataclass
from pathlib import Path

from .intraday import Candle, Decision, TradePlan
from .signal_config import SignalsCfg, SignalSettings


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
    checked_at: int = 0
    closed_at: int | None = None
    exit_price: float | None = None
    estimated_net_pnl: float | None = None
    stop_updated_at: int = 0


def evaluate_exit(
    trade: TrackedTrade,
    decision: Decision | None,
    candles: list[Candle],
    now: int,
    cfg: SignalsCfg,
) -> TrackedTrade:
    if trade.closed_at or trade.state.startswith("EXIT_"):
        return trade
    side = trade.plan.side
    sign = 1 if side == "long" else -1
    age = now - trade.opened_at
    if age >= trade.plan.hold_hours * 3600:
        trade.state, trade.reason = (
            f"EXIT_{side.upper()}",
            "Maximum holding time reached",
        )
        return trade
    if (
        decision is None
        or decision.price is None
        or not 0 <= now - decision.as_of <= cfg.max_data_age_seconds
    ):
        trade.state, trade.reason = (
            "REVIEW",
            "Market data unavailable; check the position and exchange-side stop",
        )
        return trade
    price = decision.price
    # Only fully post-entry candles are eligible; never count a wick that preceded the entry.
    new_bars = [c for c in candles if c.time >= max(trade.opened_at, trade.checked_at)]
    adverse = [price] + [c.low if side == "long" else c.high for c in new_bars]
    favorable = [price] + [c.high if side == "long" else c.low for c in new_bars]
    stop_bars = [c for c in new_bars if c.time >= trade.stop_updated_at]
    trailing_adverse = [price] + [
        c.low if side == "long" else c.high for c in stop_bars
    ]
    stop_hit = any(
        sign * (p - trade.current_stop) <= 0 for p in trailing_adverse
    ) or any(sign * (p - trade.plan.stop) <= 0 for p in adverse)
    target_hit = any(sign * (p - trade.plan.target) >= 0 for p in favorable)
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
    elif decision.metrics.get("trend_1h") == ("short" if side == "long" else "long"):
        trade.state, trade.reason = (
            f"EXIT_{side.upper()}",
            "Completed 1h structure reversed against the trade",
        )
    else:
        trade.best_price = max([trade.best_price] + favorable, key=lambda p: sign * p)
        initial_risk = abs(trade.plan.entry - trade.plan.stop)
        progress_r = sign * (trade.best_price - trade.plan.entry) / initial_risk
        if age >= cfg.stale_trade_hours * 3600 and progress_r < cfg.stale_progress_r:
            trade.state, trade.reason = (
                f"EXIT_{side.upper()}",
                "Trade failed to progress within the review window",
            )
        else:
            trade.state, trade.reason = (
                f"HOLD_{side.upper()}",
                "Original setup remains active",
            )
            proposed = trade.current_stop
            if progress_r >= cfg.breakeven_at_r:
                covered = (
                    trade.plan.entry
                    + sign * trade.plan.entry * trade.plan.cost_pct / 100
                )
                if sign * (covered - proposed) > 0:
                    proposed = covered
            if progress_r >= cfg.trailing_activate_r:
                # Ratchet on observed closes; keep at least the original risk distance.
                trail = price - sign * initial_risk
                if sign * (trail - proposed) > 0:
                    proposed = trail
            if sign * (proposed - trade.current_stop) > 0:
                trade.current_stop = proposed
                trade.stop_updated_at = now
                trade.reason = (
                    "Protective stop advanced; update the exchange-side stop manually"
                )
    # Retain the start of the current 15m candle so its full range is examined once closed.
    trade.checked_at = max(trade.opened_at, now // 900 * 900)
    return trade


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
            value = json.loads(row[0])
            value["plan"] = TradePlan(**value["plan"])
            trade = TrackedTrade(**value)
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
