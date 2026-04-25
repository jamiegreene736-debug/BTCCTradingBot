"""Trade-quality journal — JSONL log of structured per-trade context.

Every entry is one line of JSON, written to logs/trades.jsonl. After ~200-300
trades the journal becomes the data source for the next layer of tuning:
factor-group weight calibration, dynamic threshold by recent WR, conviction
multiplier validation, regime-specific parameter tuning.

Two event kinds:
  - "entry": logged when an order is placed (signal context + sizing)
  - "exit":  logged when a position closes (outcome + lifecycle stats)

Pairing entries to exits is done by `client_id` (deterministic clientId
already used for idempotency). Offline analysis joins on that field.

Designed to fail soft — if the journal can't write (disk full, perm
denied), it logs a warning and continues. Trading must never be blocked
by an observability failure.
"""
from __future__ import annotations

import json
import logging
import threading
import time
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)


class TradeJournal:
    """Append-only JSONL logger for structured trade events.

    Thread-safe (lock around file writes). Lazy-creates the parent
    directory. Each line: a single JSON object with at least
    {ts, kind, symbol, ...}. No log rotation — file is expected to
    grow linearly with trade volume; the user can rotate offline.
    """

    def __init__(self, path: str | Path = "logs/trades.jsonl"):
        self.path = Path(path)
        self._lock = threading.Lock()
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            log.warning("TradeJournal: could not create dir %s: %s",
                        self.path.parent, e)

    def _write(self, record: dict[str, Any]) -> None:
        record.setdefault("ts", time.time())
        try:
            line = json.dumps(record, separators=(",", ":"), default=str)
        except (TypeError, ValueError) as e:
            log.warning("TradeJournal: cannot serialize record: %s", e)
            return
        try:
            with self._lock:
                with self.path.open("a", encoding="utf-8") as f:
                    f.write(line + "\n")
        except Exception as e:
            log.warning("TradeJournal: write failed for %s: %s",
                        self.path, e)

    def record_entry(
        self,
        *,
        symbol: str,
        side: str,
        client_id: str,
        order_type: str,
        score: float,
        threshold_used: float | None,
        conviction_mult: float | None,
        indicator_count: int,
        pattern_score: float,
        reasons: list[str],
        atr_pct: float | None,
        adx: float | None,
        spread_pct: float | None,
        bid_depth: float | None,
        ask_depth: float | None,
        aggression_10s: float | None,
        real_cvd: float | None,
        activity_mult: float | None,
        session_weight: float | None,
        entry_price: float,
        stop_loss: float,
        take_profit: float,
        notional: float,
        leverage: int,
    ) -> None:
        """Log all available signal + execution context at entry time.

        Many fields are optional/None when the upstream feed is cold — the
        journal preserves them as nulls so post-hoc analysis can correlate
        outcomes against feed availability."""
        self._write({
            "kind": "entry",
            "symbol": symbol,
            "side": side,
            "client_id": client_id,
            "order_type": order_type,
            "score": score,
            "threshold_used": threshold_used,
            "conviction_mult": conviction_mult,
            "indicator_count": indicator_count,
            "pattern_score": pattern_score,
            "reasons": reasons,
            "atr_pct": atr_pct,
            "adx": adx,
            "spread_pct": spread_pct,
            "bid_depth": bid_depth,
            "ask_depth": ask_depth,
            "aggression_10s": aggression_10s,
            "real_cvd": real_cvd,
            "activity_mult": activity_mult,
            "session_weight": session_weight,
            "entry_price": entry_price,
            "stop_loss": stop_loss,
            "take_profit": take_profit,
            "notional": notional,
            "leverage": leverage,
        })

    def record_exit(
        self,
        *,
        symbol: str,
        position_id: str,
        side: str,
        entry_price: float,
        exit_price: float | None,
        exit_reason: str,
        hold_time_sec: float,
        max_favor_r: float | None,
        net_pnl: float | None,
        realized_pnl: float | None,
        fee: float | None,
        funding: float | None,
    ) -> None:
        """Log position lifecycle outcome.

        `exit_reason` is one of: tp_hit / sl_hit / partial_tp /
        tape_exit / stale_exit / time_exit / unknown.
        `max_favor_r` is the highest r_favor seen during the position
        (tracked by the bot's per-position state).
        """
        self._write({
            "kind": "exit",
            "symbol": symbol,
            "position_id": position_id,
            "side": side,
            "entry_price": entry_price,
            "exit_price": exit_price,
            "exit_reason": exit_reason,
            "hold_time_sec": hold_time_sec,
            "max_favor_r": max_favor_r,
            "net_pnl": net_pnl,
            "realized_pnl": realized_pnl,
            "fee": fee,
            "funding": funding,
        })
