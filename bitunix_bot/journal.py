"""Trade-quality journal — JSONL log of structured per-trade context.

Every entry is one line of JSON, written to logs/trades.jsonl by default.

Persistence on Railway:
  Railway's filesystem is EPHEMERAL — every redeploy (every git push)
  wipes the file. To survive redeploys, mount a Railway persistent
  volume in the service Settings (e.g. at `/data`) and set the env var
  JOURNAL_PATH=/data/trades.jsonl. The journal will then accumulate
  across deploys.

  Without the volume + env var, expect the file to reset every push.
  Trade history is NOT lost (Bitunix retains it server-side); only the
  rich entry context (score, factor breakdown, ADX, tape signals, etc.)
  is journal-only and reset.

After ~200-300 trades the journal becomes the data source for the next
layer of tuning: factor-group weight calibration, dynamic threshold by
recent WR, conviction multiplier validation, regime-specific parameters.

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
import os
import threading
import time
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)


def _default_journal_path() -> Path:
    """Resolve the journal file path with env-var override.

    JOURNAL_PATH env var, if set, takes precedence — Railway / Docker /
    other deployments can point this at a persistent volume mount (e.g.
    `/data/trades.jsonl`) so the file survives redeploys. Falls back to
    `logs/trades.jsonl` (relative) which IS wiped on Railway redeploys.
    """
    override = os.environ.get("JOURNAL_PATH", "").strip()
    if override:
        return Path(override)
    return Path("logs/trades.jsonl")


class TradeJournal:
    """Append-only JSONL logger for structured trade events.

    Thread-safe (lock around file writes). Lazy-creates the parent
    directory. Each line: a single JSON object with at least
    {ts, kind, symbol, ...}. No log rotation — file is expected to
    grow linearly with trade volume; the user can rotate offline.
    """

    def __init__(self, path: str | Path | None = None):
        # If no explicit path given, resolve from env var or default.
        # This keeps tests' explicit-path constructors working while
        # letting production read JOURNAL_PATH for Railway volume mounts.
        if path is None:
            self.path = _default_journal_path()
        else:
            self.path = Path(path)
        self._lock = threading.Lock()
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            log.warning("TradeJournal: could not create dir %s: %s",
                        self.path.parent, e)
        log.info("TradeJournal writing to %s%s", self.path,
                 " (from JOURNAL_PATH env var)"
                 if os.environ.get("JOURNAL_PATH") else "")

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
        factor_trend: float | None = None,
        factor_mean_rev: float | None = None,
        factor_flow: float | None = None,
        factor_context: float | None = None,
        adaptive_adj: float | None = None,
        recent_trade_r_sum: float | None = None,
        entry_mechanism: str | None = None,
        limit_price: float | None = None,
        tob_bid: float | None = None,
        tob_ask: float | None = None,
        dynamic_timeout_secs: int | None = None,
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
            "factor_trend": factor_trend,
            "factor_mean_rev": factor_mean_rev,
            "factor_flow": factor_flow,
            "factor_context": factor_context,
            "atr_pct": atr_pct,
            "adx": adx,
            "spread_pct": spread_pct,
            "bid_depth": bid_depth,
            "ask_depth": ask_depth,
            "aggression_10s": aggression_10s,
            "real_cvd": real_cvd,
            "activity_mult": activity_mult,
            "session_weight": session_weight,
            "adaptive_adj": adaptive_adj,
            "recent_trade_r_sum": recent_trade_r_sum,
            # Execution-mechanism context — distinguishes MAKER_LIMIT_POST_ONLY
            # (rests on book at top-of-book bid/ask) from MARKET (taker, taken
            # immediately). limit_price is the actual limit-order price (None
            # for market). tob_bid / tob_ask captured at order placement.
            # dynamic_timeout_secs is the activity-scaled timeout assigned
            # to the post-only order (4-12s clamped); None for market.
            "entry_mechanism": entry_mechanism,
            "limit_price": limit_price,
            "tob_bid": tob_bid,
            "tob_ask": tob_ask,
            "dynamic_timeout_secs": dynamic_timeout_secs,
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
