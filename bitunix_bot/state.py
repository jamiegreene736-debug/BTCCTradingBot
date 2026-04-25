"""Thread-safe shared state between the trading worker and the dashboard."""
from __future__ import annotations

import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any


@dataclass
class TickEvent:
    ts: int                            # unix seconds
    kind: str                          # "signal" | "order" | "skip" | "error"
    text: str                          # human-readable summary
    extra: dict[str, Any] | None = None


class BotState:
    """Mutable, lock-protected. The bot writes; the dashboard reads."""

    def __init__(self, max_events: int = 100):
        self._lock = threading.RLock()
        self.started_at: int = int(time.time())
        self.last_tick_at: int = 0
        self.last_signal_text: str = "(none yet)"
        self.last_error_text: str = ""
        self.tick_count: int = 0
        self.signal_count: int = 0
        self.order_count: int = 0
        self.error_count: int = 0
        self.events: deque[TickEvent] = deque(maxlen=max_events)
        self.last_price: float | None = None
        self.last_klines_count: int = 0

    def record_tick(self, price: float | None, klines_n: int) -> None:
        with self._lock:
            self.last_tick_at = int(time.time())
            self.tick_count += 1
            if price:
                self.last_price = price
            self.last_klines_count = klines_n

    def record_signal(self, text: str, extra: dict[str, Any] | None = None) -> None:
        with self._lock:
            self.last_signal_text = text
            self.signal_count += 1
            self.events.append(TickEvent(int(time.time()), "signal", text, extra))

    def record_order(self, text: str, extra: dict[str, Any] | None = None) -> None:
        with self._lock:
            self.order_count += 1
            self.events.append(TickEvent(int(time.time()), "order", text, extra))

    def record_skip(self, text: str) -> None:
        with self._lock:
            self.events.append(TickEvent(int(time.time()), "skip", text))

    def record_error(self, text: str) -> None:
        with self._lock:
            self.error_count += 1
            self.last_error_text = text
            self.events.append(TickEvent(int(time.time()), "error", text))

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return {
                "started_at": self.started_at,
                "last_tick_at": self.last_tick_at,
                "last_signal_text": self.last_signal_text,
                "last_error_text": self.last_error_text,
                "tick_count": self.tick_count,
                "signal_count": self.signal_count,
                "order_count": self.order_count,
                "error_count": self.error_count,
                "last_price": self.last_price,
                "last_klines_count": self.last_klines_count,
                "events": [
                    {"ts": e.ts, "kind": e.kind, "text": e.text, "extra": e.extra}
                    for e in reversed(self.events)
                ],
            }


# Global singleton populated at startup.
_state: BotState | None = None


def get() -> BotState:
    global _state
    if _state is None:
        _state = BotState()
    return _state
