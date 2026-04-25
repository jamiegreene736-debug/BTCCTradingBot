"""WebSocket order book feed + imbalance computation.

Maintains a top-N order book per symbol via Bitunix's public depth_books
WebSocket channel and exposes:

  - get_imbalance(symbol) -> float in [-1, 1]
      Positive = bid pressure (bullish); negative = ask pressure (bearish).
      Computed as (bid_vol - ask_vol) / (bid_vol + ask_vol) over top N levels.

The feed runs in a background daemon thread. It auto-reconnects on
disconnect and pings every PING_INTERVAL seconds (Bitunix's WS uses
{"op":"ping","ping":<unix>}). On stop it closes cleanly.

Defensive parsing: Bitunix's depth payload schema isn't fully documented
publicly. We accept several plausible field-name conventions (b/a vs
bids/asks, [price, size] vs {"px":..,"sz":..}, etc.).
"""
from __future__ import annotations

import json
import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Any

import websocket  # websocket-client

log = logging.getLogger(__name__)

WS_URL = "wss://fapi.bitunix.com/public/"
CHANNEL = "depth_books"
PING_INTERVAL = 15.0
RECONNECT_BACKOFF_SECS = 5.0
# If no messages received for this long, the connection looks dead even though
# the socket may report itself open. Force a reconnect.
SILENCE_TIMEOUT_SECS = 60.0


@dataclass
class Book:
    bids: list[tuple[float, float]] = field(default_factory=list)  # (price desc)
    asks: list[tuple[float, float]] = field(default_factory=list)  # (price asc)
    last_update: float = 0.0


class OrderBookFeed:
    """Background WebSocket subscriber. Thread-safe accessors for imbalance."""

    def __init__(self, symbols: list[str], depth_levels: int = 10):
        self.symbols = [s.upper() for s in symbols]
        self.depth_levels = depth_levels
        self._books: dict[str, Book] = {s: Book() for s in self.symbols}
        self._lock = threading.RLock()
        self._stop = threading.Event()
        self._ws: websocket.WebSocketApp | None = None
        self._thread: threading.Thread | None = None
        self._ping_thread: threading.Thread | None = None
        self._last_message_at: float = 0.0  # for the silence watchdog

    # ------------------------------------------------------------------ lifecycle

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="ob-ws", daemon=True)
        self._thread.start()
        log.info("OrderBookFeed started for %d symbol(s)", len(self.symbols))

    def stop(self) -> None:
        self._stop.set()
        if self._ws:
            try:
                self._ws.close()
            except Exception:
                pass

    # ------------------------------------------------------------------ accessors

    def get_imbalance(self, symbol: str, max_age_secs: float = 30.0) -> float | None:
        sym = symbol.upper()
        with self._lock:
            book = self._books.get(sym)
            if not book or not book.bids or not book.asks:
                return None
            if (time.time() - book.last_update) > max_age_secs:
                return None
            n = self.depth_levels
            bid_vol = sum(sz for _, sz in book.bids[:n])
            ask_vol = sum(sz for _, sz in book.asks[:n])
            total = bid_vol + ask_vol
            if total <= 0:
                return None
            return (bid_vol - ask_vol) / total

    def is_connected(self) -> bool:
        return self._ws is not None and getattr(self._ws, "sock", None) is not None

    # ------------------------------------------------------------------ ws thread

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                self._ws = websocket.WebSocketApp(
                    WS_URL,
                    on_open=self._on_open,
                    on_message=self._on_message,
                    on_error=self._on_error,
                    on_close=self._on_close,
                )
                self._ws.run_forever(ping_interval=0)  # we ping ourselves
            except Exception as e:
                log.warning("OB WS run_forever exception: %s", e)
            if self._stop.is_set():
                break
            log.info("OB WS reconnecting in %ss", RECONNECT_BACKOFF_SECS)
            time.sleep(RECONNECT_BACKOFF_SECS)

    # ------------------------------------------------------------------ callbacks

    def _on_open(self, ws: websocket.WebSocketApp) -> None:
        log.info("OB WS connected; subscribing to %d symbols", len(self.symbols))
        sub_msg = {
            "op": "subscribe",
            "args": [{"symbol": s, "ch": CHANNEL} for s in self.symbols],
        }
        try:
            ws.send(json.dumps(sub_msg))
        except Exception as e:
            log.warning("OB WS subscribe send failed: %s", e)
        # Start a self-pinger to keep connection alive.
        if not self._ping_thread or not self._ping_thread.is_alive():
            self._ping_thread = threading.Thread(target=self._pinger, name="ob-ping", daemon=True)
            self._ping_thread.start()

    def _on_message(self, ws: websocket.WebSocketApp, message: str) -> None:
        self._last_message_at = time.time()
        try:
            msg = json.loads(message)
        except json.JSONDecodeError:
            return
        # Heartbeat reply: ignore.
        if isinstance(msg, dict) and msg.get("op") == "pong":
            return
        # Subscribe ack.
        if isinstance(msg, dict) and msg.get("op") == "subscribe":
            log.debug("OB WS subscribe ack: %s", msg)
            return
        # Data payload — defensive parsing.
        symbol = self._extract_symbol(msg)
        if not symbol:
            return
        bids, asks = self._extract_book(msg)
        if bids is None and asks is None:
            return
        with self._lock:
            book = self._books.setdefault(symbol.upper(), Book())
            if bids is not None:
                book.bids = bids
            if asks is not None:
                book.asks = asks
            book.last_update = time.time()

    def _on_error(self, ws: websocket.WebSocketApp, error: Any) -> None:
        log.warning("OB WS error: %s", error)

    def _on_close(self, ws: websocket.WebSocketApp, *args: Any) -> None:
        log.info("OB WS closed")

    def _pinger(self) -> None:
        """Send heartbeats AND watchdog the connection — if no messages have
        arrived for SILENCE_TIMEOUT_SECS, force-close so _run reconnects.
        Sometimes WS sockets stay nominally open but stop delivering data."""
        while not self._stop.is_set():
            time.sleep(PING_INTERVAL)
            if self._stop.is_set() or not self._ws:
                return
            # Watchdog: silence implies dead connection. Force a reconnect.
            if self._last_message_at and (
                time.time() - self._last_message_at > SILENCE_TIMEOUT_SECS
            ):
                log.warning("OB WS silent for >%.0fs; forcing reconnect",
                            SILENCE_TIMEOUT_SECS)
                try:
                    self._ws.close()
                except Exception:
                    pass
                return
            try:
                self._ws.send(json.dumps({"op": "ping", "ping": int(time.time())}))
            except Exception:
                return  # connection dead; _run will reconnect

    # ------------------------------------------------------------------ parsing

    @staticmethod
    def _extract_symbol(msg: dict) -> str | None:
        # Bitunix typically wraps payloads as {"ch":..., "symbol":..., "data":{...}}
        for key in ("symbol", "s"):
            v = msg.get(key)
            if v:
                return str(v)
        data = msg.get("data")
        if isinstance(data, dict):
            for key in ("symbol", "s"):
                v = data.get(key)
                if v:
                    return str(v)
        return None

    @staticmethod
    def _extract_book(msg: dict) -> tuple[list[tuple[float, float]] | None,
                                          list[tuple[float, float]] | None]:
        """Try several field-name conventions: b/a, bids/asks, buy/sell."""
        # The order book may be at root or nested in "data".
        sources: list[dict] = [msg]
        if isinstance(msg.get("data"), dict):
            sources.append(msg["data"])
        if isinstance(msg.get("data"), list) and msg["data"]:
            for item in msg["data"]:
                if isinstance(item, dict):
                    sources.append(item)

        for src in sources:
            bids_raw = src.get("b") or src.get("bids") or src.get("buy")
            asks_raw = src.get("a") or src.get("asks") or src.get("sell")
            bids = OrderBookFeed._parse_levels(bids_raw)
            asks = OrderBookFeed._parse_levels(asks_raw)
            if bids or asks:
                # Sort: bids desc by price, asks asc by price (defensive — some
                # exchanges already send sorted, but enforce here).
                if bids:
                    bids = sorted(bids, key=lambda x: -x[0])
                if asks:
                    asks = sorted(asks, key=lambda x: x[0])
                return bids or None, asks or None
        return None, None

    @staticmethod
    def _parse_levels(raw: Any) -> list[tuple[float, float]]:
        if not isinstance(raw, list):
            return []
        out: list[tuple[float, float]] = []
        for item in raw:
            try:
                if isinstance(item, (list, tuple)) and len(item) >= 2:
                    out.append((float(item[0]), float(item[1])))
                elif isinstance(item, dict):
                    px = item.get("px") or item.get("price") or item.get("p")
                    sz = item.get("sz") or item.get("size") or item.get("quantity") or item.get("q")
                    if px is not None and sz is not None:
                        out.append((float(px), float(sz)))
            except (ValueError, TypeError):
                continue
        return out
