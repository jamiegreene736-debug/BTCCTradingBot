"""Trade-tape WebSocket feed — executed trades from Bitunix's public stream.

This is the alpha core that the rest of the bot was missing. Where the
order book shows what's RESTING, the trade tape shows what just EXECUTED
and which side was the aggressor (took liquidity). That's where 1–3
minute predictive signal lives in crypto scalping:

  - Real CVD: cumulative volume delta from buy/sell aggressor flags.
              positive = net buy aggression; negative = net sell aggression.
              Real-volume flow, not the candle-close-position proxy used
              by `indicators.cvd`.
  - Aggression ratio: (buy_size - sell_size) / total over a short window.
              Reads as "how lopsided is the immediate order flow right now."
  - Print intensity: trade count per second — proxy for activity surge.
  - Average aggressor size: large average size = institutional flow,
              small + frequent = retail churn.

The feed runs in a daemon thread, auto-reconnects with backoff, and
maintains a bounded rolling history per symbol. All accessors are
thread-safe and return None when data is missing or stale.

Bitunix's trade payload schema isn't fully documented publicly, so the
parser is defensive: it tries multiple conventional field names for
price, size, side (or isBuyerMaker), and timestamp. The first shape
that yields a valid trade wins.
"""
from __future__ import annotations

import json
import logging
import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import Any, Deque

import websocket  # websocket-client

log = logging.getLogger(__name__)

WS_URL = "wss://fapi.bitunix.com/public/"
CHANNEL = "trade"                 # singular, per Bitunix convention
PING_INTERVAL = 15.0
RECONNECT_BACKOFF_SECS = 5.0
SILENCE_TIMEOUT_SECS = 60.0
# Bound history to keep memory predictable. 5 minutes is plenty for the
# short-window flow analysis the strategy needs (CVD/aggression are
# computed over 10–60 seconds).
HISTORY_SECS = 300.0
MAX_TRADES_PER_SYMBOL = 20_000    # hard cap; auto-evicts oldest


@dataclass
class Trade:
    ts: float        # unix seconds
    price: float
    qty: float       # base-coin size
    is_buy: bool     # True if the aggressor was a buyer (lifted the ask)


class TradeFeed:
    """Background WebSocket subscriber for executed trades.

    Maintains a thread-safe rolling deque of recent trades per symbol and
    exposes flow-derived accessors (CVD, aggression ratio, print rate,
    avg aggressor size) for the strategy layer to consume as alpha votes.
    """

    def __init__(
        self,
        symbols: list[str],
        history_secs: float = HISTORY_SECS,
        max_trades_per_symbol: int = MAX_TRADES_PER_SYMBOL,
    ):
        self.symbols = [s.upper() for s in symbols]
        self.history_secs = history_secs
        self._trades: dict[str, Deque[Trade]] = {
            s: deque(maxlen=max_trades_per_symbol) for s in self.symbols
        }
        self._lock = threading.RLock()
        self._stop = threading.Event()
        self._ws: websocket.WebSocketApp | None = None
        self._thread: threading.Thread | None = None
        self._ping_thread: threading.Thread | None = None
        self._last_message_at: float = 0.0
        self._last_trade_at: dict[str, float] = {}

    # ------------------------------------------------------------------ lifecycle

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="tape-ws", daemon=True)
        self._thread.start()
        log.info("TradeFeed started for %d symbol(s)", len(self.symbols))

    def stop(self) -> None:
        self._stop.set()
        if self._ws:
            try:
                self._ws.close()
            except Exception:
                pass

    def is_connected(self) -> bool:
        return self._ws is not None and getattr(self._ws, "sock", None) is not None

    # ------------------------------------------------------------------ accessors

    def _recent(self, symbol: str, window_secs: float) -> list[Trade]:
        """Return trades within the last `window_secs`, oldest-first.
        Caller MUST hold the lock or accept a brief snapshot view."""
        sym = symbol.upper()
        cutoff = time.time() - window_secs
        with self._lock:
            buf = self._trades.get(sym)
            if not buf:
                return []
            # Walk newest-to-oldest until past cutoff. Deques iterate left-to-
            # right (oldest-to-newest) but we use reverse for early termination.
            recent: list[Trade] = []
            for t in reversed(buf):
                if t.ts < cutoff:
                    break
                recent.append(t)
            recent.reverse()
            return recent

    def get_cvd(
        self, symbol: str, window_secs: float = 60.0, min_count: int = 5
    ) -> float | None:
        """Cumulative volume delta over the last `window_secs`.

        Positive  = net buy aggression (more volume hitting the ask)
        Negative  = net sell aggression
        None      = no data / stale feed / insufficient sample.

        Returned in BASE-COIN units (BTC, ETH, etc.) — consumers can
        convert to USD by multiplying by current price if needed.

        `min_count` guards against firing on tiny samples (1-2 trades
        right after WS connect would otherwise produce noisy signals).
        """
        recent = self._recent(symbol, window_secs)
        if len(recent) < min_count:
            return None
        return sum(t.qty if t.is_buy else -t.qty for t in recent)

    def get_aggression_ratio(
        self, symbol: str, window_secs: float = 10.0, min_count: int = 5
    ) -> float | None:
        """Buy/sell aggression ratio in [-1, +1] over short window.

        +1.0 = 100% buyer-initiated flow
        -1.0 = 100% seller-initiated flow
         0.0 = balanced

        Use a SHORT window (10s default) — this captures momentum bursts
        that happen between bar closes. `min_count` ensures we don't read
        a 1-trade sample as a perfect ±1.0 signal.
        """
        recent = self._recent(symbol, window_secs)
        if len(recent) < min_count:
            return None
        buy_sz = sum(t.qty for t in recent if t.is_buy)
        sell_sz = sum(t.qty for t in recent if not t.is_buy)
        total = buy_sz + sell_sz
        if total <= 0:
            return None
        return (buy_sz - sell_sz) / total

    def get_activity_multiplier(
        self,
        symbol: str,
        recent_window: float = 10.0,
        baseline_window: float = 300.0,
        clamp_min: float = 0.85,
        clamp_max: float = 1.10,
    ) -> float | None:
        """Activity-surge factor for use as a score multiplier.

        Compares current print rate (recent_window) to baseline (5-min
        trailing). Above-baseline = real conviction, boost; below-baseline
        = market asleep, dampen.

        Asymmetric clamp [0.85, 1.10] by default — we dampen more
        aggressively than we boost. Boosting too much in low-quality
        regimes is a way to overtrade; dampening protects against
        firing on dead-market noise.

        Returns None if either window has insufficient data.
        """
        recent_count = len(self._recent(symbol, recent_window))
        baseline_count = len(self._recent(symbol, baseline_window))
        if recent_count < 3 or baseline_count < 30:
            return None
        recent_rate = recent_count / recent_window
        baseline_rate = baseline_count / baseline_window
        if baseline_rate <= 0:
            return None
        raw = recent_rate / baseline_rate
        return max(clamp_min, min(clamp_max, raw))

    def get_print_rate(self, symbol: str, window_secs: float = 10.0) -> float | None:
        """Trades per second over the window. Spikes = activity surge.
        Useful as a regime detector — high print rate often coincides
        with breakouts or liquidation cascades."""
        recent = self._recent(symbol, window_secs)
        if not recent:
            return None
        return len(recent) / window_secs

    def get_avg_aggressor_size(
        self, symbol: str, window_secs: float = 60.0
    ) -> float | None:
        """Average size of aggressive prints. Large = institutional;
        small = retail churn. Helps disambiguate quality of recent flow."""
        recent = self._recent(symbol, window_secs)
        if not recent:
            return None
        sizes = [t.qty for t in recent]
        return sum(sizes) / len(sizes)

    def get_price_change_pct(
        self, symbol: str, window_secs: float = 10.0, min_count: int = 5
    ) -> float | None:
        """Percentage price change between the OLDEST and NEWEST trade in
        the window. Used by the absorption detector — when aggression is
        extreme but price barely moved, big players are defending the
        level and a reversal is likely.

        Returns None if fewer than `min_count` trades or invalid prices.
        """
        recent = self._recent(symbol, window_secs)
        if len(recent) < min_count:
            return None
        first_price = recent[0].price
        last_price = recent[-1].price
        if first_price <= 0:
            return None
        return (last_price - first_price) / first_price * 100.0

    def get_large_print_count(
        self,
        symbol: str,
        size_threshold: float,
        window_secs: float = 60.0,
    ) -> int:
        """Count of trades exceeding `size_threshold` in the window. Used
        for iceberg / institutional flow detection — a sequence of
        unusually large prints in one direction is meaningful."""
        recent = self._recent(symbol, window_secs)
        return sum(1 for t in recent if t.qty >= size_threshold)

    def last_trade_age(self, symbol: str) -> float | None:
        """Seconds since the most recent trade for this symbol — or None
        if none seen. Used to gate accessors on data freshness."""
        sym = symbol.upper()
        with self._lock:
            buf = self._trades.get(sym)
            if not buf:
                return None
            return time.time() - buf[-1].ts

    # ------------------------------------------------------------------ ingestion

    def _ingest(self, trade: Trade, symbol: str) -> None:
        sym = symbol.upper()
        with self._lock:
            buf = self._trades.setdefault(sym, deque(maxlen=MAX_TRADES_PER_SYMBOL))
            buf.append(trade)
            self._last_trade_at[sym] = trade.ts
            # Periodically prune by time (deque already caps by count).
            cutoff = time.time() - self.history_secs
            while buf and buf[0].ts < cutoff:
                buf.popleft()

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
                log.warning("Trade WS run_forever exception: %s", e)
            if self._stop.is_set():
                break
            log.info("Trade WS reconnecting in %ss", RECONNECT_BACKOFF_SECS)
            time.sleep(RECONNECT_BACKOFF_SECS)

    # ------------------------------------------------------------------ callbacks

    def _on_open(self, ws: websocket.WebSocketApp) -> None:
        log.info("Trade WS connected; subscribing to %d symbols", len(self.symbols))
        sub_msg = {
            "op": "subscribe",
            "args": [{"symbol": s, "ch": CHANNEL} for s in self.symbols],
        }
        try:
            ws.send(json.dumps(sub_msg))
        except Exception as e:
            log.warning("Trade WS subscribe send failed: %s", e)
        if not self._ping_thread or not self._ping_thread.is_alive():
            self._ping_thread = threading.Thread(target=self._pinger, name="tape-ping", daemon=True)
            self._ping_thread.start()

    def _on_message(self, ws: websocket.WebSocketApp, message: str) -> None:
        self._last_message_at = time.time()
        try:
            msg = json.loads(message)
        except json.JSONDecodeError:
            return
        if isinstance(msg, dict):
            op = msg.get("op")
            if op in ("pong", "subscribe", "ping"):
                return

        symbol, trades = self._extract_trades(msg)
        if not symbol or not trades:
            return
        for t in trades:
            self._ingest(t, symbol)

    def _on_error(self, ws: websocket.WebSocketApp, error: Any) -> None:
        log.warning("Trade WS error: %s", error)

    def _on_close(self, ws: websocket.WebSocketApp, *args: Any) -> None:
        log.info("Trade WS closed")

    def _pinger(self) -> None:
        """Send heartbeats and watchdog the connection — silence > 60s
        forces reconnect so a half-dead socket doesn't starve us of data."""
        while not self._stop.is_set():
            time.sleep(PING_INTERVAL)
            if self._stop.is_set() or not self._ws:
                return
            if self._last_message_at and (
                time.time() - self._last_message_at > SILENCE_TIMEOUT_SECS
            ):
                log.warning("Trade WS silent for >%.0fs; forcing reconnect",
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
        """Bitunix wraps payloads as {"ch":..., "symbol":..., "data":...}."""
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
        if isinstance(data, list) and data and isinstance(data[0], dict):
            for key in ("symbol", "s"):
                v = data[0].get(key)
                if v:
                    return str(v)
        return None

    @classmethod
    def _extract_trades(cls, msg: dict) -> tuple[str | None, list[Trade]]:
        """Defensive parser. Bitunix's trade payload exact shape isn't
        publicly documented; we try several conventional shapes:
          - {"data": {"p": ..., "v": ..., "ts": ..., "side": "BUY"}}
          - {"data": [{"p": ..., "v": ..., "t": ..., "s": "B"}, ...]}
          - {"data": {"price": "...", "size": "...", "side": "buy"}}
          - Binance-style with isBuyerMaker / m flag
        Returns (symbol, list_of_trades). Empty list = nothing parseable."""
        symbol = cls._extract_symbol(msg)
        if not symbol:
            return None, []

        data = msg.get("data")
        items: list[dict] = []
        if isinstance(data, dict):
            items = [data]
        elif isinstance(data, list):
            items = [d for d in data if isinstance(d, dict)]
        else:
            return symbol, []

        out: list[Trade] = []
        for item in items:
            t = cls._parse_trade(item)
            if t is not None:
                out.append(t)
        return symbol, out

    @staticmethod
    def _parse_trade(item: dict) -> Trade | None:
        """Parse one trade row trying multiple field-name conventions."""
        # Price.
        px_raw = (item.get("p") or item.get("price") or item.get("px")
                  or item.get("dealPrice") or item.get("lastPrice"))
        # Size / qty.
        qty_raw = (item.get("v") or item.get("q") or item.get("qty")
                   or item.get("quantity") or item.get("size") or item.get("sz")
                   or item.get("vol") or item.get("volume") or item.get("dealVolume"))
        # Timestamp (ms or seconds).
        ts_raw = (item.get("t") or item.get("ts") or item.get("time")
                  or item.get("timestamp") or item.get("dealTime"))
        # Aggressor side.
        side_raw = item.get("side") or item.get("s") or item.get("dealSide")
        is_buyer_maker = item.get("m")
        if is_buyer_maker is None:
            is_buyer_maker = item.get("isBuyerMaker")
        if is_buyer_maker is None:
            is_buyer_maker = item.get("buyerMaker")

        if px_raw is None or qty_raw is None:
            return None

        try:
            price = float(px_raw)
            qty = float(qty_raw)
        except (TypeError, ValueError):
            return None
        if price <= 0 or qty <= 0:
            return None

        # Timestamp normalization. Bitunix usually emits ms; accept seconds too.
        ts = time.time()
        if ts_raw is not None:
            try:
                tnum = float(ts_raw)
                # Heuristic: > 1e12 → ms; else seconds.
                ts = tnum / 1000.0 if tnum > 1e12 else tnum
            except (TypeError, ValueError):
                pass

        # Determine aggressor side.
        is_buy: bool | None = None
        if side_raw is not None:
            s = str(side_raw).upper()
            if s in ("BUY", "B", "BID", "LONG"):
                is_buy = True
            elif s in ("SELL", "S", "ASK", "SHORT"):
                is_buy = False
        if is_buy is None and is_buyer_maker is not None:
            # If buyer is the MAKER (resting bid), the aggressor was a SELLER.
            is_buy = not bool(is_buyer_maker)
        if is_buy is None:
            return None  # can't determine direction → useless for CVD

        return Trade(ts=ts, price=price, qty=qty, is_buy=is_buy)
