"""Main trading loop.

One-position-at-a-time model: look for a confluence signal → place order with
native tpPrice/slPrice attached (Bitunix enforces both server-side) → wait
for the position to close → repeat. No fill-tracking state machine needed.
"""
from __future__ import annotations

import contextlib
import logging
import logging.handlers
import signal
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .client import BitunixClient, BitunixError
from .config import Config
from .risk import OrderPlan, build_order
from .state import get as get_state
from .strategy import evaluate

log = logging.getLogger(__name__)


def configure_logging(cfg: Config) -> None:
    level = getattr(logging, cfg.logging.level.upper(), logging.INFO)
    Path(cfg.logging.file).parent.mkdir(parents=True, exist_ok=True)
    root = logging.getLogger()
    root.setLevel(level)
    fmt = logging.Formatter("%(asctime)s %(levelname)-7s %(name)s | %(message)s")
    fh = logging.handlers.RotatingFileHandler(cfg.logging.file, maxBytes=2_000_000, backupCount=3)
    fh.setFormatter(fmt)
    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    for h in list(root.handlers):
        root.removeHandler(h)
    root.addHandler(fh)
    root.addHandler(ch)


@dataclass
class SymbolMeta:
    base_precision: float     # qty step (e.g. 0.001)
    price_precision: int      # digits for price (e.g. 1 for BTCUSDT)
    min_qty: float


class BitunixBot:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.client = BitunixClient(
            cfg.creds.api_key,
            cfg.creds.secret_key,
            margin_coin=cfg.trading.margin_coin,
        )
        self.meta = SymbolMeta(base_precision=0.001, price_precision=1, min_qty=0.001)
        self.stop_flag = False
        self.state = get_state()

    # ------------------------------------------------------------------ setup

    def _resolve_symbol_meta(self) -> None:
        try:
            pairs = self.client.trading_pairs()
        except Exception as e:
            log.warning("trading_pairs() failed: %s — using defaults", e)
            return
        s = self.cfg.trading.symbol
        for row in pairs:
            if str(row.get("symbol", "")).upper() != s.upper():
                continue
            # Bitunix reports `basePrecision` and `quotePrecision` as decimal
            # COUNTS (e.g. 4 means 4 decimals = step 0.0001), not step sizes.
            # Treat small ints as decimal counts; pass through values already < 1.
            raw_base = row.get("basePrecision")
            if isinstance(raw_base, (int, float)) and raw_base >= 1:
                base_step = 10 ** (-int(raw_base))
            else:
                base_step = float(raw_base) if raw_base else 0.001
            price_prec = int(row.get("quotePrecision") or row.get("pricePrecision") or 1)
            min_qty = float(row.get("minTradeVolume") or base_step)
            self.meta = SymbolMeta(base_step, price_prec, min_qty)
            log.info("Symbol meta: step=%s priceDigits=%s minQty=%s (raw=%s)",
                     base_step, price_prec, min_qty, row)
            return
        log.warning("Symbol %s not found in trading_pairs; using defaults", s)

    def _configure_account(self) -> None:
        s = self.cfg.trading.symbol
        # Best-effort: these may fail if already configured or scope-restricted.
        attempts = [
            (lambda: self.client.set_position_mode("ONE_WAY"), "position_mode=ONE_WAY"),
            (lambda: self.client.set_margin_mode(s, self.cfg.trading.margin_mode),
             f"margin_mode={self.cfg.trading.margin_mode}"),
            (lambda: self.client.set_leverage(s, self.cfg.trading.leverage),
             f"leverage={self.cfg.trading.leverage}x"),
        ]
        for fn, desc in attempts:
            try:
                fn()
                log.info("Set %s", desc)
            except BitunixError as e:
                log.info("Skip %s: %s", desc, e.msg or e.code)
            except Exception as e:
                log.warning("Error setting %s: %s", desc, e)

    # ------------------------------------------------------------------ loop

    def start(self) -> None:
        log.info("Starting Bitunix TraderBot in %s mode", self.cfg.mode.upper())
        self._resolve_symbol_meta()
        if self.cfg.is_live:
            self._configure_account()

        signal.signal(signal.SIGINT, self._on_sig)
        with contextlib.suppress(AttributeError):
            signal.signal(signal.SIGTERM, self._on_sig)

        self.run_forever()

    def run_forever(self) -> None:
        """Loop without installing signal handlers (safe in a worker thread)."""
        while not self.stop_flag:
            try:
                self._tick()
            except BitunixError as e:
                log.error("Bitunix API error: %s", e)
                self.state.record_error(f"{e.code}: {e.msg}")
            except Exception as e:
                log.exception("Tick failed")
                self.state.record_error(str(e))
            for _ in range(self.cfg.loop.tick_seconds):
                if self.stop_flag:
                    break
                time.sleep(1)
        log.info("Bot stopped")

    def _on_sig(self, *_: Any) -> None:
        self.stop_flag = True

    # ------------------------------------------------------------------ tick

    def _tick(self) -> None:
        sym = self.cfg.trading.symbol

        # 1. Skip if already holding a position on this symbol.
        open_positions = [
            p for p in self.client.pending_positions(symbol=sym)
            if float(p.get("qty") or 0) != 0
        ]
        if open_positions:
            log.debug("Holding %d open position(s); waiting for TP/SL", len(open_positions))
            self.state.record_tick(None, 0)
            return

        # 2. Candles.
        rows = self.client.klines(
            sym, self.cfg.trading.timeframe, limit=self.cfg.loop.kline_lookback
        )
        if len(rows) < 30:
            log.debug("Not enough klines yet (%d)", len(rows))
            self.state.record_tick(None, len(rows))
            return
        rows = sorted(rows, key=lambda r: int(r.get("time") or 0))
        highs = [float(r["high"]) for r in rows]
        lows = [float(r["low"]) for r in rows]
        closes = [float(r["close"]) for r in rows]
        self.state.record_tick(closes[-1], len(rows))

        # 3. Signal.
        sig = evaluate(highs, lows, closes, self.cfg.strategy)
        if sig is None:
            log.debug("No signal (no confluence)")
            return
        log.info("Signal: %s score=%d reasons=%s @ %s",
                 sig.direction, sig.score, sig.reasons, sig.price)
        sig_text = f"{sig.direction.upper()} score={sig.score} @ {sig.price:.2f} ({', '.join(sig.reasons)})"
        self.state.record_signal(sig_text)

        # 4. Risk plan.
        acct = self.client.account()
        free_margin = float(acct.get("available") or 0)
        if free_margin <= 0:
            log.warning("No available margin — skipping")
            self.state.record_skip(f"no available margin (have {acct.get('available')!r})")
            return

        plan = build_order(
            sig,
            free_margin=free_margin,
            trading=self.cfg.trading,
            risk=self.cfg.risk,
            min_volume=self.meta.min_qty,
            volume_step=self.meta.base_precision,
            digits=self.meta.price_precision,
        )
        if plan is None:
            log.info("Risk manager rejected the trade")
            self.state.record_skip("risk manager rejected (volume below min)")
            return

        # 5. Execute.
        self._execute(sym, plan)

    def _execute(self, symbol: str, plan: OrderPlan) -> None:
        prefix = "LIVE" if self.cfg.is_live else "PAPER"
        order_text = (f"{prefix} {plan.side} qty={plan.volume} "
                      f"entry~{plan.price} SL={plan.stop_loss} TP={plan.take_profit} "
                      f"lev={plan.leverage}x")
        log.info("ORDER %s [%s]", order_text, plan.notes)
        if not self.cfg.is_live:
            self.state.record_order(order_text + " (paper)")
            return
        try:
            resp = self.client.place_order(
                symbol=symbol,
                side=plan.side,
                qty=str(plan.volume),
                order_type="MARKET",
                trade_side="OPEN",
                tp_price=str(plan.take_profit),
                sl_price=str(plan.stop_loss),
            )
            log.info("Placed orderId=%s clientId=%s", resp.get("orderId"), resp.get("clientId"))
            self.state.record_order(f"{order_text} → orderId={resp.get('orderId')}")
        except BitunixError as e:
            log.error("Order rejected: %s (payload=%s)", e, e.payload)
            self.state.record_error(f"order rejected: {e.code} {e.msg}")
