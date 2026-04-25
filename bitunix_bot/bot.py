"""Main trading loop — multi-symbol, cooldown-aware, bar-deduped.

Per tick:
  1. Pull all open positions and the account once.
  2. For each configured symbol:
     - Skip if already holding (max_positions_per_symbol reached).
     - Skip if cooldown window not elapsed.
     - Skip if global max_open_positions reached.
     - Skip if no fresh bar (last bar's timestamp unchanged since last eval).
     - Otherwise: evaluate signal, build risk-sized order, place with native SL/TP.
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
    max_leverage: int = 100   # Bitunix caps differ per symbol


_DEFAULT_META = SymbolMeta(base_precision=0.001, price_precision=2, min_qty=0.001, max_leverage=100)


class BitunixBot:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.client = BitunixClient(
            cfg.creds.api_key,
            cfg.creds.secret_key,
            margin_coin=cfg.trading.margin_coin,
        )
        self.metas: dict[str, SymbolMeta] = {}
        # Per-symbol state for cooldown + bar-dedupe.
        self.last_action_at: dict[str, int] = {}    # unix sec
        self.last_bar_ts: dict[str, int] = {}       # unix sec/ms
        self.stop_flag = False
        self.state = get_state()

    # ------------------------------------------------------------------ setup

    def _resolve_symbol_meta(self) -> None:
        try:
            pairs = self.client.trading_pairs()
        except Exception as e:
            log.warning("trading_pairs() failed: %s — using defaults for all", e)
            for s in self.cfg.trading.symbols:
                self.metas[s] = _DEFAULT_META
            return

        by_name = {str(r.get("symbol", "")).upper(): r for r in pairs}
        for sym in self.cfg.trading.symbols:
            row = by_name.get(sym.upper())
            if not row:
                log.warning("Symbol %s not in trading_pairs; using defaults", sym)
                self.metas[sym] = _DEFAULT_META
                continue
            raw_base = row.get("basePrecision")
            if isinstance(raw_base, (int, float)) and raw_base >= 1:
                base_step = 10 ** (-int(raw_base))
            else:
                base_step = float(raw_base) if raw_base else 0.001
            price_prec = int(row.get("quotePrecision") or row.get("pricePrecision") or 2)
            min_qty = float(row.get("minTradeVolume") or base_step)
            max_lev = int(row.get("maxLeverage") or 100)
            self.metas[sym] = SymbolMeta(base_step, price_prec, min_qty, max_lev)
            log.info("Meta %s: step=%s priceDigits=%s minQty=%s maxLev=%s",
                     sym, base_step, price_prec, min_qty, max_lev)

    def _configure_account(self) -> None:
        # Position mode is global; set once.
        try:
            self.client.set_position_mode("ONE_WAY")
            log.info("Set position_mode=ONE_WAY")
        except BitunixError as e:
            log.info("Skip position_mode: %s", e.msg or e.code)

        # Margin mode + leverage are per symbol. Cap leverage at the symbol's
        # max so Bitunix doesn't reject the call and silently leave us with
        # whatever was already set.
        for sym in self.cfg.trading.symbols:
            meta = self.metas.get(sym, _DEFAULT_META)
            eff_lev = min(self.cfg.trading.leverage, meta.max_leverage)
            for fn, desc in [
                (lambda s=sym: self.client.set_margin_mode(s, self.cfg.trading.margin_mode),
                 f"{sym} margin_mode={self.cfg.trading.margin_mode}"),
                (lambda s=sym, lev=eff_lev: self.client.set_leverage(s, lev),
                 f"{sym} leverage={eff_lev}x (cap {meta.max_leverage}x)"),
            ]:
                try:
                    fn()
                    log.info("Set %s", desc)
                except BitunixError as e:
                    log.info("Skip %s: %s", desc, e.msg or e.code)
                except Exception as e:
                    log.warning("Error setting %s: %s", desc, e)

    # ------------------------------------------------------------------ loop

    def start(self) -> None:
        log.info("Starting Bitunix TraderBot in %s mode (symbols=%s)",
                 self.cfg.mode.upper(), self.cfg.trading.symbols)
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
        # 1. Snapshot global state once per tick.
        all_open = [p for p in self.client.pending_positions() if float(p.get("qty") or 0) != 0]

        # Time-based exit: force-close any position older than max_position_age_seconds.
        max_age = self.cfg.trading.max_position_age_seconds
        if max_age > 0 and self.cfg.is_live:
            now_ms = int(time.time() * 1000)
            still_open = []
            for p in all_open:
                ctime = int(p.get("ctime") or 0)
                age_s = (now_ms - ctime) // 1000 if ctime else 0
                if age_s >= max_age:
                    pid = str(p.get("positionId") or "")
                    sym = str(p.get("symbol") or "")
                    log.info("Force-close stale position %s (%s) age=%ss", pid, sym, age_s)
                    try:
                        self.client.flash_close_position(pid)
                        self.state.record_order(f"{sym} TIME_EXIT positionId={pid} age={age_s}s")
                    except BitunixError as e:
                        log.error("Force-close failed for %s: %s", pid, e)
                        self.state.record_error(f"{sym} time-exit failed: {e.code} {e.msg}")
                        still_open.append(p)
                else:
                    still_open.append(p)
            all_open = still_open

        n_open = len(all_open)
        per_sym_count: dict[str, int] = {}
        for p in all_open:
            s = str(p.get("symbol", "")).upper()
            per_sym_count[s] = per_sym_count.get(s, 0) + 1

        self.state.record_tick(None, len(all_open))

        if n_open >= self.cfg.trading.max_open_positions:
            log.debug("Global cap reached (%d/%d open); waiting", n_open, self.cfg.trading.max_open_positions)
            return

        # Account fetched lazily — only when we're about to size an order.
        cached_acct: dict[str, Any] | None = None

        now = int(time.time())
        for sym in self.cfg.trading.symbols:
            if self.stop_flag:
                return
            sym_u = sym.upper()

            # Per-symbol cap.
            if per_sym_count.get(sym_u, 0) >= self.cfg.trading.max_positions_per_symbol:
                continue

            # Cooldown.
            last = self.last_action_at.get(sym_u, 0)
            if now - last < self.cfg.trading.cooldown_seconds:
                continue

            # Klines.
            try:
                rows = self.client.klines(sym, self.cfg.trading.timeframe,
                                          limit=self.cfg.loop.kline_lookback)
            except BitunixError as e:
                log.warning("%s klines failed: %s", sym, e)
                continue
            if len(rows) < 30:
                continue
            rows = sorted(rows, key=lambda r: int(r.get("time") or 0))
            last_bar = int(rows[-1].get("time") or 0)
            if last_bar and self.last_bar_ts.get(sym_u) == last_bar:
                # No fresh bar since we last looked — don't re-fire on the same candle.
                continue
            self.last_bar_ts[sym_u] = last_bar

            highs = [float(r["high"]) for r in rows]
            lows = [float(r["low"]) for r in rows]
            closes = [float(r["close"]) for r in rows]

            # Signal.
            sig = evaluate(highs, lows, closes, self.cfg.strategy)
            if sig is None:
                continue
            sig_text = (f"{sym} {sig.direction.upper()} score={sig.score} @ "
                        f"{sig.price:.4f} ({', '.join(sig.reasons)})")
            log.info("Signal: %s", sig_text)
            self.state.record_signal(sig_text)

            # Risk plan.
            if cached_acct is None:
                try:
                    cached_acct = self.client.account()
                except Exception as e:
                    log.error("account fetch failed: %s", e)
                    self.state.record_error(f"account fetch failed: {e}")
                    return
            free_margin = float(cached_acct.get("available") or 0)
            if free_margin <= 0:
                if not self.cfg.is_live:
                    # Paper mode: pretend we have $1k so the dashboard shows
                    # what the bot WOULD do regardless of real balance.
                    free_margin = 1000.0
                else:
                    self.state.record_skip(f"{sym}: no available margin")
                    return  # no point checking other symbols in live mode

            meta = self.metas.get(sym, _DEFAULT_META)
            # Per-symbol effective leverage: cap config at the symbol's max.
            eff_lev = min(self.cfg.trading.leverage, meta.max_leverage)
            plan = build_order(
                sig,
                free_margin=free_margin,
                trading=self.cfg.trading,
                risk=self.cfg.risk,
                min_volume=meta.min_qty,
                volume_step=meta.base_precision,
                digits=meta.price_precision,
                effective_leverage=eff_lev,
            )
            if plan is None:
                self.state.record_skip(f"{sym}: risk manager rejected (volume below min)")
                continue

            # Execute.
            placed = self._execute(sym, plan)
            if placed:
                self.last_action_at[sym_u] = now
                per_sym_count[sym_u] = per_sym_count.get(sym_u, 0) + 1
                n_open += 1
                # Reduce optimistic free margin in cache so subsequent symbols
                # don't oversize against the same dollars.
                used_margin = (plan.volume * plan.price) / max(plan.leverage, 1)
                cached_acct["available"] = str(max(0.0, free_margin - used_margin))
                if n_open >= self.cfg.trading.max_open_positions:
                    log.info("Hit global cap %d after placing %s; halting tick",
                             n_open, sym)
                    return

    def _execute(self, symbol: str, plan: OrderPlan) -> bool:
        prefix = "LIVE" if self.cfg.is_live else "PAPER"
        order_text = (f"{prefix} {symbol} {plan.side} qty={plan.volume} "
                      f"entry~{plan.price} SL={plan.stop_loss} TP={plan.take_profit} "
                      f"lev={plan.leverage}x")
        log.info("ORDER %s [%s]", order_text, plan.notes)
        if not self.cfg.is_live:
            self.state.record_order(order_text + " (paper)")
            return True
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
            return True
        except BitunixError as e:
            log.error("Order rejected: %s (payload=%s)", e, e.payload)
            self.state.record_error(f"{symbol} order rejected: {e.code} {e.msg}")
            return False
