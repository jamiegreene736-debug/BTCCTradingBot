#!/usr/bin/env python3
"""Small offline pump-fade replay using Bitunix klines.

This is intentionally simple: it replays recent 1m candles, resamples them
into 5m/15m context, runs the same overlay pump-fade decision tree, and checks
whether a fixed quick TP/SL would have hit over the next few 1m bars.

It does not include live order-book or trade-tape signals, so treat results as
a conservative structural filter check rather than a full execution backtest.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from bitunix_bot.bot import BitunixBot  # noqa: E402
from bitunix_bot.client import BitunixClient  # noqa: E402
from bitunix_bot.config import load  # noqa: E402
from bitunix_bot.strategy import compute_overlay_scores  # noqa: E402


def _resample(rows: list[dict[str, Any]], factor: int) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for i in range(0, len(rows) - factor + 1, factor):
        chunk = rows[i:i + factor]
        out.append({
            "time": chunk[-1].get("time"),
            "open": float(chunk[0]["open"]),
            "high": max(float(r["high"]) for r in chunk),
            "low": min(float(r["low"]) for r in chunk),
            "close": float(chunk[-1]["close"]),
            "baseVol": sum(float(r.get("baseVol") or 0) for r in chunk),
            "quoteVol": sum(float(r.get("quoteVol") or 0) for r in chunk),
        })
    return out


def _horizon(rows: list[dict[str, Any]], cfg, label: str) -> dict[str, Any] | None:
    if len(rows) < 60:
        return None
    opens = [float(r["open"]) for r in rows]
    highs = [float(r["high"]) for r in rows]
    lows = [float(r["low"]) for r in rows]
    closes = [float(r["close"]) for r in rows]
    volumes = [float(r.get("baseVol") or r.get("quoteVol") or 0) for r in rows]
    overlay = compute_overlay_scores(
        opens,
        highs,
        lows,
        closes,
        cfg.strategy,
        volumes=volumes,
    )
    if overlay is None:
        return None
    atr_abs = overlay.price * overlay.atr_pct / 100.0

    def move_atr(bars: int) -> float:
        if len(closes) <= bars or atr_abs <= 0:
            return 0.0
        return (closes[-1] - closes[-(bars + 1)]) / atr_abs

    def close_counts(bars: int) -> tuple[int, int]:
        pairs = list(zip(closes[-(bars + 1):-1], closes[-bars:]))
        return (
            sum(1 for prev, cur in pairs if cur < prev),
            sum(1 for prev, cur in pairs if cur > prev),
        )

    recent_lows = lows[-15:]
    recent_highs = highs[-15:]
    range_low = min(recent_lows)
    range_high = max(recent_highs)
    span = range_high - range_low
    range_pos = (closes[-1] - range_low) / span if span > 0 else 0.5
    down5, up5 = close_counts(5)
    down10, up10 = close_counts(10)
    down12, up12 = close_counts(12)
    down15, up15 = close_counts(15)
    last_open = opens[-1]
    last_close = closes[-1]
    last_high = highs[-1]
    last_low = lows[-1]
    last_range = max(0.0, last_high - last_low)
    vol_ma_values = volumes[-21:-1] if len(volumes) >= 21 else volumes[:-1]
    vol_ma = sum(vol_ma_values) / len(vol_ma_values) if vol_ma_values else 0.0
    return {
        "label": label,
        "price": overlay.price,
        "atr": atr_abs,
        "atr_pct": overlay.atr_pct,
        "long_score": overlay.long_score,
        "short_score": overlay.short_score,
        "long_reasons": overlay.long_reasons,
        "short_reasons": overlay.short_reasons,
        "adx": overlay.adx,
        "move_3_atr": move_atr(3),
        "move_5_atr": move_atr(5),
        "move_10_atr": move_atr(10),
        "move_12_atr": move_atr(12),
        "move_15_atr": move_atr(15),
        "position_in_recent_range_15": max(0.0, min(1.0, range_pos)),
        "distance_from_recent_high_atr": ((range_high - closes[-1]) / atr_abs) if atr_abs > 0 else 999.0,
        "up_closes_5": up5,
        "down_closes_5": down5,
        "up_closes_10": up10,
        "down_closes_10": down10,
        "up_closes_12": up12,
        "down_closes_12": down12,
        "up_closes_15": up15,
        "down_closes_15": down15,
        "up_candles_5": sum(1 for r in rows[-5:] if float(r["close"]) > float(r["open"])),
        "down_candles_5": sum(1 for r in rows[-5:] if float(r["close"]) < float(r["open"])),
        "last_bar_high": last_high,
        "last_bar_low": last_low,
        "last_bar_body_atr": ((last_close - last_open) / atr_abs) if atr_abs > 0 else 0.0,
        "last_bar_upper_wick_atr": ((last_high - max(last_open, last_close)) / atr_abs) if atr_abs > 0 else 0.0,
        "last_bar_close_position": ((last_close - last_low) / last_range) if last_range > 0 else 0.5,
        "volume_spike_ratio": volumes[-1] / vol_ma if vol_ma > 0 else 0.0,
    }


def _simulate_short(rows: list[dict[str, Any]], start: int, entry: float, cfg, hold_bars: int) -> float:
    lev = max(1, int(cfg.trading.pump_fade_auto_leverage))
    tp_pct = (float(cfg.risk.margin_profit_target_pct or 0.0) / lev) / 100.0
    if tp_pct <= 0:
        tp_pct = (float(cfg.risk.stop_loss_pct) * float(cfg.risk.take_profit_r)) / 100.0
    sl_pct = float(cfg.risk.stop_loss_pct) / 100.0
    tp = entry * (1 - tp_pct)
    sl = entry * (1 + sl_pct)
    end = min(len(rows), start + hold_bars + 1)
    exit_price = float(rows[end - 1]["close"])
    for row in rows[start + 1:end]:
        high = float(row["high"])
        low = float(row["low"])
        if high >= sl:
            exit_price = sl
            break
        if low <= tp:
            exit_price = tp
            break
    return (entry - exit_price) / entry * lev * 100.0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--symbols", default="", help="comma-separated override")
    parser.add_argument("--limit", type=int, default=500)
    parser.add_argument("--hold-bars", type=int, default=5)
    args = parser.parse_args()

    cfg = load(args.config, ".env")
    client = BitunixClient(
        os.environ["BITUNIX_API_KEY"],
        os.environ["BITUNIX_SECRET_KEY"],
        margin_coin=cfg.trading.margin_coin,
    )
    symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()] or cfg.trading.symbols
    lookback = max(120, int(cfg.loop.kline_lookback))
    for sym in symbols:
        rows = sorted(client.klines(sym, "1m", limit=args.limit), key=lambda r: int(r.get("time") or 0))
        wins = losses = signals = 0
        pnl = 0.0
        for i in range(lookback, len(rows) - args.hold_bars):
            w1 = rows[i - lookback:i]
            h15 = _horizon(w1, cfg, "1m entry")
            h30 = _horizon(_resample(w1, 5), cfg, "5m pump")
            h1 = _horizon(_resample(w1, 15), cfg, "trend context")
            if not h15 or not h30:
                continue
            decision = BitunixBot._build_pump_fade_only_decision({
                "h_15m": h15,
                "h_30m": h30,
                "h_1h": h1 or {},
            })
            if decision.get("action") != "short":
                continue
            conf = int(decision.get("confidence_score") or 0)
            if conf < int(cfg.trading.pump_fade_auto_min_confidence):
                continue
            signals += 1
            entry = float(rows[i]["close"])
            trade_pnl = _simulate_short(rows, i, entry, cfg, args.hold_bars)
            pnl += trade_pnl
            if trade_pnl > 0:
                wins += 1
            else:
                losses += 1
        rate = wins / signals * 100.0 if signals else 0.0
        print(f"{sym}: signals={signals} win_rate={rate:.1f}% pnl_margin_pct={pnl:.2f} wins={wins} losses={losses}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
