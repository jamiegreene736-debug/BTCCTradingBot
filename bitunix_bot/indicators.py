"""Pure-numpy technical indicators (no TA-Lib needed).

Each function takes numpy arrays of closes/highs/lows and returns the same-length
array (prefix may be NaN). Keeping this self-contained avoids native deps so the
bot runs on any Python 3.10+ without extra toolchain.
"""
from __future__ import annotations

import numpy as np


def ema(x: np.ndarray, period: int) -> np.ndarray:
    if period <= 1:
        return x.copy()
    alpha = 2.0 / (period + 1)
    out = np.empty_like(x, dtype=float)
    out[:] = np.nan
    if len(x) < period:
        return out
    # Seed with SMA over first `period` values, then recurse.
    out[period - 1] = x[:period].mean()
    for i in range(period, len(x)):
        out[i] = alpha * x[i] + (1 - alpha) * out[i - 1]
    return out


def rsi(x: np.ndarray, period: int = 14) -> np.ndarray:
    out = np.full_like(x, np.nan, dtype=float)
    if len(x) <= period:
        return out
    diffs = np.diff(x)
    gains = np.where(diffs > 0, diffs, 0.0)
    losses = np.where(diffs < 0, -diffs, 0.0)
    # Wilder smoothing.
    avg_gain = gains[:period].mean()
    avg_loss = losses[:period].mean()
    out[period] = 100 - (100 / (1 + (avg_gain / avg_loss if avg_loss else np.inf)))
    for i in range(period + 1, len(x)):
        avg_gain = (avg_gain * (period - 1) + gains[i - 1]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i - 1]) / period
        rs = avg_gain / avg_loss if avg_loss else np.inf
        out[i] = 100 - (100 / (1 + rs))
    return out


def macd(x: np.ndarray, fast: int = 12, slow: int = 26, signal: int = 9) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    line = ema(x, fast) - ema(x, slow)
    sig = ema(np.nan_to_num(line), signal)
    hist = line - sig
    return line, sig, hist


def bollinger(x: np.ndarray, period: int = 20, std: float = 2.0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = len(x)
    mid = np.full(n, np.nan)
    up = np.full(n, np.nan)
    lo = np.full(n, np.nan)
    for i in range(period - 1, n):
        window = x[i - period + 1 : i + 1]
        m = window.mean()
        s = window.std(ddof=0)
        mid[i] = m
        up[i] = m + std * s
        lo[i] = m - std * s
    return up, mid, lo


def atr(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 14) -> np.ndarray:
    n = len(close)
    out = np.full(n, np.nan)
    if n < period + 1:
        return out
    prev_close = np.concatenate([[close[0]], close[:-1]])
    tr = np.maximum.reduce([high - low, np.abs(high - prev_close), np.abs(low - prev_close)])
    # Wilder smoothing.
    out[period] = tr[1 : period + 1].mean()
    for i in range(period + 1, n):
        out[i] = (out[i - 1] * (period - 1) + tr[i]) / period
    return out
