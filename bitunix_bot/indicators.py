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


def adx(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 14) -> np.ndarray:
    """Average Directional Index — measures trend strength regardless of direction.

    >25 = trending; <20 = chop. Strongest filter against EMA-cross whipsaws.
    Standard Wilder's ADX implementation.
    """
    n = len(close)
    out = np.full(n, np.nan)
    if n < period * 2 + 1:
        return out
    up_move = np.diff(high)
    dn_move = -np.diff(low)
    plus_dm = np.where((up_move > dn_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((dn_move > up_move) & (dn_move > 0), dn_move, 0.0)
    prev_close = close[:-1]
    tr = np.maximum.reduce([high[1:] - low[1:], np.abs(high[1:] - prev_close), np.abs(low[1:] - prev_close)])
    # Wilder smoothing on tr / +DM / -DM.
    atr_ = np.zeros(len(tr))
    pdm = np.zeros(len(tr))
    mdm = np.zeros(len(tr))
    atr_[period - 1] = tr[:period].sum()
    pdm[period - 1] = plus_dm[:period].sum()
    mdm[period - 1] = minus_dm[:period].sum()
    for i in range(period, len(tr)):
        atr_[i] = atr_[i - 1] - (atr_[i - 1] / period) + tr[i]
        pdm[i] = pdm[i - 1] - (pdm[i - 1] / period) + plus_dm[i]
        mdm[i] = mdm[i - 1] - (mdm[i - 1] / period) + minus_dm[i]
    with np.errstate(invalid="ignore", divide="ignore"):
        plus_di = np.where(atr_ > 0, 100.0 * pdm / atr_, 0.0)
        minus_di = np.where(atr_ > 0, 100.0 * mdm / atr_, 0.0)
        sum_di = plus_di + minus_di
        dx = np.where(sum_di > 0, 100.0 * np.abs(plus_di - minus_di) / sum_di, 0.0)
    # Final Wilder smoothing on DX -> ADX.
    adx_arr = np.full(len(tr), np.nan)
    adx_arr[period * 2 - 2] = dx[period - 1 : period * 2 - 1].mean()
    for i in range(period * 2 - 1, len(tr)):
        adx_arr[i] = (adx_arr[i - 1] * (period - 1) + dx[i]) / period
    # Align back to original `close` indexing (tr starts at index 1).
    out[1:] = adx_arr
    return out


def supertrend(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    period: int = 10,
    multiplier: float = 3.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Returns (line, direction). direction = 1 (uptrend) or -1 (downtrend).

    Canonical supertrend per the standard reference implementation:
    - upper "ratchets" downward while we're in a downtrend (close < final_upper)
    - lower "ratchets" upward while we're in an uptrend (close > final_lower)
    - direction flips when close pierces the opposite band.
    """
    n = len(close)
    line = np.full(n, np.nan)
    direction = np.full(n, np.nan)
    if n < period + 2:
        return line, direction
    a = atr(high, low, close, period)
    hl2 = (high + low) / 2.0
    raw_upper = hl2 + multiplier * a
    raw_lower = hl2 - multiplier * a

    final_upper = raw_upper.copy()
    final_lower = raw_lower.copy()
    # Find first index with a valid ATR; that's where we start tracking.
    start = period + 1
    while start < n and np.isnan(a[start]):
        start += 1
    if start >= n:
        return line, direction

    # Initial direction: pick based on close vs hl2 at start.
    direction[start] = 1 if close[start] >= hl2[start] else -1
    line[start] = final_lower[start] if direction[start] == 1 else final_upper[start]

    for i in range(start + 1, n):
        # Upper band ratchets down while in downtrend.
        if raw_upper[i] < final_upper[i - 1] or close[i - 1] > final_upper[i - 1]:
            final_upper[i] = raw_upper[i]
        else:
            final_upper[i] = final_upper[i - 1]
        # Lower band ratchets up while in uptrend.
        if raw_lower[i] > final_lower[i - 1] or close[i - 1] < final_lower[i - 1]:
            final_lower[i] = raw_lower[i]
        else:
            final_lower[i] = final_lower[i - 1]

        prev_dir = direction[i - 1]
        if prev_dir == 1 and close[i] < final_lower[i]:
            direction[i] = -1
        elif prev_dir == -1 and close[i] > final_upper[i]:
            direction[i] = 1
        else:
            direction[i] = prev_dir
        line[i] = final_lower[i] if direction[i] == 1 else final_upper[i]
    return line, direction
