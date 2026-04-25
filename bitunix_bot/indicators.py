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


def volume_ma(volumes: np.ndarray, period: int = 20) -> np.ndarray:
    """Rolling SMA of volume — used as a baseline to spot volume spikes."""
    n = len(volumes)
    out = np.full(n, np.nan)
    if n < period:
        return out
    for i in range(period - 1, n):
        out[i] = volumes[i - period + 1 : i + 1].mean()
    return out


def vwap(high: np.ndarray, low: np.ndarray, close: np.ndarray, volumes: np.ndarray) -> np.ndarray:
    """Rolling Volume-Weighted Average Price.

    Uses (high+low+close)/3 as the typical price. This is the classic VWAP
    calculation; price above VWAP = bullish bias, price below = bearish bias.
    """
    typical = (high + low + close) / 3.0
    cum_pv = np.cumsum(typical * volumes)
    cum_v = np.cumsum(volumes)
    with np.errstate(invalid="ignore", divide="ignore"):
        out = np.where(cum_v > 0, cum_pv / cum_v, np.nan)
    return out


def stoch_rsi(
    close: np.ndarray,
    rsi_period: int = 14,
    stoch_period: int = 14,
    k_smooth: int = 3,
    d_smooth: int = 3,
) -> tuple[np.ndarray, np.ndarray]:
    """Stochastic RSI: stochastic oscillator applied to RSI values.

    Returns (%K, %D), both normalized to [0, 100]. More sensitive than raw
    RSI — common scalper momentum indicator. Crossovers in the oversold (<20)
    or overbought (>80) zones produce reversal signals.
    """
    rsi_v = rsi(close, rsi_period)
    n = len(rsi_v)
    raw = np.full(n, np.nan)
    for i in range(stoch_period - 1, n):
        window = rsi_v[max(0, i - stoch_period + 1) : i + 1]
        valid = window[~np.isnan(window)]
        if len(valid) < 2:
            continue
        lo, hi = float(valid.min()), float(valid.max())
        if hi == lo:
            raw[i] = 50.0
        else:
            raw[i] = 100.0 * (rsi_v[i] - lo) / (hi - lo)
    # %K = SMA(raw, k_smooth)
    k = np.full(n, np.nan)
    for i in range(k_smooth - 1, n):
        window = raw[max(0, i - k_smooth + 1) : i + 1]
        valid = window[~np.isnan(window)]
        if len(valid):
            k[i] = float(valid.mean())
    # %D = SMA(%K, d_smooth)
    d = np.full(n, np.nan)
    for i in range(d_smooth - 1, n):
        window = k[max(0, i - d_smooth + 1) : i + 1]
        valid = window[~np.isnan(window)]
        if len(valid):
            d[i] = float(valid.mean())
    return k, d


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


def obv(close: np.ndarray, volumes: np.ndarray) -> np.ndarray:
    """On-Balance Volume — cumulative volume signed by direction.

    Up bar adds volume, down bar subtracts. Used as a momentum-with-volume
    confirmation indicator and as an oscillator for divergence detection.
    """
    n = len(close)
    out = np.zeros(n)
    for i in range(1, n):
        if close[i] > close[i - 1]:
            out[i] = out[i - 1] + volumes[i]
        elif close[i] < close[i - 1]:
            out[i] = out[i - 1] - volumes[i]
        else:
            out[i] = out[i - 1]
    return out


def mfi(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    volumes: np.ndarray,
    period: int = 14,
) -> np.ndarray:
    """Money Flow Index — volume-weighted RSI.

    Typical price = (H+L+C)/3. Money flow = typical * volume. Positive flow
    when typical rises, negative when it falls. MFI = 100 - 100/(1 + ratio).
    Range 0-100, like RSI.
    """
    n = len(close)
    out = np.full(n, np.nan)
    if n < period + 1:
        return out
    typical = (high + low + close) / 3.0
    money_flow = typical * volumes
    pos = np.zeros(n)
    neg = np.zeros(n)
    for i in range(1, n):
        if typical[i] > typical[i - 1]:
            pos[i] = money_flow[i]
        elif typical[i] < typical[i - 1]:
            neg[i] = money_flow[i]
    for i in range(period, n):
        p = pos[i - period + 1 : i + 1].sum()
        ng = neg[i - period + 1 : i + 1].sum()
        if ng == 0:
            out[i] = 100.0
        else:
            ratio = p / ng
            out[i] = 100.0 - (100.0 / (1.0 + ratio))
    return out


def keltner_channels(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    period: int = 20,
    multiplier: float = 1.5,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Keltner Channels: EMA mid line ± multiplier × ATR. Returns (upper, mid, lower).

    Used standalone for breakout detection AND as the inner band in TTM Squeeze
    (when Bollinger Bands are INSIDE Keltner, the market is compressed).
    """
    mid = ema(close, period)
    a = atr(high, low, close, period)
    upper = mid + multiplier * a
    lower = mid - multiplier * a
    return upper, mid, lower


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
