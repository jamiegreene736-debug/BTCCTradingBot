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


def volume_profile_hvns(
    high: np.ndarray,
    low: np.ndarray,
    volumes: np.ndarray,
    current_price: float,
    *,
    lookback: int = 100,
    num_bins: int = 50,
    hvn_percentile: float = 80.0,
) -> tuple[float | None, float | None]:
    """Build a price-volume histogram from the last `lookback` bars and
    return the nearest high-volume nodes (HVNs) above and below
    `current_price`.

    HVNs are price levels where lots of volume has traded — strong
    structural support/resistance. For a long entry, the nearest HVN
    BELOW is a natural SL anchor (support); the nearest HVN ABOVE is a
    natural TP target (resistance). Mirror for shorts.

    Algorithm:
      1. Define `num_bins` evenly-spaced price bins covering [min_low, max_high]
         over the last `lookback` bars.
      2. For each bar, distribute its volume uniformly across the bins
         that overlap its [low, high] range (volume-by-price approximation
         since we don't have intra-bar tick data).
      3. Identify "high-volume" bins as those above the `hvn_percentile`
         threshold. Pick the nearest such bin centers above and below
         `current_price`.

    Returns (hvn_below, hvn_above). Either may be None when:
      - Insufficient bars (< lookback // 2)
      - No volume in the lookback window (synthetic / paper-mode data)
      - No HVNs found on that side of current price

    This is the "volume profile" piece of the Grok holistic review SL/TP
    redesign — gives `risk.build_order` real market-structure anchors
    instead of pure %-of-price stops.
    """
    if len(high) < lookback // 2 or len(volumes) != len(high):
        return None, None

    # Slice last `lookback` bars (or fewer if not enough data).
    n = min(lookback, len(high))
    h = high[-n:]
    l = low[-n:]
    v = volumes[-n:].astype(float)
    if v.sum() <= 0:
        return None, None

    p_min = float(l.min())
    p_max = float(h.max())
    if p_max <= p_min:
        return None, None

    bins = np.linspace(p_min, p_max, num_bins + 1)
    bin_centers = (bins[:-1] + bins[1:]) / 2.0
    profile = np.zeros(num_bins)

    # Distribute each bar's volume across overlapping bins.
    for i in range(n):
        bar_lo = float(l[i])
        bar_hi = float(h[i])
        bar_range = bar_hi - bar_lo
        if bar_range <= 0:
            # Doji or tiny bar — assign all volume to the bin containing close.
            idx = int(np.searchsorted(bins, bar_lo) - 1)
            if 0 <= idx < num_bins:
                profile[idx] += float(v[i])
            continue
        for j in range(num_bins):
            bin_lo = bins[j]
            bin_hi = bins[j + 1]
            overlap = max(0.0, min(bar_hi, bin_hi) - max(bar_lo, bin_lo))
            if overlap > 0:
                profile[j] += float(v[i]) * (overlap / bar_range)

    if profile.max() <= 0:
        return None, None

    # HVN threshold: bins above the given percentile.
    threshold = float(np.percentile(profile, hvn_percentile))
    hvn_mask = profile >= threshold

    # Nearest HVN below current_price.
    below_indices = np.where(hvn_mask & (bin_centers < current_price))[0]
    hvn_below: float | None = None
    if len(below_indices) > 0:
        hvn_below = float(bin_centers[below_indices[-1]])  # rightmost = closest below

    # Nearest HVN above current_price.
    above_indices = np.where(hvn_mask & (bin_centers > current_price))[0]
    hvn_above: float | None = None
    if len(above_indices) > 0:
        hvn_above = float(bin_centers[above_indices[0]])   # leftmost = closest above

    return hvn_below, hvn_above


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


def cvd(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    volumes: np.ndarray,
) -> np.ndarray:
    """Cumulative Volume Delta — kline body-position approximation.

    Without per-trade taker/maker data (which Bitunix's kline doesn't
    expose), we estimate buyer vs seller pressure from where the close
    sits within the bar's range:

      buyer_pct  = (close - low) / (high - low)
      seller_pct = (high - close) / (high - low)
      bar_delta  = volume × (buyer_pct - seller_pct)
      CVD        = cumsum(bar_delta)

    Different from OBV: OBV signs by close-vs-prev-close direction, which
    treats every up-bar as +volume even if the close was a fakeout. CVD
    by body position is more nuanced — a doji with tiny upper wick gets
    a small positive delta; a hammer (close near high after deep wick)
    gets a strongly positive delta. Better for detecting hidden pressure.
    """
    n = len(close)
    delta = np.zeros(n)
    for i in range(n):
        rng = high[i] - low[i]
        if rng <= 0:
            continue
        buyer = (close[i] - low[i]) / rng
        seller = (high[i] - close[i]) / rng
        delta[i] = volumes[i] * (buyer - seller)
    return np.cumsum(delta)


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
