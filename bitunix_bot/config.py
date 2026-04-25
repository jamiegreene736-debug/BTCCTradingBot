"""Config loader — merges .env and config.yaml."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv


@dataclass
class Credentials:
    api_key: str
    secret_key: str


@dataclass
class TradingCfg:
    symbols: list[str]               # universe to trade
    timeframe: str
    leverage: int
    margin_coin: str
    margin_mode: str                 # ISOLATION | CROSS
    risk_per_trade_pct: float
    max_open_positions: int = 1      # global cap across all symbols
    max_positions_per_symbol: int = 1
    max_same_direction: int = 2      # max concurrent LONGs (or SHORTs) — kills correlated risk
    cooldown_seconds: int = 60       # min seconds between actions on same symbol
    max_position_age_seconds: int = 0  # 0 = disabled; else force-close stale positions
    # Streak protection — pause a symbol after N consecutive losses there.
    # Catches "wrong about this symbol's regime" without manual intervention.
    streak_loss_limit: int = 3
    streak_loss_pause_seconds: int = 7200    # 2 hours
    # Profit-aware time exit: when a position reaches max_position_age_seconds,
    # only force-close if it's at a loss. Profitable positions keep running
    # under the SL ratchet (which has already locked in some gain at +1R).
    time_exit_only_if_losing: bool = True
    # Daily drawdown circuit breaker — global kill switch for the day.
    # If equity drops below session_start × (1 - max_daily_dd_pct/100), all
    # new entries halt until next UTC midnight. Existing positions still
    # managed (SL ratchet, time exit, etc).
    max_daily_dd_pct: float = 8.0
    # Pre-trade spread filter — refuse entries when (ask-bid)/mid exceeds
    # this %. Wide spread means single-tick adverse fills eat large chunks
    # of the SL budget on entry.
    max_entry_spread_pct: float = 0.05
    # Post-only (maker) entries: place LIMIT orders at top-of-book with
    # POST_ONLY effect to qualify for maker fees (~0.02% vs 0.06% taker).
    # If the order doesn't fill within post_only_timeout_secs, cancel it
    # and fall back to a market order. Saves ~0.04% per round-trip.
    use_post_only_entries: bool = True
    post_only_timeout_secs: int = 8


@dataclass
class RiskCfg:
    stop_loss_pct: float
    take_profit_r: float
    use_atr: bool
    atr_multiplier_sl: float
    atr_multiplier_tp: float
    # Dynamic SL management — once a trade moves favorably, SL moves with it.
    # Set any of these to 0 (or breakeven_at_r above trailing_activate_r) to disable.
    breakeven_at_r: float = 1.0          # at +1R favorable, SL → entry+buffer
    breakeven_buffer_pct: float = 0.05   # SL sits this % above (long) / below (short) entry
    trailing_activate_r: float = 1.5     # at +1.5R favorable, start trailing
    trailing_distance_r: float = 0.5     # SL trails this many R behind current price
    # Partial TP at +1R — close N% of position at break-even ratchet point.
    # Lock in fee-clearing profit on half, let the rest ride to TP target.
    # Mathematically transforms a 2.5R target into ~1.4R realized but with
    # higher effective hit rate (more trades end positive).
    partial_tp_enabled: bool = True
    partial_tp_at_r: float = 1.0         # fire when r_favor reaches this
    partial_tp_close_pct: float = 50.0   # close this % of qty at market
    # Round-trip fee buffer (round-trip taker fee + slippage estimate). Used
    # to enforce a minimum profitable TP target after fees. 0.17% is the
    # observed Bitunix taker round-trip + slippage on 50x.
    round_trip_fee_pct: float = 0.17     # % of notional. SL+TP must clear this for net profit.
    # Adaptive TP tightening over time. Original TP is take_profit_r × SL distance.
    # As the trade ages, if it hasn't progressed, ratchet the TP DOWN toward
    # break-even-plus-fees so the trade has a realistic chance to fire while
    # still being net-positive. Tiers expressed as (age_minutes, R-multiple).
    # Always aim for profit — never go below the fee floor.
    adaptive_tp_enabled: bool = True
    adaptive_tp_floor_r: float = 0.7     # never tighten TP below this R (covers fees + small profit)


@dataclass
class StrategyCfg:
    ema_fast: int
    ema_mid: int
    ema_slow: int
    rsi_period: int
    rsi_long_min: float
    rsi_long_max: float
    rsi_short_min: float
    rsi_short_max: float
    macd_fast: int
    macd_slow: int
    macd_signal: int
    bb_period: int
    bb_std: float
    atr_period: int
    # ADX trend-strength filter — counts for whichever side is otherwise winning.
    adx_period: int = 14
    adx_min: float = 22.0
    # Supertrend regime filter — direction-aware.
    supertrend_period: int = 10
    supertrend_mult: float = 3.0
    # Volume confirmation — current bar volume / N-period MA. A "spike" is when
    # current volume >= multiplier × MA, indicating real conviction behind the move.
    volume_ma_period: int = 20
    volume_spike_multiplier: float = 1.5
    # Stochastic RSI — more sensitive momentum oscillator (Stoch applied to RSI).
    stoch_rsi_period: int = 14
    stoch_rsi_k: int = 3
    stoch_rsi_d: int = 3
    # Higher-timeframe trend (multi-timeframe) — votes for direction of HTF trend.
    htf_timeframe: str = "15m"
    htf_ema_period: int = 50
    # Funding rate (perp futures) — vote against crowded positioning.
    # Bitunix's fundingRate is per-interval (typically 8h). 0.0005 = 0.05%/8h.
    funding_threshold: float = 0.0005
    # Support/Resistance via swing-point clustering. Vote on bounce/rejection.
    swing_lookback: int = 5            # bars on each side that confirm a swing
    sr_cluster_tol_pct: float = 0.3    # swings within 0.3% cluster into one level
    sr_min_touches: int = 2            # minimum swings to qualify as a "real" level
    sr_proximity_pct: float = 0.3      # how close current price must be to fire
    # Order book imbalance (live WebSocket).
    ob_depth_levels: int = 10          # top-N levels to sum on each side
    ob_imbalance_threshold: float = 0.30  # |(bid - ask) / (bid + ask)| above this fires
    # Money Flow Index — volume-weighted RSI. Same windows as RSI.
    mfi_period: int = 14
    mfi_long_max: float = 60.0          # MFI must be < this for long entry (room to run)
    mfi_short_min: float = 40.0
    # TTM Squeeze — Bollinger Bands inside Keltner Channels = compression.
    # Direction is determined by momentum (close vs midpoint) at release.
    keltner_period: int = 20
    keltner_atr_multiplier: float = 1.5
    # BTC-as-leader: alts shouldn't long against a BTC dump. We compute a
    # short-term BTC trend and add a vote in BTC's direction when alts
    # signal the same way (and against when opposed).
    btc_leader_symbol: str = "BTCUSDT"
    btc_leader_ema: int = 21    # EMA period on BTC 1m closes for trend direction
    # Pattern recognition: candlestick patterns are weighted heavily because
    # they're the primary technical-trading signal a human chartist uses.
    # combined_score = pattern_weight * pattern + (1 - pattern_weight) * indicator
    pattern_weight: float = 0.55          # 55% pattern, 45% indicator
    pattern_norm: float = 2.0             # divide raw pattern strength by this for 0-1 normalization
    fire_threshold: float = 0.50          # combined score must be ≥ this to fire


@dataclass
class LoopCfg:
    tick_seconds: int
    kline_lookback: int


@dataclass
class LoggingCfg:
    level: str
    file: str


@dataclass
class Config:
    mode: str
    creds: Credentials
    trading: TradingCfg
    risk: RiskCfg
    strategy: StrategyCfg
    loop: LoopCfg
    logging: LoggingCfg
    raw: dict[str, Any] = field(default_factory=dict)

    @property
    def is_live(self) -> bool:
        return self.mode == "live"


def load(path: str | Path = "config.yaml", env_path: str | Path = ".env") -> Config:
    load_dotenv(env_path)
    with open(path) as f:
        raw = yaml.safe_load(f)

    creds = Credentials(
        api_key=os.environ["BITUNIX_API_KEY"],
        secret_key=os.environ["BITUNIX_SECRET_KEY"],
    )

    # Backwards-compat: accept singular `symbol` and lift to `symbols: [..]`.
    trading_raw = dict(raw["trading"])
    if "symbol" in trading_raw and "symbols" not in trading_raw:
        trading_raw["symbols"] = [trading_raw.pop("symbol")]
    elif "symbol" in trading_raw:
        trading_raw.pop("symbol")
    if isinstance(trading_raw.get("symbols"), str):
        trading_raw["symbols"] = [trading_raw["symbols"]]
    # Normalize symbols to UPPER throughout the bot's lifecycle to avoid
    # case-mismatch bugs in caches keyed by symbol.
    trading_raw["symbols"] = [str(s).upper() for s in trading_raw["symbols"]]

    cfg = Config(
        mode=raw["mode"],
        creds=creds,
        trading=TradingCfg(**trading_raw),
        risk=RiskCfg(**raw["risk"]),
        strategy=StrategyCfg(**raw["strategy"]),
        loop=LoopCfg(**raw["loop"]),
        logging=LoggingCfg(**raw["logging"]),
        raw=raw,
    )
    _validate(cfg)
    return cfg


def _validate(cfg: Config) -> None:
    """Catch config sins early — fail loudly at startup, not mid-trade."""
    errs: list[str] = []
    t, r, s, l = cfg.trading, cfg.risk, cfg.strategy, cfg.loop
    if cfg.mode not in ("paper", "live"):
        errs.append(f"mode must be 'paper' or 'live', got {cfg.mode!r}")
    if not t.symbols:
        errs.append("trading.symbols must be non-empty")
    if not (1 <= t.leverage <= 200):
        errs.append(f"trading.leverage must be 1..200, got {t.leverage}")
    if t.margin_mode not in ("ISOLATION", "CROSS"):
        errs.append(f"trading.margin_mode must be ISOLATION/CROSS, got {t.margin_mode!r}")
    if not (0 < t.risk_per_trade_pct <= 10):
        errs.append(f"trading.risk_per_trade_pct must be 0..10 (refuse >10% per trade), "
                    f"got {t.risk_per_trade_pct}")
    if t.max_open_positions < 1:
        errs.append(f"trading.max_open_positions must be >=1, got {t.max_open_positions}")
    if t.max_positions_per_symbol < 1:
        errs.append("trading.max_positions_per_symbol must be >=1")
    if t.cooldown_seconds < 0:
        errs.append("trading.cooldown_seconds must be >=0")
    if not (0 < r.stop_loss_pct < 5):
        errs.append(f"risk.stop_loss_pct must be 0..5%, got {r.stop_loss_pct}")
    if not (0 < r.take_profit_r < 20):
        errs.append(f"risk.take_profit_r must be 0..20, got {r.take_profit_r}")
    if r.breakeven_at_r < 0 or r.trailing_activate_r < 0 or r.trailing_distance_r < 0:
        errs.append("risk.breakeven_at_r / trailing_* must be >=0")
    if s.fire_threshold < 0 or s.fire_threshold > 1:
        errs.append("strategy.fire_threshold must be 0..1")
    if s.pattern_weight < 0 or s.pattern_weight > 1:
        errs.append("strategy.pattern_weight must be 0..1")
    if s.adx_min < 0 or s.adx_min > 100:
        errs.append("strategy.adx_min must be 0..100")
    if l.tick_seconds < 1:
        errs.append("loop.tick_seconds must be >=1")
    if l.kline_lookback < 60:
        errs.append("loop.kline_lookback must be >=60 for indicators to warm up")
    if errs:
        raise ValueError("Config validation failed:\n  - " + "\n  - ".join(errs))
