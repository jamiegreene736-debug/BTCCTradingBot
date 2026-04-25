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
    min_confluence: int
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
    # Dead-market filter: skip if ATR/price < this (no point trading when
    # the market can't move enough to clear fees + reach TP within hours).
    min_atr_pct: float = 0.08    # 0.08% = 8 bps
    # Pattern recognition: candlestick patterns are weighted heavily because
    # they're the primary technical-trading signal a human chartist uses.
    # combined_score = pattern_weight * pattern + (1 - pattern_weight) * indicator
    pattern_weight: float = 0.55          # 55% pattern, 45% indicator
    pattern_norm: float = 2.0             # divide raw pattern strength by this for 0-1 normalization
    fire_threshold: float = 0.30          # combined score must be ≥ this to fire


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

    return Config(
        mode=raw["mode"],
        creds=creds,
        trading=TradingCfg(**trading_raw),
        risk=RiskCfg(**raw["risk"]),
        strategy=StrategyCfg(**raw["strategy"]),
        loop=LoopCfg(**raw["loop"]),
        logging=LoggingCfg(**raw["logging"]),
        raw=raw,
    )
