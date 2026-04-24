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
    symbol: str
    timeframe: str
    leverage: int
    margin_coin: str
    margin_mode: str       # ISOLATION | CROSS
    risk_per_trade_pct: float


@dataclass
class RiskCfg:
    stop_loss_pct: float
    take_profit_r: float
    use_atr: bool
    atr_multiplier_sl: float
    atr_multiplier_tp: float


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

    return Config(
        mode=raw["mode"],
        creds=creds,
        trading=TradingCfg(**raw["trading"]),
        risk=RiskCfg(**raw["risk"]),
        strategy=StrategyCfg(**raw["strategy"]),
        loop=LoopCfg(**raw["loop"]),
        logging=LoggingCfg(**raw["logging"]),
        raw=raw,
    )
