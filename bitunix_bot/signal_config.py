"""Validated settings for the isolated, alerts-only intraday scanner."""

from __future__ import annotations

import math
from dataclasses import dataclass, fields


@dataclass(frozen=True)
class SignalSettings:
    planning_equity: float = 1000.0
    risk_pct: float = 0.5
    leverage: int = 25
    hold_hours: int = 24

    def validate(self) -> None:
        if (
            not math.isfinite(self.planning_equity)
            or not 10 <= self.planning_equity <= 100_000_000
        ):
            raise ValueError("Planning equity must be between 10 and 100,000,000 USDT")
        if not math.isfinite(self.risk_pct) or not 0 < self.risk_pct <= 2:
            raise ValueError("Planned risk must be greater than zero and at most 2%")
        if type(self.leverage) is not int or not 1 <= self.leverage <= 40:
            raise ValueError("Leverage must be a whole number from 1 to 40")
        if type(self.hold_hours) is not int or self.hold_hours not in (12, 24):
            raise ValueError("Maximum holding time must be 12 or 24 hours")

    @classmethod
    def from_dict(cls, values: dict[str, object]) -> SignalSettings:
        if set(values) != {f.name for f in fields(cls)}:
            raise ValueError(
                "Provide planning_equity, risk_pct, leverage and hold_hours"
            )
        if any(type(v) not in (int, float) for v in values.values()):
            raise ValueError("Planning settings must be numbers")
        settings = cls(**values)  # type: ignore[arg-type]
        settings.validate()
        return settings


@dataclass
class SignalsCfg:
    enabled: bool = False
    refresh_seconds: int = 15
    max_data_age_seconds: int = 60
    min_quote_volume: float = 10_000_000.0
    max_symbols: int = 12
    min_reward_risk: float = 2.0
    relative_volume_min: float = 1.2
    breakout_volume_min: float = 1.5
    stop_atr_buffer: float = 0.2
    max_spread_pct: float = 0.08
    round_trip_fee_pct: float = 0.12
    slippage_pct: float = 0.06
    min_depth_ratio: float = 5.0
    liquidation_buffer_pct: float = 0.5
    stale_trade_hours: int = 4
    stale_progress_r: float = 0.25
    trailing_activate_r: float = 1.5
    breakeven_at_r: float = 1.0
    max_total_risk_pct: float = 1.5
    max_same_direction: int = 2
    # 24h / 25-40x planning: skip dead or blow-off hours, keep targets inside
    # a hold-window travel budget, and reject funding that eats the edge.
    min_hourly_atr_pct: float = 0.12
    max_hourly_atr_pct: float = 5.0
    max_target_atr_multiple: float = 8.0
    max_target_4h_atr_multiple: float = 3.0
    impulse_atr_min: float = 1.1
    max_funding_cost_pct: float = 0.40
    queue_size: int = 5
    handoff_seconds: int = 20
    expiry_warn_seconds: int = 45

    def validate(self) -> None:
        if type(self.enabled) is not bool:
            raise ValueError("signals.enabled must be true or false")
        for name in (
            "refresh_seconds",
            "max_data_age_seconds",
            "max_symbols",
            "stale_trade_hours",
            "max_same_direction",
            "queue_size",
            "handoff_seconds",
            "expiry_warn_seconds",
        ):
            if type(getattr(self, name)) is not int:
                raise ValueError(f"signals.{name} must be a whole number")
        for field in fields(self):
            if field.name == "enabled":
                continue
            value = getattr(self, field.name)
            if (
                type(value) not in (int, float)
                or not math.isfinite(value)
                or value <= 0
            ):
                raise ValueError(
                    f"signals.{field.name} must be a finite positive number"
                )
        if not 5 <= self.refresh_seconds <= 30:
            raise ValueError("signals.refresh_seconds must be between 5 and 30")
        if not self.refresh_seconds * 2 <= self.max_data_age_seconds <= 120:
            raise ValueError(
                "signals.max_data_age_seconds must allow two refreshes, at most 120 seconds"
            )
        if type(self.max_symbols) is not int or not 1 <= self.max_symbols <= 30:
            raise ValueError("signals.max_symbols must be an integer from 1 to 30")
        if not 1 <= self.queue_size <= 8:
            raise ValueError("signals.queue_size must be an integer from 1 to 8")
        if not 5 <= self.handoff_seconds <= 60:
            raise ValueError("signals.handoff_seconds must be between 5 and 60")
        if not 15 <= self.expiry_warn_seconds <= 180:
            raise ValueError("signals.expiry_warn_seconds must be between 15 and 180")
