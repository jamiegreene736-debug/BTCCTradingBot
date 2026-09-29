"""Validated settings for the isolated, alerts-only intraday scanner."""

from __future__ import annotations

import math
from dataclasses import dataclass, field, fields

# "trend": 1h EMA bias, 15m HH/HL structure, completed 5m continuation trigger;
# 1-2h hold at 20-100x (50x default). The stop is the 5m structural level,
# capped at 0.60%, and must sit inside the estimated isolated liquidation.
# "swing" is the pre-1.8 name and is read as "trend".
# "scalp": parabolic-exhaustion fade, long or short, on 1m/3m bars, 1-2h hold,
# up to the exchange tier maximum. The stop must sit inside the liquidation
# distance. "scalp_short" is the pre-1.7 name and is read as "scalp".
# profile -> (allowed hold hours, minimum leverage, maximum planning leverage)
PROFILES: dict[str, tuple[tuple[int, ...], int, int]] = {
    "trend": ((1, 2), 20, 100),
    "scalp": ((1, 2), 1, 125),
    "scalp_short": ((1, 2), 1, 125),
}
LEGACY_PROFILE_NAMES: dict[str, str] = {"scalp_short": "scalp", "swing": "trend"}
DEFAULT_PROFILE = "trend"
SCALP_PROFILES: tuple[str, ...] = ("scalp", "scalp_short")
NUMERIC_SETTINGS = ("planning_equity", "risk_pct", "leverage", "hold_hours")


@dataclass(frozen=True)
class SignalSettings:
    planning_equity: float = 1000.0
    risk_pct: float = 0.5
    leverage: int = 50
    hold_hours: int = 2
    profile: str = DEFAULT_PROFILE

    def validate(self) -> None:
        if self.profile not in PROFILES:
            raise ValueError("Profile must be trend or scalp")
        holds, lo, hi = PROFILES[self.profile]
        if (
            not math.isfinite(self.planning_equity)
            or not 10 <= self.planning_equity <= 100_000_000
        ):
            raise ValueError("Planning equity must be between 10 and 100,000,000 USDT")
        if not math.isfinite(self.risk_pct) or not 0 < self.risk_pct <= 2:
            raise ValueError("Planned risk must be greater than zero and at most 2%")
        if type(self.leverage) is not int or not lo <= self.leverage <= hi:
            raise ValueError(f"Leverage must be a whole number from {lo} to {hi}")
        if type(self.hold_hours) is not int or self.hold_hours not in holds:
            raise ValueError(
                "Maximum holding time must be "
                + " or ".join(str(h) for h in holds)
                + " hours"
            )

    @classmethod
    def from_dict(cls, values: dict[str, object]) -> SignalSettings:
        payload = dict(values)
        # Settings saved before profiles existed carry only the numeric fields.
        raw = payload.pop("profile", None)
        if set(payload) != set(NUMERIC_SETTINGS):
            raise ValueError(
                "Provide planning_equity, risk_pct, leverage and hold_hours"
            )
        if any(type(v) not in (int, float) for v in payload.values()):
            raise ValueError("Planning settings must be numbers")
        if raw is not None and type(raw) is not str:
            raise ValueError("Profile must be trend or scalp")
        name = raw or DEFAULT_PROFILE
        profile = LEGACY_PROFILE_NAMES.get(name, name)
        if raw in (None, "swing") and profile in PROFILES:
            # Legacy swing rows (12-24h, 25-40x) rebase to the trend defaults;
            # equity and risk are kept.
            holds, lo, hi = PROFILES[profile]
            leverage = payload["leverage"]
            if payload["hold_hours"] not in holds:
                payload["hold_hours"] = 2
            if isinstance(leverage, (int, float)) and not lo <= leverage <= hi:
                payload["leverage"] = 50
        settings = cls(profile=profile, **payload)  # type: ignore[arg-type]
        settings.validate()
        return settings


@dataclass
class ScalpCfg:
    """Gates for the 1-2h parabolic-exhaustion short profile."""

    trigger_interval: str = "1m"
    # Candidate discovery: extension, climax volume and crowded longs.
    min_gain_1h_pct: float = 1.5
    min_gain_4h_pct: float = 3.0
    min_extension_atr: float = 2.0
    min_climax_volume: float = 3.0
    climax_lookback: int = 15
    min_funding_rate_pct: float = 0.01
    min_oi_change_pct: float = 1.0
    oi_window_seconds: int = 3600
    # Failed-high trigger and stop.
    spike_lookback: int = 30
    spike_min_age_bars: int = 2
    spike_max_age_bars: int = 12
    stop_atr_buffer: float = 0.15
    max_stop_pct: float = 0.45
    min_stop_atr: float = 0.5
    # Mean-reversion targets inside a 1-2h travel budget.
    travel_atr_multiple: float = 1.5
    min_reward_risk: float = 2.0
    # Execution gates tightened for the leverage.
    max_spread_pct: float = 0.03
    min_depth_ratio: float = 8.0
    liquidation_buffer_pct: float = 0.15
    entry_expiry_seconds: int = 180
    # Trade management.
    stale_minutes: int = 20
    stale_progress_r: float = 0.3
    breakeven_at_r: float = 0.75
    trailing_activate_r: float = 1.0

    def validate(self) -> None:
        if self.trigger_interval not in ("1m", "3m", "5m"):
            raise ValueError("signals.scalp.trigger_interval must be 1m, 3m or 5m")
        for item in fields(self):
            if item.name == "trigger_interval":
                continue
            value = getattr(self, item.name)
            if (
                type(value) not in (int, float)
                or not math.isfinite(value)
                or value <= 0
            ):
                raise ValueError(
                    f"signals.scalp.{item.name} must be a finite positive number"
                )
        for name in (
            "climax_lookback",
            "spike_lookback",
            "spike_min_age_bars",
            "spike_max_age_bars",
            "oi_window_seconds",
            "entry_expiry_seconds",
            "stale_minutes",
        ):
            if type(getattr(self, name)) is not int:
                raise ValueError(f"signals.scalp.{name} must be a whole number")
        if not self.spike_min_age_bars < self.spike_max_age_bars < self.spike_lookback:
            raise ValueError(
                "signals.scalp spike ages must satisfy min < max < lookback"
            )
        if self.max_stop_pct > 2:
            raise ValueError("signals.scalp.max_stop_pct must be at most 2%")
        if not 1 <= self.stale_minutes <= 120:
            raise ValueError("signals.scalp.stale_minutes must be 1 to 120")


@dataclass
class TrendCfg:
    """Gates for the 1-2h trend-continuation profile at 20-100x."""

    trigger_interval: str = "5m"
    # Hold-window volatility: the 1h ATR floor depends on the hold length.
    min_hourly_atr_pct_1h: float = 0.60
    min_hourly_atr_pct_2h: float = 0.35
    max_hourly_atr_pct: float = 1.2
    # Market quality.
    max_spread_pct: float = 0.04
    max_mark_basis_pct: float = 0.15
    funding_blackout_seconds: int = 300
    session_volume_ratio: float = 0.5
    # Do not buy the squeeze the scalp profile fades.
    max_extension_atr: float = 2.0
    max_funding_rate_pct: float = 0.05
    max_oi_change_pct: float = 3.0
    crowd_extension_atr: float = 1.5
    # Trigger candle (ATR of the trigger frame).
    relative_volume_min: float = 1.2
    breakout_volume_min: float = 1.5
    impulse_atr_min: float = 1.1
    stop_atr_buffer: float = 0.2
    # Stop size.
    min_stop_atr: float = 1.0
    min_stop_atr_15m: float = 0.35
    min_stop_pct: float = 0.20
    max_stop_pct: float = 0.60
    # Entry zone around the trigger close.
    entry_pullback_atr: float = 0.30
    entry_chase_atr: float = 0.25
    entry_chase_risk_fraction: float = 0.25
    # Targets and costs.
    travel_atr_multiple: float = 1.5
    min_reward_risk: float = 1.5
    max_funding_cost_pct: float = 0.10
    min_depth_ratio: float = 8.0
    liquidation_buffer_pct: float = 0.25
    entry_expiry_seconds: int = 600
    # Trade management, as fractions of the hold and R multiples.
    stale_hold_fraction: float = 0.35
    stale_progress_r: float = 0.3
    late_hold_fraction: float = 0.75
    late_hold_min_r: float = 0.5
    breakeven_at_r: float = 1.0
    trailing_activate_r: float = 1.25
    hope_exit_r: float = 0.75

    def validate(self) -> None:
        if self.trigger_interval not in ("3m", "5m"):
            raise ValueError("signals.trend.trigger_interval must be 3m or 5m")
        for item in fields(self):
            if item.name == "trigger_interval":
                continue
            value = getattr(self, item.name)
            if (
                type(value) not in (int, float)
                or not math.isfinite(value)
                or value <= 0
            ):
                raise ValueError(
                    f"signals.trend.{item.name} must be a finite positive number"
                )
        for name in ("funding_blackout_seconds", "entry_expiry_seconds"):
            if type(getattr(self, name)) is not int:
                raise ValueError(f"signals.trend.{name} must be a whole number")
        if not (
            self.min_hourly_atr_pct_2h
            <= self.min_hourly_atr_pct_1h
            < self.max_hourly_atr_pct
        ):
            raise ValueError(
                "signals.trend hourly ATR band must satisfy 2h floor <= 1h floor < max"
            )
        if not self.min_stop_pct < self.max_stop_pct <= 2:
            raise ValueError(
                "signals.trend stop band must satisfy min_stop_pct < max_stop_pct <= 2"
            )
        if not 0 < self.stale_hold_fraction < self.late_hold_fraction < 1:
            raise ValueError(
                "signals.trend hold fractions must satisfy 0 < stale < late < 1"
            )
        if self.stale_hold_fraction * 60 < 10:
            raise ValueError(
                "signals.trend.stale_hold_fraction must allow at least 10 minutes on a 1h hold"
            )
        if not self.breakeven_at_r < self.trailing_activate_r < self.min_reward_risk:
            raise ValueError(
                "signals.trend R ladder must satisfy breakeven < trailing < min_reward_risk"
            )
        if self.late_hold_min_r >= self.min_reward_risk:
            raise ValueError(
                "signals.trend.late_hold_min_r must be below min_reward_risk"
            )
        if self.hope_exit_r >= 1:
            raise ValueError("signals.trend.hope_exit_r must be below 1R")
        if not 60 <= self.entry_expiry_seconds <= 1800:
            raise ValueError("signals.trend.entry_expiry_seconds must be 60 to 1800")
        if self.max_spread_pct > 0.1:
            raise ValueError("signals.trend.max_spread_pct must be at most 0.1%")
        if self.entry_chase_risk_fraction > 1:
            raise ValueError(
                "signals.trend.entry_chase_risk_fraction must be at most 1"
            )


@dataclass
class SignalsCfg:
    enabled: bool = False
    refresh_seconds: int = 15
    max_data_age_seconds: int = 60
    min_quote_volume: float = 10_000_000.0
    max_symbols: int = 12
    universe_size: int = 80
    evaluate_batch: int = 10
    round_trip_fee_pct: float = 0.12
    slippage_pct: float = 0.06
    max_total_risk_pct: float = 1.5
    max_same_direction: int = 2
    queue_size: int = 5
    handoff_seconds: int = 20
    expiry_warn_seconds: int = 45
    # Scalp market-quality gates; the trend profile carries its own.
    max_mark_basis_pct: float = 0.25
    funding_blackout_seconds: int = 180
    scalp: ScalpCfg = field(default_factory=ScalpCfg)
    trend: TrendCfg = field(default_factory=TrendCfg)

    def __post_init__(self) -> None:
        if isinstance(self.scalp, dict):
            self.scalp = ScalpCfg(**self.scalp)
        if isinstance(self.trend, dict):
            self.trend = TrendCfg(**self.trend)

    def validate(self) -> None:
        if type(self.enabled) is not bool:
            raise ValueError("signals.enabled must be true or false")
        if not isinstance(self.scalp, ScalpCfg):
            raise ValueError("signals.scalp must be a mapping")
        self.scalp.validate()
        if not isinstance(self.trend, TrendCfg):
            raise ValueError("signals.trend must be a mapping")
        self.trend.validate()
        for name in (
            "refresh_seconds",
            "max_data_age_seconds",
            "max_symbols",
            "universe_size",
            "evaluate_batch",
            "max_same_direction",
            "queue_size",
            "handoff_seconds",
            "expiry_warn_seconds",
            "funding_blackout_seconds",
        ):
            if type(getattr(self, name)) is not int:
                raise ValueError(f"signals.{name} must be a whole number")
        for item in fields(self):
            if item.name in ("enabled", "scalp", "trend"):
                continue
            value = getattr(self, item.name)
            if (
                type(value) not in (int, float)
                or not math.isfinite(value)
                or value <= 0
            ):
                raise ValueError(
                    f"signals.{item.name} must be a finite positive number"
                )
        if not 5 <= self.refresh_seconds <= 30:
            raise ValueError("signals.refresh_seconds must be between 5 and 30")
        if not self.refresh_seconds * 2 <= self.max_data_age_seconds <= 120:
            raise ValueError(
                "signals.max_data_age_seconds must allow two refreshes, at most 120 seconds"
            )
        if type(self.max_symbols) is not int or not 1 <= self.max_symbols <= 30:
            raise ValueError("signals.max_symbols must be an integer from 1 to 30")
        if type(self.universe_size) is not int or not 6 <= self.universe_size <= 120:
            raise ValueError("signals.universe_size must be an integer from 6 to 120")
        if self.universe_size < self.max_symbols:
            raise ValueError(
                "signals.universe_size must be at least signals.max_symbols"
            )
        if type(self.evaluate_batch) is not int or not 1 <= self.evaluate_batch <= 30:
            raise ValueError("signals.evaluate_batch must be an integer from 1 to 30")
        if not 1 <= self.queue_size <= 8:
            raise ValueError("signals.queue_size must be an integer from 1 to 8")
        if not 5 <= self.handoff_seconds <= 60:
            raise ValueError("signals.handoff_seconds must be between 5 and 60")
        if not 15 <= self.expiry_warn_seconds <= 180:
            raise ValueError("signals.expiry_warn_seconds must be between 15 and 180")


def profile_cfg(cfg: SignalsCfg, profile: str) -> TrendCfg | ScalpCfg:
    """The gate block a profile reads: scalp for the scalp names, trend otherwise."""
    return cfg.scalp if profile in SCALP_PROFILES else cfg.trend


def min_hourly_atr_pct(t: TrendCfg, hold_minutes: int) -> float:
    """1h ATR floor that can still carry the profile R inside the hold window."""
    return t.min_hourly_atr_pct_1h if hold_minutes <= 60 else t.min_hourly_atr_pct_2h
