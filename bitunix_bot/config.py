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
    # Correlation / beta-adjusted sizing per symbol. Crypto alts are heavily
    # correlated to BTC (typically 0.7+ on 1m), so a portfolio of "1 long BTC
    # + 3 long alts" carries effective BTC exposure of ~3.5x, not 4x of
    # independent risk. Down-weight high-beta alts to keep portfolio risk
    # closer to nominal. BTC = 1.0 baseline; alts are scaled by inverse-beta
    # estimates from realized 1m correlations:
    #   ETHUSDT — high cap, ~0.85 to BTC, slightly lower beta → 0.85
    #   SOLUSDT — mid cap, ~0.75 corr, higher beta → 0.70
    #   XRPUSDT — mid cap, ~0.70 corr, higher idiosyncratic vol → 0.70
    # Symbols not listed default to 1.0 (no adjustment).
    symbol_risk_mult: dict[str, float] = field(default_factory=lambda: {
        "BTCUSDT": 1.0,
        "ETHUSDT": 0.85,
        "SOLUSDT": 0.70,
        "XRPUSDT": 0.70,
    })
    # Minimum top-5 book depth (base-coin units) per symbol — thin-book
    # microstructure filter. Standard HFT/desk filter: thin books cause
    # post-only limits to sit forever or get picked off via adverse
    # selection. Skip new entries when min(bid_depth, ask_depth) < threshold.
    # Calibrate from observed Bitunix per-symbol liquidity. Symbols not
    # listed have no depth filter (None / 0 = disabled).
    # Live-tuned 2026-04-26: BTC dropped 8.0 → 3.0. The 8.0 floor was
    # calibrated against larger trade sizes; on a $35 account at 10x
    # leverage our notional is ~$150 = 0.002 BTC. Top-5 depth of 3 BTC
    # is still ~1500× our trade size — plenty of cushion against
    # adverse selection on a maker fill. The 8.0 threshold was blocking
    # ~70% of would-be entries (live data over 9.8h: 9 thin-book skips,
    # 0 entries). Other symbols left at original calibration.
    symbol_min_depth: dict[str, float] = field(default_factory=lambda: {
        "BTCUSDT": 3.0,
        "ETHUSDT": 60.0,
        "SOLUSDT": 400.0,
        "XRPUSDT": 15000.0,
    })


@dataclass
class RiskCfg:
    stop_loss_pct: float
    take_profit_r: float
    use_atr: bool
    atr_multiplier_sl: float
    atr_multiplier_tp: float
    # Dynamic SL management — once a trade moves favorably, SL moves with it.
    # Set any of these to 0 (or breakeven_at_r above trailing_activate_r) to disable.
    # Grok review v9: BE/trailing thresholds lowered after seeing trades hit
    # 0.012R max-favor and stale-exit before any protection kicked in. Letting
    # the move develop a tiny bit (0.5R) and locking in BE there is worth more
    # than waiting for 1R+ that 1m chop rarely produces.
    breakeven_at_r: float = 0.5          # at +0.5R favorable, SL → entry+buffer
    breakeven_buffer_pct: float = 0.05   # SL sits this % above (long) / below (short) entry
    trailing_activate_r: float = 1.0     # at +1.0R favorable, start trailing
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
    # Stale-trade early exit. If a position has been alive for stale_exit_min
    # minutes AND has never reached more than stale_exit_max_favor_r favorable,
    # flash-close it. Pro-desk rule: a 1m scalp signal that hasn't moved in 6
    # minutes has lost its edge — the regime that birthed it has shifted or
    # the move never materialized. Pay the small loss and free the slot.
    # Distinct from time_exit_only_if_losing (90-min, profit-aware) and from
    # the tape-driven exit (immediate, flow-flip based).
    # Grok review v9: 6-min / 0.5R was strangling winners — ETH 0.65-score
    # trade hit max_favor 0.012R then got stale-killed before it could
    # develop. 12 min gives 1m signals time to actually unfold; 0.2R floor
    # only kills genuinely dead trades (zero-conviction sideways drift).
    stale_exit_enabled: bool = True
    stale_exit_min: float = 12.0         # minutes
    stale_exit_max_favor_r: float = 0.2  # if max_favor below this, exit
    # Tape-driven exit. DISABLED by default after live data showed it firing
    # on microstructure noise within 10–15 seconds of entry — closing trades
    # at 25-50% of full SL distance instead of letting them develop or hit
    # SL fairly. Aggression naturally swings in 10s windows; the ±0.50
    # threshold catches normal noise, not real regime flips.
    #
    # Set to True only if (a) you've seen the journal data, (b) tape_feed is
    # reliably providing data, and (c) you've increased the threshold or
    # added a min_hold_seconds guard that protects against sub-30s exits.
    tape_exit_enabled: bool = False
    tape_exit_threshold: float = 0.50    # contrary aggression magnitude
    tape_exit_min_hold_secs: int = 30    # don't fire before this many seconds


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
    # Hard ADX floor — don't trade at all when below this. The regime-
    # adaptive threshold (raise to 0.58 in chop) wasn't enough; live data
    # showed the bot still firing 0.64+ scores in ADX=15-20 chop and
    # bleeding via 6-min stale exits because price never moved in the
    # predicted direction. In deep chop, confluence agreement DOESN'T
    # predict direction — it's pattern artifacts of flat candles.
    # 22 matches the existing chop band (used to raise fire_threshold);
    # going below the band = fully skip rather than just raise the bar.
    min_adx_for_trade: float = 22.0
    # Post-signal confirmation gate (Grok review v8). After a signal
    # passes all gates, fetch the live ticker and require price to have
    # moved in the trade direction past the signal-bar close. This is
    # the missing piece that the in-bar continuation gate didn't catch:
    # the signal bar's close is by definition the LAST data we have, so
    # checking it confirms what just happened. Live tape continuation
    # checks what's happening NOW, post-signal — the only way to filter
    # exhaustion entries that look perfect on the bar that just closed
    # but immediately reverse on the bar that's now opening.
    confirm_with_ticker: bool = True
    # Signal inversion — wire for forward-looking experiment. When True,
    # every signal that would otherwise fire LONG fires SHORT and vice versa.
    # Use case: if 130+ live trades show the bot is systematically entering at
    # exhaustion points (high net-loss rate after fees, but with directional
    # bias detectable in net_pnl distribution), inverting empirically tests
    # whether the strategy has negative direction edge. Set in config.yaml as
    # `invert_signals: true` to enable. All other gates (continuation, ticker
    # confirmation, absorption veto) still run on the ORIGINAL direction —
    # we trust the signal's strength, just bet the opposite side.
    invert_signals: bool = False
    # Factor-group scoring — replaces raw vote-count normalization. Each
    # vote is classified into one of four groups; counts within a group are
    # capped at the saturation value (so 6 correlated trend votes don't
    # inflate confidence vs 1 strong signal). Weighted-average across
    # groups gives the indicator-half of combined score.
    #
    # Grok holistic review: rebalance per the structural critique that
    # trend-group dominance picks exhaustion. Flow promoted further (0.45
    # → 0.50), Trend demoted further (0.20 → 0.15), and trend saturation
    # cut from 6 → 3 so a chorus of correlated lagging signals (EMA stack
    # + MACD + Supertrend all firing on the same move) can't inflate
    # confidence above 3-vote saturation. Mean-rev saturation also cut
    # 6 → 3 for symmetry with flow (the actual leading signal). Sum 1.0.
    factor_weights: dict[str, float] = field(default_factory=lambda: {
        "trend": 0.15,      # was 0.20 — lagging confirmation only
        "mean_rev": 0.25,   # counterbalance / reversal setups
        "flow": 0.50,       # was 0.45 — leading edge dominates
        "context": 0.10,    # vol / regime / session modulators
    })
    factor_saturation: dict[str, int] = field(default_factory=lambda: {
        "trend": 3,         # was 6 — kill chorus-inflation of correlated trend votes
        "mean_rev": 3,      # was 6 — symmetry with flow's 3-cap
        "flow": 3,          # only 4 flow inputs total → saturate at 3
        "context": 3,
    })


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
