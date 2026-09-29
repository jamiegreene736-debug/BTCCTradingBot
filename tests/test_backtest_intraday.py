"""Offline checks for the trend replay: report shape, settings and CLI guards.

Histories are built from the deterministic frames in ``test_intraday`` and
extended with flat bars, so the replay enters on the fixture's completed 5m
pullback and then exits on the stale review without any network access.
"""

from __future__ import annotations

import importlib.util
import json
from dataclasses import replace
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest
from test_intraday import NOW, PRICE, TARGET_HIGH, market_frames

from bitunix_bot.intraday import INTERVALS, Candle
from bitunix_bot.signal_config import SignalSettings

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "backtest_intraday.py"
STEP = INTERVALS["5m"]
# The fixture's last completed 5m bar closes at NOW // 300 * 300; the replay
# evaluates that instant plus the two steps before it, then three flat hours.
START = NOW // STEP * STEP - 2 * STEP
END = NOW // STEP * STEP + 3 * 3600
REPORT_KEYS = (
    "closed_count",
    "entries_per_symbol_day",
    "exit_reasons",
    "watch_blockers",
    "median_minutes_held",
    "share_time_stop",
    "liquidation_touches",
    "worst_adverse_share_of_liquidation",
    "hit_rate",
    "average_net_r",
    "estimated_net_usdt",
    "share_plans_ceiling_ge_50",
    "unresolved_trades_due_to_gaps",
    "assumptions",
)


def load_script() -> ModuleType:
    spec = importlib.util.spec_from_file_location("backtest_intraday", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def fixture_history() -> dict[str, list[Candle]]:
    """The ready-to-enter long fixture, continued flat at PRICE until END."""
    frames: dict[str, list[Candle]] = market_frames()[1]  # type: ignore[no-untyped-call]
    frames["15m"][120] = replace(frames["15m"][120], high=TARGET_HIGH)
    for name in ("5m", "15m", "1h"):
        last = frames[name][-1]
        seconds = INTERVALS[name]
        frames[name] = frames[name] + [
            Candle(t, PRICE, PRICE, PRICE, PRICE, last.volume)
            for t in range(last.time + seconds, END + seconds, seconds)
        ]
    return frames


def run_replay(**overrides: Any) -> dict[str, Any]:
    histories = {"BTCUSDT": fixture_history(), "ETHUSDT": fixture_history()}
    report: dict[str, Any] = load_script().replay(
        histories, START, END, SignalSettings(), 0.02, 0.0001, **overrides
    )
    return report


def test_replay_reports_trend_settings_and_aggregates(tmp_path: Path) -> None:
    report = run_replay()
    assert report["settings"] == {
        "planning_equity": 1000.0,
        "risk_pct": 0.5,
        "leverage": 50,
        "hold_hours": 2,
        "profile": "trend",
    }
    assert report["hold_minutes"] == 120
    assert report["trend"]["trigger_interval"] == "5m"
    for key in REPORT_KEYS:
        assert key in report, key
    steps = (END - START) // STEP
    assert sum(report["evaluations"].values()) == steps * 2
    assert report["evaluations"]["ENTER_LONG"] == 2
    assert "Completed trigger candle" in report["watch_blockers"]
    assert report["assumptions"]["maintenance_rate"] == 0.005
    assert report["assumptions"]["tier_leverage"] == 100
    assert report["assumptions"]["depth_usdt_per_side"] == 500_000
    assert report["assumptions"]["spread_pct"] == 0.02
    # Serialisable without NaN so --output never fails after a long fetch.
    (tmp_path / "report.json").write_text(json.dumps(report, allow_nan=False))


def test_replay_closes_fixture_entries_inside_the_hold() -> None:
    report = run_replay()
    assert report["entries"] == 2 and report["closed_count"] == 2
    assert report["open_at_end"] == 0
    assert report["unresolved_trades_due_to_gaps"] == 0
    assert report["exit_reasons"] == {"stale": 2}
    assert report["share_time_stop"] == 0.0
    assert report["hit_rate"] == 0.0
    assert report["liquidation_touches"] == 0
    assert 42 <= report["median_minutes_held"] < 120
    assert 0 <= report["worst_adverse_share_of_liquidation"] < 0.6
    assert report["share_plans_ceiling_ge_50"] == 1.0
    assert report["entries_per_symbol_day"] == pytest.approx(
        2 / (2 * (END - START) / 86400)
    )
    assert report["average_net_r"] < 0 and report["estimated_net_usdt"] < 0
    for trade in report["closed_trades"]:
        assert trade["exit_reason"] == "stale"
        assert trade["minutes_held"] <= 120
        assert trade["stop_pct"] <= 0.60
        assert trade["max_leverage"] >= 50 and trade["leverage"] == 50
        assert trade["liquidation_distance_pct"] > trade["stop_pct"] + 0.25
        assert trade["adverse_share_of_liquidation"] == pytest.approx(
            trade["max_adverse_pct"] / trade["liquidation_distance_pct"]
        )
        assert trade["liquidation_touched"] is False
        assert trade["net_r"] == pytest.approx(
            trade["estimated_net_usdt"] / 5.0, rel=0.05
        )


def test_replay_reads_maintenance_and_tier_assumptions() -> None:
    report = run_replay(maintenance_rate=0.01, tier_leverage=60)
    assert report["assumptions"]["maintenance_rate"] == 0.01
    assert report["assumptions"]["tier_leverage"] == 60
    # A 1% tier still fits the fixture's 0.35% stop at 50x (ceiling 56x).
    assert report["entries"] == 2
    assert all(50 <= t["max_leverage"] <= 60 for t in report["closed_trades"])


@pytest.mark.parametrize(
    ("reason", "advanced", "expected"),
    [
        ("Stop level reached; verify your exchange fill", False, "stop"),
        ("Stop level reached; verify your exchange fill", True, "trail"),
        ("Structural profit target reached", False, "target"),
        ("Maximum holding time reached", False, "time"),
        ("Trade failed to progress within the review window", False, "stale"),
        ("Hold window closing without progress; close on Bitunix", False, "late"),
        ("Completed 15m structure reversed against the trade", False, "structure"),
        ("Estimated liquidation buffer is gone. Close on Bitunix now.", False, "liquidation_buffer"),
        ("Do not wait for a reversal. Live -0.80R; the structural stop is next.", False, "hope"),
        ("Unprotected position is already losing.", False, "unprotected"),
        ("Something new", True, "other"),
    ],
)
def test_exit_class(reason: str, advanced: bool, expected: str) -> None:
    assert load_script().exit_class(reason, advanced) == expected


def test_history_intervals_exclude_four_hour_frame() -> None:
    from bitunix_bot.signal_config import SignalsCfg

    assert load_script().intervals_for(SignalsCfg()) == ("5m", "15m", "1h")


@pytest.mark.parametrize(
    "argv",
    [
        ["--hold-hours", "3"],
        ["--hold-hours", "24"],
        ["--leverage", "101"],
        ["--leverage", "19"],
        ["--days", "31"],
        ["--days", "0"],
        ["--tier-leverage", "126"],
        ["--symbols", "BTC-USDT"],
    ],
)
def test_parser_rejects_out_of_range_values(argv: list[str], tmp_path: Path) -> None:
    module = load_script()
    with pytest.raises(SystemExit) as exc:
        module.main([*argv, "--output", str(tmp_path / "out.json")])
    assert exc.value.code == 2
    assert not (tmp_path / "out.json").exists()


def test_parser_requires_output_and_defaults_to_fifty_x_two_hours() -> None:
    module = load_script()
    with pytest.raises(SystemExit):
        module.build_parser().parse_args([])
    args = module.build_parser().parse_args(["--output", "x.json"])
    assert (args.leverage, args.hold_hours, args.days) == (50, 2, 7)
    assert (args.equity, args.risk_pct, args.spread_pct) == (1000.0, 0.5, 0.03)
    assert (args.funding_rate, args.maintenance_rate, args.tier_leverage) == (
        0.0001,
        0.005,
        100,
    )
    assert args.symbols == "BTCUSDT,ETHUSDT,SOLUSDT,DOGEUSDT,XRPUSDT"
