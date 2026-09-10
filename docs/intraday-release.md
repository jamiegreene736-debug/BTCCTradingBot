# Intraday signal redesign — release checks

Scope: replace the pump-fade overlay with manual long/short signals, completed
4h/1h/15m rules, 12/24h tracking and an estimated leverage/risk gate. Automatic
exchange actions are disabled. Initial planning values are hypothetical.

- [x] Symmetric pullback and breakout/retest engine; structural stops and targets.
- [x] Completed-candle, volume, BTC, funding, spread, depth and maintenance-tier checks.
- [x] Persistent manual/paper tracking; latched exits and non-widening trailing stops.
- [x] New overlay, editable planning values, alert history and recorded closures.
- [x] Regression suite, strict typing for new Python modules, worker/browser tests.
- [x] Public Bitunix smoke check: BTC/ETH/SOL returned current WAIT decisions.
- [x] Seven-day BTC/ETH candle replay at 25x under stated assumptions.
- [ ] Establish forward paper performance before interpreting signals as an edge.

The seven-day replay run on 2026-09-10 had 1,344 symbol/time evaluations:
929 lacked complete source history, and the remaining 415 returned WAIT.
No trades were opened. This supplies no win-rate or profitability evidence;
it demonstrates that missing history and unaligned conditions do not force entries.
The exact rolling window varies when rerun. Source gaps were confirmed in raw API
responses; no candles were interpolated or fabricated.

The shipped baseline exits at the first structural target. A second target is
context only; partial-profit schemes are not enabled without separate testing.
There is no news feed or automatic exchange-position import. Recorded trade
state needs a persistent volume for continuity across Railway deployments.

## Extension startup repair — 1.0.1

The blank-panel report on 2026-09-10 came from a mixed local installation:
the 0.3.16 manifest, content script and stylesheet had been restored alongside
the 1.0.0 background worker. The old panel listened for `momentum-update` while
the new worker broadcasts `signals-update`, and the old manifest omitted `alarms`.
The affected local files exactly matched commit `2d09f3c`; main remained correct.

The repair restores a consistent local release and adds an immediate loading
state, a bounded worker wait, recovery buttons, malformed-response handling,
and real connection checks in Settings and the popup. Regression tests cover
the manifest's actual permissions, missing credentials, HTTP/non-JSON errors,
old and incomplete responses, timeouts, recovery and retained stale plans.
The backend is still required. A successful local/browser test does not prove
that a Railway deployment exists or that the user's loaded tab has refreshed.
