# Intraday signal redesign — release checks

Scope: replace the pump-fade overlay with manual long/short signals, completed
4h/1h/15m rules, 12/24h tracking and an estimated leverage/risk gate. Automatic
exchange actions are disabled. Initial planning values are hypothetical.

- [x] Symmetric pullback and breakout/retest engine; structural stops and targets.
- [x] Completed-candle, volume, BTC, funding, spread, depth and maintenance-tier checks.
- [x] Stable 19-gate checklist plus mark/last and funding-print window.
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
Live Bitunix positions are imported read-only when API keys are present. The
scanner still does not open or flatten trades. **Set Bitunix stop** can attach
a position-level protective stop. Recorded trade state needs a persistent
volume for continuity across Railway deployments.

## 24h / 25-40x methodology revision

The first seven-day BTC/ETH replay produced no entries. Dual confirmed 4h and
1h swing structure was the dominant blocker: a 4h pivot needs two closed bars
on each side (16 hours) and often prints after a 12/24h hold is already over.
Nearest-target selection also rejected valid 2R trades when a nearby swing sat
inside the cost-adjusted stop.

This revision keeps alerts-only execution and the same risk accounting:

- 4h is EMA bias only. 1h confirmed swings still win, but a matching 1h EMA
  stack can admit a side before those swings print, and a confirmed 1h
  structure is enough when 4h is mixed. Opposite 1h structure still blocks.
- 15m pullbacks reclaim the prior close (not the prior high) and may use the
  15m EMA. Impulse continuation is a third completed-candle trigger.
- Structural targets skip sub-2R swings and stay inside a hold-window travel
  budget. Projected funding above 0.40% of notional blocks a new entry.
- 1h ATR must sit between 0.12% and 5% so 25-40x isolated margin is usable
  and fees do not dominate.
- Tracked stops can cover estimated costs after 1R; the 1.5R trail still
  never widens.

These changes increase the number of valid 12/24h candidates. They do not
create a measured edge. Forward paper observation is still required before
treating signals as profitable.

## Live hold suggestion

Open tracked trades now carry a live hold/close suggestion, a hold-confidence
checklist score, and a fixed 13-gate close-out list. Hard EXIT rules latch on
the original stop/target/time/structure/stale gates plus liquidation-buffer
loss, −0.75R (do not wait for a reversal), and an unprotected losing position.
Missing exchange-stop confirmation shows SET STOP instead of HOLD. Soft
failures only change the suggestion to CONSIDER CLOSE. The score is a
checklist percentage, not a measured win rate; SET STOP and CLOSE hide it.

## Exit discipline — 1.5.0

The blank-panel and hold-suggestion work still left one failure mode open:
waiting for a bounce without a Bitunix stop, then getting liquidated. This
release does not place that stop. It makes the missing stop and the “get out”
call impossible to treat as optional.

- Recording a real fill can confirm a stop you already placed, or leave the
  card on SET STOP until you click **Set Bitunix stop**.
- Imported live positions start as SET STOP until that confirmation.
- **Set Bitunix stop** is the only exchange write in signal mode: a
  position-level Bitunix stop at the working stop. It does not open or flatten
  a trade. Existing tighter stops are left alone; wider stops are tightened.
- The overlay speaks “Set the Bitunix stop now” and “Close the trade now. Do
  not wait for a reversal.”
- Hard EXIT now includes estimated liquidation-buffer loss and −0.75R, with
  copy that a reversal will not beat liquidation.

## Plan checklist — 1.5.2

WATCH cards used to show Plan 0/8 forever because later gates were left as
unscored waiting rows, which looked the same as failed. Waiting is now a
separate state in the overlay. After a completed 15m setup, funding, size,
depth and leverage are scored even when the 2R target is missing. ENTER still
requires every gate, including the 15m trigger.

## One-click protective stop — 1.5.1

Waiting for a bounce without an exchange stop was still one extra step too
many. The overlay now places that stop for you. Trade permission is required
on the Bitunix API key. Entries, leverage changes and flash-closes stay
blocked.

## Two-tier scan — wide universe, hot set

The overlay still shows five ranked setups. The backend no longer limits
discovery to those twelve names. Tickers build a liquid universe of up to 80
USDT perps. Every refresh fully evaluates the hot set (BTC, live/tracked
positions, current WATCH/ENTER names, the displayed queue, and top-volume
fillers up to `max_symbols`) and rotates `evaluate_batch` more universe names.
A name that prints WATCH or ENTER is promoted onto the 15-second lane. Stale
rotation WAIT rows stay out of the published queue. A confirmed ENTER is not
dropped to WAIT when the 15m entry window closes; it stays listed as WATCH
until alignment breaks. A live ENTER is not handed off to a WATCH-only name.
A failed re-read keeps the last good decision for two data-age windows.

## Signal queue, timestamps and handoff

The snapshot now includes a ranked `queue` of up to five markets, `state_since`
on each decision, and a `handoff` countdown when the featured card is about
to change. WATCH alerts are persisted alongside ENTER and exit alerts, keyed
per symbol / state / setup / completed bar so refreshes do not duplicate them.
History rows keep a unix `time` plus setup and side. The overlay formats
those as local date-time and a relative age, and holds the current featured
setup for 20 seconds after a better rank appears.

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
