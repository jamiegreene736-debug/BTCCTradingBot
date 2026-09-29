# Intraday signal redesign — release checks

Scope: manual long/short trend signals from a 1h EMA bias, completed 15m
structure and a completed 5m trigger, held 1-2 hours at 20-100x (50x default),
with an estimated liquidation/leverage-ceiling gate; see "Trend 50x / 60-120
min revision" below. The original 4h/1h/15m, 12/24h, 25x design that the
checklist under it was written against is superseded and kept as history.
Automatic exchange actions are disabled. Initial planning values are
hypothetical.

- [x] Symmetric pullback and breakout/retest engine; structural stops and targets.
- [x] Completed-candle, volume, BTC, funding, spread, depth and maintenance-tier checks.
- [x] Stable 21-gate checklist plus mark/last and funding-print window.
- [x] Persistent manual/paper tracking; latched exits and non-widening trailing stops.
- [x] New overlay, editable planning values, alert history and recorded closures.
- [x] Regression suite, strict typing for new Python modules, worker/browser tests.
- [x] Public Bitunix smoke check: BTC/ETH/SOL returned current WAIT decisions.
- [x] Seven-day BTC/ETH candle replay at 25x under stated assumptions (historical; the trend revision's replay command is below).
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

## Trend 50x / 60-120 min revision

The `swing` profile (12/24h hold, 25-40x, 4h/1h/15m frames) is replaced by
`trend`: the same engine shape (bias -> structure -> completed-candle
continuation trigger -> structural stop -> structural target inside a travel
budget) with every frame one step down, so a trade resolves inside a 60 or
120 minute hold at 50x-class leverage. Alerts-only execution and risk-based
sizing are unchanged. Every threshold below is read from `signals.trend` in
`config.yaml` (`TrendCfg` in `signal_config.py`).

### Frames and settings

- Bias: 1h EMA20/50 stack and slope. Structure: completed 15m HH/HL or LH/LL.
  Trigger: the last completed 5m bar (`trigger_interval: 5m`; `3m` allowed).
  The 4h frame is not fetched for this profile.
- Session VWAP, prior UTC-day high/low and current-session high/low come from
  the 15m frame.
- Planning settings: profile `trend` (default), leverage 20-100 (50 default),
  hold 1 or 2 hours (`hold_minutes` 60 / 120 inside the code), 1,000 USDT
  planning equity, 0.5% risk. Scalp stays `((1, 2), 1, 125)`.
- Cost `c` = `round_trip_fee_pct 0.12 + slippage_pct 0.06 = 0.18%` plus
  projected adverse funding over the hold.
- The card lives 10 minutes after the trigger candle closes
  (`expires_at = trigger open + 300 + 600 s`).

### The 21 gates (fixed order)

| # | Label | Pass condition | Threshold |
|---|---|---|---|
| 1 | Fresh market data | `now - as_of` inside the data-age limit | 60 s |
| 2 | Liquid market | 24h quote volume floor and last four 15m bars' volume at least `session_volume_ratio` x (24h volume / 24) | 10M USDT; 0.5 |
| 3 | Spread | `(ask - bid) / price` | 0.04% |
| 4 | Hold-window volatility | 1h ATR between the hold floor and the cap | 0.60% (1h) / 0.35% (2h) to 1.2% |
| 5 | Mark vs last | `abs(mark - price) / price` | 0.15% |
| 6 | Funding print window | next funding further away than the blackout | 300 s |
| 7 | 1h bias / 15m structure | `ema_bias(1h) == trend(15m)`, both long or both short; on failure every later gate waits (state WAIT) | - |
| 8 | BTC context | BTC 1h bias not opposed and 2h relative strength >= 0 (BTCUSDT passes; missing BTC candles fail) | 2h, >= 0 |
| 9 | Not extended | `sign x (price - EMA20(1h)) / ATR(1h)` | <= 2.0 ATR |
| 10 | Crowding headwind | funding in the trade direction per print, and not (OI up >= 3% over the hour while extended >= 1.5 ATR) | 0.05%; 3.0% / 1.5 ATR |
| 11 | Completed trigger candle | a completed 5m pullback reclaim, impulse continuation (body >= 1.1 ATR5m) or breakout retest (breakout volume >= 1.5x) | - |
| 12 | Volume confirmation | trigger bar relative volume | >= 1.2x |
| 13 | Entry zone | ask (long) / bid (short) inside `[close - 0.30 ATR5m, close + min(0.25 ATR5m, 0.25 x stop distance)]` | - |
| 14 | Stop size | stop distance >= `max(1.0 ATR5m, 0.35 ATR15m, 0.20%)` and stop_pct <= `max_stop_pct` | 0.20% to 0.60% |
| 15 | Target inside hold budget | a structural level within `1.5 x ATR(1h) x sqrt(hold_minutes / 60)` that clears 1.5R net | 1.50 ATR1h at 60 min, 2.12 at 120 min |
| 16 | Reward after costs | `(move - c) / (stop_pct + c) >= min_reward_risk` | 1.5R |
| 17 | Funding drag | projected funding paid over the hold | <= 0.10% of notional |
| 18 | Order size | quantity >= the pair minimum | - |
| 19 | Execution depth | both sides of the book >= notional x ratio | 8.0x |
| 20 | Leverage ceiling | planned leverage <= the highest leverage whose estimated liquidation sits at least the buffer beyond the stop | buffer `max(0.25%, 0.5 ATR5m)`; cap 100x |
| 21 | Tracked exposure | portfolio gate in the scanner | 2 same-direction, 1.5% total risk |

Labels 13-20 are the eight plan gates. Metrics on the card: `trend_1h`,
`trend_15m`, `atr_pct` (5m), `atr_15m_pct`, `hourly_atr_pct`,
`extension_atr`, `session_volume_ratio`, `oi_change_pct`,
`relative_strength_pct`, `relative_volume`, `tier_max_leverage`.

### Liquidation and 1.5R arithmetic

Sizing: `risk_fraction = (stop_pct + c) / 100`;
`notional = min(equity x risk_pct / risk_fraction, equity x leverage x 0.9)`;
`margin = notional / leverage`. Leverage changes margin and the liquidation
line, never size.

Estimated isolated liquidation at leverage `L` with maintenance rate `m`:
`entry x (1 - sign / L + sign x c) / (1 - sign x m)`. Ceiling (long):
`floor(1 / ((stop + buffer) x (1 - m) + c + m))`; short uses `(1 + m)`.
`select_targets` keeps levels with `0 < sign x (level - entry) <= travel`,
nearest first, whose net R >= 1.5. Minimum move for 1.5R net is
`1.5 x stop_pct + 2.5 x c = 1.5 x stop + 0.45%`.

Alt, entry 100.000 long, ATR1h 0.60%, tier maintenance 0.5%, 1,000 USDT,
0.5% risk, 50x, no funding print inside the hold (ATR15m ~0.30%, ATR5m
~0.173%):

- Structural stop from the 5m pullback low: 99.650 (0.35%). Floor
  `max(0.173, 0.105, 0.20) = 0.20%` <= 0.35 <= 0.60: Stop size passes.
- `risk_fraction 0.53%`; notional 943.40; margin 18.87; risk 5.00 USDT =
  26.5% of posted margin. At 20x the margin is 47.17, at 100x 9.43; notional
  and the 5.00 USDT risk are the same.
- Liquidation at 50x: `100 x (1 - 0.02 + 0.0018) / 0.995 = 98.673`
  (1.327% away). Room `99.650 - 98.673 = 0.977%` >= buffer
  `max(0.25, 0.087) = 0.25%`: fits. Ceiling
  `floor(1 / ((0.0035 + 0.0025) x 0.995 + 0.0018 + 0.005)) = 78x`.
  Short mirror: liquidation `100 x 1.0182 / 1.005 = 101.313`, ceiling 77x.
- At 100x the liquidation is 0.322% away, the 0.35% stop fails and the card
  reports "estimated ceiling 78x". At 20x the liquidation is 4.34% away.
- Target: minimum move `1.5 x 0.35 + 0.45 = 0.975%`, so a level >= 100.975.
  Travel at 2h is `1.5 x 0.60 x 1.414 = 1.273%`; a 15m swing high at 101.00
  qualifies and gives `(1.00 - 0.18) / 0.53 = 1.55R` net. Travel at 1h is
  0.90% < 0.975%: no target at the 1h hold with this stop; a 0.30% stop
  needs 0.90% and fits exactly. That is why the 1h ATR floor is 0.60% at
  the 1h hold.
- Entry zone: `[100.000 - 0.052, 100.000 + min(0.043, 0.0875)]` =
  `[99.948, 100.043]`.
- Ceiling table (maintenance 0.5%, buffer 0.25%): stop 0.20% -> 88x,
  0.30 -> 81x, 0.35 -> 78x, 0.40 -> 75x, 0.50 -> 70x, 0.60 -> 65x. On a
  1.0% maintenance tier: 0.35 -> 56x, 0.50 -> 52x, 0.60 -> 49x (WATCH at
  50x; the card says so). On a 0.4% tier: 0.35 -> 84x, 0.60 -> 70x.

BTC, ATR1h 0.25%, tier maintenance 0.3% (ATR5m ~0.072%, ATR15m ~0.125%):

- Hold-window volatility: 0.25 < 0.35 (2h floor), so the card is WAIT with
  `1h ATR 0.25% must sit in 0.35-1.2% to carry a 1.5R target inside 120 min
  at 50x`. The smallest admissible stop is the 0.20% floor, needing a
  `1.5 x 0.20 + 0.45 = 0.75%` move, while the 2h travel budget is
  `1.5 x 0.25 x 1.414 = 0.53%` (1h: 0.375%). No target can exist, so the
  gate refuses up front.
- For reference if the ATR were >= 0.35%: at a 0.20% stop `risk_fraction
  0.38%`, notional 1315.8, margin 26.3, loss 5.00 = 19% of margin;
  liquidation `100 x 0.9818 / 0.997 = 98.475` (1.525% away); ceiling
  `floor(1 / (0.0045 x 0.997 + 0.0018 + 0.003)) = 107x`, reported as 100x
  (profile cap). A 0.15% stop gives 113x.

Floor derivation: with `stop ~ 0.5 x ATR1h`, the budget must satisfy
`1.5 x A x sqrt(h / 60) >= 0.75 x A + 0.45`, so `A >= 0.328%` at 120 min
(config 0.35) and `A >= 0.60%` at 60 min (config 0.60).

### Exits (minutes of the hold, R of the initial risk)

| Exit | Rule | 1h hold | 2h hold |
|---|---|---|---|
| Time stop | `age >= hold_hours x 3600` -> EXIT "Maximum holding time reached" | 60 min | 120 min |
| Entry expiry | card flips to WAIT 600 s after the trigger candle closes | 10 min | 10 min |
| Stale review | `max(600, round(0.35 x hold_minutes) x 60)`; EXIT "Trade failed to progress within the review window" when best progress < 0.3R | 21 min | 42 min |
| Late-hold exit | after the stale check: EXIT "Hold window closing without progress; close on Bitunix" when `age >= 0.75 x hold` and live R < 0.5R | 45 min | 90 min |
| Breakeven | stop moves to cover estimated costs at 1.0R | 1.0R | 1.0R |
| Trailing | from 1.25R the stop trails one initial risk behind the last 5m close, only tighter | 1.25R | 1.25R |
| Hope exit | live R <= -0.75R -> EXIT "Do not wait for a reversal" | -0.75R | -0.75R |
| Unprotected | manual trade without a confirmed exchange stop at <= -0.5R | -0.5R | -0.5R |
| Liquidation room | estimated room to liquidation < 0.25% of entry (scalp 0.15%) | 0.25% | 0.25% |
| Structure reversal | completed 15m structure (`trend_15m`) opposite the trade -> EXIT "Completed 15m structure reversed against the trade" | - | - |
| Stop / target touch | completed 5m bars plus live price; reasons unchanged | - | - |

The late exit is checked only after the stale check, so a trade that never
made 0.3R exits at the stale time, not the late time. Hold cards print
`{n} of {hold_minutes} min hold left` and `need 0.3R within 21m / 42m`. The
13 hold checks are unchanged in count; `1h structure` is now `Structure
intact` (15m) and `4h bias` is `Bias intact` (1h).

### Settings migration (swing -> trend)

`SignalSettings.from_dict` maps `profile: "swing"` (and a missing profile) to
`trend`, then rebases only what the new range rejects: a hold outside
(1, 2) becomes 2 hours; a leverage outside 20-100 becomes 50x; equity and
risk are kept. So `{leverage 25, hold_hours 24, profile "swing"}` loads as
trend / 25x / 2h and `{leverage 10, hold_hours 12}` as trend / 50x / 2h.
Rows already saying `trend` with an invalid hold or leverage still raise.
`scalp_short` continues to alias to `scalp`. The overlay accepts `swing` and
renders the Trend strip; the strip prints the migrated values on first
render. Imported exchange positions older than the new hold flip to EXIT
"Maximum holding time reached" immediately after the upgrade, and that state
is sticky. Backend, `config.yaml`, extension 1.8.0 and fixtures ship in one
commit; an old overlay against the new backend fails Save with the backend's
range text.

An imported position with no matching card gets the profile's tightest stop
and a liquidation estimate from the pair's real maintenance tier (the same
hourly tier read the scanner already makes; one extra request for a pair
outside the universe), with the leverage ceiling from the same fit. Only when
the exchange cannot supply tiers is a maintenance rate assumed, at most 1% and
never more than keeps the estimate one buffer beyond that stop at the
position's leverage, so a guessed tier cannot latch a liquidation EXIT by
itself; if even a 0% rate leaves the stop inside the buffer the alarm holds
for every tier.

Removed top-level `signals` keys (now under `signals.trend` or gone):
`min_reward_risk, relative_volume_min, breakout_volume_min, stop_atr_buffer,
max_spread_pct, min_depth_ratio, liquidation_buffer_pct, hope_exit_r,
stale_trade_hours, stale_progress_r, trailing_activate_r, breakeven_at_r,
min_hourly_atr_pct, max_hourly_atr_pct, max_target_atr_multiple,
max_target_4h_atr_multiple, impulse_atr_min, max_funding_cost_pct`. An
unedited `config.yaml` with those keys raises `TypeError` at startup.

### Backtest

```sh
python3 scripts/backtest_intraday.py --days 14 --symbols ETHUSDT,SOLUSDT,DOGEUSDT,XRPUSDT,SUIUSDT --leverage 50 --hold-hours 2 --output /tmp/trend-replay-2h.json
python3 scripts/backtest_intraday.py --days 14 --symbols ETHUSDT,SOLUSDT,DOGEUSDT,XRPUSDT,SUIUSDT --leverage 50 --hold-hours 1 --output /tmp/trend-replay-1h.json
python3 scripts/backtest_intraday.py --days 14 --symbols ETHUSDT,SOLUSDT,DOGEUSDT,XRPUSDT,SUIUSDT --leverage 50 --hold-hours 2 --maintenance-rate 0.01 --output /tmp/trend-replay-mmr1.json
python3 scripts/backtest_intraday.py --days 14 --symbols ETHUSDT,SOLUSDT,DOGEUSDT,XRPUSDT,SUIUSDT --leverage 100 --hold-hours 2 --output /tmp/trend-replay-100x.json
```

The replay steps on completed 5m bars, fetches only 5m/15m/1h, fills small
gaps, and assumes one maintenance tier (`--maintenance-rate`, default 0.005;
`--tier-leverage` 100), `--spread-pct` 0.03 and `--funding-rate` 0.0001.
Per closed trade it records `minutes_held`, `stop_pct`, `max_leverage`,
`liquidation_distance_pct`, `max_adverse_pct`,
`adverse_share_of_liquidation`, `liquidation_touched` and the exit class.
The report aggregates `closed_count`, `entries_per_symbol_day`,
`exit_reasons`, `watch_blockers`, `median_minutes_held`, `share_time_stop`,
`liquidation_touches`, `worst_adverse_share_of_liquidation`, `hit_rate`,
`average_net_r`, `estimated_net_usdt` and `share_plans_ceiling_ge_50`.

Acceptance criteria (geometry and gate behaviour, not edge):

1. `liquidation_touches == 0` in every run and
   `worst_adverse_share_of_liquidation < 0.6`. Non-negotiable; if violated,
   lower `max_stop_pct` to 0.50 before touching anything else.
2. Every closed trade has `stop_pct <= 0.60` and `minutes_held <= hold`;
   `median_minutes_held < hold`; `share_time_stop <= 0.30`.
3. `entries_per_symbol_day` between 0.3 and 3 on the 2h run. Below 0.3, the
   `watch_blockers` histogram must name one dominant gate (Target inside hold
   budget or Hold-window volatility), not Leverage ceiling.
4. The 1h run produces fewer entries than the 2h run and no
   `minutes_held > 60`.
5. The `--maintenance-rate 0.01` run shows `share_plans_ceiling_ge_50`
   falling and Leverage ceiling rising in the blocker histogram; the
   `--leverage 100` run makes Leverage ceiling the top blocker.
6. BTCUSDT produces zero entries on days when its 1h ATR is under 0.35%;
   `unresolved_trades_due_to_gaps` stays near zero.
7. `average_net_r` and `estimated_net_usdt` are reported. A positive value is
   not required to accept the design, but `average_net_r < -0.3` over 14
   days on five alts is a reason to revisit `breakeven_at_r`,
   `trailing_activate_r` or `stale_hold_fraction`, never the survival rules.

### What the trader should know

- Leverage does not enlarge the position. Risk-based sizing gives about
  640-1,300 USDT notional on 1,000 USDT equity at 0.5% risk. 50x only sets
  the margin (about 10-26 USDT) and the liquidation line. A stop-out costs
  19-39% of posted margin and never more than 5 USDT.
- The liquidation line is an estimate. It uses the tier maintenance rate,
  0.18% cost and the mark basis at evaluation. It ignores funding accrual,
  ADL and insurance effects. On 1% maintenance tiers a 0.60% stop does not
  fit 50x; the ceiling reads 49x. Treat any ceiling below your leverage as
  "use less". Verify the exchange's liquidation price on every position.
- Cost drag dominates. 0.18% round trip is 30-45% of the stop. The 1.5R
  floor is the minimum that leaves positive expectancy near a 45% hit rate.
  Use limit entries inside the zone. The 0.06% slippage assumption is
  optimistic for market orders at 50x.
- BTC and other pairs with 1h ATR under 0.35% will mostly sit at WAIT with
  the volatility reason shown. That is correct behaviour.
- Nothing here measures hit rate. The forward test and backtest resolve
  first touch or expiry and ignore breakeven, trail, stale and late exits.
  Compare exit-reason histograms, not raw hit rate.
- Migration is silent. Saved `swing` rows load as trend / 2h, and 50x if the
  old leverage was outside 20-100. Imported positions older than the new
  hold exit as "Maximum holding time reached" right after the upgrade and
  stay latched. Ship backend, `config.yaml`, extension 1.8.0 and fixtures
  together.
- Timing depends on polling. Stale (21 / 42 min), late (45 / 90 min) and
  time (60 / 120 min) exits fire on the next scan or overlay poll. A closed
  laptop delays them.
- Gap-filled zero-volume 5m bars depress the 20-bar volume baseline on quiet
  minutes. The session-activity term and the 10M/24h floor limit this but do
  not remove it.
- Funding on 1h-interval pairs can add two prints inside 120 minutes. The
  0.10% cap blocks obvious cases. The rate is not re-checked after the card
  is shown.
- The sqrt travel budget and the 35% / 75% exit fractions are heuristics.
  The backtest's `entries_per_symbol_day`, `share_time_stop` and exit
  histogram are the checks that they are neither unreachable nor permissive.

## 24h / 25-40x methodology revision

Superseded by the Trend 50x / 60-120 min revision above; kept as history.

The first seven-day BTC/ETH replay produced no entries. Dual confirmed 4h and
1h swing structure was the dominant blocker: a 4h pivot needs two closed bars
on each side (16 hours) and often prints after a 12/24h hold is already over.
Nearest-target selection also rejected valid 2R trades when a nearby swing sat
inside the cost-adjusted stop.

This revision keeps alerts-only execution and the same risk accounting:

- 4h is EMA bias only; 1h still requires confirmed HH/HL or LH/LL.
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
rotation WAIT rows stay out of the published queue.

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
