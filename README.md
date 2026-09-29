# Bitunix Intraday Signals

An alerts-only scanner and Chrome overlay with two profiles. **Trend** (the
default) finds long and short continuation trades from 1h EMA bias, 15m
HH/HL structure and a completed 5m trigger, held 1 or 2 hours at 20-100x
(50x default). **Scalp** fades parabolic exhaustion on 1m bars, long or
short, for a 1h or 2h hold at 1-125x, up to the pair's leverage cap. Neither
opens trades, changes leverage, or closes exchange positions. The one
exception is the overlay **Set Bitunix stop** button, which places a
position-level protective stop.

## Start the backend

```sh
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
export DASHBOARD_PASSWORD='choose-a-private-dashboard-password'
.venv/bin/python run.py
```

`signals.enabled: true` is the shipped default. It overrides legacy `mode: live`
and pump-fade execution flags. A read-only exchange-client guard also rejects
every exchange POST. The old automatic entry/exit loop is bypassed, and old
extension admin POSTs receive HTTP 403. Public market scanning needs no API key.
Legacy account/dashboard reads may still need read credentials.

Load `chrome-extensions/bitunix-momentum-overlay` unpacked through
`chrome://extensions`. Connect it to your HTTPS Railway dashboard and password.
After updating, click **Reload** on the extension in `chrome://extensions`, then
reload the Bitunix tab. An in-page Bitunix refresh is not enough; the header
must show the current version, a drag grip, and ⤢. Reloading the extension
replaces any leftover immovable panel. The new overlay requires the matching
backend release; it does not interpret old pump-fade scores.
Drag the title to move the panel, the bottom-right corner to resize it, or
double-click the header / use ⤢ to restore the default position. The layout is
stored in local Chrome storage.

## Signals

1. 1h supplies directional bias only (20/50 EMA stack and slope). 15m must
   show matching confirmed HH/HL or LH/LL structure. Mixed 15m structure or
   an opposite 1h bias means WAIT. The 4h frame is not fetched.
2. A completed 5m pullback must touch the 15m or 5m EMA20, the 1h EMA20, a
   recent 15m swing, or UTC-session VWAP, then reclaim the prior close in the
   trend direction with at least 1.2× relative volume. Entries do not wait
   for a break of the prior high.
3. Alternatively, a volume-backed 5m breakout (1.5× volume) must precede a
   separate retest that holds the old range boundary, or a 5m impulse of at
   least 1.1 ATR must pull back without breaking its origin and then reclaim.
   Long and short rules are symmetric.
4. The stop sits beyond the setup's 5m structural low/high plus 0.2 ATR(5m).
   It must be at least 1.0 ATR(5m), 0.35 ATR(15m) and 0.20% of entry, at most
   0.60% of entry, and must sit at least 0.25% of entry (or half a 5m ATR)
   inside the estimated isolated liquidation at the planned leverage. The
   nearest structural target (15m and 1h swings, prior-day and session
   high/low) that clears 1.5R after estimated costs inside a travel budget of
   1.5 × ATR(1h) × sqrt(hold minutes / 60) is used. Near swings that fail
   1.5R are skipped instead of blocking the trade. A second target is
   contextual only; the default exit is the first target.
5. The 1h ATR must sit between 0.60% (1h hold) or 0.35% (2h hold) and 1.2%,
   so the travel budget can carry a 1.5R target; BTC usually fails this and
   shows WAIT with that reason. Price may not be more than 2 ATR(1h) past the
   1h EMA20, and funding in the trade direction may not exceed 0.05% per
   print (or open interest up 3% in an hour while extended 1.5 ATR).
   Altcoins additionally require BTC not opposed and non-negative relative
   strength over two hours. Spread (0.04%), 24h USDT volume plus last-hour
   session activity, mark/last basis (0.15%), a 5-minute funding-print
   blackout, planned order size, depth (8× notional), projected funding drag
   (0.10% of notional) and the liquidation buffer must pass. Leverage never
   narrows the stop; it only changes margin and liquidation distance.

The scanner keeps a wide liquid universe of about 80 USDT perpetuals and a hot
set of about 12. Each 15-second refresh fully rescans the hot set — BTC, open or
tracked positions, current WATCH/ENTER names, the displayed top five, and the
most liquid majors — then rotates through about 10 more universe names so the
full list is covered every couple of minutes. Names that print a WATCH or ENTER
are promoted onto the fast lane. The overlay still shows only the ranked top
five. Every trend card uses the same 21-gate checklist
(market, setup, plan, portfolio). Later stages stay marked waiting until the
prior stage prints, so a WATCH does not jump from 4/5 to 5/7. Once a 5m setup
exists, all eight plan gates are scored even if the 1.5R target fails. Market
gates such as mark vs last, the funding-print blackout, extension and crowding
use data the scanner already has. RSI, MACD and news are not added: they are
either redundant with the 1h/15m/5m stack or need history we do not store.
Funding and open interest remain context, not standalone buy/sell triggers;
missing open interest is labeled warming up or unavailable. Missing funding, depth,
or maintenance tiers blocks a new entry. This release does not include a
news/event feed.

## Scalp profile

Select **Scalp** in the overlay's planning settings. It is a separate
evaluator (`bitunix_bot/scalp_short.py`) with its own `signals.scalp` block in
`config.yaml`; the trend profile reads `signals.trend`.

1. **Universe by tier.** Only liquid USDT perpetuals whose exchange leverage
   cap allows the planned leverage are scanned, and the card's "Leverage tier"
   gate re-checks the position tiers. Open interest from each ticker refresh is
   kept for three hours so a one-hour OI delta is available.
2. **Candidate discovery.** A market must be extended (at least +1.5% over 1h,
   +3% over 4h, and 2 hourly ATRs above the 1h EMA20), climactic (a bar in the
   last 15 at 3× the prior 20-bar volume baseline), crowded (funding at or
   above +0.01% per interval, or open interest up at least 1% over the hour),
   and, for alts, have outrun BTC over the last two hours.
3. **Failed-high trigger on 1m bars.** The spike is the highest high of the
   last 30 bars, between 2 and 12 bars old. The last completed bar must close
   below the spike bar's body on a down bar with no new high since the spike.
   A close-weighted volume-delta proxy since the spike must be net selling
   (the public kline feed has no taker split). The stop is the spike high plus
   0.15 ATR; it must be at least 0.5 ATR and at most 0.45% of entry.
4. **Stop inside liquidation.** The scanner estimates the isolated short
   liquidation from the maintenance tier, costs and mark basis, and blocks the
   entry if the planned leverage would liquidate before the stop plus a 0.15%
   buffer. It reports the highest leverage that fits. With a typical 0.4%
   maintenance rate, 100x liquidates about 0.4% above entry after fees, so
   most cards show a lower ceiling; that number is the honest answer.
5. **Mean-reversion targets.** The nearest of session VWAP, the 15m EMA20,
   the 1h EMA20 and the spike base that sits within 1.5 hourly ATR and clears
   2R net of fees, slippage and any negative funding. Shorts receive positive
   funding but that is never counted as reward.
6. **Execution gates.** Spread at most 0.03%, both sides of the book at least
   eight times the planned notional, a three-minute entry window, plus the
   usual fresh data, liquidity, mark basis and funding-print checks.
7. **Management and measurement.** Tracked scalps exit at the hold cap, on
   stop or target, at −0.75R, and after 20 minutes below 0.3R. The 1h trend is
   faded by design and does not latch an exit. Every ENTER alert (both
   profiles) is forward-tested against the completed trigger candles that
   follow: maximum favorable and adverse excursion in R, and the first touch
   of target, stop or estimated liquidation. The summary is in the API
   snapshot under `forward_test` and on the overlay status line, and
   `scripts/backtest_scalp_short.py` replays the profile over recent 1m
   candles of the top 24h gainers.

## Planning and tracking

The overlay's Edit button sets the profile, planning equity, risk per trade,
leverage (20–100x trend, 1–125x scalp), and maximum hold (1 or 2 hours for
both profiles). Initial planning defaults are explicitly hypothetical:
1,000 USDT equity, 0.5% risk, 50x, 2 hours, trend. These are not an exchange
balance. Settings saved under the old `swing` profile load as trend; a hold
outside 1-2 hours becomes 2 hours and a leverage outside 20-100 becomes 50x.

The overlay shows a ranked queue of the top five markets, each with the time
the current state started. WATCH and ENTER alerts are stored with that
timestamp so you can look back. When a setup flips to ENTER, the laptop
speakers say “Trade entry waiting” plus the market and side, followed by the
planned leverage and hold window when those settings are configured (for
example “50 x, 120 minute hold”). When a tracked
trade needs an exchange stop or a latched exit, they say “Set the Bitunix stop
now” or “Close the trade now. Do not wait for a reversal.” Click the panel
once if Chrome blocks speech until a gesture. When the featured setup is about
to change — a higher-ranked market is ready, or the entry window is under 45
seconds — the panel counts down before switching.

Position size accounts for the structural stop plus estimated costs. The scanner
counts funding settlements over the maximum hold using the provider's actual
interval and next-settlement timestamp, then projects the current funding rate.
Expected funding receipts do not subsidize the trade. Fees and slippage remain
configurable estimates in `config.yaml`, and future funding rates can change.

Leverage never narrows a stop. The scanner estimates a leverage ceiling using the
position's maintenance tier, mark/last basis, costs, and a buffer of at least
0.25% of entry or half a 5m ATR (0.15% for scalp). This is an estimate for
isolated margin without extra collateral, not the exchange's exact liquidation
price or a guarantee. Risk-based sizing means 50x changes only the posted
margin and the liquidation distance, never the position size.

- **Track paper trade** records a simulated entry without placing an order.
- **Record my fill** records a fill you already executed, within the confirmed
  entry zone and planned size/risk. Until a Bitunix stop is confirmed, the card
  shows SET STOP instead of HOLD. Imported live positions start the same way.
- **Set Bitunix stop** places the quantity stop-loss the Bitunix ticket shows,
  rounded to that market's tick. One click. If hit, Bitunix closes that size at
  market. This is the only exchange write in signal mode. It cannot open a trade
  or flash-close a position. API keys need Trade permission.
- **Live Bitunix positions** are imported read-only when `BITUNIX_API_KEY` and
  `BITUNIX_SECRET_KEY` are set. The scanner still does not open or flatten
  trades. **Set Bitunix stop** can attach or tighten that ticket stop.
  A vanished position closes that live track automatically.
- Original stop, target, risk and holding limit are frozen when recorded.
- Tracked cards show a live hold/close suggestion, a hold-confidence checklist
  score (not a measured win rate), and a fixed 13-gate close-out list: fresh
  data, stop, target, hold time, drawdown, room to the working stop, liquidation
  buffer, confirmed exchange stop, structure intact (15m), bias intact (1h),
  progress vs review window, session VWAP, and funding carry. Hard EXIT alerts
  latch on stop/target touches, opposite completed 15m structure, failure to
  make 0.3R within 35% of the hold (21 min at 1h, 42 min at 2h), being under
  0.5R at 75% of the hold (45 / 90 min), maximum age (60 / 120 min), a
  missing liquidation buffer (0.25% of entry for trend), −0.75R (do not wait
  for a reversal), and an unprotected position that is already −0.5R.
  Soft failures drop hold confidence and can switch the live suggestion to
  CONSIDER CLOSE without latching an exit. After 1R the stop can advance to
  cover estimated costs; a trailing stop can advance after 1.25R and never
  widen.
  Stop changes are suggestions to apply manually. SET STOP and CLOSE hide the
  confidence bar so a checklist percentage cannot talk you into holding.
- Exit alerts remain latched until you record closure. They do not reverse into
  an opposite entry or assume that an exchange order filled.
- **Record closure** ends tracking only. Estimated net results use the recorded
  exit price and planned costs; they are not exchange-settled P&L.

Tracked exposure is capped at 1.5% of planning equity and two trades in the same
direction. Live imported positions count toward that cap so an open HYPE short
blocks a second HYPE entry alert. Put actual protective stops on Bitunix;
browser alerts are not a replacement for exchange-side protection.

## Data integrity and persistence

Market reads are paced below eight requests per second, cached by timeframe,
invalidated at candle boundaries, and backed off after provider failures. A
provider circuit breaker prevents retry storms. Invalid, missing, incomplete,
or stale data cannot produce a confirmed entry. Tracking shows REVIEW on an
outage; the maximum-hold exit still applies. Candle wicks before entry never
trigger a tracked exit, and trailing stops are not applied retroactively.

Bitunix candle payloads use `quoteVol` for coin volume and `baseVol` for turnover;
tickers use `quoteVol` for USDT turnover. Some opens carry the prior close outside
the traded high/low; normalization includes this opening gap in the range.

Planning settings, trades and deduplicated alert history are stored in SQLite at
`logs/intraday-signals.sqlite3`. Set `SIGNAL_STATE_PATH` to a path on a persistent
Railway volume to retain them across deployments. Without a persistent volume,
container replacement can lose local state. Existing tracks survive ordinary
process restarts. Passwords are stored locally in Chrome, migrated out of sync
storage, and are never sent to the page or an arbitrary dashboard origin.

## Validation

```sh
python3 -m pytest tests/test_e2e.py tests/test_intraday.py tests/test_scalp_short.py -q
node --test tests/extension_worker.test.cjs
# Requires Playwright; CHROME_PATH can override the local Chrome executable.
node tests/extension_ui.cjs
python3 scripts/backtest_intraday.py --days 14 \
  --symbols ETHUSDT,SOLUSDT,DOGEUSDT,XRPUSDT,SUIUSDT \
  --leverage 50 --hold-hours 2 --output /tmp/trend-replay-2h.json
python3 scripts/backtest_scalp_short.py --hours 48 --leverage 100 \
  --symbols auto --top 15 --output /tmp/scalp-short-replay.json
```

The replay is chronological and uses only candles completed at each decision.
It declares its spread, funding, margin and depth assumptions. Missing-data
windows are counted and excluded; an open trade crossing a gap is unresolved,
not assigned an invented result. Ambiguous candles test stops before targets.
Historical book depth, changing funding/tier rules, mark basis, liquidation
paths and actual fills are not available from candles. This is a structural
check, not evidence of profitability. Forward paper observation remains necessary.

## Components and API

- `intraday.py`: pure signal/plan calculations.
- `signal_scanner.py`: public data, freshness, ranking and planning actions.
- `signal_store.py`: persistent tracking and exit state.
- `signal_routes.py`: authenticated `/api/signals` and tracking endpoints.
- `chrome-extensions/bitunix-momentum-overlay`: display and manual records.

`GET /api/signals` returns the latest snapshot without making blocking market
requests. `POST /api/signals/settings`, `/track`, and `/close` update local
planning/tracking state only. `/api/momentum` returns the new snapshot in signal
mode for compatibility detection. All require existing dashboard Basic auth.
The old strategy modules remain for explicit legacy regression/replay use.

Provider references: [candles](https://www.bitunix.com/api-docs/futures/market/get_kline.html),
[funding](https://www.bitunix.com/api-docs/futures/market/get_funding_rate.html),
[depth](https://www.bitunix.com/api-docs/futures/market/get_depth.html),
[position tiers](https://www.bitunix.com/api-docs/futures/position/get_position_tiers.html).
