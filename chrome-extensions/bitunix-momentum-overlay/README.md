# Bitunix Intraday Signals — Chrome extension v1.8.2

Long and short entry, hold, review and exit alerts for trades held 1-2 hours
at 50x-class leverage. The backend computes the strategy; this extension displays it.
No extension action sends an exchange order or modifies a position.

1. Deploy/start the matching backend with `signals.enabled: true`.
2. Open `chrome://extensions`, enable Developer mode, and Load unpacked this folder.
3. Open Settings, enter the HTTPS Railway dashboard URL and dashboard password,
   and click **Save and test connection**. Wait for **Connected**.
   To skip this step, copy `config.local.example.json` to `config.local.json`
   in this folder and fill in the URL and password. The extension reads it on
   startup whenever nothing is saved yet, so a fresh Load unpacked connects on
   its own. The file is git-ignored; saved Settings still take precedence.
4. Reload the Bitunix tab. For an existing installation, reload the extension first.
5. Edit the displayed profile, planning equity, risk, leverage and maximum
   holding time. **Trend** runs on 1h EMA bias, 15m structure and a completed
   5m continuation trigger for a 1 or 2 hour hold at 20-100x (50x default).
   The stop is the 5m structural level, capped at 0.60%, and must sit at least
   0.25% inside the estimated isolated liquidation; the card reports the
   highest leverage that still fits. **Scalp** fades parabolic exhaustion on
   1m bars, long or short, for a 1h or 2h hold up to the pair's leverage cap
   (125x maximum); its stop sits just past the failed high or low and must fit
   inside the estimated liquidation distance. Leverage never tightens the stop.

## If the panel is empty

- **Pump Fade Radar** or version **0.3.16** means old files or an old tab are
  still loaded. Reload the extension from this folder in `chrome://extensions`,
  then reload the Bitunix tab. The popup must say **Bitunix Intraday Signals**,
  version **1.8.2**. Unpacked extensions do not refresh themselves after a git pull.
  Click **Reload** on this extension in `chrome://extensions`, then reload the
  Bitunix tab. A Bitunix in-page refresh is not enough. The panel header must
  show **v1.8.2**, a drag grip, and ⤢. Drag the title or grip to move; drag the
  bottom-right corner to resize. Empty header chrome lets Bitunix menus
  (timeframes, short/long tickets) receive clicks through the overlay.
- Use the complete extension folder from one release. Mixing the old manifest
  and panel with the new worker breaks message delivery and omits its alarms permission.
- The panel displays **Connecting** immediately. A stalled or disconnected worker
  shows reload instructions within 12 seconds, with Settings and Retry buttons.
- Settings and the popup check the actual connection. A saved URL/password does
  not establish that the backend exists or has the right version. Missing endpoints,
  wrong passwords, outages and incompatible responses show their own messages.
- **Connected; waiting for complete market data** means the backend is reachable
  but has not produced usable market data yet. No entry is enabled during an outage.

The header shows a grip and the extension version. Drag the title to move, the
bottom-right corner to resize, or double-click the header / use ⤢ to restore
the default spot. Reloading the extension replaces any leftover immovable
panel from an older script. Size and position persist in local Chrome storage.

The top-five queue lists the current ranked markets with the time each state
started. A countdown warning appears before the featured card switches to the
next setup. Recent alerts keep WATCH and ENTER rows with full timestamps.
Under the "what it needs" price line, each fresh row and the card show
**Next 5 min ≈**: the last hour's drift on trigger-candle closes extrapolated
five minutes, with a band of one trigger ATR either side and the drift as a
percentage. It is anchored to the live price and recomputed every scan. Treat
it as a volatility envelope around recent momentum, not a forecast of direction.

The card shows WAIT, WATCH LONG/SHORT or ENTER LONG/SHORT, an entry zone,
structural stop, profit target, estimated net reward/risk, planning size,
planned leverage, hold window, time-stop and leverage ceiling, plus the stop's
distance inside the estimated liquidation and the share of posted margin one
stop-out costs. Open the checklist for the underlying evidence. Every card
lists the same gates for its profile (21 trend, 20 scalp), grouped as
market / setup / plan / book.

Trend rules (all in minutes of the chosen hold, 60 or 120):

- Market: fresh data, 10M USDT/24h liquidity plus last-hour session activity,
  spread ≤ 0.04%, 1h ATR inside 0.60-1.2% (1h hold) or 0.35-1.2% (2h hold),
  mark within 0.15% of last, no funding print inside 5 minutes, 1h EMA bias
  equal to the completed 15m HH/HL structure, BTC not opposed with non-negative
  2h relative strength, price no more than 2 hourly ATR past the 1h EMA20, and
  no crowding (funding ≤ 0.05% per print against the trade; OI not up 3% with
  the price 1.5 ATR extended).
- Setup: a completed 5m pullback reclaim, impulse continuation or breakout
  retest with 1.2x baseline volume.
- Plan: entry inside a zone that adds at most 25% to planned risk; stop at the
  5m structural level minus 0.2 ATR5m, at least 1 ATR5m / 0.35 ATR15m / 0.20%
  and at most 0.60%; a structural target that clears 1.5R net of 0.18% costs
  inside 1.5 × ATR1h × sqrt(hold / 60); funding drag ≤ 0.10%; eight times depth;
  and the stop at least 0.25% (or half a 5m ATR) inside the estimated isolated
  liquidation at the planned leverage, which sets the reported ceiling.
- Exits: stale review at 35% of the hold (21 / 42 min) below 0.3R, late-hold
  exit at 75% (45 / 90 min) below 0.5R, hard time-stop at 100%, breakeven at
  1R, trailing stop from 1.25R on completed 5m closes, hope exit at -0.75R,
  liquidation-buffer exit under 0.25% of room, and a completed 15m structure
  reversal against the trade.

In the scalp profile a short needs an extended coin (1h and 4h gain, distance
above the 1h EMA20 in hourly ATRs), a climactic volume bar, crowded longs
(positive funding or open interest built over the last hour), and an alt that
outran BTC. The setup is a completed 1m close back through a spike bar's body
with no new high (failed high) and net selling since the spike. A long is the
mirror: a capitulation dump, climax volume, crowded shorts (negative funding or
open-interest build), an alt that fell harder than BTC, and a completed close
back above a spike low's body with no new low (failed low) and net buying. Both
sides are scored every scan and the card shows the more advanced one; a WAIT
card spells out both thresholds. The plan needs the stop inside the liquidation distance at the planned
leverage, a VWAP / EMA / spike-base target inside 1.5 hourly ATR that clears
2R net, a 0.03% spread cap and eight times depth. Tracked scalps exit at the
hold cap or after 20 minutes below 0.3R; the 1h trend is faded by design and
does not trigger an exit. The status line shows the running forward test of
every ENTER alert (resolved/count, hit rate, average R, liquidation touches).
Stale data disables entry tracking. The Best setup selector ranks eligible markets.
The laptop speakers say “Trade entry waiting” plus the market and side when a
setup flips to ENTER, followed by the planned leverage and hold window when
those settings are configured (for example “50 x, 120 minute hold”), “Set the
Bitunix stop now” when a live or recorded fill
has no confirmed exchange stop, and “Close the trade now. Do not wait for a
reversal.” when an exit latches. Click the overlay once if Chrome blocks speech
until a gesture.

Track paper trade simulates a record. Record my fill records a trade you already
executed, within the currently confirmed plan. **Set Bitunix stop** then places
the quantity stop-loss the Bitunix ticket shows, rounded to that market's tick;
if it hits, Bitunix closes that size at market. That is the only exchange write.
It does not open a trade or flatten immediately. You can still mark a stop you
already placed by hand.
Tracked trades show a live hold/close suggestion (HOLD, CONSIDER CLOSE, SET STOP,
CLOSE, or REVIEW). SET STOP and CLOSE hide the hold-confidence bar and show the
working stop in a get-out banner. The close-out list is 13 gates, including
whether the exchange stop is confirmed. Imported live positions start as SET STOP
until the stop is placed or confirmed.
Open futures positions are imported read-only from the backend when API keys are
configured, labeled **LIVE**, and show mark and unrealized P&L. Record closure
only ends overlay tracking; a still-open Bitunix position is imported again on
the next scan.

Alert history does not claim trading P&L. Recorded closures show estimated net
results with paper/user-recorded/live labels; fees and funding are not actual settlement.

Connection credentials use local Chrome storage. Older sync settings migrate
automatically. The service worker allows only HTTPS Railway origins, refuses
redirects, and has no legacy close-symbol action. No script gets the dashboard
password through page messages.

See the root README for exact rules, costs, persistence, limitations and tests.
