# Bitunix Intraday Signals — Chrome extension v1.4.0

Long and short entry, hold, review and exit alerts for trades lasting up to
12–24 hours. The backend computes the strategy; this extension displays it.
No extension action sends an exchange order or modifies a position.

1. Deploy/start the matching backend with `signals.enabled: true`.
2. Open `chrome://extensions`, enable Developer mode, and Load unpacked this folder.
3. Open Settings, enter the HTTPS Railway dashboard URL and dashboard password,
   and click **Save and test connection**. Wait for **Connected**.
4. Reload the Bitunix tab. For an existing installation, reload the extension first.
5. Edit the displayed planning equity, risk, leverage and maximum holding time.
   The scanner is built for isolated 25-40x and a 12h or 24h hold. Leverage
   never tightens the stop.

## If the panel is empty

- **Pump Fade Radar** or version **0.3.16** means old files or an old tab are
  still loaded. Reload the extension from this folder in `chrome://extensions`,
  then reload the Bitunix tab. The popup must say **Bitunix Intraday Signals**,
  version **1.4.0**. Unpacked extensions do not refresh themselves after a git pull.
  Click **Reload** on this extension in `chrome://extensions`, then reload the
  Bitunix tab. A Bitunix in-page refresh is not enough. The panel header must
  show **v1.4.0**, a drag grip, and ⤢. Drag the title to move; drag the
  bottom-right corner to resize.
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

The card shows WAIT, WATCH LONG/SHORT or ENTER LONG/SHORT, an entry zone,
structural stop, profit target, estimated net reward/risk, planning size and
estimated leverage ceiling. Open the checklist for the underlying evidence.
Stale data disables entry tracking. The Best setup selector ranks eligible markets.

Track paper trade simulates a record. Record my fill records a trade you already
executed, within the currently confirmed plan. Neither button places an order.
Tracked trades display HOLD, REVIEW or EXIT; stops and closures on Bitunix remain
manual. Open futures positions are imported read-only from the backend when API
keys are configured, labeled **LIVE**, and show mark and unrealized P&L.
Set protective stops on the exchange. Record closure only ends overlay tracking;
a still-open Bitunix position is imported again on the next scan.

Alert history does not claim trading P&L. Recorded closures show estimated net
results with paper/user-recorded/live labels; fees and funding are not actual settlement.

Connection credentials use local Chrome storage. Older sync settings migrate
automatically. The service worker allows only HTTPS Railway origins, refuses
redirects, and has no legacy close-symbol action. No script gets the dashboard
password through page messages.

See the root README for exact rules, costs, persistence, limitations and tests.
