# Bitunix Intraday Signals — Chrome extension v1.0.1

Long and short entry, hold, review and exit alerts for trades lasting up to
12–24 hours. The backend computes the strategy; this extension displays it.
No extension action sends an exchange order or modifies a position.

1. Deploy/start the matching backend with `signals.enabled: true`.
2. Open `chrome://extensions`, enable Developer mode, and Load unpacked this folder.
3. Open Settings, enter the HTTPS Railway dashboard URL and dashboard password,
   and click **Save and test connection**. Wait for **Connected**.
4. Reload the Bitunix tab. For an existing installation, reload the extension first.
5. Edit the displayed planning equity, risk, leverage and maximum holding time.

## If the panel is empty

- **Pump Fade Radar** or version **0.3.16** means old files or an old tab are
  still loaded. Reload the extension from this folder in `chrome://extensions`,
  then reload the Bitunix tab. The popup must say **Bitunix Intraday Signals**,
  version **1.0.1**. Unpacked extensions do not refresh themselves after a git pull.
- Use the complete extension folder from one release. Mixing the old manifest
  and panel with the new worker breaks message delivery and omits its alarms permission.
- The panel displays **Connecting** immediately. A stalled or disconnected worker
  shows reload instructions within 12 seconds, with Settings and Retry buttons.
- Settings and the popup check the actual connection. A saved URL/password does
  not establish that the backend exists or has the right version. Missing endpoints,
  wrong passwords, outages and incompatible responses show their own messages.
- **Connected; waiting for complete market data** means the backend is reachable
  but has not produced usable market data yet. No entry is enabled during an outage.

The card shows WAIT, WATCH LONG/SHORT or ENTER LONG/SHORT, an entry zone,
structural stop, profit target, estimated net reward/risk, planning size and
estimated leverage ceiling. Open the checklist for the underlying evidence.
Stale data disables entry tracking. The Best setup selector ranks eligible markets.

Track paper trade simulates a record. Record my fill records a trade you already
executed, within the currently confirmed plan. Neither button places an order.
Tracked trades display HOLD, REVIEW or EXIT; stops and closures on Bitunix remain
manual. Set protective stops on the exchange. Record closure only ends tracking.

Alert history does not claim trading P&L. Recorded closures show estimated net
results with paper/user-recorded labels; fees and funding are not actual settlement.
The extension does not import unrecorded positions from your exchange account.

Connection credentials use local Chrome storage. Older sync settings migrate
automatically. The service worker allows only HTTPS Railway origins, refuses
redirects, and has no legacy close-symbol action. No script gets the dashboard
password through page messages.

See the root README for exact rules, costs, persistence, limitations and tests.
