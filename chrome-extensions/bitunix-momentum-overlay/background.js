// Authenticated requests stay in the service worker; no exchange trading routes.
const POLL_MS = 5000;
let latest = null, fetchedAt = 0, inFlight = null;
let settings = { dashboardUrl: '', password: '' };

function validDashboardUrl(raw) {
  const url = new URL(raw);
  if (url.protocol !== 'https:' || !url.hostname.endsWith('.up.railway.app') || url.username || url.password || url.search || url.hash || !['', '/'].includes(url.pathname)) {
    throw new Error('Use your HTTPS Railway dashboard address, without a path.');
  }
  return url.origin;
}
async function loadSettings() {
  const local = await chrome.storage.local.get(['dashboardUrl', 'password']);
  const old = await chrome.storage.sync.get(['dashboardUrl', 'password']);
  settings = { dashboardUrl: local.dashboardUrl || old.dashboardUrl || '', password: local.password || old.password || '' };
  if (old.password || old.dashboardUrl) {
    await chrome.storage.local.set(settings);
    await chrome.storage.sync.remove(['dashboardUrl', 'password']);
  }
}
let settingsReady = loadSettings();
async function request(path, body) {
  await settingsReady;
  if (!settings.dashboardUrl || !settings.password) throw new Error('Open Settings and connect your dashboard.');
  const origin = validDashboardUrl(settings.dashboardUrl);
  const bytes = new TextEncoder().encode('admin:' + settings.password);
  const response = await fetch(origin + path, {
    method: body === undefined ? 'GET' : 'POST',
    headers: { Authorization: 'Basic ' + btoa(String.fromCharCode(...bytes)), 'Content-Type': 'application/json' },
    body: body === undefined ? undefined : JSON.stringify(body),
    signal: AbortSignal.timeout(8000), cache: 'no-store', redirect: 'error',
  });
  const payload = await response.json().catch(() => ({}));
  if (!response.ok) throw new Error(payload.error || (response.status === 401 ? 'Dashboard password rejected.' : `Dashboard returned ${response.status}.`));
  return payload;
}
async function refresh() {
  if (inFlight) return inFlight;
  inFlight = (async () => {
    try {
      const payload = await request('/api/signals');
      if (payload.strategy !== 'intraday' || payload.mode !== 'alerts_only') throw new Error('Update the backend to the intraday signals release.');
      latest = payload;
    } catch (error) { latest = { ...(latest?.strategy === 'intraday' ? latest : {}), error: error.message || 'Dashboard unavailable' }; }
    fetchedAt = Date.now();
    const tabs = await chrome.tabs.query({ url: 'https://*.bitunix.com/*' });
    await Promise.allSettled(tabs.map(tab => chrome.tabs.sendMessage(tab.id, { type: 'signals-update', payload: latest, fetchedAt })));
    return { payload: latest, fetchedAt };
  })().finally(() => { inFlight = null; });
  return inFlight;
}
chrome.runtime.onMessage.addListener((message, sender, respond) => {
  if (sender.id !== chrome.runtime.id) return false;
  if (message.type === 'open-options') {
    chrome.runtime.openOptionsPage(); respond({ ok: true }); return false;
  }
  if (message.type === 'request-latest' || message.type === 'force-refresh') {
    const result = !latest || Date.now() - fetchedAt >= POLL_MS || message.type === 'force-refresh'
      ? refresh() : Promise.resolve({ payload: latest, fetchedAt });
    result.then(respond).catch(error => respond({ payload: { error: error.message } }));
    return true;
  }
  const paths = { 'save-planning': '/api/signals/settings', 'track-entry': '/api/signals/track', 'close-track': '/api/signals/close' };
  if (Object.hasOwn(paths, message.type)) {
    request(paths[message.type], message.body).then(async result => {
      await refresh(); respond(result);
    }).catch(error => respond({ ok: false, error: error.message }));
    return true;
  }
  return false;
});
chrome.storage.onChanged.addListener((changes, area) => {
  if (area === 'local' && (changes.dashboardUrl || changes.password)) {
    settingsReady = loadSettings(); latest = null; settingsReady.then(refresh);
  }
});
// Alarms survive MV3 worker suspension; open tabs also refresh every five seconds.
chrome.alarms.create('intraday-refresh', { periodInMinutes: 0.5 });
chrome.alarms.onAlarm.addListener(alarm => { if (alarm.name === 'intraday-refresh') refresh(); });
settingsReady.then(refresh);
