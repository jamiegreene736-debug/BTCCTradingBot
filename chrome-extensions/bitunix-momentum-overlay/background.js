// Service worker — owns all network I/O and the polling loop. Content scripts
// ask for the latest scores via runtime messaging instead of fetching directly,
// because:
//   1. host_permissions on the manifest let the SW bypass page CORS
//   2. one polling loop is shared across all open Bitunix tabs
//   3. settings live here in chrome.storage and are read once at startup

const NORMAL_POLL_INTERVAL_MS = 5000;
const FAST_POLL_INTERVAL_MS = 2000;
const FETCH_TIMEOUT_MS = 5000;

let latest = null;          // most recent /api/momentum payload, or { error }
let lastFetchAt = 0;
let pollTimer = null;
let currentPollIntervalMs = NORMAL_POLL_INTERVAL_MS;
let fetchInFlight = null;
let settings = { dashboardUrl: '', password: '' };

async function loadSettings() {
  const stored = await chrome.storage.sync.get(['dashboardUrl', 'password']);
  settings.dashboardUrl = (stored.dashboardUrl || '').replace(/\/+$/, '');
  settings.password = stored.password || '';
}

function basicAuthHeader(password) {
  // Username is hardcoded "admin" by the bot's dashboard.
  return 'Basic ' + btoa('admin:' + password);
}

function stageFromDecision(decision) {
  const action = String(decision?.action || '').toLowerCase();
  const setup = String(decision?.setup || '').toLowerCase();
  const stage = String(decision?.setup_stage || decision?.setupStage || '').toLowerCase();
  if (action === 'short' && setup === 'parabolic_pump_fade') return 'short';
  if (stage === 'pump_watch') return 'watch';
  if (stage === 'pump_building') return 'building';
  return 'hunting';
}

function payloadNeedsFastPoll(payload) {
  if (!payload || payload.error) return false;
  const best = payload.best_candidate || payload.bestCandidate;
  if (['short', 'watch', 'building'].includes(String(best?.stage || '').toLowerCase())) return true;
  const symbols = payload.symbols || {};
  return Object.values(symbols).some((row) => (
    ['short', 'watch', 'building'].includes(stageFromDecision(row?.decision || row?.recommendation || row?.next_1h || row?.next1h))
  ));
}

function desiredPollIntervalMs() {
  return payloadNeedsFastPoll(latest) ? FAST_POLL_INTERVAL_MS : NORMAL_POLL_INTERVAL_MS;
}

function ensurePollInterval() {
  const desired = desiredPollIntervalMs();
  if (pollTimer && desired === currentPollIntervalMs) return;
  if (pollTimer) clearInterval(pollTimer);
  currentPollIntervalMs = desired;
  pollTimer = setInterval(fetchOnce, currentPollIntervalMs);
}

async function doFetchOnce() {
  if (!settings.dashboardUrl || !settings.password) {
    latest = { error: 'not_configured', message: 'Open the extension settings and add your dashboard URL + password.' };
    lastFetchAt = Date.now();
    broadcast();
    ensurePollInterval();
    return;
  }
  const url = settings.dashboardUrl + '/api/momentum';
  const ctrl = new AbortController();
  const t = setTimeout(() => ctrl.abort(), FETCH_TIMEOUT_MS);
  try {
    const resp = await fetch(url, {
      method: 'GET',
      headers: { 'Authorization': basicAuthHeader(settings.password) },
      signal: ctrl.signal,
      cache: 'no-store',
    });
    if (resp.status === 401) {
      latest = { error: 'auth', message: 'Dashboard rejected the password (401).' };
    } else if (resp.status === 503) {
      latest = { error: 'disabled', message: 'Dashboard returned 503 — DASHBOARD_PASSWORD env var unset on Railway?' };
    } else if (!resp.ok) {
      latest = { error: 'http', message: `Dashboard returned HTTP ${resp.status}.` };
    } else {
      latest = await resp.json();
    }
  } catch (e) {
    latest = { error: 'network', message: 'Could not reach dashboard: ' + (e?.message || String(e)) };
  } finally {
    clearTimeout(t);
    lastFetchAt = Date.now();
  }
  // Push to any listening content scripts so the UI updates instantly.
  broadcast();
  ensurePollInterval();
}

async function fetchOnce() {
  if (fetchInFlight) return fetchInFlight;
  fetchInFlight = doFetchOnce().finally(() => {
    fetchInFlight = null;
  });
  return fetchInFlight;
}

async function postDashboardJson(path, body) {
  if (!settings.dashboardUrl || !settings.password) {
    return { ok: false, error: 'not_configured', message: 'Dashboard URL/password missing.' };
  }
  const url = settings.dashboardUrl + path;
  const ctrl = new AbortController();
  const t = setTimeout(() => ctrl.abort(), FETCH_TIMEOUT_MS);
  try {
    const resp = await fetch(url, {
      method: 'POST',
      headers: {
        'Authorization': basicAuthHeader(settings.password),
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(body || {}),
      signal: ctrl.signal,
      cache: 'no-store',
    });
    let payload = {};
    try { payload = await resp.json(); } catch {}
    if (!resp.ok) {
      return {
        ok: false,
        error: 'http',
        message: payload?.error || payload?.message || `Dashboard returned HTTP ${resp.status}.`,
        payload,
      };
    }
    return { ok: true, payload, ...payload };
  } catch (e) {
    return { ok: false, error: 'network', message: e?.message || String(e) };
  } finally {
    clearTimeout(t);
  }
}

function broadcast() {
  chrome.tabs.query({ url: 'https://*.bitunix.com/*' }, (tabs) => {
    for (const tab of tabs) {
      chrome.tabs.sendMessage(tab.id, {
        type: 'momentum-update',
        payload: latest,
        fetchedAt: lastFetchAt,
        pollMs: currentPollIntervalMs,
      })
        .catch(() => { /* tab may not have content script ready yet */ });
    }
  });
}

function startPolling() {
  if (pollTimer) clearInterval(pollTimer);
  pollTimer = null;
  ensurePollInterval();
  fetchOnce();
}

chrome.runtime.onInstalled.addListener(async () => {
  await loadSettings();
  startPolling();
});
chrome.runtime.onStartup.addListener(async () => {
  await loadSettings();
  startPolling();
});

// Settings changed → reload immediately so the user sees feedback.
chrome.storage.onChanged.addListener(async (changes, area) => {
  if (area !== 'sync') return;
  await loadSettings();
  fetchOnce();
});

// Content scripts ask for the latest snapshot when they mount.
chrome.runtime.onMessage.addListener((msg, sender, sendResponse) => {
  if (msg?.type === 'request-latest') {
    const stale = !latest || (Date.now() - lastFetchAt) > Math.max(1500, desiredPollIntervalMs() * 1.25);
    if (stale) {
      fetchOnce().then(() => sendResponse({ payload: latest, fetchedAt: lastFetchAt, pollMs: currentPollIntervalMs }));
      return true;
    }
    sendResponse({ payload: latest, fetchedAt: lastFetchAt, pollMs: currentPollIntervalMs });
    return false;
  }
  if (msg?.type === 'force-refresh') {
    fetchOnce().then(() => sendResponse({ payload: latest, fetchedAt: lastFetchAt, pollMs: currentPollIntervalMs }));
    return true; // async response
  }
  if (msg?.type === 'close-symbol') {
    postDashboardJson('/api/admin/close-symbol', {
      symbol: msg.symbol,
      positionId: msg.positionId || null,
      source: 'bitunix-momentum-extension',
    }).then((resp) => {
      fetchOnce().finally(() => sendResponse(resp));
    });
    return true; // async response
  }
  if (msg?.type === 'open-options') {
    // chrome.runtime.openOptionsPage isn't available from content scripts in
    // MV3 — has to be called from an extension context like this SW.
    chrome.runtime.openOptionsPage?.();
    sendResponse({ ok: true });
    return false;
  }
});

// Cold-start path for when the SW is woken by a message before
// onStartup/onInstalled have fired this session.
loadSettings().then(startPolling);
