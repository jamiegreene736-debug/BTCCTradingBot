const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
const source = fs.readFileSync(require('node:path').join(__dirname, '../chrome-extensions/bitunix-momentum-overlay/background.js'), 'utf8');
const manifest = JSON.parse(fs.readFileSync(require('node:path').join(__dirname, '../chrome-extensions/bitunix-momentum-overlay/manifest.json'), 'utf8'));
const fixture = JSON.parse(fs.readFileSync(require('node:path').join(__dirname, 'fixtures/intraday_snapshot.json'), 'utf8'));

function worker({ response = { ok: true, json: async () => structuredClone(fixture) }, configured = true } = {}) {
  const messages = [], requests = [];
  const local = configured ? { dashboardUrl: 'https://example.up.railway.app', password: 'secret' } : {};
  const flags = { reloaded: 0 };
  const chrome = {
    runtime: {
      id: 'extension-id',
      onMessage: { addListener: fn => messages.push(fn) },
      onInstalled: { addListener() {} },
      openOptionsPage() {},
      getManifest: () => ({ version: manifest.version }),
      reload() { flags.reloaded += 1; },
    },
    storage: { local: {
      get: async keys => {
        if (keys == null) return { ...local };
        const list = Array.isArray(keys) ? keys : [keys];
        return Object.fromEntries(list.map(key => [key, local[key]]));
      },
      set: async value => { Object.assign(local, value); },
    }, sync: { get: async () => ({}), remove: async () => {} }, onChanged: { addListener() {} } },
    tabs: { query: async () => [], sendMessage: async () => {}, reload: async () => {} },
    // Mirror Chrome: an omitted manifest permission does not expose the API.
    alarms: manifest.permissions.includes('alarms') ? { create() {}, onAlarm: { addListener() {} } } : undefined,
  };
  const context = vm.createContext({ chrome, URL, TextEncoder, btoa, AbortSignal, console,
    fetch: async (url, options) => { requests.push({ url, options }); return typeof response === 'function' ? response() : response; },
  });
  vm.runInContext(source, context);
  return { context, requests, local, message: messages[0], flags };
}
test('dashboard credentials are constrained to the allowed HTTPS origin', async () => {
  const { context } = worker();
  for (const url of ['http://example.up.railway.app', 'https://evil.com', 'https://example.up.railway.app.evil.com', 'https://a:b@example.up.railway.app', 'https://example.up.railway.app/path']) {
    assert.throws(() => vm.runInContext(`validDashboardUrl(${JSON.stringify(url)})`, context));
  }
  assert.equal(vm.runInContext("validDashboardUrl('https://example.up.railway.app/')", context), 'https://example.up.railway.app');
});
test('missing configuration produces a useful status without making requests', async () => {
  const { context, requests, local } = worker({ configured: false });
  const { payload } = await vm.runInContext('refresh()', context);
  assert.match(payload.error, /Open Settings/);
  assert.equal(requests.length, 0);
  assert.equal(local.dashboardUrl, 'https://btcc-trading-bot-production.up.railway.app');
  assert.equal(local.password, undefined);
});
test('saved password uses the production Railway URL when no URL is stored', async () => {
  const { context, requests, local } = worker({ configured: false });
  local.password = 'secret';
  await vm.runInContext('settingsReady = loadSettings(); await settingsReady; refresh()', context);
  const { payload } = await vm.runInContext('refresh()', context);
  assert.equal(payload.error, undefined);
  assert.equal(requests.at(-1).url, 'https://btcc-trading-bot-production.up.railway.app/api/signals');
  assert.equal(local.dashboardUrl, 'https://btcc-trading-bot-production.up.railway.app');
  assert.equal(local.password, 'secret');
});
test('a newer backend version reloads the overlay without dropping credentials', async () => {
  const snapshot = structuredClone(fixture);
  snapshot.extension_version = '99.0.0';
  const { context, local, flags } = worker({
    response: { ok: true, json: async () => structuredClone(snapshot) },
  });
  await vm.runInContext('refresh()', context);
  assert.equal(flags.reloaded, 1);
  assert.equal(local.dashboardUrl, 'https://example.up.railway.app');
  assert.equal(local.password, 'secret');
});
test('connection failures are actionable, including non-JSON error pages', async () => {
  const cases = [
    [{ ok: false, status: 401, json: async () => ({}) }, /password rejected/i],
    [{ ok: false, status: 404, json: async () => { throw new Error('HTML'); } }, /endpoint not found/i],
    [{ ok: false, status: 502, json: async () => { throw new Error('HTML'); } }, /backend deployment/i],
    [{ ok: true, json: async () => { throw new Error('HTML'); } }, /did not return signal data/i],
    [{ ok: true, json: async () => ({ strategy: 'pump_fade' }) }, /Update the backend/],
    [{ ok: true, json: async () => ({ strategy: 'intraday', mode: 'alerts_only' }) }, /incomplete/i],
    [() => { throw new Error('Network unavailable'); }, /Cannot reach the dashboard/],
    [() => { const error = new Error(); error.name = 'TimeoutError'; throw error; }, /timed out/],
  ];
  for (const [response, expected] of cases) {
    const { context } = worker({ response });
    const { payload } = await vm.runInContext('refresh()', context);
    assert.match(payload.error, expected);
    assert.equal(payload.symbols, undefined);
  }
});
test('an outage retains the last validated plan with an error that disables entry', async () => {
  let failed = false;
  const { context } = worker({ response: () => {
    if (failed) throw new Error('Network unavailable');
    return { ok: true, json: async () => structuredClone(fixture) };
  } });
  await vm.runInContext('refresh()', context);
  failed = true;
  const { payload } = await vm.runInContext('refresh()', context);
  assert.equal(payload.symbols.BTCUSDT.plan.entry, fixture.symbols.BTCUSDT.plan.entry);
  assert.match(payload.error, /Cannot reach/);
});
test('connection test uses newly saved credentials after an older request completes', async () => {
  const { context, local, requests, message } = worker();
  await vm.runInContext('refresh()', context);
  local.dashboardUrl = 'https://new-backend.up.railway.app';
  local.password = 'updated';
  const result = await new Promise(resolve => message({ type: 'check-connection' }, { id: 'extension-id' }, resolve));
  assert.equal(result.payload.error, undefined);
  assert.equal(requests.at(-1).url, 'https://new-backend.up.railway.app/api/signals');
  assert.equal(requests.at(-1).options.headers.Authorization, 'Basic ' + btoa('admin:updated'));
});
test('legacy close requests and foreign senders cannot reach the backend', () => {
  const { message } = worker();
  assert.equal(message({ type: 'close-symbol' }, { id: 'extension-id' }, () => {}), false);
  assert.equal(message({ type: 'track-entry' }, { id: 'foreign' }, () => {}), false);
});
test('place-stop posts only the tracking id', async () => {
  const { requests, message } = worker();
  const result = await new Promise(resolve => message(
    { type: 'place-stop', body: { id: 'exchange:HYPE1' } },
    { id: 'extension-id' },
    resolve,
  ));
  assert.equal(result.error, undefined);
  assert.equal(requests.some(item => item.url.endsWith('/api/signals/place-stop')), true);
  assert.equal(requests.find(item => item.url.endsWith('/api/signals/place-stop')).options.method, 'POST');
});
test('confirm-stop posts only the tracking id', async () => {
  const { requests, message } = worker();
  const result = await new Promise(resolve => message(
    { type: 'confirm-stop', body: { id: 'exchange:HYPE1' } },
    { id: 'extension-id' },
    resolve,
  ));
  assert.equal(result.error, undefined);
  assert.equal(requests.some(item => item.url.endsWith('/api/signals/confirm-stop')), true);
  assert.equal(requests.find(item => item.url.endsWith('/api/signals/confirm-stop')).options.method, 'POST');
});
test('polling is deduplicated and refuses redirects', async () => {
  const { context, requests } = worker();
  await vm.runInContext('Promise.all([refresh(), refresh(), refresh()])', context);
  assert.equal(requests.length, 1);
  assert.equal(requests[0].url, 'https://example.up.railway.app/api/signals');
  assert.equal(requests[0].options.redirect, 'error');
  assert.equal(requests[0].options.method, 'GET');
});
