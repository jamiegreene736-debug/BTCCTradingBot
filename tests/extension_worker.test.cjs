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
  const chrome = {
    runtime: { id: 'extension-id', onMessage: { addListener: fn => messages.push(fn) }, openOptionsPage() {} },
    storage: { local: { get: async () => local, set: async () => {} }, sync: { get: async () => ({}), remove: async () => {} }, onChanged: { addListener() {} } },
    tabs: { query: async () => [], sendMessage: async () => {} },
    // Mirror Chrome: an omitted manifest permission does not expose the API.
    alarms: manifest.permissions.includes('alarms') ? { create() {}, onAlarm: { addListener() {} } } : undefined,
  };
  const context = vm.createContext({ chrome, URL, TextEncoder, btoa, AbortSignal, console,
    fetch: async (url, options) => { requests.push({ url, options }); return typeof response === 'function' ? response() : response; },
  });
  vm.runInContext(source, context);
  return { context, requests, local, message: messages[0] };
}
test('dashboard credentials are constrained to the allowed HTTPS origin', async () => {
  const { context } = worker();
  for (const url of ['http://example.up.railway.app', 'https://evil.com', 'https://example.up.railway.app.evil.com', 'https://a:b@example.up.railway.app', 'https://example.up.railway.app/path']) {
    assert.throws(() => vm.runInContext(`validDashboardUrl(${JSON.stringify(url)})`, context));
  }
  assert.equal(vm.runInContext("validDashboardUrl('https://example.up.railway.app/')", context), 'https://example.up.railway.app');
});
test('missing configuration produces a useful status without making requests', async () => {
  const { context, requests } = worker({ configured: false });
  const { payload } = await vm.runInContext('refresh()', context);
  assert.match(payload.error, /Open Settings/);
  assert.equal(requests.length, 0);
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
test('polling is deduplicated and refuses redirects', async () => {
  const { context, requests } = worker();
  await vm.runInContext('Promise.all([refresh(), refresh(), refresh()])', context);
  assert.equal(requests.length, 1);
  assert.equal(requests[0].url, 'https://example.up.railway.app/api/signals');
  assert.equal(requests[0].options.redirect, 'error');
  assert.equal(requests[0].options.method, 'GET');
});
