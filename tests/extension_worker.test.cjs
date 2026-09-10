const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
const source = fs.readFileSync(require('node:path').join(__dirname, '../chrome-extensions/bitunix-momentum-overlay/background.js'), 'utf8');

function worker() {
  const messages = [], requests = [];
  const local = { dashboardUrl: 'https://example.up.railway.app', password: 'secret' };
  const chrome = {
    runtime: { id: 'extension-id', onMessage: { addListener: fn => messages.push(fn) }, openOptionsPage() {} },
    storage: { local: { get: async () => local, set: async () => {} }, sync: { get: async () => ({}), remove: async () => {} }, onChanged: { addListener() {} } },
    tabs: { query: async () => [], sendMessage: async () => {} },
    alarms: { create() {}, onAlarm: { addListener() {} } },
  };
  const context = vm.createContext({ chrome, URL, TextEncoder, btoa, AbortSignal, console,
    fetch: async (url, options) => { requests.push({ url, options }); return { ok: true, json: async () => ({ strategy: 'intraday', mode: 'alerts_only' }) }; },
  });
  vm.runInContext(source, context);
  return { context, requests, message: messages[0] };
}
test('dashboard credentials are constrained to the allowed HTTPS origin', async () => {
  const { context } = worker();
  for (const url of ['http://example.up.railway.app', 'https://evil.com', 'https://example.up.railway.app.evil.com', 'https://a:b@example.up.railway.app', 'https://example.up.railway.app/path']) {
    assert.throws(() => vm.runInContext(`validDashboardUrl(${JSON.stringify(url)})`, context));
  }
  assert.equal(vm.runInContext("validDashboardUrl('https://example.up.railway.app/')", context), 'https://example.up.railway.app');
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
