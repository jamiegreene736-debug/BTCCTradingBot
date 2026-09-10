// Run with NODE_PATH pointing to an installed Playwright package directory.
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const os = require('node:os');
const { chromium } = require('playwright');

async function main() {
  const fixture = JSON.parse(fs.readFileSync(process.argv[2] || path.join(__dirname, 'fixtures/intraday_snapshot.json'), 'utf8'));
  const root = path.resolve(__dirname, '../chrome-extensions/bitunix-momentum-overlay');
  const macChrome = '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome';
  const output = path.join(os.tmpdir(), 'bitunix-extension-qa');
  fs.mkdirSync(output, { recursive: true });
  const browser = await chromium.launch({ headless: true, executablePath: process.env.CHROME_PATH || (fs.existsSync(macChrome) ? macChrome : chromium.executablePath()) });
  try {
    const page = await browser.newPage({ viewport: { width: 1440, height: 1000 } });
    const errors = [];
    page.on('pageerror', error => errors.push(error.message));
    await page.setContent('<html><body style="background:#080e17;color:#b2c3d9;font:16px sans-serif;padding:35px"><h1>Trading workspace</h1><p>Isolated extension fixture — no exchange connection</p></body></html>');
    await page.evaluate(data => {
      window.testPayload = data;
      window.messages = [];
      window.listeners = [];
      window.chrome = { runtime: {
        sendMessage: async message => {
          window.messages.push(message);
          if (['save-planning', 'track-entry', 'close-track'].includes(message.type)) return { ok: true };
          return { payload: structuredClone(window.testPayload) };
        },
        onMessage: { addListener: listener => window.listeners.push(listener) },
      } };
      const now = Math.floor(Date.now() / 1000);
      for (const row of Object.values(data.symbols)) { row.as_of = now; row.plan.expires_at = now + 900; }
    }, fixture);
    // A slow worker must never leave only an empty header on screen.
    await page.evaluate(() => {
      window.originalSend = chrome.runtime.sendMessage;
      chrome.runtime.sendMessage = () => new Promise(resolve => { window.releaseStartup = resolve; });
    });
    await page.addStyleTag({ path: path.join(root, 'content.css') });
    await page.addScriptTag({ path: path.join(root, 'content.js') });
    assert.match(await page.locator('#bis-status').textContent(), /Connecting/);
    await page.evaluate(() => {
      chrome.runtime.sendMessage = window.originalSend;
      window.releaseStartup({ payload: window.testPayload });
    });
    await page.locator('.bis-state').waitFor();
    assert.equal(await page.locator('.bis-state').textContent(), 'ENTER LONG');
    await page.locator('[data-action="paper"]').click();
    assert.match(await page.locator('.bis-modal').textContent(), /No order is submitted/);
    await page.locator('.bis-modal [type="submit"]').click();
    await page.locator('.bis-modal').waitFor({ state: 'detached' });
    assert.equal(await page.evaluate(() => window.messages.find(m => m.type === 'track-entry').body.kind), 'paper');
    await page.locator('[data-action="planning"]').click();
    await page.locator('[name="leverage"]').fill('30');
    await page.locator('[name="hold_hours"]').selectOption('12');
    await page.locator('.bis-modal [type="submit"]').click();
    await page.locator('.bis-modal').waitFor({ state: 'detached' });
    const saved = await page.evaluate(() => window.messages.find(m => m.type === 'save-planning').body);
    assert.equal(saved.leverage, 30); assert.equal(saved.hold_hours, 12);
    await page.locator('#bis-symbol').selectOption('ETHUSDT');
    assert.equal(await page.locator('.bis-state').textContent(), 'ENTER SHORT');
    await page.locator('#bis-card summary').click();
    await page.locator('[data-action="refresh"]').click();
    assert.equal(await page.locator('#bis-card details').getAttribute('open'), '');
    await page.locator('#bis-card summary').click();
    await page.screenshot({ path: path.join(output, 'desktop.png'), fullPage: true });
    await page.setViewportSize({ width: 390, height: 844 });
    assert.equal(await page.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth), true);
    assert.equal(await page.locator('#bis-panel').evaluate(el => el.scrollWidth <= el.clientWidth), true);
    await page.screenshot({ path: path.join(output, 'mobile.png'), fullPage: true });
    await page.evaluate(() => {
      window.testPayload.symbols.ETHUSDT.as_of -= 120;
      window.listeners[0]({ type: 'signals-update', payload: window.testPayload });
    });
    assert.equal(await page.locator('.bis-state').textContent(), 'WAIT');
    assert.equal(await page.locator('[data-action="manual"]').isDisabled(), true);
    await page.evaluate(() => {
      window.testPayload.symbols.ETHUSDT.as_of = Math.floor(Date.now() / 1000);
      window.testPayload.symbols.ETHUSDT.reasons = ['<img src=x onerror="window.injected=true">'];
      window.listeners[0]({ type: 'signals-update', payload: window.testPayload });
    });
    assert.equal(await page.locator('#bis-card img').count(), 0);
    // Incomplete and old backend responses must fail visibly and block entries.
    await page.evaluate(() => window.listeners[0]({ type: 'signals-update', payload: { symbols: {} } }));
    assert.match(await page.locator('#bis-status').textContent(), /incomplete|incompatible/i);
    assert.equal(await page.locator('[data-action="manual"]').count(), 0);
    await page.evaluate(() => window.listeners[0]({ type: 'signals-update', payload: { error: 'Open Settings and connect your dashboard.' } }));
    assert.match(await page.locator('#bis-status').textContent(), /Open Settings/);
    assert.equal(await page.locator('[data-action="settings"]').last().isVisible(), true);
    await page.evaluate(() => { chrome.runtime.sendMessage = async () => undefined; });
    await page.locator('[data-action="refresh"]').click();
    assert.match(await page.locator('#bis-status').textContent(), /Reload.*Bitunix/i);
    await page.evaluate(() => {
      chrome.runtime.sendMessage = window.originalSend;
      window.listeners[0]({ type: 'signals-update', payload: window.testPayload });
    });
    assert.equal(await page.locator('.bis-state').textContent(), 'ENTER SHORT');
    await page.locator('[data-action="collapse"]').click();
    assert.equal(await page.locator('#bis-body').isVisible(), false);
    assert.deepEqual(errors, []);
    assert.equal(await page.evaluate(() => window.messages.some(m => m.type === 'close-symbol')), false);
    const options = await browser.newPage();
    options.on('pageerror', error => errors.push(error.message));
    await options.setContent(fs.readFileSync(path.join(root, 'options.html'), 'utf8').replace(/<script[^>]*><\/script>/g, ''));
    await options.evaluate(() => {
      window.saved = {};
      window.result = { error: 'Signal endpoint not found. Deploy the intraday backend.' };
      window.chrome = {
        storage: {
          local: { get: async () => ({}), set: async value => { window.saved = value; } },
          sync: { get: async () => ({ dashboardUrl: 'https://old.up.railway.app', password: 'old' }), remove: async () => {} },
        },
        runtime: { sendMessage: async () => ({ payload: window.result }) },
      };
    });
    await options.addScriptTag({ path: path.join(root, 'options.js') });
    assert.equal(await options.locator('#dashboardUrl').inputValue(), 'https://old.up.railway.app');
    await options.locator('#dashboardUrl').fill('https://new.up.railway.app');
    await options.locator('#password').fill('new');
    await options.locator('#save').click();
    assert.match(await options.locator('#status').textContent(), /Settings saved.*endpoint not found/);
    assert.equal(await options.evaluate(() => window.saved.dashboardUrl), 'https://new.up.railway.app');
    await options.evaluate(() => { window.result = { strategy: 'intraday', mode: 'alerts_only', status: { ready: true } }; });
    await options.locator('#save').click();
    assert.match(await options.locator('#status').textContent(), /Connected.*Reload/);
    await options.locator('#dashboardUrl').fill('https://untrusted.example');
    await options.locator('#save').click();
    assert.match(await options.locator('#status').textContent(), /HTTPS Railway/);
    assert.equal(await options.locator('#save').isEnabled(), true);
    const popup = await browser.newPage();
    popup.on('pageerror', error => errors.push(error.message));
    await popup.setContent(fs.readFileSync(path.join(root, 'popup.html'), 'utf8').replace(/<script[^>]*><\/script>/g, ''));
    await popup.evaluate(() => {
      window.chrome = { runtime: {
        getManifest: () => ({ version: '1.0.1' }),
        sendMessage: async () => ({ payload: { error: 'Cannot reach the dashboard.' } }),
      } };
    });
    await popup.addScriptTag({ path: path.join(root, 'popup.js') });
    assert.match(await popup.locator('#status').textContent(), /Cannot reach/);
    assert.equal(await popup.locator('#version').textContent(), 'Version 1.0.1');
    const stalled = await browser.newPage();
    stalled.on('pageerror', error => errors.push(error.message));
    await stalled.clock.install();
    await stalled.setContent('<html><body></body></html>');
    await stalled.evaluate(() => {
      window.chrome = { runtime: { sendMessage: () => new Promise(() => {}), onMessage: { addListener() {} } } };
    });
    await stalled.addStyleTag({ path: path.join(root, 'content.css') });
    await stalled.addScriptTag({ path: path.join(root, 'content.js') });
    await stalled.clock.runFor(13000);
    assert.match(await stalled.locator('#bis-status').textContent(), /Reload the extension/);
    assert.equal(await stalled.locator('[data-action="settings"]').last().isVisible(), true);
    await stalled.screenshot({ path: path.join(output, 'connection-error.png') });
    assert.deepEqual(errors, []);
    console.log('Browser checks passed: startup, malformed data, missing settings, disconnected worker, recovery, long/short cards, tracking, settings, mobile, stale data, escaping, collapse, no exchange actions.');
  } finally { await browser.close(); }
}
main().catch(error => { console.error(error); process.exitCode = 1; });
