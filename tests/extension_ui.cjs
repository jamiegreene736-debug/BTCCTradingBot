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
    await page.addStyleTag({ path: path.join(root, 'content.css') });
    await page.addScriptTag({ path: path.join(root, 'content.js') });
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
    await page.locator('[data-action="collapse"]').click();
    assert.equal(await page.locator('#bis-body').isVisible(), false);
    assert.deepEqual(errors, []);
    assert.equal(await page.evaluate(() => window.messages.some(m => m.type === 'close-symbol')), false);
    console.log('Browser checks passed: long/short cards, paper tracking, settings, persistent details, mobile bounds, stale data, escaped content, collapse, no legacy trade messages.');
  } finally { await browser.close(); }
}
main().catch(error => { console.error(error); process.exitCode = 1; });
