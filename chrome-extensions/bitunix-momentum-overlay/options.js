async function load() {
  try {
    const stored = await chrome.storage.local.get(["dashboardUrl", "password"]);
    const old = await chrome.storage.sync.get(["dashboardUrl", "password"]);
    document.getElementById("dashboardUrl").value = stored.dashboardUrl || old.dashboardUrl || "";
    document.getElementById("password").value = stored.password || old.password || "";
  } catch { document.getElementById('status').textContent = 'Could not load settings. Reload the extension and reopen Settings.'; }
}

async function save() {
  const dashboardUrl = document.getElementById("dashboardUrl").value.trim().replace(/\/+$/, "");
  const password = document.getElementById("password").value;
  const status = document.getElementById("status");
  const button = document.getElementById('save');
  button.disabled = true;
  status.textContent = 'Checking connection…';
  let timer;
  try {
    const url = new URL(dashboardUrl);
    if (url.protocol !== 'https:' || !url.hostname.endsWith('.up.railway.app') || url.username || url.password || url.search || url.hash || !['', '/'].includes(url.pathname)) {
      throw new Error('Use an HTTPS Railway dashboard URL without a path.');
    }
    if (!password) throw new Error('Enter your dashboard password.');
    await chrome.storage.local.set({ dashboardUrl: url.origin, password });
    await chrome.storage.sync.remove(['dashboardUrl', 'password']);
    const response = await Promise.race([
      chrome.runtime.sendMessage({ type: 'check-connection' }),
      new Promise((_, reject) => { timer = setTimeout(() => reject(new Error('Settings saved, but the connection check timed out. Reload the extension, then retry.')), 20000); }),
    ]);
    if (!response?.payload) throw new Error('Settings saved, but the worker did not respond. Reload the extension, then retry.');
    if (response.payload.error) throw new Error('Settings saved. ' + response.payload.error);
    if (response.payload.strategy !== 'intraday' || response.payload.mode !== 'alerts_only') throw new Error('Settings saved, but the backend is outdated. Deploy the intraday release.');
    status.textContent = response.payload.status?.ready
      ? 'Connected. Reload your Bitunix tab to see signals.'
      : 'Connected. The scanner is waiting for complete market data.';
  } catch (error) { status.textContent = error.message; }
  finally { clearTimeout(timer); button.disabled = false; }
}

document.getElementById("save").addEventListener("click", save);
load();
