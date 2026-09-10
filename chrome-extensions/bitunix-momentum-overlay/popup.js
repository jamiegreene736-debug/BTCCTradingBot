async function loadStatus() {
  document.getElementById('version').textContent = 'Version ' + chrome.runtime.getManifest().version;
  const status = document.getElementById('status');
  let timer;
  try {
    const response = await Promise.race([
      chrome.runtime.sendMessage({ type: 'request-latest' }),
      new Promise((_, reject) => { timer = setTimeout(() => reject(new Error('Connection check timed out. Reload the extension and try again.')), 12000); }),
    ]);
    if (!response?.payload) throw new Error('Extension worker unavailable. Reload the extension and your Bitunix tab.');
    if (!response.payload.error && (response.payload.strategy !== 'intraday' || response.payload.mode !== 'alerts_only')) throw new Error('Update the backend to the intraday signals release.');
    status.textContent = response.payload.error || response.payload.status?.error || (response.payload.status?.ready
      ? 'Connected. Open or refresh Bitunix to see your signals.'
      : 'Connected. The scanner is waiting for complete market data.');
  } catch (error) { status.textContent = error.message; }
  finally { clearTimeout(timer); }
}

document.getElementById("options").addEventListener("click", () => {
  chrome.runtime.openOptionsPage();
});

document.getElementById("openBitunix").addEventListener("click", () => {
  chrome.tabs.create({ url: "https://www.bitunix.com/" });
});

loadStatus();
