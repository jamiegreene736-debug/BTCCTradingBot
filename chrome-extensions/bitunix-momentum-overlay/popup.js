async function loadStatus() {
  const stored = await chrome.storage.local.get(["dashboardUrl", "password"]);
  const configured = Boolean(stored.dashboardUrl && stored.password);
  document.getElementById("status").textContent = configured
    ? "Configured. Open bitunix.com and refresh the page if the panel is not visible."
    : "Not configured yet. Add your dashboard URL and password.";
}

document.getElementById("options").addEventListener("click", () => {
  chrome.runtime.openOptionsPage();
});

document.getElementById("openBitunix").addEventListener("click", () => {
  chrome.tabs.create({ url: "https://www.bitunix.com/" });
});

loadStatus();
