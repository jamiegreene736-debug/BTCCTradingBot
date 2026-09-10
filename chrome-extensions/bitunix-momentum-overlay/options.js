async function load() {
  const stored = await chrome.storage.local.get(["dashboardUrl", "password"]);
  document.getElementById("dashboardUrl").value = stored.dashboardUrl || "";
  document.getElementById("password").value = stored.password || "";
}

async function save() {
  const dashboardUrl = document.getElementById("dashboardUrl").value.trim().replace(/\/+$/, "");
  const password = document.getElementById("password").value;
  const status = document.getElementById("status");
  try {
    const url = new URL(dashboardUrl);
    if (url.protocol !== 'https:' || !url.hostname.endsWith('.up.railway.app') || url.username || url.password || url.search || url.hash || !['', '/'].includes(url.pathname)) {
      throw new Error('Use an HTTPS Railway dashboard URL without a path.');
    }
    if (!password) throw new Error('Enter your dashboard password.');
    await chrome.storage.local.set({ dashboardUrl: url.origin, password });
    await chrome.storage.sync.remove(['dashboardUrl', 'password']);
  } catch (error) { status.textContent = error.message; return; }
  status.textContent = "Saved";
  setTimeout(() => { status.textContent = ""; }, 1800);
}

document.getElementById("save").addEventListener("click", save);
load();
