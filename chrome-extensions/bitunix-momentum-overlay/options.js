async function load() {
  const stored = await chrome.storage.sync.get(["dashboardUrl", "password"]);
  document.getElementById("dashboardUrl").value = stored.dashboardUrl || "";
  document.getElementById("password").value = stored.password || "";
}

async function save() {
  const dashboardUrl = document.getElementById("dashboardUrl").value.trim().replace(/\/+$/, "");
  const password = document.getElementById("password").value;
  await chrome.storage.sync.set({ dashboardUrl, password });
  const status = document.getElementById("status");
  status.textContent = "Saved";
  setTimeout(() => { status.textContent = ""; }, 1800);
}

document.getElementById("save").addEventListener("click", save);
load();
