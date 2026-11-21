const COLUMN_TITLES = {
  upcoming: "即将到期",
  todo: "待办",
  done: "已完成",
};

const KIND_LABEL = { reminder: "提醒", task: "任务", countdown: "倒计时" };

function formatDueAt(iso) {
  if (!iso) return "";
  const d = new Date(iso);
  if (Number.isNaN(d.getTime())) return iso;
  const now = new Date();
  const hm = d.toLocaleTimeString("zh-CN", { hour: "2-digit", minute: "2-digit", hour12: false });
  const today = new Date(now.getFullYear(), now.getMonth(), now.getDate());
  const dueDay = new Date(d.getFullYear(), d.getMonth(), d.getDate());
  const diffDays = Math.round((dueDay - today) / 86400000);
  if (diffDays === 0) return `今天 ${hm}`;
  if (diffDays === 1) return `明天 ${hm}`;
  if (diffDays === 2) return `后天 ${hm}`;
  const y = d.getFullYear();
  const m = d.getMonth() + 1;
  const day = d.getDate();
  if (y === now.getFullYear()) return `${m}月${day}日 ${hm}`;
  return `${y}年${m}月${day}日 ${hm}`;
}

function formatCountdown(sec) {
  if (sec == null || sec < 0) return "";
  if (sec < 60) return `剩余 ${sec} 秒`;
  const m = Math.floor(sec / 60);
  if (m < 60) return `剩余 ${m} 分钟`;
  const h = Math.floor(m / 60);
  const rm = m % 60;
  return rm ? `剩余 ${h} 小时 ${rm} 分钟` : `剩余 ${h} 小时`;
}

function api() {
  return window.pywebview?.api;
}

function el(id) {
  return document.getElementById(id);
}

function renderState(data) {
  const running = data.running;
  el("pill-running").textContent = running ? "助手：运行中" : "助手：已停止";
  el("pill-running").className = "pill " + (running ? "on" : "off");
  el("pill-connected").textContent = data.connected ? "云端：已连接" : "云端：未连接";
  el("pill-connected").className = "pill " + (data.connected ? "on" : "off");
  el("pill-state").textContent = "状态：" + (data.conversation_state || "IDLE");

  const btn = el("btn-toggle");
  btn.textContent = running ? "停止助手" : "启动助手";
  btn.classList.toggle("primary", !running);

  const stats = data.stats || {};
  el("stats-panel").innerHTML = `
    <dt>运行时长 (h)</dt><dd>${stats.run_hours ?? 0}</dd>
    <dt>唤醒次数</dt><dd>${stats.wakes ?? 0}</dd>
    <dt>对话轮次</dt><dd>${stats.turns ?? 0}</dd>
    <dt>断线次数</dt><dd>${stats.disconnects ?? 0}</dd>
    <dt>音频重启</dt><dd>${stats.audio_restarts ?? 0}</dd>
  `;

  const board = data.board || {};
  el("last-notify").textContent = board.last_notify_text
    ? "最近提醒：" + board.last_notify_text
    : "";

  const cols = board.columns || {};
  const container = el("board-columns");
  container.innerHTML = "";
  for (const key of ["upcoming", "todo", "done"]) {
    const col = document.createElement("div");
    col.className = "column";
    col.innerHTML = `<h3>${COLUMN_TITLES[key]}</h3>`;
    for (const item of cols[key] || []) {
      const card = document.createElement("div");
      card.className = "card" + (item.highlighted ? " highlight" : "");
      const meta = [];
      if (item.kind) meta.push(KIND_LABEL[item.kind] || item.kind);
      if (item.due_at) meta.push(formatDueAt(item.due_at));
      const cd = formatCountdown(item.countdown_sec);
      if (cd) meta.push(cd);
      card.innerHTML = `<div>${escapeHtml(item.title)}</div><div class="meta">${escapeHtml(meta.join(" · "))}</div>`;
      col.appendChild(card);
    }
    container.appendChild(col);
  }
}

function escapeHtml(s) {
  return String(s)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;");
}

async function refresh() {
  const bridge = api();
  if (!bridge) return;
  try {
    const data = await bridge.get_state();
    renderState(data);
  } catch (e) {
    console.error(e);
  }
}

async function toggleAssistant() {
  const bridge = api();
  if (!bridge) return;
  const state = await bridge.get_state();
  if (state.running) {
    await bridge.stop_assistant();
  } else {
    await bridge.start_assistant();
  }
  await refresh();
}

async function openSettings() {
  const bridge = api();
  if (!bridge) return;
  const s = await bridge.get_settings();
  const form = el("settings-form");
  for (const name of ["server_url", "device_id", "auth_token", "input_device", "output_device"]) {
    if (form.elements[name]) form.elements[name].value = s[name] ?? "";
  }
  el("settings-dialog").showModal();
}

async function saveSettings(ev) {
  ev.preventDefault();
  const bridge = api();
  if (!bridge) return;
  const form = el("settings-form");
  const payload = Object.fromEntries(new FormData(form).entries());
  const res = await bridge.save_settings(payload);
  if (res.error) alert(res.error);
  else if (res.note) alert(res.note);
  el("settings-dialog").close();
}

function init() {
  el("btn-toggle").addEventListener("click", () => toggleAssistant().catch(console.error));
  el("btn-settings").addEventListener("click", () => openSettings().catch(console.error));
  el("btn-settings-cancel").addEventListener("click", () => el("settings-dialog").close());
  el("settings-form").addEventListener("submit", (e) => saveSettings(e).catch(console.error));

  window.addEventListener("pywebviewready", () => {
    refresh();
    setInterval(() => refresh().catch(console.error), 800);
  });
}

init();