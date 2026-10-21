// PersonalWisper to-do page. Talks to todo_server.py on 127.0.0.1.

const STATUS_LABELS = {
  dirty: "Dirty", research: "Research", researching: "Researching", researched: "Researched",
  work: "Work", working: "In progress", done: "Done", archived: "Archived",
};
const state = {
  items: [], projects: [], selectedId: null, filter: "dirty", detail: null, listening: null,
  researchModels: [], effortLabels: {}, researchDefault: null,
  choice: null, // { itemId, model, effort, place } picked for the selected item's next T3 thread
  followupDraft: "", // the follow-up box, kept across redraws
  view: localStorage.getItem("pw.view") || "list", // "list" or "board"
  board: JSON.parse(localStorage.getItem("pw.board") || "null"), // { model, effort, place, project } for drops
  dragging: null, // the id of the card being dragged on the board
};

const $ = (sel) => document.querySelector(sel);
const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[c]);

async function api(method, path, body) {
  const res = await fetch(path, {
    method,
    headers: { "Content-Type": "application/json", "X-PersonalWisper": "1" },
    body: body === undefined ? undefined : JSON.stringify(body),
  });
  const data = await res.json().catch(() => ({}));
  if (!res.ok) throw new Error(data.error || `Request failed (${res.status})`);
  return data;
}

function toast(message, ms = 3500) {
  const el = $("#toast");
  el.textContent = message;
  el.hidden = false;
  clearTimeout(toast.timer);
  toast.timer = setTimeout(() => (el.hidden = true), ms);
}

function when(iso) {
  const d = new Date(iso);
  return d.toLocaleDateString(undefined, { month: "short", day: "numeric" }) + " " +
    d.toLocaleTimeString(undefined, { hour: "numeric", minute: "2-digit" });
}

// ---- list ----

async function refresh() {
  try {
    const data = await api("GET", "/api/state");
    state.items = data.items;
    state.projects = data.projects;
    state.defaultProjectId = data.default_project_id;
    state.researchModels = data.research_models;
    state.effortLabels = data.effort_labels;
    state.researchDefault = data.research_default;
    $("#hint").textContent = data.hotkey
      ? `Hold ${data.hotkey} anywhere to capture a dirty item.`
      : "Hold your to-do hotkey anywhere to capture a dirty item.";
    const banner = $("#banner");
    banner.hidden = !data.t3_error;
    banner.textContent = data.t3_error ? `T3 Code: ${data.t3_error}` : "";
    renderList();
    if (state.view === "board" && state.dragging === null) renderBoard();
    const current = state.items.find((i) => i.id === state.selectedId);
    if (state.selectedId && !current) select(null);
    else if (current && state.detail && !isEditing() && (current.updated_at !== state.detail.updated_at ||
      current.eval_state !== state.detail.eval_state)) loadDetail();
  } catch (e) {
    $("#banner").hidden = false;
    $("#banner").textContent = "Can't reach PersonalWisper. Is it running?";
  }
}

const FOLDER_ICON = `<svg class="folder-icon" viewBox="0 0 16 16" aria-hidden="true"><path d="M1.5 3.5A1.5 1.5 0 0 1 3 2h3.2l1.6 1.6H13A1.5 1.5 0 0 1 14.5 5v7.5A1.5 1.5 0 0 1 13 14H3a1.5 1.5 0 0 1-1.5-1.5z"/></svg>`;
const collapsed = new Set(JSON.parse(localStorage.getItem("pw.collapsedFolders") || "[]"));

// One folder per T3 project: the default project first, then the rest by name, then any project an item
// still points at that T3 no longer lists, then items with no project.
function projectFolders(items) {
  const folders = [...state.projects]
    .sort((a, b) => (b.id === state.defaultProjectId) - (a.id === state.defaultProjectId) || a.title.localeCompare(b.title))
    .map((p) => ({ id: p.id, title: p.title, items: [] }));
  const byId = new Map(folders.map((f) => [f.id, f]));
  for (const item of items) {
    const key = item.project_id || "none";
    if (!byId.has(key)) {
      const folder = { id: key, title: item.project_title || "No project", items: [] };
      byId.set(key, folder);
      folders.push(folder);
    }
    byId.get(key).items.push(item);
  }
  return folders;
}

const WORK_STATUSES = ["work", "working", "done"];
const EVALUATED = ["dirty", "research", "work"];

function itemRow(i) {
  const work = WORK_STATUSES.includes(i.status);
  const runState = work ? i.work_state : i.research_state;
  return `<li data-id="${i.id}" class="item ${i.id === state.selectedId ? "selected" : ""}">
      <div class="title">${esc(i.title)}</div>
      <div class="meta">${["researching", "working"].includes(i.status) && runState === "running" ? '<span class="spinner"></span>' : ""}
        ${runState === "needs-input" ? '<span class="error">needs your input in T3</span>' : ""}
        ${runState === "error" ? `<span class="error">${work ? "work" : "research"} failed</span>` : ""}
        ${evalChip(i)}
        <span>${when(i.updated_at)}</span></div>
    </li>`;
}

// The local evaluation's state, for items it works on (dirty, research and work).
function evalChip(i) {
  if (!EVALUATED.includes(i.status)) return "";
  if (i.eval_state === "running") return '<span class="chip"><span class="spinner"></span> evaluating</span>';
  if (i.eval_state === "queued") return '<span class="chip">waiting to evaluate</span>';
  if (i.eval_state === "failed") return '<span class="error">evaluation failed</span>';
  if (i.open_questions) return `<span class="chip ask">${i.open_questions} question${i.open_questions > 1 ? "s" : ""}</span>`;
  if (i.open_questions === 0) return '<span class="chip ready">ready</span>';
  return "";
}

function renderList() {
  const counts = Object.fromEntries(Object.keys(STATUS_LABELS).map((s) => [s, 0]));
  state.items.forEach((i) => counts[i.status]++);
  $("#tabs").innerHTML = Object.entries(STATUS_LABELS).map(([s, label]) =>
    `<button class="tab ${state.filter === s ? "active" : ""}" data-filter="${s}">${label} ${counts[s]}</button>`).join("");
  const folders = projectFolders(state.items.filter((i) => i.status === state.filter));
  $("#list").innerHTML = folders.length ? folders.map((f) => {
    const open = !collapsed.has(f.id);
    return `<li class="folder ${f.items.length ? "" : "empty"}" data-folder="${esc(f.id)}">
        <span class="caret">${open ? "▾" : "▸"}</span>${FOLDER_ICON}
        <span class="name">${esc(f.title)}</span><span class="count">${f.items.length}</span>
      </li>${open ? f.items.map(itemRow).join("") : ""}`;
  }).join("") : `<li class="none">Nothing here.</li>`;
}

$("#tabs").addEventListener("click", (e) => {
  const tab = e.target.closest("[data-filter]");
  if (tab) { state.filter = tab.dataset.filter; renderList(); }
});
$("#list").addEventListener("click", (e) => {
  const folder = e.target.closest("li[data-folder]");
  if (folder) {
    const id = folder.dataset.folder;
    collapsed.has(id) ? collapsed.delete(id) : collapsed.add(id);
    localStorage.setItem("pw.collapsedFolders", JSON.stringify([...collapsed]));
    return renderList();
  }
  const li = e.target.closest("li[data-id]");
  if (li) select(Number(li.dataset.id));
});

// ---- detail ----

function isEditing() {
  const el = document.activeElement;
  return el && $("#detail").contains(el) && ["INPUT", "TEXTAREA"].includes(el.tagName);
}

function select(id) {
  if (id !== state.selectedId) state.followupDraft = "";
  state.selectedId = id;
  state.detail = null;
  document.body.classList.toggle("has-selection", Boolean(id));
  renderList();
  if (state.view === "board") renderColumns();
  if (id) loadDetail();
  else $("#detail").innerHTML = `<div class="empty">Select an item, or hold your to-do hotkey anywhere and start talking.</div>`;
}

async function loadDetail() {
  try {
    state.detail = await api("GET", `/api/items/${state.selectedId}`);
    if (state.choice?.itemId !== state.detail.id) {
      // Start from what this item's thread last used, else the default (Opus 5.5 · High).
      const d = state.detail;
      const [model, effort] = WORK_STATUSES.includes(d.status) && d.work_model
        ? [d.work_model, d.work_effort] : [d.research_model, d.research_effort];
      state.choice = { itemId: d.id, ...(model ? { model, effort } : state.researchDefault),
        place: d.work_place || d.work_place_default };
    }
    renderDetail();
  } catch (e) { toast(e.message); }
}

function talkButton(target, label) {
  const listening = state.listening === target;
  return `<button class="talk secondary ${listening ? "listening" : ""}" data-talk="${target}">
    ${listening ? "■ Stop and add" : "🎙 " + label}</button>`;
}

const bullets = (items) => items.length ? `<ul>${items.map((x) => `<li>${esc(x)}</li>`).join("")}</ul>` : "";

function evaluationCard(it) {
  const ev = it.evaluation;
  const working = it.eval_state === "running" || it.eval_state === "queued";
  const status = working
    ? `<p class="muted"><span class="spinner"></span> ${it.eval_state === "running" ? "Evaluating" : "Waiting to evaluate"}${ev ? " again with your changes" : ""}…</p>`
    : it.eval_state === "failed"
      ? `<p class="error">Evaluation failed: ${esc(it.eval_error)}</p>` : "";
  if (!ev) {
    return `<div class="card eval"><h3>Local evaluation</h3>${status ||
      '<p class="muted">Not evaluated yet.</p>'}<button class="secondary" data-act="evaluate">Evaluate now</button></div>`;
  }
  const titleHint = ev.title && ev.title !== it.title
    ? `<p class="muted">Suggested title: <b>${esc(ev.title)}</b> <button class="link" data-act="use-title">Use it</button></p>` : "";
  const questions = ev.questions.length ? `
      <h4>Questions for you</h4>
      <ol class="questions">${ev.questions.map((q) => `<li>${esc(q.question)}${q.why ? `<div class="muted">${esc(q.why)}</div>` : ""}</li>`).join("")}</ol>
      <textarea id="answer" placeholder="Answer any or all of them, in any order…" style="min-height:70px"></textarea>
      <div class="row" style="margin-top:8px">
        <button data-act="answer">Send answer</button>${talkButton("answer", "Talk to answer")}
      </div>`
    : `<p class="ready-note">✓ No open questions. It's clear enough to research or do.</p>`;
  return `<div class="card eval">
      <h3>Local evaluation <span class="badge kind">${esc(ev.kind)}</span></h3>
      ${status}${titleHint}
      <p><b>Goal:</b> ${esc(ev.goal)}</p>
      <b>Details</b>${bullets(ev.details)}
      ${ev.done_when.length ? `<b>Done when</b>${bullets(ev.done_when)}` : ""}
      ${questions}
      ${ev.research.length ? `<details class="sub"><summary>For Claude to check in the code (${ev.research.length})</summary>${bullets(ev.research)}</details>` : ""}
      ${it.eval_rounds.length ? `<details class="sub"><summary>Your answers so far (${it.eval_rounds.length})</summary>
        ${it.eval_rounds.map((r) => `<div class="event">${esc(r.answer)}</div>`).join("")}</details>` : ""}
      <div class="row" style="margin-top:12px"><button class="secondary" data-act="evaluate" ${working ? "disabled" : ""}>Re-evaluate</button>
        <span class="muted">Runs on this Mac. The brief and your answers go along to T3.</span></div>
    </div>`;
}

function modelEntry(id) {
  return state.researchModels.find((m) => m.id === id) || state.researchModels[0];
}

function workPlace() {
  const places = state.detail.work_places || [];
  return places.find((p) => p.id === state.choice.place) || places.find((p) => p.id === state.detail.work_place_default) || places[0];
}

function choiceLabel(kind) {
  const m = modelEntry(state.choice.model);
  const label = `${m.label} · ${state.choice.effort ? state.effortLabels[state.choice.effort] : "default effort"}`;
  const place = kind === "work" && workPlace();
  return place ? `${label} · ${place.id === "worktree" ? "new worktree" : "project folder"}` : label;
}

// Where a work thread ran: its worktree branch, or the project folder's branch.
function workWhere(it) {
  if (it.work_worktree) return ` in worktree <b>${esc(it.work_branch || "")}</b> <span class="muted" title="${esc(it.work_worktree)}">(${esc(it.work_worktree.replace(/^\/Users\/[^/]+/, "~"))})</span>`;
  if (it.work_place === "worktree") return " in a new worktree (being set up)";
  if (it.work_place === "local") return ` in the project folder${it.work_branch ? ` on <b>${esc(it.work_branch)}</b>` : ""}`;
  return "";
}

// Model and effort (and, for work, where it runs), shown next to every button that starts a T3 thread.
function researchPicker(kind) {
  const c = state.choice;
  const m = modelEntry(c.model);
  const efforts = m.efforts.length
    ? m.efforts.map((e) => `<option value="${e}" ${e === c.effort ? "selected" : ""}>${esc(state.effortLabels[e])}</option>`).join("")
    : "<option>Default</option>";
  return `<div class="row picker">
      <label class="inline">Model <select data-choice="model">${state.researchModels.map((x) =>
        `<option value="${x.id}" ${x.id === m.id ? "selected" : ""}>${esc(x.label)}</option>`).join("")}</select></label>
      <label class="inline">Effort <select data-choice="effort" ${m.efforts.length ? "" : "disabled"}>${efforts}</select></label>
      ${kind === "work" && state.detail.work_places?.length ? `<label class="inline">Where <select data-choice="place">${
        state.detail.work_places.map((p) => `<option value="${p.id}" ${p.id === workPlace().id ? "selected" : ""}>${esc(p.label)}</option>`).join("")}</select></label>` : ""}
    </div>
    ${c.effort === "ultracode" ? '<p class="warn">Ultracode is extra-high effort plus multi-agent orchestration: much slower and much heavier on your usage.</p>' : ""}
    ${m.efforts.length ? "" : `<p class="muted">T3 doesn't list effort levels for ${esc(m.label)} yet, so it runs at Claude Code's default.</p>`}`;
}

// The fields of an item's research or work thread.
function runFields(it, kind) {
  return kind === "work"
    ? { state: it.work_state, note: it.work_note, label: it.work_label, html: it.work_result_html,
        plan: it.work_plan, planHtml: it.work_plan_html, title: "Work", verb: "Working", done: "Done",
        followup: it.work_followup }
    : { state: it.research_state, note: it.research_note, label: it.research_label, html: it.result_html,
        plan: it.result_plan, planHtml: it.plan_html, title: "Research", verb: "Researching", done: "Researched",
        followup: it.followup };
}

function startButton(kind, label, cls = "") {
  return `${researchPicker(kind)}<button class="${cls}" data-act="${kind}">${label} · ${esc(choiceLabel(kind))}</button>`;
}

// Research and work continue the item's T3 thread once it has one; starting over is the exception.
function runButtons(it, kind) {
  const verb = kind === "work" ? "work" : "research";
  if (!it.has_thread) return startButton(kind, kind === "work" ? "Start work in T3" : "Start research in T3");
  return `<p class="muted">Continues this item's T3 thread, which already has everything said and found so far.
      Only what changed since is sent.</p>
    ${startButton(kind, kind === "work" ? "Continue as work in T3" : "Continue research in T3")}
    <details class="sub"><summary>Or start over in a new thread…</summary>
      <p class="muted">A fresh thread gets your notes and brief${kind === "work" && it.result ? " and the research report" : ""}, without the conversation so far.</p>
      <button class="secondary" data-act="${kind}" data-fresh="1">Start ${verb} in a new thread · ${esc(choiceLabel(kind))}</button></details>`;
}

function report(r) {
  return `<div class="result">${r.html || ""}</div>
    ${r.planHtml ? `<details class="report"><summary>Full report (${Math.max(1, Math.round(r.plan.split(/\s+/).length / 250))} min read)</summary>
      <div class="result">${r.planHtml}</div></details>` : ""}`;
}

function runningCard(it, kind) {
  const r = runFields(it, kind);
  return `<div class="card"><h3>${r.state === "running" ? `<span class="spinner"></span> ${r.verb} in T3…` : `${r.title} needs attention`}</h3>
    ${r.note ? `<p class="${r.state === "running" ? "muted" : "error"}">${esc(r.note)}</p>` : ""}
    ${r.followup ? `<p><b>Follow-up:</b> ${esc(r.followup)}</p>` : ""}
    <p class="muted">Thread “${r.title}: ${esc(it.title)}” in ${esc(it.project_title)}${r.label ? `, using <b>${esc(r.label)}</b>` : ""}${kind === "work" ? workWhere(it) : ""}.
    Results appear here automatically.</p>
    <div class="row"><button class="secondary" data-act="check">Check now</button>
    <button class="secondary" data-act="open-t3">Open T3 Code</button></div>
    ${r.state === "error" || r.state === "missing" ? startButton(kind, `Retry ${kind}`) : ""}</div>`;
}

// Ask more of a finished thread. The follow-up goes to the same T3 thread, which already has everything it found
// or did, so only the follow-up itself is sent.
function followUpBox(kind) {
  const work = kind === "work";
  return `<div class="followup">
    <label>Follow up <small>sent to the same ${work ? "work thread (and worktree), which keeps what it already did" :
      "research thread, which already has everything it found"}; only your follow-up is sent</small></label>
    <textarea id="followup" placeholder="${work ? "What else should it change or fix?" : "What should it dig into next?"}"
      style="min-height:70px">${esc(state.followupDraft)}</textarea>
    <div class="row" style="margin-top:8px">${talkButton("followup", "Talk to add")}</div>
    ${researchPicker("followup")}
    <button data-act="followup">Send follow-up · ${esc(choiceLabel("followup"))}</button>
    <details class="sub"><summary>Or start over in a new thread…</summary>
      <p class="muted">Starts a fresh thread with your notes and brief, without anything found so far.</p>
      <button class="secondary" data-act="${kind}" data-fresh="1">Start over in a new thread · ${esc(choiceLabel(kind))}</button></details>
  </div>`;
}

// Results that a follow-up or a fresh start replaced, oldest first.
function earlierResults(it) {
  const runs = it.run_history || [];
  if (!runs.length) return "";
  return `<details class="sub"><summary>Earlier results (${runs.length})</summary>${runs.map((h) => `
    <div class="event"><div class="when">${h.finished_at ? when(h.finished_at) + " · " : ""}${
      h.question ? `Follow-up: ${esc(h.question)}` : `First ${h.kind}`}${h.label ? ` · ${esc(h.label)}` : ""}</div>
      <div class="result">${h.html}</div>
      ${h.plan_html ? `<details class="report"><summary>Full report</summary><div class="result">${h.plan_html}</div></details>` : ""}
    </div>`).join("")}</details>`;
}

function resultCard(it, kind) {
  const r = runFields(it, kind);
  return `<div class="card"><h3>${r.title} result</h3>
    ${r.label ? `<p class="muted">${r.done} with <b>${esc(r.label)}</b>${kind === "work" ? workWhere(it) : ""}.</p>` : ""}
    <div class="row"><button class="secondary" data-act="open-t3">Continue in T3 Code</button>
    <button class="secondary" data-act="check">Refresh from T3</button></div>
    ${r.state === "running" ? `<p class="muted"><span class="spinner"></span> T3 is still adding to this thread${
      r.note ? ` (${esc(r.note)})` : ""}. The result will update automatically.</p>` : ""}
    ${r.followup ? `<p><b>Follow-up:</b> ${esc(r.followup)}</p>` : ""}
    ${report(r)}
    ${earlierResults(it)}
    ${followUpBox(kind)}</div>`;
}

// What can happen next: dirty items can be researched or done; research items can skip to work; researched
// items can be done with the research going along.
function nextCard(it) {
  const project = `<b>${esc(it.project_title || "?")}</b>`;
  switch (it.status) {
    case "dirty":
      return `<div class="card"><h3>What's next?</h3>
        <p class="muted"><b>Research it</b> when Claude should look into it first and report back (read-only).
        <b>Just do it</b> when it's a clear change, like making a card cost 10 energy instead of 5.</p>
        <div class="row"><button data-act="to-research">Research it first →</button>
        <button data-act="to-work">Just do it: make work item →</button></div></div>`;
    case "research":
      return `<div class="card"><h3>Ready for research</h3>
        <p class="muted">Starts a read-only (plan mode) thread in T3 Code, in the ${project} project.
        It counts against your Claude usage.</p>
        ${runButtons(it, "research")}
        <p class="muted" style="margin:14px 0 0">Doesn't need research after all? <button class="link" data-act="to-work">Make it a work item →</button></p></div>`;
    case "researching": return runningCard(it, "research");
    case "researched":
      return `<div class="card"><h3>Ready to do it?</h3>
        <p class="muted">Make this a work item. The work continues in the same T3 thread, which already has the research.</p>
        <button data-act="to-work">Make work item →</button></div>${resultCard(it, "research")}`;
    case "work":
      return `<div class="card"><h3>Ready to do</h3>
        <p class="muted">Starts a regular thread in T3 Code, in the ${project} project, that does the work and can change
        files: in a <b>new worktree</b> (its own <code>t3code/…</code> branch, so your checkout isn't touched) or in the
        <b>project folder</b> on whatever branch it has checked out.
        It counts against your Claude usage.</p>
        ${runButtons(it, "work")}</div>`;
    case "working": return runningCard(it, "work");
    case "done": return resultCard(it, "work");
  }
  return "";
}

function renderDetail() {
  const it = state.detail;
  const projectOptions = state.projects.map((p) =>
    `<option value="${esc(p.id)}" ${p.id === it.project_id ? "selected" : ""}>${esc(p.title)}</option>`).join("");
  const showQuestions = ["research", "researching", "researched"].includes(it.status);
  const earlierResearch = WORK_STATUSES.includes(it.status) && it.result_html
    ? `<details class="sub card"><summary>Research it's based on</summary>${report(runFields(it, "research"))}</details>` : "";

  $("#detail").innerHTML = `
    ${state.view === "board" ? '<div class="row" style="justify-content:flex-end;margin:0"><button class="link" data-act="close">Close ✕</button></div>' : ""}
    <input class="title" data-field="title" value="${esc(it.title)}">
    <div class="row">
      <span class="badge ${it.status}">${STATUS_LABELS[it.status]}</span>
      <span class="muted">T3 project</span><select data-field="project_id">${projectOptions}</select>
      <span class="saved" id="saved"></span>
    </div>
    <label>Notes <small>edit freely, or talk to add details</small></label>
    <textarea data-field="notes">${esc(it.notes)}</textarea>
    <div class="row" style="margin-top:8px">${talkButton("notes", "Talk to refine")}</div>
    ${EVALUATED.includes(it.status) ? evaluationCard(it) : ""}
    ${showQuestions ? `
      <label>Research questions <small>optional, sent to the research thread with your notes</small></label>
      <textarea data-field="brief" style="min-height:90px">${esc(it.brief)}</textarea>
      <div class="row" style="margin-top:8px">${talkButton("questions", "Talk to add questions")}</div>` : ""}
    ${nextCard(it)}
    ${earlierResearch}
    <div class="row" style="margin-top:24px">
      ${it.status === "archived"
        ? '<button class="secondary" data-act="restore">Restore</button>'
        : '<button class="secondary" data-act="archive">Archive</button>'}
      <button class="danger" data-act="delete">Delete</button>
      ${["research", "work"].includes(it.status) ? '<button class="secondary" data-act="to-dirty">Back to dirty</button>' : ""}
    </div>
    <details><summary>History (${it.events.length})</summary>
      ${it.events.map((e) => `<div class="event"><div class="when">${when(e.created_at)} · ${esc(e.kind)}</div>${esc(e.text)}</div>`).join("")}
    </details>`;
}

async function save(fields) {
  try {
    state.detail = await api("PATCH", `/api/items/${state.selectedId}`, fields);
    const item = state.items.find((i) => i.id === state.selectedId);
    if (item) Object.assign(item, state.detail);
    const saved = $("#saved");
    if (saved) { saved.textContent = "Saved"; setTimeout(() => (saved.textContent = ""), 1500); }
    renderList();
  } catch (e) { toast(e.message); }
}

$("#detail").addEventListener("input", (e) => {
  if (e.target.id === "followup") state.followupDraft = e.target.value;
});

$("#detail").addEventListener("change", (e) => {
  const choice = e.target.dataset.choice;
  if (choice) {
    state.choice[choice] = e.target.value;
    if (choice === "model") {
      const efforts = modelEntry(e.target.value).efforts;
      if (!efforts.includes(state.choice.effort)) {
        state.choice.effort = efforts.includes(state.researchDefault.effort) ? state.researchDefault.effort : efforts[0] ?? null;
      }
    }
    const open = e.target.closest("details")?.open;
    renderDetail();
    if (open) document.querySelector("#detail details.sub:has(.picker)")?.setAttribute("open", "");
    return;
  }
  const field = e.target.dataset.field;
  if (!field || !state.detail || e.target.value === state.detail[field]) return;
  // A project change moves the item to another folder and changes the research card text, so redraw it.
  save({ [field]: e.target.value }).then(() => field === "project_id" && renderDetail());
});

$("#detail").addEventListener("click", async (e) => {
  const talk = e.target.closest("[data-talk]");
  if (talk) return toggleTalk(talk.dataset.talk);
  const act = e.target.closest("[data-act]")?.dataset.act;
  if (!act) return;
  const id = state.selectedId;
  try {
    if (act === "to-research") { await save({ status: "research" }); state.filter = "research"; renderDetail(); }
    if (act === "to-work") { await save({ status: "work" }); state.filter = "work"; renderDetail(); }
    if (act === "to-dirty") { await save({ status: "dirty" }); state.filter = "dirty"; renderDetail(); }
    if (act === "archive") { await save({ status: "archived" }); select(null); }
    if (act === "restore") { await save({ status: "dirty" }); state.filter = "dirty"; renderDetail(); }
    if (act === "delete") {
      const t3Note = state.detail.thread_id || state.detail.work_thread_id ? "\n\nIts T3 threads stay in T3 Code." : "";
      if (!confirm(`Delete “${state.detail.title}” forever? This can't be undone.${t3Note}`)) return;
      await api("DELETE", `/api/items/${id}`);
      select(null);
      toast("Item deleted.");
    }
    if (act === "research" || act === "work") {
      const fresh = e.target.closest("[data-act]").dataset.fresh === "1";
      const continuing = state.detail.has_thread && !fresh;
      const what = continuing ? `the next step (${act}) in this item's existing T3 thread` : act === "work"
        ? `a work thread in T3 Code that can change files, ${workPlace()?.label.replace(/^./, (c) => c.toLowerCase())}`
        : "a read-only research thread in T3 Code";
      if (!confirm(`Start ${what} (${state.detail.project_title}) with ${choiceLabel()}?\nIt uses your Claude usage.`)) return;
      e.target.disabled = true;
      state.detail = await api("POST", `/api/items/${id}/${act}`,
        { model: state.choice.model, effort: state.choice.effort, place: workPlace()?.id, fresh });
      state.filter = act === "work" ? "working" : "researching";
      renderDetail();
      toast(continuing ? "Sent to the same T3 thread." : act === "work" ? "Work thread started in T3 Code." : "Research thread started in T3 Code.");
    }
    if (act === "followup") {
      const question = $("#followup").value.trim();
      if (!question) return toast("Type or say your follow-up first.");
      if (!confirm(`Send this follow-up to the same T3 thread with ${choiceLabel("followup")}?\nIt uses your Claude usage.`)) return;
      e.target.disabled = true;
      state.detail = await api("POST", `/api/items/${id}/followup`,
        { question, model: state.choice.model, effort: state.choice.effort });
      state.followupDraft = "";
      state.filter = WORK_STATUSES.includes(state.detail.status) ? "working" : "researching";
      renderDetail();
      toast("Follow-up sent to the same T3 thread.");
    }
    if (act === "check") {
      const turn = (d) => (WORK_STATUSES.includes(d.status) ? d.work_turn_id : d.result_turn_id);
      const before = turn(state.detail);
      state.detail = await api("POST", `/api/items/${id}/check`);
      renderDetail();
      if (["researched", "done"].includes(state.detail.status)) {
        toast(turn(state.detail) !== before ? "Result updated from T3." : "Already up to date.");
      }
    }
    if (act === "close") return select(null);
    if (act === "open-t3") await api("POST", "/api/open-t3");
    if (act === "evaluate") { state.detail = await api("POST", `/api/items/${id}/evaluate`); renderDetail(); }
    if (act === "use-title") { await save({ title: state.detail.evaluation.title }); renderDetail(); }
    if (act === "answer") {
      const answer = $("#answer").value.trim();
      if (!answer) return toast("Type an answer first.");
      state.detail = await api("POST", `/api/items/${id}/answer`, { answer });
      renderDetail();
      toast("Thanks. Evaluating again with your answer.");
    }
    refresh();
  } catch (err) { toast(err.message, 6000); e.target.disabled = false; }
});

// ---- board ----
// Dirty items are the backlog. Dropping a card on Research or Work starts that agent in T3 right away, with the
// model, effort and place picked at the top of the board.

const BOARD_COLUMNS = [
  { id: "backlog", title: "Backlog", statuses: ["dirty"], hint: "Dirty to-dos. Drag one to Research or Work to start an agent in T3." },
  { id: "research", title: "Research", statuses: ["research", "researching"], hint: "Drop here: a read-only research agent starts." },
  { id: "researched", title: "Researched", statuses: ["researched"], hint: "Research done. Drag to Work to have it done." },
  { id: "work", title: "Work", statuses: ["work", "working"], hint: "Drop here: an agent starts doing it (it can change files)." },
  { id: "done", title: "Done", statuses: ["done"], hint: "Finished work. Open a card to review or follow up." },
];
// Where a card with each status can be dropped. Running and finished cards stay put.
const DROPS = { dirty: ["research", "work"], research: ["backlog", "work"], work: ["backlog", "research"], researched: ["work"] };
const PLACES = [["default", "Project default"], ["worktree", "New worktree"], ["local", "Project folder"]];

function boardChoice() {
  if (!state.board && state.researchDefault) state.board = { ...state.researchDefault, place: "default", project: "" };
  return state.board;
}

function saveBoardChoice() {
  localStorage.setItem("pw.board", JSON.stringify(state.board));
}

function runLabel(model, effort) {
  const m = state.researchModels.find((x) => x.id === model);
  return m ? `${m.label}${effort ? " · " + (state.effortLabels[effort] || effort) : ""}` : "";
}

function renderBoard() {
  const bar = $("#board-bar");
  if (!bar.contains(document.activeElement)) renderBoardBar();  // don't close a menu that's open
  renderColumns();
}

function renderBoardBar() {
  const b = boardChoice();
  if (!b) return;
  const m = modelEntry(b.model);
  const options = (pairs, value) => pairs.map(([v, l]) => `<option value="${esc(v)}" ${v === value ? "selected" : ""}>${esc(l)}</option>`).join("");
  const html = `
    <span>Dropping on <b>Research</b> or <b>Work</b> starts the agent with</span>
    <label class="inline">Model <select data-board="model">${options(state.researchModels.map((x) => [x.id, x.label]), m.id)}</select></label>
    <label class="inline">Effort <select data-board="effort" ${m.efforts.length ? "" : "disabled"}>${
      m.efforts.length ? options(m.efforts.map((e) => [e, state.effortLabels[e]]), b.effort) : "<option>Default</option>"}</select></label>
    <label class="inline">Work in <select data-board="place">${options(PLACES, b.place)}</select></label>
    <label class="inline" style="margin-left:auto">Project <select data-board="project">${
      options([["", "All projects"], ...state.projects.map((p) => [p.id, p.title])], b.project)}</select></label>
    ${b.effort === "ultracode" ? '<span class="warn">Ultracode is much slower and much heavier on your usage.</span>' : ""}`;
  if (html !== renderBoardBar.last) $("#board-bar").innerHTML = renderBoardBar.last = html;
}

function boardCard(i) {
  const work = WORK_STATUSES.includes(i.status);
  const run = work ? i.work_state : i.research_state;
  const label = work ? runLabel(i.work_model, i.work_effort) : runLabel(i.research_model, i.research_effort);
  let status = "";
  if (i.status === "dirty") status = evalChip(i);
  else if (["research", "work"].includes(i.status)) {
    status = `<span>Not started</span><button class="link" data-start="${i.status}">Start ▸</button>`;
  } else if (["researching", "working"].includes(i.status)) {
    status = run === "running" ? `<span><span class="spinner"></span> ${work ? "Working" : "Researching"}</span>`
      : run === "needs-input" ? '<span class="error">needs your input in T3</span>'
      : run === "error" || run === "missing" ? `<span class="error">${work ? "work" : "research"} failed</span>` : "";
  }
  const showLabel = label && !["dirty", "research", "work"].includes(i.status);
  const project = state.board?.project ? "" : `<span class="project">${esc(i.project_title || "No project")}</span>`;
  return `<div class="bcard ${i.status} ${i.id === state.selectedId ? "selected" : ""}" data-id="${i.id}" draggable="${Boolean(DROPS[i.status])}">
      <div class="title">${esc(i.title)}</div>
      <div class="meta">${status}${showLabel ? `<span>${esc(label)}</span>` : ""}${project}</div>
    </div>`;
}

function renderColumns() {
  const project = state.board?.project;
  const items = state.items.filter((i) => i.status !== "archived" && (!project || i.project_id === project));
  const html = BOARD_COLUMNS.map((c) => {
    const cards = items.filter((i) => c.statuses.includes(i.status));
    return `<div class="column" data-column="${c.id}">
        <h2>${c.title} <span class="count">${cards.length}</span></h2>
        <div class="hint">${c.hint}</div>
        <div class="cards">${cards.map(boardCard).join("")}</div>
      </div>`;
  }).join("");
  // Redraw only when something changed, so a card isn't replaced under the mouse as you pick it up.
  if (html !== renderColumns.last) $("#columns").innerHTML = renderColumns.last = html;
}

// Start a research or work agent in T3 for an item on the board.
async function startFromBoard(id, kind) {
  const b = boardChoice();
  const item = state.items.find((i) => i.id === id);
  if (item && item.status !== kind) await api("PATCH", `/api/items/${id}`, { status: kind });
  if (item) { item.status = kind === "work" ? "working" : "researching"; item[kind === "work" ? "work_state" : "research_state"] = "running"; renderColumns(); }
  const detail = await api("POST", `/api/items/${id}/${kind}`, { model: b.model, effort: b.effort, place: b.place });
  toast(`${kind === "work" ? "Work" : "Research"} started in T3 (${detail.project_title}) with ${runLabel(b.model, b.effort)}.`);
  if (state.selectedId === id) { state.detail = detail; renderDetail(); }
}

async function moveCard(id, target) {
  const item = state.items.find((i) => i.id === id);
  if (!item || !(DROPS[item.status] || []).includes(target)) return;
  try {
    if (target === "backlog") {
      await api("PATCH", `/api/items/${id}`, { status: "dirty" });
      toast("Back in the backlog.");
    } else {
      await startFromBoard(id, target);
    }
  } catch (e) { toast(e.message, 6000); }
  await refresh();
  if (state.selectedId === id) loadDetail();
}

$("#views").addEventListener("click", (e) => {
  const view = e.target.closest("[data-view]")?.dataset.view;
  if (view) setView(view);
});

function setView(view) {
  state.view = view;
  localStorage.setItem("pw.view", view);
  document.body.classList.toggle("board", view === "board");
  document.querySelectorAll("#views [data-view]").forEach((b) => b.classList.toggle("active", b.dataset.view === view));
  if (view === "board") renderBoard();
  else if (state.detail) {  // back in the list: show the selected item's tab
    state.filter = state.detail.status;
    renderList();
  }
  if (state.detail) renderDetail();
}

$("#board-bar").addEventListener("change", (e) => {
  const key = e.target.dataset.board;
  if (!key) return;
  state.board[key] = e.target.value;
  if (key === "model") {
    const efforts = modelEntry(e.target.value).efforts;
    if (!efforts.includes(state.board.effort)) {
      state.board.effort = efforts.includes(state.researchDefault.effort) ? state.researchDefault.effort : efforts[0] ?? null;
    }
  }
  saveBoardChoice();
  renderBoardBar();
  renderColumns();
});

$("#columns").addEventListener("click", async (e) => {
  const start = e.target.closest("[data-start]");
  const card = e.target.closest(".bcard");
  if (!card) return;
  const id = Number(card.dataset.id);
  if (start) {
    start.disabled = true;
    try { await startFromBoard(id, start.dataset.start); } catch (err) { toast(err.message, 6000); }
    return refresh();
  }
  select(id === state.selectedId ? null : id);
});

$("#columns").addEventListener("dragstart", (e) => {
  const card = e.target.closest(".bcard");
  if (!card) return;
  state.dragging = Number(card.dataset.id);
  e.dataTransfer.setData("text/plain", card.dataset.id);
  e.dataTransfer.effectAllowed = "move";
  const item = state.items.find((i) => i.id === state.dragging);
  document.querySelectorAll(".column").forEach((c) => c.classList.toggle("drop-ok", (DROPS[item?.status] || []).includes(c.dataset.column)));
  document.body.classList.add("dragging");
});

$("#columns").addEventListener("dragend", () => {
  state.dragging = null;
  document.body.classList.remove("dragging");
  document.querySelectorAll(".column").forEach((c) => c.classList.remove("drop-ok", "over"));
});

$("#columns").addEventListener("dragover", (e) => {
  const column = e.target.closest(".column.drop-ok");
  if (!column) return;
  e.preventDefault();
  document.querySelectorAll(".column.over").forEach((c) => c !== column && c.classList.remove("over"));
  column.classList.add("over");
});

$("#columns").addEventListener("drop", (e) => {
  const column = e.target.closest(".column.drop-ok");
  if (!column) return;
  e.preventDefault();
  const id = Number(e.dataTransfer.getData("text/plain"));
  state.dragging = null;
  document.body.classList.remove("dragging");
  moveCard(id, column.dataset.column);
});

// ---- voice ----

async function toggleTalk(target) {
  const id = state.selectedId;
  if (state.listening === null) {
    try {
      await api("POST", `/api/items/${id}/voice/start`);
      state.listening = target;
      renderDetail();
    } catch (e) { toast(e.message); }
    return;
  }
  if (state.listening !== target) return;
  const button = document.querySelector(`[data-talk="${target}"]`);
  button.classList.remove("listening");
  button.disabled = true;
  button.textContent = "Processing…";
  try {
    const res = await api("POST", `/api/items/${id}/voice/stop`, { target });
    state.detail = res.item;
    if (target === "followup" && res.text) state.followupDraft = `${state.followupDraft} ${res.text}`.trim();
    toast(`Heard: “${res.heard}”`, 5000);
  } catch (e) { toast(e.message); }
  state.listening = null;
  renderDetail();
  refresh();
}

document.addEventListener("keydown", async (e) => {
  if (e.key === "Escape" && state.listening !== null) {
    await api("POST", `/api/items/${state.selectedId}/voice/cancel`).catch(() => {});
    state.listening = null;
    renderDetail();
    toast("Stopped listening.");
  }
});

// ---- typed items ----

$("#new-item").addEventListener("click", async () => {
  const title = prompt("New to-do item:");
  if (!title || !title.trim()) return;
  try {
    const item = await api("POST", "/api/items", { title });
    state.filter = "dirty";
    await refresh();
    select(item.id);
  } catch (e) { toast(e.message); }
});

setView(state.view);
refresh();
setInterval(refresh, 3000);
