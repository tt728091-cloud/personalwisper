"""Minimal client for the local T3 Code server: list projects, start research threads, read results.

Uses T3's orchestration protocol v2. State comes from GET /api/orchestration/shell and
/api/orchestration/threads/:id; commands go over the /ws WebSocket (Effect RPC, one JSON message per frame) as
orchestration.launchThread and orchestration.dispatchCommand, the calls T3's own web app makes. These are T3
internals, not a public API, so check_compatible() verifies the protocol version before anything is sent.

Auth: a bearer token from `t3 auth session issue --label personalwisper --token-only`, stored in secrets/t3-token.
"""
import json
import subprocess
import urllib.error
import urllib.parse
import urllib.request
import uuid
from pathlib import Path

from websockets.exceptions import WebSocketException
from websockets.sync.client import connect

ROOT = Path(__file__).resolve().parent
TOKEN_PATH = ROOT / "secrets" / "t3-token"
RUNTIME_PATH = Path.home() / ".t3" / "userdata" / "server-runtime.json"
WORKTREES_DIR = Path.home() / ".t3" / "worktrees"  # where T3 keeps its own thread worktrees
SUPPORTED_PROTOCOL = 2
PROTOCOL_HEADER = "x-t3-orchestration-protocol"
# Research threads may read the code but never change it: plan mode blocks edits until you approve a plan in T3.
# Work threads are ordinary threads, like the ones you start in T3, so they can change files.
# The runtime mode matches T3's own threads; "approval-required" would stall on every command.
INTERACTION_MODES = {"research": "plan", "work": "default"}
RUNTIME_MODE = "full-access"
CLAUDE_INSTANCE = "claudeAgent"
# Claude models offered for research, with the effort levels T3 (0.0.45) lists for each. T3's catalog doesn't
# include Haiku 5.5 yet, so it runs at Claude Code's default effort. Ultracode is extra-high effort plus
# multi-agent orchestration: much slower and much heavier on usage.
EFFORT_LABELS = {"low": "Low", "medium": "Medium", "high": "High", "xhigh": "Extra high", "max": "Max",
                 "ultracode": "Ultracode"}
RESEARCH_MODELS = [
    {"id": "claude-fable-5-1", "label": "Fable 5.1", "efforts": ["low", "medium", "high", "xhigh", "max", "ultracode"]},
    {"id": "claude-opus-5-5", "label": "Opus 5.5", "efforts": ["low", "medium", "high", "xhigh", "max", "ultracode"]},
    {"id": "claude-sonnet-5-5", "label": "Sonnet 5.5", "efforts": ["low", "medium", "high", "xhigh", "max"]},
    {"id": "claude-haiku-5-5", "label": "Haiku 5.5", "efforts": []},
]
DEFAULT_RESEARCH_MODEL, DEFAULT_RESEARCH_EFFORT = "claude-opus-5-5", "high"


def research_choice(model, effort):
    """Validate a model and effort from the to-do page. Returns (model, effort or None)."""
    entry = next((m for m in RESEARCH_MODELS if m["id"] == model), None)
    if entry is None:
        raise T3Error(f"Unknown research model {model}.")
    if not entry["efforts"]:
        return model, None
    if effort not in entry["efforts"]:
        raise T3Error(f"{entry['label']} doesn't offer {effort} effort.")
    return model, effort


def describe_choice(model, effort):
    """"Opus 5.5 · High", for showing which model a research thread uses."""
    label = next((m["label"] for m in RESEARCH_MODELS if m["id"] == model), model)
    return f"{label} · {EFFORT_LABELS.get(effort, effort)}" if effort else f"{label} · default effort"


class T3Error(Exception):
    pass


def _git(cwd, *args, timeout=120):
    out = subprocess.run(["git", "-C", str(cwd), *args], capture_output=True, text=True, timeout=timeout)
    if out.returncode != 0:
        raise T3Error(f"git {' '.join(args[:2])} failed: {(out.stderr or out.stdout).strip()[:300]}")
    return out.stdout


def create_worktree(project_root, base):
    """Check out a new t3code/<id> branch from `base` in its own worktree, named and placed the way T3 names and
    places its own. For moving an existing thread; T3 makes a new thread's worktree itself."""
    branch = f"t3code/{uuid.uuid4().hex[:8]}"
    path = WORKTREES_DIR / Path(project_root).name / branch.replace("/", "-")
    path.parent.mkdir(parents=True, exist_ok=True)
    _git(project_root, "worktree", "add", "-b", branch, str(path), base)
    if (path / ".gitmodules").exists():
        _git(path, "submodule", "update", "--init", "--recursive", timeout=600)
    return branch, str(path)


def remove_worktree(project_root, path, branch):
    """Remove a worktree and its branch. Refuses (git's own safety checks) if either holds unsaved work."""
    _git(project_root, "worktree", "remove", path)
    _git(project_root, "branch", "-d", branch)


class T3Client:
    def __init__(self):
        try:
            self.origin = json.loads(RUNTIME_PATH.read_text())["origin"]
        except (OSError, ValueError, KeyError):
            self.origin = "http://127.0.0.1:3773"
        try:
            self.token = TOKEN_PATH.read_text().strip()
        except OSError:
            raise T3Error(f"No T3 token at {TOKEN_PATH}. Create one with: "
                          "t3 auth session issue --ttl 365d --label personalwisper --token-only")

    def _request(self, method, path, body=None, auth=True, headers=None):
        headers = {"Content-Type": "application/json", **(headers or {})}
        if auth:
            headers["Authorization"] = f"Bearer {self.token}"
        data = json.dumps(body).encode() if body is not None else None
        req = urllib.request.Request(self.origin + path, data=data, method=method, headers=headers)
        try:
            with urllib.request.urlopen(req, timeout=20) as resp:
                return json.loads(resp.read() or b"null")
        except urllib.error.HTTPError as e:
            detail = e.read().decode(errors="replace")[:500]
            if e.code == 401:
                raise T3Error("T3 rejected the token (expired or revoked). Issue a new one.") from e
            raise T3Error(f"T3 {method} {path} failed: HTTP {e.code} {detail}") from e
        except urllib.error.URLError as e:
            raise T3Error(f"Can't reach T3 Code at {self.origin}. Is T3 Code running?") from e

    def check_compatible(self):
        env = self._request("GET", "/.well-known/t3/environment", auth=False)
        version = env.get("orchestrationProtocolVersion")
        if version != SUPPORTED_PROTOCOL:
            raise T3Error(f"T3 Code {env.get('serverVersion')} uses orchestration protocol v{version}; "
                          f"PersonalWisper supports v{SUPPORTED_PROTOCOL}. The T3 integration needs updating.")
        return env

    def _rpc(self, method, payload):
        """One call over T3's WebSocket. Returns its result or raises T3Error with T3's message."""
        url = self.origin.replace("http", "ws", 1) + f"/ws?orchestrationProtocol={SUPPORTED_PROTOCOL}"
        try:
            with connect(url, additional_headers={"Authorization": f"Bearer {self.token}"}, open_timeout=10,
                         max_size=None) as ws:
                ws.send(json.dumps({"_tag": "Request", "id": "1", "tag": method, "payload": payload, "headers": []}))
                while True:
                    msg = json.loads(ws.recv(timeout=60))
                    if msg.get("_tag") == "Ping":
                        ws.send(json.dumps({"_tag": "Pong"}))
                    elif msg.get("_tag") == "Defect":
                        raise T3Error(f"T3 {method} failed: {str(msg.get('defect'))[:300]}")
                    elif msg.get("_tag") == "Exit" and msg.get("requestId") == "1":
                        break
        except (OSError, TimeoutError, WebSocketException) as e:
            if "401" in str(e):
                raise T3Error("T3 rejected the token (expired or revoked). Issue a new one.") from e
            raise T3Error(f"Can't reach T3 Code at {self.origin} ({e}). Is T3 Code running?") from e
        exit_ = msg["exit"]
        if exit_["_tag"] == "Success":
            return exit_.get("value")
        cause = (exit_.get("cause") or [{}])[0]
        detail = (cause.get("error") or {}).get("message") or cause.get("defect") or cause
        raise T3Error(f"T3 {method} failed: {str(detail)[:300]}")

    def _dispatch(self, command):
        return self._rpc("orchestration.dispatchCommand", {"commandId": str(uuid.uuid4()), **command})

    def shell(self):
        return self._request("GET", "/api/orchestration/shell", headers={PROTOCOL_HEADER: str(SUPPORTED_PROTOCOL)})

    def projects(self):
        return [{"id": p["id"], "title": p["title"], "workspaceRoot": p.get("workspaceRoot")}
                for p in self.shell()["projects"] if not p.get("deletedAt")]

    def thread(self, thread_id):
        """The thread's full projection: thread, runs, messages, plans and more."""
        return self._request("GET", f"/api/orchestration/threads/{urllib.parse.quote(thread_id, safe='')}",
                             headers={PROTOCOL_HEADER: str(SUPPORTED_PROTOCOL)})["projection"]

    @staticmethod
    def _model(model, effort):
        return {"instanceId": CLAUDE_INSTANCE, "model": model,
                "options": [{"id": "effort", "value": effort}] if effort else []}

    def start_thread(self, kind, project_id, title, prompt, model, effort, worktree_base=None):
        """Create a research (read-only, plan mode) or work thread in the project and send it the prompt.
        With worktree_base, T3 makes the thread a new worktree on a new branch made from that branch.
        Returns (thread id, branch, worktree path); T3 may still be making the worktree, so they can be None
        until thread_status reports them."""
        self.check_compatible()
        if not any(p["id"] == project_id for p in self.projects()):
            raise T3Error(f"T3 project {project_id} no longer exists.")
        workspace = {"type": "worktree", "baseRef": worktree_base, "startFromOrigin": False} \
            if worktree_base else {"type": "root"}
        launched = self._rpc("orchestration.launchThread", {
            "commandId": str(uuid.uuid4()), "creationSource": "web", "projectId": project_id, "title": title,
            "modelSelection": self._model(model, effort), "runtimeMode": RUNTIME_MODE,
            "interactionMode": INTERACTION_MODES[kind], "workspaceStrategy": workspace,
            "initialMessage": {"messageId": str(uuid.uuid4()), "text": prompt, "attachments": []},
        })
        thread = launched["projection"]["thread"]
        return launched["threadId"], thread.get("branch"), thread.get("worktreePath")

    @staticmethod
    def _summary(shell, thread_id):
        return next((t for t in shell["threads"] + shell.get("archivedThreads", []) if t["id"] == thread_id), None)

    def find_thread(self, thread_id):
        """The thread's summary, unarchiving it first if it's archived. None if it was deleted."""
        found = self._summary(self.shell(), thread_id)
        if found is None:  # the shell leaves archived threads out
            try:
                self._dispatch({"type": "thread.unarchive", "threadId": thread_id})
            except T3Error:
                return None
            found = self._summary(self.shell(), thread_id)
        return None if found is None or found.get("deletedAt") else found

    def move_to_worktree(self, thread_id, project_root, base):
        """Give an existing thread a new worktree. T3 restarts the agent's session there and resumes the same
        conversation on its next turn. Returns (branch, worktree path)."""
        branch, path = create_worktree(project_root, base)
        try:
            self._dispatch({"type": "thread.metadata.update", "threadId": thread_id, "branch": branch,
                            "worktreePath": path})
        except T3Error:
            remove_worktree(project_root, path, branch)
            raise
        return branch, path

    def send_turn(self, kind, thread_id, text, model, effort):
        """Send another message to an existing thread, as research (plan mode) or work. Returns the id of the run
        that was latest before this one, so its answer isn't mistaken for the new one."""
        self.check_compatible()
        thread = self.find_thread(thread_id)
        if thread is None:
            raise T3Error("That thread no longer exists in T3. Start over in a new thread instead.")
        if thread.get("interactionMode") != INTERACTION_MODES[kind]:
            self._dispatch({"type": "thread.interaction-mode.set", "threadId": thread_id,
                            "interactionMode": INTERACTION_MODES[kind]})
        if thread.get("runtimeMode") != RUNTIME_MODE:
            self._dispatch({"type": "thread.runtime-mode.set", "threadId": thread_id, "runtimeMode": RUNTIME_MODE})
        busy = thread.get("activeRunId") is not None
        self._dispatch({
            "type": "message.dispatch", "threadId": thread_id, "messageId": str(uuid.uuid4()), "text": text,
            "attachments": [], "modelSelection": self._model(model, effort), "createdBy": "user",
            "creationSource": "web", "dispatchMode": {"type": "queue_after_active" if busy else "start_immediately"},
        })
        return thread.get("latestRunId")

    def archive_thread(self, thread_id):
        self._dispatch({"type": "thread.archive", "threadId": thread_id})

    def thread_status(self, thread_id, shell=None):
        """Returns a dict with "state" (running, needs-input, completed, error, missing), the thread's "branch" and
        "worktree_path", and, when completed, "turn_id" (its latest run's id), "reply" (the agent's last message)
        and "plan" (that run's plan-mode report, if any)."""
        summary = self._summary(shell or self.shell(), thread_id)
        if summary is None or summary.get("deletedAt"):
            return {"state": "missing", "note": "The thread was deleted in T3."}
        where = {"branch": summary.get("branch"), "worktree_path": summary.get("worktreePath")}
        return {**where, **self._thread_state(thread_id, summary)}

    def _thread_state(self, thread_id, summary):
        request = summary.get("pendingRuntimeRequest")
        if request and request.get("blocking") is not False:
            return {"state": "needs-input", "note": "The thread is waiting for your answer in T3."}
        status, run_id = summary.get("status"), summary.get("latestRunId")
        if status in ("preparing", "queued", "starting", "running", "waiting") or run_id is None:
            return {"state": "running", "note": None}
        if status in ("interrupted", "failed", "cancelled", "rolled_back"):
            return {"state": "error", "note": summary.get("lastError") or f"The run was {status.replace('_', ' ')}."}
        # T3 starts a new run on its own when subagents or commands the agent waits on end. (A command it doesn't
        # wait on, like a dev server left running, doesn't hold the thread.)
        pending = [t.get("description") or "background task" for t in summary.get("pendingBackgroundTasks") or []
                   if t.get("kind") == "subagent" or t.get("awaited")]
        if pending:
            return {"state": "running", "note": f"Waiting on background work: {', '.join(pending)}"}

        detail = self.thread(thread_id)
        replies = [m for m in detail.get("messages", []) if m["role"] == "assistant" and m.get("text", "").strip()]
        reply = next((m for m in reversed(replies) if m.get("runId") == run_id), replies[-1] if replies else None)
        plans = [p for p in detail.get("plans", []) if p.get("runId") == run_id and p.get("markdown")]
        return {"state": "completed", "turn_id": run_id, "reply": (reply or {}).get("text", "").strip(),
                "plan": plans[-1]["markdown"] if plans else None}
