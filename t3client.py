"""Minimal client for the local T3 Code server: list projects, start research threads, read results.

Uses T3's orchestration HTTP API (protocol v1): POST /api/orchestration/dispatch for commands and
GET /api/orchestration/shell and /api/orchestration/threads/:id for state. These are T3 internals, not a
public API, so check_compatible() verifies the protocol version before anything is sent.

Auth: a bearer token from `t3 auth session issue --label personalwisper --token-only`, stored in secrets/t3-token.
"""
import json
import subprocess
import urllib.error
import urllib.request
import uuid
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent
TOKEN_PATH = ROOT / "secrets" / "t3-token"
RUNTIME_PATH = Path.home() / ".t3" / "userdata" / "server-runtime.json"
WORKTREES_DIR = Path.home() / ".t3" / "worktrees"  # where T3 keeps its own thread worktrees
SUPPORTED_PROTOCOL = 1
# T3 0.1.0 moved to protocol v2: listing projects still works (with this header), but the HTTP dispatch endpoint
# is gone, so starting threads stays blocked by check_compatible() until the integration is ported.
LISTING_PROTOCOL = 2
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
    places its own (T3's HTTP API can't create worktrees; only its app's live connection can)."""
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


def _now():
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


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

    def shell(self, protocol=SUPPORTED_PROTOCOL):
        return self._request("GET", "/api/orchestration/shell", headers={PROTOCOL_HEADER: str(protocol)})

    def projects(self):
        # Protocol v2 (T3 0.1.0) still lists projects the same way, but rejects shell requests without its header.
        shell = self.shell(LISTING_PROTOCOL)
        return [{"id": p["id"], "title": p["title"], "workspaceRoot": p.get("workspaceRoot")}
                for p in shell["projects"] if not p.get("deletedAt")]

    def thread(self, thread_id):
        return self._request("GET", f"/api/orchestration/threads/{thread_id}")["thread"]

    def _dispatch(self, command):
        return self._request("POST", "/api/orchestration/dispatch", command)

    def start_thread(self, kind, project_id, title, prompt, model, effort, worktree_base=None):
        """Create a research (read-only, plan mode) or work thread in the project and send it the prompt.
        With worktree_base, the thread works in a new worktree on a new t3code/<id> branch made from that branch.
        Returns (thread id, branch, worktree path); branch and path are None without a worktree."""
        mode = INTERACTION_MODES[kind]
        self.check_compatible()
        project = next((p for p in self.shell()["projects"] if p["id"] == project_id), None)
        if project is None:
            raise T3Error(f"T3 project {project_id} no longer exists.")
        model = {"instanceId": CLAUDE_INSTANCE, "model": model,
                 "options": [{"id": "effort", "value": effort}] if effort else []}
        thread_id = str(uuid.uuid4())
        branch = worktree_path = None
        if worktree_base:
            branch, worktree_path = create_worktree(project["workspaceRoot"], worktree_base)
        try:
            self._dispatch({
                "type": "thread.create", "commandId": str(uuid.uuid4()), "threadId": thread_id,
                "projectId": project_id, "title": title, "modelSelection": model,
                "runtimeMode": RUNTIME_MODE, "interactionMode": mode,
                "branch": branch, "worktreePath": worktree_path, "createdAt": _now(),
            })
        except T3Error:
            if worktree_path:  # brand new and unused: don't leave it behind
                remove_worktree(project["workspaceRoot"], worktree_path, branch)
            raise
        self._dispatch({
            "type": "thread.turn.start", "commandId": str(uuid.uuid4()), "threadId": thread_id,
            "message": {"messageId": str(uuid.uuid4()), "role": "user", "text": prompt, "attachments": []},
            "modelSelection": model, "runtimeMode": RUNTIME_MODE,
            "interactionMode": mode, "createdAt": _now(),
        })
        return thread_id, branch, worktree_path

    def find_thread(self, thread_id):
        """The thread's summary, unarchiving it first if it's archived. None if it was deleted."""
        def summary():
            return next((t for t in self.shell()["threads"] if t["id"] == thread_id), None)
        found = summary()
        if found is None:
            try:
                self._dispatch({"type": "thread.unarchive", "commandId": str(uuid.uuid4()), "threadId": thread_id})
            except T3Error:
                return None
            found = summary()
        return found

    def move_to_worktree(self, thread_id, project_root, base):
        """Give an existing thread a new worktree. T3 restarts the agent's session there and resumes the same
        conversation on its next turn. Returns (branch, worktree path)."""
        branch, path = create_worktree(project_root, base)
        try:
            self._dispatch({"type": "thread.meta.update", "commandId": str(uuid.uuid4()), "threadId": thread_id,
                            "branch": branch, "worktreePath": path})
        except T3Error:
            remove_worktree(project_root, path, branch)
            raise
        return branch, path

    def send_turn(self, kind, thread_id, text, model, effort):
        """Send another message to an existing thread, as research (plan mode) or work. Returns the id of the turn
        that was latest before this one, so its answer isn't mistaken for the new one."""
        self.check_compatible()
        thread = self.find_thread(thread_id)
        if thread is None:
            raise T3Error("That thread no longer exists in T3. Start over in a new thread instead.")
        self._dispatch({
            "type": "thread.turn.start", "commandId": str(uuid.uuid4()), "threadId": thread_id,
            "message": {"messageId": str(uuid.uuid4()), "role": "user", "text": text, "attachments": []},
            "modelSelection": {"instanceId": CLAUDE_INSTANCE, "model": model,
                               "options": [{"id": "effort", "value": effort}] if effort else []},
            "runtimeMode": RUNTIME_MODE, "interactionMode": INTERACTION_MODES[kind], "createdAt": _now(),
        })
        return (thread.get("latestTurn") or {}).get("turnId")

    def archive_thread(self, thread_id):
        self._dispatch({"type": "thread.archive", "commandId": str(uuid.uuid4()), "threadId": thread_id})

    @staticmethod
    def _pending_tasks(detail):
        """Background tasks (e.g. subagents) the agent started but that haven't ended. When they finish,
        T3 starts a new turn automatically, so the research isn't done while any are pending."""
        started, ended = {}, set()
        for activity in detail.get("activities", []):
            payload = activity.get("payload") or {}
            task_id = payload.get("taskId")
            if not task_id:
                continue
            if activity["kind"] == "task.started":
                started[task_id] = payload.get("title") or "background task"
            elif activity["kind"] in ("task.completed", "task.failed", "task.stopped", "task.cancelled") \
                    or payload.get("endedAt"):
                ended.add(task_id)
        return [title for task_id, title in started.items() if task_id not in ended]

    def thread_status(self, thread_id, shell=None):
        """Returns a dict with "state" (running, needs-input, completed, error, missing), the thread's "branch" and
        "worktree_path", and, when completed, "turn_id", "reply" (the agent's last message) and "plan" (its latest
        plan-mode report, if any)."""
        shell = shell or self.shell()
        summary = next((t for t in shell["threads"] if t["id"] == thread_id), None)
        if summary is None:
            return {"state": "missing", "note": "The thread was deleted in T3."}
        where = {"branch": summary.get("branch"), "worktree_path": summary.get("worktreePath")}
        status = self._thread_state(thread_id, summary)
        return {**where, **status}

    def _thread_state(self, thread_id, summary):
        if summary.get("hasPendingApprovals") or summary.get("hasPendingUserInput"):
            return {"state": "needs-input", "note": "The thread is waiting for your answer in T3."}
        turn = summary.get("latestTurn") or {}
        session = summary.get("session") or {}
        if turn.get("state") in (None, "running") or session.get("status") in ("starting", "running"):
            return {"state": "running", "note": None}
        if turn["state"] in ("error", "interrupted"):
            return {"state": "error", "note": session.get("lastError") or f"The turn was {turn['state']}."}

        detail = self.thread(thread_id)
        pending = self._pending_tasks(detail)
        if pending:
            return {"state": "running", "note": f"Waiting on background work: {', '.join(pending)}"}
        messages = detail.get("messages", [])
        reply = next((m for m in messages if m["id"] == turn.get("assistantMessageId")), None) or next(
            (m for m in reversed(messages) if m["role"] == "assistant"), None)
        plans = sorted(detail.get("proposedPlans", []), key=lambda p: p.get("updatedAt") or p.get("createdAt"))
        return {"state": "completed", "turn_id": turn.get("turnId"),
                "reply": (reply or {}).get("text", "").strip(), "plan": plans[-1]["planMarkdown"] if plans else None}
