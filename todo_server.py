"""The to-do list: voice capture, the local web page, and T3 research threads.

The web page is served on 127.0.0.1 only. State-changing requests must carry an X-PersonalWisper header, which
browsers won't let other websites send cross-origin, so random web pages can't start research threads.
"""
import json
import mimetypes
import re
import subprocess
import threading
import time
import traceback
from datetime import datetime, timedelta, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from markdown_it import MarkdownIt

from pipeline import todo_eval_message
from t3client import (DEFAULT_RESEARCH_EFFORT, DEFAULT_RESEARCH_MODEL, EFFORT_LABELS, RESEARCH_MODELS, T3Client,
                      T3Error, describe_choice, research_choice)
from todos import STATUSES, TodoStore, now

ROOT = Path(__file__).resolve().parent
WEB_DIR = ROOT / "web"
PORT = 47615
DEFAULT_PROJECT_TITLE = None  # None: the first project T3 lists
POLL_SECONDS = 15
# After research finishes, keep watching this long in case T3 continues the thread with another turn.
WATCH_AFTER_FINISH = timedelta(minutes=30)
PROJECTS_CACHE_SECONDS = 60
T3_BUNDLE_ID = "com.t3tools.t3code"

RESEARCH_PROMPT = """Research request from my PersonalWisper to-do list. This is research only: do not modify any files.

# {title}

## My notes
{notes}
{prepared}{questions}
## Please report back with
1. What I'm asking for, in your own words, and anything ambiguous you had to assume.
2. The existing systems, files, and data this would touch (with paths).
3. A recommended implementation approach.
4. Difficulty (small / medium / large / very large), a rough effort estimate, and why.
5. Risks, open questions, and decisions I need to make.
"""

WORK_PROMPT = """Work request from my PersonalWisper to-do list. Please do this in this project.

# {title}

## My notes
{notes}
{prepared}{research}
## When you're done
Tell me what you changed (with file paths), how you checked it works, and anything I should test or decide.
"""

FOLLOWUP_PROMPTS = {
    "research": """Follow-up from my PersonalWisper to-do list on your research above. This is still research only: \
do not modify any files.

{question}

Build on what you already found and don't repeat it: report only what's new for this follow-up.""",
    "work": """Follow-up from my PersonalWisper to-do list on the work above.

{question}

When you're done, tell me what you changed (with file paths), how you checked it, and anything I should test.""",
}

# The next step for an item whose T3 thread already exists: sent as a new turn in that thread, which already
# has everything said and found so far, so only what changed is included.
NEXT_STEP_PROMPTS = {
    "research": """Next step from my PersonalWisper to-do list: research this, building on everything above. This is \
research only: do not modify any files.

# {title}
{context}{questions}
Report back with what's new: what I'm asking for and anything you had to assume, what it would touch (with paths), \
a recommended approach, difficulty and effort, and the risks and decisions I need to make.""",
    "work": """Next step from my PersonalWisper to-do list: please do this work now, building on everything above. \
You can change files now.{where}

# {title}
{context}
When you're done, tell me what you changed (with file paths), how you checked it works, and anything I should test \
or decide.""",
}

# Items the local evaluator works on (it re-runs when their notes, title or answers change).
EVALUATED_STATUSES = ("dirty", "research", "work")

# A research or work T3 thread: the item's fields that track it, and its statuses while running and when finished.
RUNS = {
    "research": {"thread": "thread_id", "state": "research_state", "note": "research_note", "result": "result",
                 "plan": "result_plan", "turn": "result_turn_id", "finished": "finished_at",
                 "model": "research_model", "effort": "research_effort", "followup": "followup",
                 "running": "researching", "finished_status": "researched", "title": "Research"},
    "work": {"thread": "work_thread_id", "state": "work_state", "note": "work_note", "result": "work_result",
             "plan": "work_plan", "turn": "work_turn_id", "finished": "work_finished_at",
             "model": "work_model", "effort": "work_effort", "followup": "work_followup",
             "running": "working", "finished_status": "done", "title": "Work"},
}


def run_kind(item):
    """Which T3 thread an item's status is about: its work thread once it's a work item, else its research."""
    return "work" if item["status"] in ("work", "working", "done") else "research"

_markdown = MarkdownIt("commonmark", {"html": False}).enable("table").enable("strikethrough")


def log(message):
    print(f"{time.strftime('%Y-%m-%d %H:%M:%S')}  [todo] {message}", flush=True)


class TodoService:
    """Everything the to-do list does. Model work is handed to `run_on_model_thread(fn) -> Future`,
    where fn receives the loaded Pipeline (MLX must stay on one thread)."""

    def __init__(self, run_on_model_thread, recorder_factory, on_change, config):
        self.store = TodoStore()
        self.run_on_model_thread = run_on_model_thread
        self.recorder = recorder_factory()
        self.recording_item = None
        self.on_change = on_change
        self.config = config
        self._projects = []
        self._projects_at = 0.0
        self._settling = {}  # item id -> turn id that looked finished on the previous check
        self._eval_lock = threading.Lock()
        # Item ids waiting for local evaluation, oldest first. Items that never finished one are picked up again.
        self._eval_queue = [i["id"] for i in reversed(self.store.list()) if i["status"] in EVALUATED_STATUSES
                            and (not i["evaluation"] or i["eval_state"] in ("queued", "running"))]
        self.t3_error = None
        try:
            self.t3 = T3Client()
        except T3Error as e:
            self.t3, self.t3_error = None, str(e)

    # ---- T3 projects ----

    def projects(self, refresh=False):
        if self.t3 and (refresh or time.time() - self._projects_at > PROJECTS_CACHE_SECONDS):
            try:
                self._projects = [p for p in self.t3.projects() if p["title"] != "No project"]
                self._projects_at = time.time()
                self.t3_error = None
            except T3Error as e:
                self.t3_error = str(e)
        return self._projects

    def default_project(self):
        wanted = self.config.get("todo_project", DEFAULT_PROJECT_TITLE)
        projects = self.projects()
        return next((p for p in projects if p["title"] == wanted), projects[0] if projects else None)

    # ---- capture and refine (called on the model thread) ----

    def capture(self, pipe, raw_text):
        title, notes = pipe.todo_from_speech(raw_text)
        project = self.default_project()
        item_id = self.store.create(title, notes, raw_text,
                                    project and project["id"], project and project["title"])
        log(f"captured #{item_id}: {title}")
        self.queue_evaluation(item_id)
        return title

    def _apply_voice(self, pipe, item_id, target, raw_text):
        item = self.store.get(item_id)
        if item is None:
            raise ValueError("That item no longer exists.")
        if target == "questions":
            cleaned = pipe.cleanup(raw_text)[0]
            self.store.update(item_id, brief=f"{item['brief']}\n{cleaned}".strip())
            self.store.add_event(item_id, "spoke-questions", raw_text)
        elif target == "answer":
            self.answer_questions(item_id, pipe.cleanup(raw_text)[0])
        elif target == "followup":
            return pipe.cleanup(raw_text)[0]  # goes into the follow-up box; nothing is saved until you send it
        else:
            title, notes, _ = pipe.refine_todo(item["title"], item["notes"], raw_text)
            self.store.update(item_id, title=title, notes=notes)
            self.store.add_event(item_id, "refined", raw_text)
            self.queue_evaluation(item_id)

    # ---- local evaluation (a clearer brief plus questions for you, made by the local model) ----

    def queue_evaluation(self, item_id):
        """Evaluate an item again when the model thread is free. Safe to call from any thread."""
        with self._eval_lock:
            if item_id not in self._eval_queue:
                self._eval_queue.append(item_id)
        self.store.update(item_id, eval_state="queued", eval_error=None, touch=False)
        self.on_change()
        self.run_on_model_thread(lambda pipe: None)  # wakes the model thread if it's idle

    def next_evaluation(self, pipe):
        """Called on the model thread when it has nothing else to do. Returns a generator that evaluates the next
        queued item a few tokens at a time (so dictation can run in between), or None if nothing is queued."""
        with self._eval_lock:
            while self._eval_queue:
                item = self.store.get(self._eval_queue.pop(0))
                if item and item["status"] in EVALUATED_STATUSES:
                    return self._evaluate(pipe, item)
        return None

    def _evaluate(self, pipe, item):
        rounds = json.loads(item["eval_rounds"] or "[]")
        evaluated = todo_eval_message(item["title"], item["notes"], rounds)
        self.store.update(item["id"], eval_state="running", touch=False)
        self.on_change()
        done = False
        try:
            try:
                result = yield from pipe.evaluate_todo_steps(item["title"], item["notes"], rounds,
                                                             item["project_title"])
            except ValueError:  # reply not in the expected format; one retry usually fixes it
                result = yield from pipe.evaluate_todo_steps(item["title"], item["notes"], rounds,
                                                             item["project_title"])
            done = True
        except Exception as e:
            done = True
            log(f"evaluation failed for #{item['id']}: {e}")
            self.store.update(item["id"], eval_state="failed", eval_error=str(e), touch=False)
            self.on_change()
            return
        finally:
            if not done:  # interrupted (the cleanup model was switched): try again later
                self.queue_evaluation(item["id"])
        latest = self.store.get(item["id"])
        if latest is None:
            return
        self.store.update(item["id"], evaluation=json.dumps(result), eval_state="done", eval_input=evaluated,
                          touch=False)
        questions = len(result["questions"])
        self.store.add_event(item["id"], "evaluated", f"{questions} question{'s' * (questions != 1)} for you"
                             if questions else "Ready for research")
        log(f"evaluated #{item['id']}: {questions} question{'s' * (questions != 1)}")
        if todo_eval_message(latest["title"], latest["notes"], json.loads(latest["eval_rounds"] or "[]")) != evaluated:
            self.queue_evaluation(item["id"])  # edited while it was being evaluated
        else:
            self.on_change()

    def answer_questions(self, item_id, answer):
        """Record your answer to the current questions, then evaluate again with it."""
        item = self.store.get(item_id)
        if item is None:
            raise ValueError("That item no longer exists.")
        evaluation = json.loads(item["evaluation"] or "{}")
        rounds = json.loads(item["eval_rounds"] or "[]")
        rounds.append({"questions": [q["question"] for q in evaluation.get("questions", [])], "answer": answer})
        self.store.update(item_id, eval_rounds=json.dumps(rounds))
        self.store.add_event(item_id, "answered", answer)
        self.queue_evaluation(item_id)

    # ---- web voice ----

    def voice_start(self, item_id):
        if self.recording_item is not None:
            raise ValueError("Already listening.")
        self.recorder.start()
        self.recording_item = item_id

    def voice_stop(self, item_id, target):
        if self.recording_item != item_id:
            raise ValueError("Not listening for this item.")
        audio, _ = self.recorder.stop()
        self.recording_item = None
        if len(audio) < 16000 * 0.3:
            raise ValueError("That was too short to hear anything.")

        def work(pipe):
            raw = pipe.transcribe(audio)
            if not raw:
                raise ValueError("I didn't catch any words.")
            return raw, self._apply_voice(pipe, item_id, target, raw)

        return self.run_on_model_thread(work).result(timeout=120)

    def voice_cancel(self):
        if self.recording_item is not None:
            self.recorder.stop()
            self.recording_item = None

    # ---- research and work threads in T3 ----

    def work_places(self, item):
        """Where a work thread can run. An item whose thread already works in a worktree keeps it. Otherwise: a new
        worktree (from the project's base branch in config.json's work_worktree_base, else from the branch the
        project folder is on) or the project folder itself."""
        if item.get("work_worktree"):
            return [{"id": "same", "label": f"Its worktree ({item['work_branch']})", "branch": item["work_branch"]}]
        project = next((p for p in self.projects() if p["id"] == item.get("project_id")), None)
        root = project and project.get("workspaceRoot")
        if not root:
            return []
        current = git_branch(root)
        places = [{"id": "local", "label": f"Project folder, on {current or 'its current branch'}", "branch": current}]
        base = self.config.get("work_worktree_base", {}).get(project["title"]) or current
        if base:
            places.insert(0, {"id": "worktree", "label": f"New worktree from {base}", "base": base, "root": root})
        return places

    def default_work_place(self, item):
        """Its existing worktree; else a new worktree for projects in config.json's work_worktree_base; else the
        project folder."""
        ids = [p["id"] for p in self.work_places(item)]
        if "same" in ids:
            return "same"
        wants_worktree = item.get("project_title") in self.config.get("work_worktree_base", {})
        return "worktree" if wants_worktree and "worktree" in ids else (ids[-1] if ids else None)

    def _thread_context(self, item):
        """The notes and local brief as last sent to the item's thread, to send only what changed."""
        return f"## My notes\n{item['notes'].strip() or '(none)'}\n{prepared_brief(item)}"

    def start_run(self, item_id, kind, model, effort, place=None, fresh=False):
        """Research or work on an item in T3. If the item already has a thread, this continues it (it already has
        everything so far), unless fresh is set or the thread was deleted; then it starts a new thread."""
        if self.t3 is None:
            raise T3Error(self.t3_error)
        model, effort = research_choice(model, effort)
        item = self.store.get(item_id)
        if not item.get("project_id"):
            raise ValueError("Pick a T3 project for this item first.")
        existing = None if fresh else (item["work_thread_id"] or item["thread_id"])
        if existing and self.t3.find_thread(existing) is None:
            existing = None  # deleted in T3: start a new one
        chosen = None
        if kind == "work":
            # A new thread never reuses an old thread's worktree: it gets a new one (or the project folder).
            places = {p["id"]: p for p in self.work_places(item if existing else {**item, "work_worktree": None})}
            if not existing and place == "same":
                place = "worktree" if "worktree" in places else "local"
            if place not in places:
                raise ValueError("Pick where the work happens.")
            chosen = places[place]
        run = RUNS[kind]
        context = self._thread_context(item)
        branch, worktree_path = (item["work_branch"], item["work_worktree"]) if existing else (None, None)
        self._keep_result(item, kind)
        if existing:
            where = ""
            if chosen and chosen["id"] == "worktree":
                branch, worktree_path = self.t3.move_to_worktree(existing, chosen["root"], chosen["base"])
                where = (f" You're now working in a new worktree at {worktree_path} (branch {branch}, made from "
                         f"{chosen['base']}); make your changes there, not in the folder you used before.")
            elif chosen and chosen["id"] == "local":
                branch = chosen.get("branch")
            same = context == item["thread_context"]
            body = "My notes and brief haven't changed since I last sent them.\n" if same else \
                f"Updated since I last sent them:\n\n{context}"
            questions = f"\n## Questions I want answered\n{item['brief'].strip()}\n" \
                if kind == "research" and item["brief"].strip() else ""
            prompt = NEXT_STEP_PROMPTS[kind].format(title=item["title"], context=body, questions=questions,
                                                    where=where)
            previous_turn = self.t3.send_turn(kind, existing, prompt, model, effort)
            thread_id = existing
            started = f"continued T3 thread {thread_id}"
        else:
            notes = item["notes"].strip() or "(none)"
            if kind == "research":
                questions = f"\n## Questions I want answered\n{item['brief'].strip()}\n" if item["brief"].strip() else ""
                prompt = RESEARCH_PROMPT.format(title=item["title"], notes=notes, prepared=prepared_brief(item),
                                                questions=questions)
            else:
                prompt = WORK_PROMPT.format(title=item["title"], notes=notes, prepared=prepared_brief(item),
                                            research=research_section(item))
            thread_id, branch, worktree_path = self.t3.start_thread(
                kind, item["project_id"], f"{run['title']}: {item['title']}"[:80], prompt, model, effort,
                chosen.get("base") if chosen and chosen["id"] == "worktree" else None)
            if chosen and chosen["id"] == "local":
                branch = chosen.get("branch")
            previous_turn = None
            started = f"new T3 thread {thread_id}"
        self.store.update(item_id, status=run["running"], thread_context=context, **{run["followup"]: None}, **{
            run["thread"]: thread_id, run["state"]: "running", run["note"]: None, run["result"]: None,
            run["plan"]: None, run["turn"]: previous_turn, run["finished"]: None, run["model"]: model,
            run["effort"]: effort})
        if kind == "work":
            self.store.update(item_id, work_place=place, work_branch=branch, work_worktree=worktree_path)
        self.store.add_event(item_id, f"{kind}-started", f"{started} in {item['project_title']} "
                             f"({describe_choice(model, effort)})")
        log(f"{kind} started for #{item_id}: {started}")
        self.on_change()

    def _keep_result(self, item, kind):
        """Move the item's current result for this kind into its history before a new one replaces it."""
        run = RUNS[kind]
        if not item[run["result"]]:
            return
        history = json.loads(item["run_history"] or "[]")
        history.append({"kind": kind, "question": item[run["followup"]], "result": item[run["result"]],
                        "plan": item[run["plan"]], "label": describe_choice(item[run["model"]], item[run["effort"]])
                        if item[run["model"]] else None, "finished_at": item[run["finished"]]})
        self.store.update(item["id"], run_history=json.dumps(history))

    def follow_up(self, item_id, question, model, effort):
        """Ask the item's finished research or work thread a follow-up, in the same T3 thread (which already
        has everything it found), instead of starting over."""
        if self.t3 is None:
            raise T3Error(self.t3_error)
        question = question.strip()
        if not question:
            raise ValueError("Type or say your follow-up first.")
        model, effort = research_choice(model, effort)
        item = self.store.get(item_id)
        kind = "work" if item["status"] == "done" else "research"
        run = RUNS[kind]
        if item["status"] != run["finished_status"] or not item[run["thread"]]:
            raise ValueError("Follow-ups are for finished research or work.")
        previous_turn = self.t3.send_turn(kind, item[run["thread"]], FOLLOWUP_PROMPTS[kind].format(question=question),
                                          model, effort)
        self._keep_result(item, kind)
        self.store.update(item_id, status=run["running"], **{run["followup"]: question}, **{
            run["state"]: "running", run["note"]: None, run["result"]: None, run["plan"]: None,
            run["turn"]: previous_turn, run["finished"]: None, run["model"]: model, run["effort"]: effort})
        self.store.add_event(item_id, f"{kind}-followup", f"{question} ({describe_choice(model, effort)})")
        log(f"{kind} follow-up for #{item_id}")
        self.on_change()

    def check_run(self, item, shell=None, confirm_now=False):
        """Update an item from its current T3 thread. A finished-looking turn is only saved once it's still the
        latest turn on the next check, because T3 starts a follow-up turn a few seconds after background
        work ends. confirm_now (the Refresh button) skips that wait."""
        kind = run_kind(item)
        run = RUNS[kind]
        if not item[run["thread"]]:
            return
        status = self.t3.thread_status(item[run["thread"]], shell)
        state, note = status["state"], status.get("note")
        if kind == "work" and status.get("branch") and \
                (status["branch"], status.get("worktree_path")) != (item["work_branch"], item["work_worktree"]):
            self.store.update(item["id"], work_branch=status["branch"], work_worktree=status.get("worktree_path"))
        settle_key = (item["id"], kind)
        if state != "completed":
            self._settling.pop(settle_key, None)
            if (state, note) != (item[run["state"]], item[run["note"]]):
                self.store.update(item["id"], **{run["state"]: state, run["note"]: note})
                self.on_change()
            return
        finished = item["status"] == run["finished_status"]
        if finished and status["turn_id"] == item[run["turn"]]:
            if item[run["state"]] != "completed":
                self.store.update(item["id"], **{run["state"]: "completed", run["note"]: None})
                self.on_change()
            return
        if item["status"] == run["running"] and status["turn_id"] == item[run["turn"]]:
            return  # the new turn in a continued thread hasn't started yet; this is still the previous answer
        if not confirm_now and self._settling.get(settle_key) != status["turn_id"]:
            self._settling[settle_key] = status["turn_id"]
            return
        self._settling.pop(settle_key, None)
        self.store.update(item["id"], status=run["finished_status"], **{
            run["state"]: "completed", run["note"]: None, run["plan"]: status["plan"], run["turn"]: status["turn_id"],
            run["result"]: status["reply"] or f"(The {kind} thread finished without a written reply.)",
            run["finished"]: now()})
        self.store.add_event(item["id"], f"{kind}-updated" if finished else f"{kind}-finished",
                             f"{run['title']} result updated from T3." if finished else f"{run['title']} result saved.")
        log(f"{kind} {'updated' if finished else 'finished'} for #{item['id']}")
        self.on_change()

    def poll_forever(self):
        while True:
            try:
                since = (datetime.now(timezone.utc) - WATCH_AFTER_FINISH).isoformat(timespec="seconds")
                items = self.store.watched(since) if self.t3 else []
                if items:
                    shell = self.t3.shell()
                    for item in items:
                        self.check_run(item, shell)
            except T3Error as e:
                log(f"T3 check failed: {e}")
            except Exception:
                log(traceback.format_exc())
            time.sleep(POLL_SECONDS)

    # ---- views ----

    def counts(self):
        counts = {s: 0 for s in STATUSES}
        for item in self.store.list():
            counts[item["status"]] += 1
        return counts

    def list_view(self):
        items = self.store.list()
        for item in items:
            evaluation = json.loads(item.pop("evaluation") or "null")
            item["open_questions"] = len(evaluation["questions"]) if evaluation else None
            for key in ("notes", "brief", "result", "result_plan", "work_result", "work_plan", "eval_rounds",
                        "eval_input", "run_history"):
                item.pop(key, None)  # the list only needs titles and states
        return items

    def item_view(self, item_id):
        item = self.store.get(item_id)
        if item is None:
            return None
        item["research_label"] = describe_choice(item["research_model"], item["research_effort"]) \
            if item["research_model"] else None
        item["work_label"] = describe_choice(item["work_model"], item["work_effort"]) if item["work_model"] else None
        item["has_thread"] = bool(item["work_thread_id"] or item["thread_id"])
        if item["status"] in ("dirty", "research", "researched", "work") or item["work_place"]:
            item["work_places"] = self.work_places(item)
            item["work_place_default"] = self.default_work_place(item)
        item["work_result_html"] = _markdown.render(item["work_result"]) if item.get("work_result") else None
        item["work_plan_html"] = _markdown.render(item["work_plan"]) if item.get("work_plan") else None
        item["run_history"] = [{**h, "html": _markdown.render(h["result"]),
                                "plan_html": _markdown.render(h["plan"]) if h.get("plan") else None}
                               for h in json.loads(item["run_history"] or "[]")]
        item["evaluation"] = json.loads(item["evaluation"] or "null")
        item["eval_rounds"] = json.loads(item["eval_rounds"] or "[]")
        item["result_html"] = _markdown.render(item["result"]) if item.get("result") else None
        item["plan_html"] = _markdown.render(item["result_plan"]) if item.get("result_plan") else None
        return item


class Handler(BaseHTTPRequestHandler):
    service: TodoService = None  # set by start_server

    def log_message(self, *args):
        pass

    # ---- plumbing ----

    def _send(self, code, body, content_type="application/json"):
        data = body if isinstance(body, bytes) else json.dumps(body).encode()
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def _json_body(self):
        length = int(self.headers.get("Content-Length") or 0)
        return json.loads(self.rfile.read(length) or b"{}") if length else {}

    def _allowed(self, write):
        # Reject requests addressed to another hostname (DNS rebinding) and unmarked writes (CSRF).
        host = (self.headers.get("Host") or "").split(":")[0]
        if host not in ("127.0.0.1", "localhost"):
            return False
        return not write or self.headers.get("X-PersonalWisper") == "1"

    def _route(self, method):
        if not self._allowed(write=method != "GET"):
            return self._send(403, {"error": "Forbidden"})
        try:
            self._dispatch(method)
        except (ValueError, T3Error) as e:
            self._send(400, {"error": str(e)})
        except Exception as e:
            log(traceback.format_exc())
            self._send(500, {"error": str(e)})

    def do_GET(self):
        self._route("GET")

    def do_POST(self):
        self._route("POST")

    def do_PATCH(self):
        self._route("PATCH")

    def do_DELETE(self):
        self._route("DELETE")

    # ---- routes ----

    def _dispatch(self, method):
        s = self.service
        path = self.path.split("?")[0]
        item_match = re.fullmatch(r"/api/items/(\d+)(/[a-z/-]+)?", path)
        item_id = int(item_match.group(1)) if item_match else None
        action = item_match.group(2) if item_match else None

        if method == "GET" and path in ("/", "/index.html"):
            return self._static("index.html")
        if method == "GET" and path.startswith("/static/"):
            return self._static(path[len("/static/"):])

        if method == "GET" and path == "/api/state":
            default = s.default_project()
            return self._send(200, {
                "items": s.list_view(), "projects": s.projects(), "t3_error": s.t3_error,
                "default_project_id": default and default["id"], "listening_item": s.recording_item,
                "hotkey": s.config.get("todo_hotkey"),
                "research_models": RESEARCH_MODELS, "effort_labels": EFFORT_LABELS,
                "research_default": {"model": DEFAULT_RESEARCH_MODEL, "effort": DEFAULT_RESEARCH_EFFORT},
            })
        if method == "POST" and path == "/api/projects/refresh":
            return self._send(200, {"projects": s.projects(refresh=True), "t3_error": s.t3_error})
        if method == "POST" and path == "/api/open-t3":
            subprocess.Popen(["/usr/bin/open", "-b", T3_BUNDLE_ID])
            return self._send(200, {"ok": True})
        if method == "POST" and path == "/api/items":
            body = self._json_body()
            title = (body.get("title") or "").strip()
            if not title:
                raise ValueError("A title is required.")
            project = s.default_project()
            new_id = s.store.create(title, (body.get("notes") or "").strip(), "(typed)",
                                    project and project["id"], project and project["title"])
            s.queue_evaluation(new_id)
            return self._send(200, s.item_view(new_id))

        if item_id is None:
            return self._send(404, {"error": "Not found"})
        if s.store.get(item_id) is None:
            return self._send(404, {"error": "That item no longer exists."})

        if method == "GET" and action is None:
            return self._send(200, s.item_view(item_id))
        if method == "PATCH" and action is None:
            body = self._json_body()
            fields = {k: body[k] for k in ("title", "notes", "brief", "status") if k in body}
            if "project_id" in body:
                project = next((p for p in s.projects() if p["id"] == body["project_id"]), None)
                if project is None:
                    raise ValueError("Unknown T3 project.")
                fields.update(project_id=project["id"], project_title=project["title"])
            if fields.get("status") in ("researching", "working"):
                raise ValueError("Use Start research or Start work to begin.")
            item = s.store.get(item_id)
            if fields.get("status") == "research" and not item["brief"].strip() and item["evaluation"]:
                research = json.loads(item["evaluation"])["research"]  # start from the evaluator's suggestions
                fields["brief"] = "\n".join(f"- {r}" for r in research)
            s.store.update(item_id, **fields)
            if any(fields.get(k, item[k]) != item[k] for k in ("title", "notes")) and \
                    s.store.get(item_id)["status"] in EVALUATED_STATUSES:
                s.queue_evaluation(item_id)
            if "status" in fields:
                s.store.add_event(item_id, "status", fields["status"])
            s.on_change()
            return self._send(200, s.item_view(item_id))
        if method == "DELETE" and action is None:
            s.store.delete(item_id)
            s.on_change()
            return self._send(200, {"ok": True})
        if method == "POST" and action == "/voice/start":
            s.voice_start(item_id)
            return self._send(200, {"ok": True})
        if method == "POST" and action == "/voice/stop":
            target = self._json_body().get("target", "notes")
            heard, text = s.voice_stop(item_id, target)
            s.on_change()
            return self._send(200, {"heard": heard, "text": text, "item": s.item_view(item_id)})
        if method == "POST" and action == "/voice/cancel":
            s.voice_cancel()
            return self._send(200, {"ok": True})
        if method == "POST" and action == "/evaluate":
            s.queue_evaluation(item_id)
            return self._send(200, s.item_view(item_id))
        if method == "POST" and action == "/answer":
            answer = (self._json_body().get("answer") or "").strip()
            if not answer:
                raise ValueError("Type an answer first.")
            s.answer_questions(item_id, answer)
            return self._send(200, s.item_view(item_id))
        if method == "POST" and action in ("/research", "/work"):
            body = self._json_body()
            s.start_run(item_id, action[1:], body.get("model", DEFAULT_RESEARCH_MODEL),
                        body.get("effort", DEFAULT_RESEARCH_EFFORT), body.get("place"), bool(body.get("fresh")))
            return self._send(200, s.item_view(item_id))
        if method == "POST" and action == "/followup":
            body = self._json_body()
            s.follow_up(item_id, body.get("question") or "", body.get("model", DEFAULT_RESEARCH_MODEL),
                        body.get("effort", DEFAULT_RESEARCH_EFFORT))
            return self._send(200, s.item_view(item_id))
        if method == "POST" and action == "/check":
            s.check_run(s.store.get(item_id), confirm_now=True)
            return self._send(200, s.item_view(item_id))
        return self._send(404, {"error": "Not found"})

    def _static(self, name):
        path = (WEB_DIR / name).resolve()
        if WEB_DIR.resolve() not in path.parents or not path.is_file():
            return self._send(404, {"error": "Not found"})
        content_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        self._send(200, path.read_bytes(), content_type + ("; charset=utf-8" if content_type.startswith("text") else ""))


def prepared_brief(item):
    """The local evaluation and your answers, as a section of the research prompt (empty if there is none)."""
    evaluation = json.loads(item["evaluation"] or "null")
    rounds = json.loads(item["eval_rounds"] or "[]")
    if not evaluation:
        return ""
    lines = ["", "## Clarified locally",
             "A local model turned my notes and answers into this brief. If it disagrees with my notes or answers, "
             "they win.", "", f"**Goal:** {evaluation['goal']}", "", "**Details:**"]
    lines += [f"- {d}" for d in evaluation["details"]]
    if evaluation["done_when"]:
        lines += ["", "**Done when:**"] + [f"- {d}" for d in evaluation["done_when"]]
    if rounds:
        lines += ["", "**Questions I answered:**"]
        for round_ in rounds:
            lines += [f"- {q}" for q in round_["questions"]] + [f"  My answer: {round_['answer']}"]
    open_questions = [q["question"] for q in evaluation["questions"]]
    if open_questions:
        lines += ["", "**Still undecided (use your judgment and tell me what you assumed):**"]
        lines += [f"- {q}" for q in open_questions]
    return "\n".join(lines) + "\n"


def git_branch(root):
    """The branch a project folder has checked out, or None."""
    try:
        out = subprocess.run(["git", "-C", root, "branch", "--show-current"], capture_output=True, text=True,
                             timeout=5)
    except (OSError, subprocess.TimeoutExpired):
        return None
    return out.stdout.strip() or None


def research_section(item):
    """An earlier research result, as a section of the work prompt (empty if the item wasn't researched)."""
    report = (item["result_plan"] or item["result"] or "").strip()
    if not report:
        return ""
    return f"\n## Research already done\nAn earlier read-only thread researched this. Its report:\n\n{report}\n"


def start_server(service):
    Handler.service = service
    server = ThreadingHTTPServer(("127.0.0.1", PORT), Handler)
    server.daemon_threads = True
    threading.Thread(target=server.serve_forever, daemon=True).start()
    threading.Thread(target=service.poll_forever, daemon=True).start()
    log(f"To-do list at http://127.0.0.1:{PORT}/")
    return f"http://127.0.0.1:{PORT}/"
