"""To-do storage (SQLite in this folder).

Lifecycle: dirty (captured by voice) -> research (refined into a research brief) -> researching (T3 thread running)
-> researched (result saved). Items that need doing rather than researching go to work (from dirty, research or
researched) -> working (T3 thread doing it) -> done. Any item can be archived.
Dirty, research and work items also get a local evaluation (see TodoService): a clearer brief plus questions for you.
"""
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DB_PATH = ROOT / "todo.db"
STATUSES = ("dirty", "research", "researching", "researched", "work", "working", "done", "archived")

# The work thread's own copy of the research fields, so doing the work never overwrites the research result.
WORK_COLUMNS = ("work_thread_id", "work_state", "work_note", "work_result", "work_plan", "work_turn_id",
                "work_finished_at", "work_model", "work_effort", "work_place", "work_branch", "work_worktree")

SCHEMA = """
CREATE TABLE IF NOT EXISTS items (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    status TEXT NOT NULL DEFAULT 'dirty',
    title TEXT NOT NULL,
    notes TEXT NOT NULL DEFAULT '',
    brief TEXT NOT NULL DEFAULT '',
    project_id TEXT,
    project_title TEXT,
    thread_id TEXT,
    research_state TEXT,
    research_note TEXT,
    result TEXT,
    result_plan TEXT,
    result_turn_id TEXT,
    finished_at TEXT,
    evaluation TEXT,
    eval_state TEXT,
    eval_error TEXT,
    eval_rounds TEXT,
    eval_input TEXT,
    eval_locked TEXT,
    research_model TEXT,
    research_effort TEXT,
    work_thread_id TEXT,
    work_state TEXT,
    work_note TEXT,
    work_result TEXT,
    work_plan TEXT,
    work_turn_id TEXT,
    work_finished_at TEXT,
    work_model TEXT,
    work_effort TEXT,
    work_place TEXT,
    work_branch TEXT,
    work_worktree TEXT,
    run_history TEXT,
    followup TEXT,
    work_followup TEXT,
    thread_context TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
-- Everything you said or did to an item, oldest first, so refinements never lose earlier context.
CREATE TABLE IF NOT EXISTS events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    item_id INTEGER NOT NULL REFERENCES items(id) ON DELETE CASCADE,
    kind TEXT NOT NULL,
    text TEXT NOT NULL,
    created_at TEXT NOT NULL
);
"""
EDITABLE = ("status", "title", "notes", "brief", "project_id", "project_title", "thread_id",
            "research_state", "research_note", "result", "result_plan", "result_turn_id", "finished_at",
            "evaluation", "eval_state", "eval_error", "eval_rounds", "eval_input", "eval_locked", "research_model",
            "research_effort", *WORK_COLUMNS, "run_history", "followup", "work_followup", "thread_context")
# Columns added after the first release, created on startup for existing databases.
MIGRATED_COLUMNS = ("result_plan", "result_turn_id", "finished_at",
                    "evaluation", "eval_state", "eval_error", "eval_rounds", "eval_input",
                    "research_model", "research_effort", *WORK_COLUMNS, "run_history", "followup", "work_followup",
                    "thread_context", "eval_locked")


def now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class TodoStore:
    def __init__(self, path=DB_PATH):
        self._lock = threading.Lock()
        self._db = sqlite3.connect(path, check_same_thread=False)
        self._db.row_factory = sqlite3.Row
        self._db.execute("PRAGMA foreign_keys = ON")
        self._db.executescript(SCHEMA)
        existing = {row["name"] for row in self._db.execute("PRAGMA table_info(items)")}
        for column in MIGRATED_COLUMNS:
            if column not in existing:
                self._db.execute(f"ALTER TABLE items ADD COLUMN {column} TEXT")

    def create(self, title, notes, raw_transcript, project_id=None, project_title=None):
        with self._lock, self._db:
            ts = now()
            cur = self._db.execute(
                "INSERT INTO items (title, notes, project_id, project_title, created_at, updated_at) "
                "VALUES (?, ?, ?, ?, ?, ?)", (title, notes, project_id, project_title, ts, ts))
            self._db.execute("INSERT INTO events (item_id, kind, text, created_at) VALUES (?, 'captured', ?, ?)",
                             (cur.lastrowid, raw_transcript, ts))
            return cur.lastrowid

    def update(self, item_id, touch=True, **fields):
        """touch=False leaves updated_at alone (for background bookkeeping that shouldn't reorder the list)."""
        unknown = set(fields) - set(EDITABLE)
        if unknown:
            raise ValueError(f"Can't edit {', '.join(sorted(unknown))}")
        if "status" in fields and fields["status"] not in STATUSES:
            raise ValueError(f"Unknown status {fields['status']}")
        if not fields:
            return
        with self._lock, self._db:
            assignments = ", ".join(f"{k} = ?" for k in fields)
            if touch:
                self._db.execute(f"UPDATE items SET {assignments}, updated_at = ? WHERE id = ?",
                                 (*fields.values(), now(), item_id))
            else:
                self._db.execute(f"UPDATE items SET {assignments} WHERE id = ?", (*fields.values(), item_id))

    def add_event(self, item_id, kind, text):
        with self._lock, self._db:
            self._db.execute("INSERT INTO events (item_id, kind, text, created_at) VALUES (?, ?, ?, ?)",
                             (item_id, kind, text, now()))

    def delete(self, item_id):
        with self._lock, self._db:
            self._db.execute("DELETE FROM items WHERE id = ?", (item_id,))

    def get(self, item_id):
        with self._lock:
            row = self._db.execute("SELECT * FROM items WHERE id = ?", (item_id,)).fetchone()
            if row is None:
                return None
            events = self._db.execute("SELECT kind, text, created_at FROM events WHERE item_id = ? ORDER BY id",
                                      (item_id,)).fetchall()
        return {**dict(row), "events": [dict(e) for e in events]}

    def list(self):
        with self._lock:
            rows = self._db.execute("SELECT * FROM items ORDER BY updated_at DESC, id DESC").fetchall()
        return [dict(r) for r in rows]

    def watched(self, since):
        """Items whose T3 thread may still change: running, or finished after `since` (ISO time)."""
        with self._lock:
            rows = self._db.execute(
                "SELECT * FROM items WHERE status IN ('researching', 'working') OR "
                "(status = 'researched' AND finished_at >= ?) OR (status = 'done' AND work_finished_at >= ?)",
                (since, since)).fetchall()
        return [dict(r) for r in rows]

    def with_status(self, status):
        with self._lock:
            rows = self._db.execute("SELECT * FROM items WHERE status = ?", (status,)).fetchall()
        return [dict(r) for r in rows]
