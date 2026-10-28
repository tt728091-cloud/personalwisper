"""Meeting transcripts: record a meeting (your microphone plus the sound other apps play), transcribe it on this Mac,
save the transcript as a text file in meetings/, and add to-dos from it to the to-do list as dirty items.

To-dos come from marks: during a meeting, the to-do hotkey marks the moment instead of saving a to-do (a quick tap,
or hold it and say what the to-do is). After the meeting, the meeting model writes one to-do for each mark from what
was said around it. With "Find them all" chosen in the menu, it also reads the whole meeting for to-dos.

The audio stays in memory and is dropped as soon as it's transcribed; only the text is saved. "Me" is the microphone
and "Others" is everything other apps played (Google Meet, a Slack huddle, ...). Wear headphones: with speakers the
microphone hears the others too. Those repeats are dropped, but not perfectly.
The transcript is read by the meeting model (the bigger one by default), loaded next to the cleanup model just for
this and unloaded once the to-dos it found have been evaluated.
"""
import difflib
import os
import platform
import re
import subprocess
import threading
import time
import traceback
from datetime import datetime
from pathlib import Path

import numpy as np

from pipeline import LLM_MODELS, MEETING_ASR_MODELS, ROOT, _words

MEETINGS_DIR = ROOT / "meetings"
HELPER = Path("Contents/MacOS/systemaudio")  # inside PersonalWisper.app; see app/systemaudio.m
SILENCE_PEAK = 1e-4
WINDOW_WORDS = 1500  # the model reads the transcript this many words at a time
CONTEXT_LINES = 6  # the end of the previous part, shown with each part for context
# Where you said a to-do out loud. If the model misses one of these, it's added anyway.
SAID_TODO = re.compile(r"(?i)\b(?:to-?dos?\b|to do (?:item|for me)\b|action items?\b|note to self\b)")
# A mark's to-do is written from the transcript from this long before it until this long after you let go.
MARK_BEFORE, MARK_AFTER = 120, 15
SOUND_SAVED = "/System/Library/Sounds/Glass.aiff"
SOUND_MARKED = "/System/Library/Sounds/Purr.aiff"


def log(message):
    print(f"{time.strftime('%Y-%m-%d %H:%M:%S')}  [meeting] {message}", flush=True)


def clock(seconds):
    """Seconds into the meeting as m:ss, or h:mm:ss."""
    seconds = int(seconds)
    h, m, s = seconds // 3600, seconds // 60 % 60, seconds % 60
    return f"{h}:{m:02}:{s:02}" if h else f"{m}:{s:02}"


def parse_clock(text):
    parts = re.findall(r"\d+", text or "")
    if not 2 <= len(parts) <= 3:
        return None
    seconds = 0
    for part in parts:
        seconds = seconds * 60 + int(part)
    return seconds


def heard_from_speakers(start, end, text, others):
    """True if this microphone line is mostly what Others said at the same time (speakers, not headphones)."""
    words = _words(text)
    heard = [w for s, e, t in others if s < end + 2 and e > start - 2 for w in _words(t)]
    if not words or not heard:
        return False
    blocks = difflib.SequenceMatcher(None, words, heard, autojunk=False).get_matching_blocks()
    return sum(b.size for b in blocks) >= 0.6 * len(words)


def merge_transcript(me, others):
    """Both sides as one list of (start, end, speaker, text), in time order."""
    lines = [(s, e, "Me", t) for s, e, t in me if not heard_from_speakers(s, e, t, others)]
    lines += [(s, e, "Others", t) for s, e, t in others]
    return sorted(lines)


def format_lines(lines, marks=()):
    """The transcript as text, with a MARK line where you pressed the to-do hotkey."""
    rows = [(start, 1, f"[{clock(start)}] {speaker}: {text}") for start, _, speaker, text in lines]
    rows += [(start, 0, f"[{clock(start)}] >>> MARK <<<") for start, _ in marks]
    return "\n".join(row for _, _, row in sorted(rows))


def windows(lines):
    """The transcript in parts of about WINDOW_WORDS words: [(part's lines, the lines just before it)]."""
    parts, current, words = [], [], 0
    for line in lines:
        if current and words + len(line[3].split()) > WINDOW_WORDS:
            parts.append(current)
            current, words = [], 0
        current.append(line)
        words += len(line[3].split())
    if current:
        parts.append(current)
    return [(part, parts[i - 1][-CONTEXT_LINES:] if i else []) for i, part in enumerate(parts)]


def excerpt(lines, at):
    """The transcript around a moment, so the to-do keeps what was actually said."""
    if at is None:
        return ""
    return format_lines([line for line in lines if at - 60 <= line[0] <= at + 20])


def helper_path():
    app = os.environ.get("PERSONALWISPER_APP")  # set by the .app launcher
    return (Path(app) if app else ROOT / "PersonalWisper.app") / HELPER


def unavailable_reason():
    """Why meetings can't be recorded on this Mac, or None."""
    version = tuple(int(x) for x in platform.mac_ver()[0].split(".")[:2])
    if version < (14, 2):
        return "Meeting transcripts need macOS 14.2 or later"
    if not helper_path().exists():
        return "Run ./build_app.sh to record meetings"
    return None


class SystemAudio:
    """Runs the systemaudio helper and keeps what it streams (16 kHz mono float32) in memory."""

    def __init__(self):
        self.error = None
        self._chunks = []
        self._process = subprocess.Popen([str(helper_path())], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                         stderr=subprocess.PIPE)
        self._reader = threading.Thread(target=self._read, daemon=True)
        self._reader.start()
        threading.Thread(target=self._read_errors, daemon=True).start()

    def _read(self):
        fd = self._process.stdout.fileno()
        while data := os.read(fd, 1 << 16):
            self._chunks.append(data)

    def _read_errors(self):
        for line in self._process.stderr:
            line = line.decode(errors="replace").strip()
            log(line)
            if "recording" not in line:
                self.error = line

    def stop(self):
        self._process.terminate()
        try:
            self._process.wait(3)
        except subprocess.TimeoutExpired:
            self._process.kill()
            self._process.wait()
        self._process.stdin.close()
        self._reader.join(3)
        data = b"".join(self._chunks)
        self._chunks = []
        return np.frombuffer(data[:len(data) // 4 * 4], dtype=np.float32)


class Meetings:
    """Start/stop from the menu. Stopping transcribes and reads the meeting in the background (see _process)."""

    def __init__(self, dictation, todo, recorder_factory, config, on_change):
        self.dictation = dictation
        self.todo = todo
        self.recorder_factory = recorder_factory
        self.config = config
        self.on_change = on_change
        self.started_at = None  # set while recording
        self.marks = []  # (seconds in, seconds in when the hotkey was let go) for each to-do hotkey press
        self.status = None  # what processing is doing, shown in the menu
        self._mic = self._system = None

    @property
    def recording(self):
        return self.started_at is not None

    def start(self):
        mic = self.recorder_factory()
        mic.start()
        try:
            self._system = SystemAudio()
        except OSError as e:
            mic.stop()
            raise RuntimeError(f"Couldn't record other apps' sound: {e}") from None
        self._mic, self.marks, self.started_at = mic, [], datetime.now()
        log("Recording a meeting")
        self.on_change()

    def mark(self, pressed_at, held):
        """The to-do hotkey during a meeting: mark this moment for a to-do (see _mark_todos). pressed_at: time.time()
        when it was pressed. Returns False if no meeting is being recorded."""
        if not self.recording:
            return False
        start = pressed_at - self.started_at.timestamp()
        self.marks.append((start, start + held))
        subprocess.Popen(["afplay", SOUND_MARKED])
        log(f"Marked {clock(start)}")
        self.on_change()
        return True

    def stop(self):
        if not self.recording:
            return
        started_at, self.started_at = self.started_at, None
        marks = self.marks
        audio = {}
        try:
            audio["me"], seconds = self._mic.stop()
        except Exception as e:  # the other side may still be fine
            log(f"Couldn't stop the microphone: {e}")
            audio["me"], seconds = np.zeros(0, dtype=np.float32), (datetime.now() - started_at).total_seconds()
        audio["others"] = self._system.stop()
        system_error = self._system.error
        self._mic = self._system = None
        log(f"Stopped after {clock(seconds)}")
        self.status = "Transcribing the meeting…"
        self.on_change()
        threading.Thread(target=self._process, args=(audio, started_at, seconds, system_error, marks),
                         daemon=True).start()

    def _process(self, audio, started_at, seconds, system_error, marks):
        d = self.dictation
        try:
            others_silent = len(audio["others"]) == 0 or np.abs(audio["others"]).max() < SILENCE_PEAK
            asr = self.config["meeting_asr"]
            label = next(label for label, name in MEETING_ASR_MODELS if name == asr).split(":")[0]
            d.download_models([asr], label)
            self.status = f"Transcribing the meeting with {label}…"
            self.on_change()
            # In small steps, so dictation keeps working while a long meeting is transcribed and read
            sides = d.submit_steps(lambda pipe: pipe.transcribe_meeting_steps(audio, asr)).result()
            audio.clear()  # the recording is never kept
            lines = merge_transcript(sides["me"], sides["others"])
            path = self._write_transcript(started_at, seconds, lines, marks)
            log(f"Transcript saved: {path.name} ({len(lines)} lines, {len(marks)} marks)")
            d.pipe.keep_extra = True  # the meeting model stays loaded until the to-dos it found are evaluated
            try:
                todos = self._find_todos(lines, marks)
                ids = [self.todo.add_from_meeting(t["title"], self._notes(t, started_at, path),
                                                  t.get("excerpt") or excerpt(lines, t["time"])) for t in todos]
            finally:
                d.pipe.keep_extra = False
                d.submit(lambda pipe: None)  # lets the model thread unload it if there's nothing to evaluate
            with path.open("a") as f:
                f.write("\n\nTo-dos added to the to-do list:\n" + "".join(
                    f"- #{i} {t['title']} ({t['at'] or 'time unknown'}, "
                    f"{self._who(t)})\n" for i, t in zip(ids, todos))
                        if todos else "\n\nNo to-dos added.\n")
            if system_error:
                d.problem = f"Couldn't record other apps' sound: {system_error}"
            elif others_silent and seconds > 30:
                d.problem = ("No sound from other apps was recorded. If the meeting had sound, allow PersonalWisper "
                             "in System Settings › Privacy & Security › Screen & System Audio Recording, then "
                             "restart it.")
            d.notice = (f"Meeting: {len(todos)} to-do{'s' * (len(todos) != 1)} added" if lines or marks
                        else "Meeting: no speech heard")
            subprocess.Popen(["afplay", SOUND_SAVED])
        except Exception as e:
            log(traceback.format_exc())
            d._fail(f"Meeting transcript failed: {e}")
        finally:
            self.status = None
            self.on_change()

    def _write_transcript(self, started_at, seconds, lines, marks):
        MEETINGS_DIR.mkdir(exist_ok=True)
        path = MEETINGS_DIR / f"{started_at:%Y-%m-%d_%H%M}.txt"
        n = 2
        while path.exists():
            path, n = MEETINGS_DIR / f"{started_at:%Y-%m-%d_%H%M}-{n}.txt", n + 1
        header = (f"Meeting transcript: {started_at:%A, %B %-d, %Y, %-I:%M %p} ({clock(seconds)} long)\n"
                  "Me: this Mac's microphone. Others: sound from other apps (the meeting). "
                  ">>> MARK <<<: where you pressed the to-do hotkey.\n\n")
        path.write_text(header + (format_lines(lines, marks) if lines or marks else "(no speech heard)") + "\n")
        return path

    def _find_todos(self, lines, marks):
        """One to-do per mark, written by the meeting model from what was said around it. With "Find them all", the
        whole transcript is also read for to-dos first (plus any spoken "to-do" it missed)."""
        everything = self.config["meeting_todos"] == "all"
        if not marks and not (everything and lines):
            return []
        d, model = self.dictation, self.config["meeting_model"]
        label = next(label for label, name in LLM_MODELS if name == model).split(":")[0]
        self.status = f"Getting {label} ready to read the meeting…"
        self.on_change()
        d.download_models([model], label)
        found = self._all_todos(lines, model, label) if everything else []
        todos = list(found)
        for number, (start, end) in enumerate(marks, 1):
            if any(t["time"] is not None and abs(t["time"] - start) <= 45 for t in found):
                continue  # Find Them All already found one there
            self.status = f"Writing to-dos for your marks with {label} ({number} of {len(marks)})…"
            self.on_change()
            around = [line for line in lines if start - MARK_BEFORE <= line[0] <= end + MARK_AFTER]
            # What you said while holding the hotkey, if you held it (a tap is just a mark)
            note = " ".join(text for at, _, speaker, text in around
                            if speaker == "Me" and start - 1 <= at <= end and end - start >= 1)
            todo = d.submit_steps(lambda pipe, text=format_lines(around, [(start, end)]), note=note:
                                  pipe.meeting_mark_todo_steps(model, text, note)).result()
            todo = todo or {"title": f"Follow up on the meeting at {clock(start)}", "from": "me", "asked_by": None,
                            "notes": "Nothing in the transcript around this mark made a clear to-do."}
            # Who it's from comes only from your note ("Andreas said ..."): the transcript doesn't know names.
            asked_by = todo["asked_by"]
            if asked_by and asked_by.split()[0].lower() in note.lower():
                todo = {**todo, "from": "others",
                        "title": todo["title"] if asked_by.lower() in todo["title"].lower()
                        else f"{todo['title']} (from {asked_by})"}
            else:
                todo = {**todo, "asked_by": None}
            todos.append({**todo, "at": clock(start), "time": start, "excerpt": format_lines(around, [(start, end)])})
        return sorted(todos, key=lambda t: t["time"] if t["time"] is not None else float("inf"))

    def _all_todos(self, lines, model, label):
        """Read the whole transcript in parts for to-dos, then add any spoken to-do it missed."""
        d, found = self.dictation, []
        parts = windows(lines)
        for number, (part, earlier) in enumerate(parts, 1):
            self.status = f"Finding to-dos in the meeting with {label} ({number} of {len(parts)})…"
            self.on_change()
            found += d.submit_steps(lambda pipe, part=part, earlier=earlier: pipe.meeting_todos_steps(
                model, format_lines(part), format_lines(earlier))).result()
        todos, titles = [], set()
        for todo in found:
            key = " ".join(_words(todo["title"]))
            if key not in titles:
                titles.add(key)
                todos.append({**todo, "time": parse_clock(todo["at"])})
        for start, _, speaker, text in lines:
            if speaker == "Me" and SAID_TODO.search(text) and not any(
                    t["time"] is not None and abs(t["time"] - start) <= 45 for t in todos):
                title, notes = d.submit(lambda pipe, text=text: pipe.todo_from_speech(text), urgent=False).result()
                todos.append({"title": title, "at": clock(start), "time": start, "from": "me", "notes": notes})
        return todos

    @staticmethod
    def _who(todo):
        if todo.get("asked_by"):
            return f"asked by {todo['asked_by']}"
        return "asked by others" if todo["from"] == "others" else "Me"

    @staticmethod
    def _notes(todo, started_at, path):
        asked = (f"{todo['asked_by']} asked for this. " if todo.get("asked_by") else
                 "Someone else in the meeting asked for this. " if todo["from"] == "others" else "")
        source = "You marked this in" if "excerpt" in todo else "From"
        return (f"{todo['notes']}\n\n{asked}{source} the meeting on {started_at:%b %-d at %-I:%M %p}"
                f"{', ' + todo['at'] + ' in' if todo['at'] else ''} (transcript: meetings/{path.name}).")
