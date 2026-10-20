"""Meeting transcripts: record a meeting (your microphone plus the sound other apps play), transcribe it on this Mac,
save the transcript as a text file in meetings/, and add the to-dos in it to the to-do list as dirty items.

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

from pipeline import LLM_MODELS, ROOT, _words

MEETINGS_DIR = ROOT / "meetings"
HELPER = Path("Contents/MacOS/systemaudio")  # inside PersonalWisper.app; see app/systemaudio.m
SILENCE_PEAK = 1e-4
WINDOW_WORDS = 1500  # the model reads the transcript this many words at a time
CONTEXT_LINES = 6  # the end of the previous part, shown with each part for context
# Where you said a to-do out loud. If the model misses one of these, it's added anyway.
SAID_TODO = re.compile(r"(?i)\b(?:to-?dos?\b|to do (?:item|for me)\b|action items?\b|note to self\b)")
SOUND_SAVED = "/System/Library/Sounds/Glass.aiff"


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


def format_lines(lines):
    return "\n".join(f"[{clock(start)}] {speaker}: {text}" for start, _, speaker, text in lines)


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
        self._mic, self.started_at = mic, datetime.now()
        log("Recording a meeting")
        self.on_change()

    def stop(self):
        if not self.recording:
            return
        started_at, self.started_at = self.started_at, None
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
        threading.Thread(target=self._process, args=(audio, started_at, seconds, system_error), daemon=True).start()

    def _process(self, audio, started_at, seconds, system_error):
        d = self.dictation
        try:
            others_silent = len(audio["others"]) == 0 or np.abs(audio["others"]).max() < SILENCE_PEAK
            me = d.submit(lambda pipe: pipe.transcribe_lines(audio["me"])).result()
            others = d.submit(lambda pipe: pipe.transcribe_lines(audio["others"])).result()
            audio.clear()  # the recording is never kept
            lines = merge_transcript(me, others)
            path = self._write_transcript(started_at, seconds, lines)
            log(f"Transcript saved: {path.name} ({len(lines)} lines)")
            d.pipe.keep_extra = True  # the meeting model stays loaded until the to-dos it found are evaluated
            try:
                todos = self._find_todos(lines) if lines else []
                ids = [self.todo.add_from_meeting(t["title"], self._notes(t, started_at, path),
                                                  excerpt(lines, t["time"])) for t in todos]
            finally:
                d.pipe.keep_extra = False
                d.submit(lambda pipe: None)  # lets the model thread unload it if there's nothing to evaluate
            with path.open("a") as f:
                f.write("\n\nTo-dos added to the to-do list:\n" + "".join(
                    f"- #{i} {t['title']} ({t['at'] or 'time unknown'}, "
                    f"{'asked by others' if t['from'] == 'others' else 'Me'})\n" for i, t in zip(ids, todos))
                        if todos else "\n\nNo to-dos found.\n")
            if system_error:
                d.problem = f"Couldn't record other apps' sound: {system_error}"
            elif others_silent and seconds > 30:
                d.problem = ("No sound from other apps was recorded. If the meeting had sound, allow PersonalWisper "
                             "in System Settings › Privacy & Security › Screen & System Audio Recording, then "
                             "restart it.")
            d.notice = (f"Meeting: {len(todos)} to-do{'s' * (len(todos) != 1)} added" if lines
                        else "Meeting: no speech heard")
            subprocess.Popen(["afplay", SOUND_SAVED])
        except Exception as e:
            log(traceback.format_exc())
            d._fail(f"Meeting transcript failed: {e}")
        finally:
            self.status = None
            self.on_change()

    def _write_transcript(self, started_at, seconds, lines):
        MEETINGS_DIR.mkdir(exist_ok=True)
        path = MEETINGS_DIR / f"{started_at:%Y-%m-%d_%H%M}.txt"
        n = 2
        while path.exists():
            path, n = MEETINGS_DIR / f"{started_at:%Y-%m-%d_%H%M}-{n}.txt", n + 1
        header = (f"Meeting transcript: {started_at:%A, %B %-d, %Y, %-I:%M %p} ({clock(seconds)} long)\n"
                  "Me: this Mac's microphone. Others: sound from other apps (the meeting).\n\n")
        path.write_text(header + (format_lines(lines) if lines else "(no speech heard)") + "\n")
        return path

    def _find_todos(self, lines):
        """Read the transcript in parts with the meeting model, then add any spoken to-do it missed."""
        d, model = self.dictation, self.config["meeting_model"]
        label = next(label for label, name in LLM_MODELS if name == model).split(":")[0]
        self.status = f"Getting {label} ready to read the meeting…"
        self.on_change()
        d.download_models([model], label)
        found = []
        parts = windows(lines)
        for number, (part, earlier) in enumerate(parts, 1):
            self.status = f"Finding to-dos in the meeting with {label} ({number} of {len(parts)})…"
            self.on_change()
            found += d.submit(lambda pipe, part=part, earlier=earlier: pipe.meeting_todos(
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
                title, notes = d.submit(lambda pipe, text=text: pipe.todo_from_speech(text)).result()
                todos.append({"title": title, "at": clock(start), "time": start, "from": "me", "notes": notes})
        return sorted(todos, key=lambda t: t["time"] if t["time"] is not None else float("inf"))

    @staticmethod
    def _notes(todo, started_at, path):
        asked = "Someone else in the meeting asked for this. " if todo["from"] == "others" else ""
        return (f"{todo['notes']}\n\n{asked}From the meeting on {started_at:%b %-d at %-I:%M %p}"
                f"{', ' + todo['at'] + ' in' if todo['at'] else ''} (transcript: meetings/{path.name}).")
