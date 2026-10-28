"""PersonalWisper: a menu-bar, hold-to-talk dictation app that runs entirely on this Mac.

Hold the dictation hotkey, speak, release, and the cleaned-up text is pasted at your cursor.
Hold the to-do hotkey instead and what you say becomes a dirty item in the to-do list (see todo_server.py).
Normally launched via PersonalWisper.app (see build_app.sh); ./run.sh runs it from a terminal for debugging.
"""
import fcntl
import itertools
import json
import os
import plistlib
import queue
import subprocess
import sys
import threading
import time
import traceback
from concurrent.futures import Future
from pathlib import Path

import numpy as np
import objc
import Quartz
import sounddevice as sd
from AppKit import (NSApplication, NSApplicationActivationPolicyAccessory, NSColor, NSImage, NSMenu,
                    NSMenuItem, NSPasteboard, NSPasteboardItem, NSPasteboardTypeString, NSStatusBar,
                    NSVariableStatusItemLength, NSWorkspace)
from AVFoundation import AVCaptureDevice, AVMediaTypeAudio
from Foundation import NSURL, NSObject, NSTimer
from PyObjCTools import AppHelper

from meetings import Meetings, unavailable_reason
from pipeline import (ASR_MODEL, LLM_MODELS, MEETING_ASR_MODEL, MEETING_ASR_MODELS, MEETING_MODEL, ROOT, SAMPLE_RATE,
                      Pipeline, is_downloaded, model_path, stored_bytes)
from todo_server import TodoService, start_server

# Hold-to-talk hotkeys. A modifier hotkey is one modifier key, detected by its keycode plus its device-specific
# flag bit. A chord hotkey is a key held with exactly these modifiers; its key presses never reach the focused app.
OPTION, COMMAND, CONTROL, SHIFT = (Quartz.kCGEventFlagMaskAlternate, Quartz.kCGEventFlagMaskCommand,
                                   Quartz.kCGEventFlagMaskControl, Quartz.kCGEventFlagMaskShift)
HOTKEYS = {
    "Right Option": {"key": 61, "device_flag": 0x40, "modifiers": OPTION},
    "Right Command": {"key": 54, "device_flag": 0x10, "modifiers": COMMAND},
    "Right Control": {"key": 62, "device_flag": 0x2000, "modifiers": CONTROL},
    "Fn (Globe)": {"key": 63, "device_flag": Quartz.kCGEventFlagMaskSecondaryFn,
                   "modifiers": Quartz.kCGEventFlagMaskSecondaryFn},
    "Option+Z": {"key": 6, "modifiers": OPTION, "chord": True},
    "Home": {"key": 115, "modifiers": 0, "chord": True},  # for keyboards without Fn
}


def hotkeys_conflict(a, b):
    """Two hotkeys can't be used together if they're the same, or one is a modifier the other chord needs."""
    one, other = HOTKEYS[a], HOTKEYS[b]
    return a == b or (one.get("chord") != other.get("chord") and bool(one["modifiers"] & other["modifiers"]))
MIN_SECONDS = 0.3  # ignore accidental taps
SILENCE_PEAK = 1e-4  # a recording this quiet means the mic is blocked, not that you were quiet
HISTORY_SIZE = 10

CONFIG_PATH = ROOT / "config.json"
DEFAULT_CONFIG = {"hotkey": "Fn (Globe)", "todo_hotkey": "Option+Z", "cleanup": True,
                  # A second dictation hotkey, or None. For example Home, for a keyboard without Fn.
                  "hotkey2": None,
                  # The T3 project new to-do items go to (by title); unset means the first project T3 lists.
                  "todo_project": None, "llm_model": LLM_MODELS[0][1],
                  # The model that reads meeting transcripts for to-dos; loaded just for that.
                  "meeting_model": MEETING_MODEL,
                  # The speech-to-text model for meetings; loaded just while a finished meeting is transcribed.
                  "meeting_asr": MEETING_ASR_MODEL,
                  # Meeting to-dos: "marks" (only where you pressed the to-do hotkey) or "all" (everything it finds).
                  "meeting_todos": "marks",
                  # T3 projects whose work items default to a new worktree, and the branch it starts from,
                  # e.g. {"my-project": "main"}.
                  "work_worktree_base": {},
                  # Write what was heard and typed to the log, not just timings.
                  "log_dictations": False}
MEETING_TODO_MODES = {"marks": "Only Where I Mark (To-Do Hotkey)", "all": "Find Them All"}
LOG_PATH = ROOT / "logs" / "personalwisper.log"
LOCK_PATH = ROOT / "logs" / ".personalwisper.lock"
APP_PATH = os.environ.get("PERSONALWISPER_APP")  # set by the .app launcher
AGENT_LABEL = "local.personalwisper"
AGENT_PATH = Path.home() / "Library" / "LaunchAgents" / f"{AGENT_LABEL}.plist"

SOUND_START = "/System/Library/Sounds/Tink.aiff"
SOUND_STOP = "/System/Library/Sounds/Pop.aiff"
SOUND_ERROR = "/System/Library/Sounds/Basso.aiff"
SOUND_TODO_START = "/System/Library/Sounds/Morse.aiff"
SOUND_TODO_SAVED = "/System/Library/Sounds/Glass.aiff"
CHROME_PATH = Path("/Applications/Google Chrome.app")

# Tells clipboard managers not to record our temporary clipboard contents.
TRANSIENT_TYPE = "org.nspasteboard.TransientType"
V_KEYCODE = 9

PRIVACY_URLS = {
    "Input Monitoring": "x-apple.systempreferences:com.apple.preference.security?Privacy_ListenEvent",
    "Accessibility": "x-apple.systempreferences:com.apple.preference.security?Privacy_Accessibility",
    "Microphone": "x-apple.systempreferences:com.apple.preference.security?Privacy_Microphone",
}


def log(message):
    print(f"{time.strftime('%Y-%m-%d %H:%M:%S')}  {message}", flush=True)


def play(sound):
    subprocess.Popen(["afplay", sound])


def load_config():
    try:
        config = {**DEFAULT_CONFIG, **json.loads(CONFIG_PATH.read_text())}
    except (OSError, ValueError):
        config = dict(DEFAULT_CONFIG)
    for key in ("hotkey", "todo_hotkey"):
        if config[key] not in HOTKEYS:
            config[key] = DEFAULT_CONFIG[key]
    if hotkeys_conflict(config["todo_hotkey"], config["hotkey"]):  # the two hotkeys must not overlap
        config["todo_hotkey"] = next(name for name in HOTKEYS if not hotkeys_conflict(name, config["hotkey"]))
    if config["hotkey2"] not in HOTKEYS or any(hotkeys_conflict(config["hotkey2"], config[key])
                                               for key in ("hotkey", "todo_hotkey")):
        config["hotkey2"] = None
    for key, models in (("llm_model", LLM_MODELS), ("meeting_model", LLM_MODELS), ("meeting_asr", MEETING_ASR_MODELS)):
        if config[key] not in {model for _, model in models}:
            config[key] = DEFAULT_CONFIG[key]
    if config["meeting_todos"] not in MEETING_TODO_MODES:
        config["meeting_todos"] = DEFAULT_CONFIG["meeting_todos"]
    return config


def save_config(config):
    CONFIG_PATH.write_text(json.dumps(config, indent=2) + "\n")


def start_at_login_enabled():
    return AGENT_PATH.exists()


def set_start_at_login(enabled):
    if not enabled:
        AGENT_PATH.unlink(missing_ok=True)
        return
    AGENT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with AGENT_PATH.open("wb") as f:
        plistlib.dump({"Label": AGENT_LABEL, "ProgramArguments": ["/usr/bin/open", "-a", APP_PATH],
                       "RunAtLoad": True}, f)


def set_clipboard(text, transient=False):
    pb = NSPasteboard.generalPasteboard()
    pb.clearContents()
    pb.setString_forType_(text, NSPasteboardTypeString)
    if transient:
        pb.setString_forType_("", TRANSIENT_TYPE)


def paste(text):
    """Paste text at the cursor via the clipboard, then restore whatever was on the clipboard."""
    pb = NSPasteboard.generalPasteboard()
    saved = [{t: item.dataForType_(t) for t in item.types()} for item in (pb.pasteboardItems() or [])]

    set_clipboard(text, transient=True)
    for key_down in (True, False):
        event = Quartz.CGEventCreateKeyboardEvent(None, V_KEYCODE, key_down)
        Quartz.CGEventSetFlags(event, Quartz.kCGEventFlagMaskCommand)
        Quartz.CGEventPost(Quartz.kCGHIDEventTap, event)

    # Give the target app time to read the clipboard before restoring it.
    time.sleep(0.3)
    pb.clearContents()
    restored = []
    for types in saved:
        item = NSPasteboardItem.alloc().init()
        for t, data in types.items():
            if data is not None:
                item.setData_forType_(data, t)
        restored.append(item)
    if restored:
        pb.writeObjects_(restored)


class Recorder:
    """Records the microphone between start() and stop().

    Stopping a PortAudio stream from our own thread can deadlock inside CoreAudio: the stopping thread holds the
    audio unit's lock and waits for the device's I/O lock, while CoreAudio's I/O thread holds that one and waits for
    the audio unit's lock in PortAudio's start/stop listener. The mic then stays on and every later recording hangs.
    So the stream stops itself from its own callback, on CoreAudio's I/O thread, where that can't happen. And no
    PortAudio call is waited on for more than a few seconds: if one still hangs, on_stuck is called, because only
    restarting the app frees the microphone."""

    TIMEOUT = 3.0
    stuck = False
    on_stuck = None  # set by the app; called once if the microphone gets stuck

    def __init__(self):
        self._lock = threading.Lock()
        self._current = None  # (stream, chunks, stop requested, finished) while recording
        self._started_at = 0.0

    def start(self):
        with self._lock:
            chunks, stop_requested, finished = [], threading.Event(), threading.Event()

            def callback(data, frames, time_info, status):
                if stop_requested.is_set():
                    raise sd.CallbackStop
                chunks.append(data.copy())

            def open_stream():
                stream = sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype="float32", callback=callback,
                                        finished_callback=finished.set)
                stream.start()
                return stream
            self._current = (self._within(open_stream), chunks, stop_requested, finished)
            self._started_at = time.perf_counter()

    def stop(self):
        """Returns (audio, seconds held)."""
        with self._lock:
            if self._current is None:
                return np.zeros(0, dtype=np.float32), 0.0
            stream, chunks, stop_requested, finished = self._current
            self._current = None
            held = time.perf_counter() - self._started_at
            stop_requested.set()

            def close():
                finished.wait(1.0)  # the callback stops the stream within one block of audio
                stream.close()  # (aborts it first if it didn't stop)
            try:
                self._within(close)
            except Exception as e:  # what was recorded is still good
                log(f"Couldn't close the microphone stream: {e}")
            audio = np.concatenate(chunks)[:, 0] if chunks else np.zeros(0, dtype=np.float32)
            return audio, held

    @classmethod
    def _within(cls, fn):
        """Run a PortAudio call on a helper thread and return its result, giving up after TIMEOUT seconds."""
        result = Future()

        def run():
            try:
                result.set_result(fn())
            except Exception as e:
                result.set_exception(e)
        threading.Thread(target=run, daemon=True).start()
        try:
            return result.result(cls.TIMEOUT)
        except TimeoutError:
            if not cls.stuck:
                cls.stuck = True
                log("The microphone is stuck in macOS's audio system")
                if cls.on_stuck:
                    cls.on_stuck()
            raise RuntimeError("The microphone got stuck.") from None


class Dictation:
    """Hotkeys, recording, and the model thread. UI-agnostic; calls on_change when state changes.

    Two hold-to-talk modes: "dictate" pastes cleaned-up text at the cursor; "todo" saves a dirty to-do item.
    """

    def __init__(self, config, on_change):
        self.config = config
        self.on_change = on_change
        self.recorder = Recorder()
        self.mic_commands = queue.Queue()
        # (0 for urgent work or 1 for a meeting's processing, order submitted, fn, Future): urgent jobs go first.
        self.jobs = queue.PriorityQueue()
        self._job_order = itertools.count()
        self.holding_mode = None  # "dictate" or "todo" while a hotkey is held
        self.holding_hotkey = None  # the name of the hotkey being held
        self._chord_key_held = None  # keycode of a chord hotkey's key whose release we still need to swallow
        self.cancelled = False
        self.loaded = False
        self.loading_status = "Loading models…"
        self.download_status = None  # progress of a model download, shown in the menu
        self.pipe = None
        self.busy = False
        self.running_job = False  # a job is running on the model thread
        self.problem = None
        self.notice = None  # last success message, shown in the menu
        self.history = []  # most recent dictations first
        self.todo_capture = None  # set by the controller: fn(pipe, raw_text) -> title
        # Set by the controller: fn(pressed_at, seconds held) -> True if the to-do hotkey marked a meeting instead.
        self.todo_mark = None
        # Set by the controller: fn(pipe) -> a generator of low-priority work (to-do evaluation), or None.
        self.background_work = None
        self._background = None  # the generator being stepped through between jobs

    def start(self):
        threading.Thread(target=self._mic_loop, daemon=True).start()
        # MLX must run on the thread that loaded the models, so loading and all model work share one thread.
        threading.Thread(target=self._worker, daemon=True).start()

    def submit(self, fn, urgent=True):
        """Run fn(pipeline) on the model thread. Returns a Future with its result. A job that isn't urgent waits
        while any urgent one (a dictation) is waiting."""
        future = Future()
        self.jobs.put((0 if urgent else 1, next(self._job_order), fn, future))
        return future

    def submit_steps(self, make_steps):
        """Run the generator make_steps(pipeline) on the model thread one step at a time, as jobs that aren't urgent,
        so dictation runs in between steps. Returns a Future with the generator's return value."""
        result, steps = Future(), None

        def step(pipe):
            nonlocal steps
            try:
                if steps is None:
                    steps = make_steps(pipe)
                next(steps)
            except StopIteration as done:
                result.set_result(done.value)
            except Exception as e:  # the caller handles it
                log(traceback.format_exc())
                result.set_exception(e)
            else:
                self.submit(step, urgent=False)
        self.submit(step, urgent=False)
        return result

    def on_key_event(self, event_type, event):
        """Called on the main thread for every key-down, key-up and modifier change. Must return quickly.
        Returns True to keep the event from the focused app (a chord hotkey's own key presses)."""
        keycode = Quartz.CGEventGetIntegerValueField(event, Quartz.kCGKeyboardEventKeycode)
        flags = Quartz.CGEventGetFlags(event)
        hotkeys = ((self.config["hotkey"], "dictate"), (self.config["hotkey2"], "dictate"),
                   (self.config["todo_hotkey"], "todo"))
        modes = [(mode, name, HOTKEYS[name]) for name, mode in hotkeys if name]
        if event_type == Quartz.kCGEventFlagsChanged:
            for mode, name, hotkey in modes:
                if not hotkey.get("chord") and keycode == hotkey["key"]:
                    self._hotkey_changed(mode, name, bool(flags & hotkey["device_flag"]))
                    return False
                if (hotkey.get("chord") and hotkey["modifiers"] and self.holding_hotkey == name
                        and not flags & hotkey["modifiers"]):
                    self._hotkey_changed(mode, name, False)  # let go of the chord's modifier: stop
                    return False
        else:
            for mode, name, hotkey in modes:
                if not hotkey.get("chord") or keycode != hotkey["key"]:
                    continue
                if event_type == Quartz.kCGEventKeyUp:
                    if self._chord_key_held == keycode:
                        self._chord_key_held = None
                        self._hotkey_changed(mode, name, False)
                        return True
                    return False
                if self._chord_key_held == keycode:
                    return True  # key repeat while held
                if flags & (OPTION | COMMAND | CONTROL | SHIFT) == hotkey["modifiers"] and self.holding_mode is None:
                    self._hotkey_changed(mode, name, True)
                    if self.holding_hotkey == name:
                        self._chord_key_held = keycode
                        return True
                break
        if self.holding_mode is not None and event_type != Quartz.kCGEventKeyUp:
            # Another key while holding means a keyboard shortcut, not dictation.
            self.cancelled = True
        return False

    def _hotkey_changed(self, mode, name, down):
        if self.holding_mode is None:
            if down and self.loaded:
                self.holding_mode, self.holding_hotkey, self.cancelled = mode, name, False
                play(SOUND_START if mode == "dictate" else SOUND_TODO_START)
                self.mic_commands.put(("start", mode))
                self.on_change()
        elif self.holding_hotkey == name:
            if not down:
                self.holding_mode = self.holding_hotkey = None
                self.mic_commands.put(("stop", mode))
                self.on_change()
        elif down:
            self.cancelled = True  # pressed another hotkey mid-recording

    def _mic_loop(self):
        # Key events must return fast, so mic start/stop runs here, in order.
        pressed_at = time.time()
        while True:
            command, mode = self.mic_commands.get()
            try:
                if command == "start":
                    pressed_at = time.time()
                    self.recorder.start()
                    continue
                audio, held = self.recorder.stop()
            except Exception as e:
                self._fail(f"Microphone error: {e}")
                continue
            if mode == "todo" and not self.cancelled and self.todo_mark and self.todo_mark(pressed_at, held):
                continue  # during a meeting, the to-do hotkey marks the moment instead
            if self.cancelled or held < MIN_SECONDS:
                continue
            if len(audio) == 0 or np.abs(audio).max() < SILENCE_PEAK:
                self._fail("No audio. Check Microphone permission.")
                continue
            play(SOUND_STOP)
            self.busy = True
            self.on_change()
            handler = self._dictate if mode == "dictate" else self._capture_todo
            self.submit(lambda pipe, audio=audio, handler=handler: handler(pipe, audio))

    def _dictate(self, pipe, audio):
        t0 = time.perf_counter()
        raw = pipe.transcribe(audio)
        t1 = time.perf_counter()
        if not raw:
            return
        text = pipe.cleanup(raw)[0] if self.config["cleanup"] else raw
        t2 = time.perf_counter()
        paste(text)
        self.history = [text] + self.history[:HISTORY_SIZE - 1]
        self.problem = self.notice = None
        timings = f"{len(audio) / SAMPLE_RATE:.1f}s audio | stt {t1 - t0:.2f}s + cleanup {t2 - t1:.2f}s"
        log(f"{timings}\n    raw:   {raw}\n    typed: {text}" if self.config["log_dictations"] else timings)

    def _capture_todo(self, pipe, audio):
        raw = pipe.transcribe(audio)
        if not raw:
            raise ValueError("I didn't catch any words for that to-do.")
        title = self.todo_capture(pipe, raw)
        play(SOUND_TODO_SAVED)
        self.problem, self.notice = None, f"Saved to-do: {title}"

    def download_models(self, names, label):
        """Download any of these models that aren't on this Mac yet, showing progress in the menu."""
        missing = [name for name in names if not is_downloaded(name)]
        if not missing:
            return
        log(f"Downloading {', '.join(missing)} ...")
        finished, before = threading.Event(), stored_bytes()

        def show_progress():
            while not finished.wait(2):
                self.download_status = f"Downloading {label}… {(stored_bytes() - before) / 1e9:.1f} GB so far"
                self.on_change()
        self.download_status = f"Downloading {label}…"
        self.on_change()
        threading.Thread(target=show_progress, daemon=True).start()
        try:
            for name in missing:
                model_path(name)
        finally:
            finished.set()
            self.download_status = None
            self.on_change()
        log(f"Downloaded {', '.join(missing)}")

    def _worker(self):
        try:
            self.download_models([ASR_MODEL, self.config["llm_model"]], "models (first run only)")
            log("Loading models ...")
            self.pipe = Pipeline(self.config["llm_model"])
            # Warm up both models so the first real dictation isn't slow.
            self.pipe.transcribe(np.zeros(SAMPLE_RATE, dtype=np.float32))
            self.pipe.cleanup("hello world")
        except Exception as e:
            log(traceback.format_exc())
            self.problem = f"Couldn't download or load models: {e}"
            self.on_change()
            return
        self.loaded = True
        log("Ready")
        self.on_change()

        while True:
            # Background work only runs a few tokens at a time while no job is waiting.
            if self.jobs.empty() and self._background_step():
                continue
            _, _, fn, future = self.jobs.get()
            self.running_job = True
            try:
                future.set_result(fn(self.pipe))
            except Exception as e:
                log(traceback.format_exc())
                future.set_exception(e)
                self._fail(f"Error: {e}")
            finally:
                self.running_job = False
                self.busy = not self.jobs.empty()
                self.on_change()

    def _background_step(self):
        """Advance the background work a little. Returns False when there's none left."""
        if self._background is None:
            self._background = self.background_work(self.pipe) if self.background_work else None
            if self._background is None:
                self.pipe.release_evaluator()  # nothing to evaluate: give its memory back
                return False
        try:
            next(self._background)
        except StopIteration:
            self._background = None
        except Exception:
            log(traceback.format_exc())
            self._background = None
        return True

    def switch_model(self, model, label):
        """Swap the cleanup model, downloading it first if it isn't on this Mac yet (dictation keeps working while it
        downloads, and pauses while it loads). Returns a Future that fails if the download or the load fails."""
        name = label.split(":")[0]
        result = Future()

        def work(pipe):
            if self.config["llm_model"] != model or pipe.llm_model == model:
                return  # another model was chosen while this one was downloading
            self.loaded, self.loading_status = False, f"Switching to {name}…"
            self.on_change()
            if self._background is not None:  # an evaluation in progress restarts on the new setup
                self._background.close()
                self._background = None
            try:
                started = time.perf_counter()
                pipe.switch_llm(model)
                pipe.cleanup("hello world")
                log(f"Cleanup model is now {model} ({time.perf_counter() - started:.1f}s)")
                self.problem, self.notice = None, f"Cleanup model: {name}"
            finally:
                self.loaded, self.loading_status = True, "Loading models…"

        def download_then_switch():
            try:
                self.download_models([model], name)
            except Exception as e:
                log(traceback.format_exc())
                self._fail(f"Couldn't download {name}. Check the internet connection, then pick it again.")
                result.set_exception(e)
                return
            self.submit(work).add_done_callback(
                lambda done: result.set_exception(done.exception()) if done.exception() else result.set_result(None))

        threading.Thread(target=download_then_switch, daemon=True).start()
        return result

    def _fail(self, message):
        log(message)
        play(SOUND_ERROR)
        self.problem = message
        self.on_change()


def handle_tap_event(proxy, event_type, event, refcon):
    controller = CONTROLLER
    if event_type in (Quartz.kCGEventTapDisabledByTimeout, Quartz.kCGEventTapDisabledByUserInput):
        Quartz.CGEventTapEnable(controller.tap, True)
    elif controller.dictation.on_key_event(event_type, event):
        return None  # a chord hotkey's key: don't type it
    return event


CONTROLLER = None  # the MenuController; the event tap callback needs it


class MenuController(NSObject):
    def init(self):
        self = objc.super(MenuController, self).init()
        if self is None:
            return None
        self.config = load_config()
        self.dictation = Dictation(self.config, self.request_refresh)
        self.todo = TodoService(self.dictation.submit, Recorder, self.request_refresh, self.config)
        self.dictation.todo_capture = self.todo.capture
        self.dictation.background_work = self.todo.next_evaluation
        self.meetings = Meetings(self.dictation, self.todo, Recorder, self.config, self.request_refresh)
        self.dictation.todo_mark = self.meetings.mark
        Recorder.on_stuck = self.mic_stuck
        self.todo_url = None
        self.tap = None
        self.missing_permissions = []
        self.asked_permissions = False
        return self

    # ---- lifecycle ----

    def applicationDidFinishLaunching_(self, notification):
        self.build_menu()
        self.check_permissions()
        self.dictation.start()
        try:
            self.todo_url = start_server(self.todo)
        except OSError as e:
            log(f"Couldn't start the to-do server: {e}")
        # Re-check every few seconds so granting a permission takes effect without a restart where possible.
        NSTimer.scheduledTimerWithTimeInterval_target_selector_userInfo_repeats_(
            3.0, self, "permissionTimer:", None, True)
        self.refresh()

    def permissionTimer_(self, timer):
        if self.missing_permissions:
            self.check_permissions()
            self.refresh()

    @objc.python_method
    def check_permissions(self):
        missing = []
        if not Quartz.CGPreflightListenEventAccess():
            missing.append("Input Monitoring")
        if not Quartz.CGPreflightPostEventAccess():
            missing.append("Accessibility")
        mic = AVCaptureDevice.authorizationStatusForMediaType_(AVMediaTypeAudio)
        if mic in (1, 2):  # restricted / denied
            missing.append("Microphone")

        if not self.asked_permissions:
            self.asked_permissions = True
            if "Input Monitoring" in missing:
                Quartz.CGRequestListenEventAccess()
            if "Accessibility" in missing:
                Quartz.CGRequestPostEventAccess()
            if mic == 0:  # not determined yet: ask now rather than on the first dictation
                AVCaptureDevice.requestAccessForMediaType_completionHandler_(AVMediaTypeAudio, lambda granted: None)

        if self.tap is None and "Input Monitoring" not in missing:
            self.start_tap()
        self.missing_permissions = missing

    @objc.python_method
    def start_tap(self):
        mask = (Quartz.CGEventMaskBit(Quartz.kCGEventFlagsChanged)
                | Quartz.CGEventMaskBit(Quartz.kCGEventKeyDown) | Quartz.CGEventMaskBit(Quartz.kCGEventKeyUp))
        # An active tap can keep a chord hotkey's key (Option+Z would type Ω) from the focused app.
        # It needs Accessibility; without it, fall back to listening only.
        tap = Quartz.CGEventTapCreate(Quartz.kCGSessionEventTap, Quartz.kCGHeadInsertEventTap,
                                      Quartz.kCGEventTapOptionDefault, mask, handle_tap_event, None)
        if tap is None:
            log("Listening to the keyboard only (no Accessibility yet), so a chord hotkey's key also types")
            tap = Quartz.CGEventTapCreate(Quartz.kCGSessionEventTap, Quartz.kCGHeadInsertEventTap,
                                          Quartz.kCGEventTapOptionListenOnly, mask, handle_tap_event, None)
        if tap is None:
            log("Could not create keyboard event tap (Input Monitoring not granted yet?)")
            return
        source = Quartz.CFMachPortCreateRunLoopSource(None, tap, 0)
        Quartz.CFRunLoopAddSource(Quartz.CFRunLoopGetCurrent(), source, Quartz.kCFRunLoopCommonModes)
        Quartz.CGEventTapEnable(tap, True)
        self.tap = tap
        log("Keyboard listener active")

    # ---- menu ----

    @objc.python_method
    def add_item(self, menu, title, action=None, key=""):
        item = NSMenuItem.alloc().initWithTitle_action_keyEquivalent_(title, action, key)
        if action:
            item.setTarget_(self)
        menu.addItem_(item)
        return item

    @objc.python_method
    def build_menu(self):
        self.status_item = NSStatusBar.systemStatusBar().statusItemWithLength_(NSVariableStatusItemLength)
        menu = NSMenu.alloc().init()
        menu.setAutoenablesItems_(False)

        self.status_line = self.add_item(menu, "")
        self.status_line.setEnabled_(False)
        self.permissions_item = self.add_item(menu, "Open Privacy Settings…", "openPrivacySettings:")
        menu.addItem_(NSMenuItem.separatorItem())
        self.todo_item = self.add_item(menu, "Open To-Do List", "openTodo:", "t")
        self.meeting_item = self.add_item(menu, "Start Meeting Transcript", "toggleMeeting:")
        menu.addItem_(NSMenuItem.separatorItem())

        recent_item = self.add_item(menu, "Recent (click to copy)")
        self.recent_menu = NSMenu.alloc().init()
        self.recent_menu.setAutoenablesItems_(False)
        recent_item.setSubmenu_(self.recent_menu)
        menu.addItem_(NSMenuItem.separatorItem())

        self.cleanup_item = self.add_item(menu, "Clean Up Text with AI", "toggleCleanup:")
        model_menu = NSMenu.alloc().init()
        self.model_items = {}
        for label, model in LLM_MODELS:
            self.model_items[model] = self.add_item(model_menu, label, "chooseModel:")
            self.model_items[model].setRepresentedObject_(model)
        self.add_item(menu, "Cleanup Model").setSubmenu_(model_menu)
        meeting_model_menu = NSMenu.alloc().init()
        self.meeting_model_items = {}
        for label, model in LLM_MODELS:
            self.meeting_model_items[model] = self.add_item(meeting_model_menu, label, "chooseMeetingModel:")
            self.meeting_model_items[model].setRepresentedObject_(model)
        self.add_item(menu, "Meeting To-Do Model").setSubmenu_(meeting_model_menu)
        meeting_asr_menu = NSMenu.alloc().init()
        self.meeting_asr_items = {}
        for label, model in MEETING_ASR_MODELS:
            self.meeting_asr_items[model] = self.add_item(meeting_asr_menu, label, "chooseMeetingAsr:")
            self.meeting_asr_items[model].setRepresentedObject_(model)
        self.add_item(menu, "Meeting Speech-to-Text").setSubmenu_(meeting_asr_menu)
        meeting_todos_menu = NSMenu.alloc().init()
        self.meeting_todos_items = {}
        for mode, label in MEETING_TODO_MODES.items():
            self.meeting_todos_items[mode] = self.add_item(meeting_todos_menu, label, "chooseMeetingTodos:")
            self.meeting_todos_items[mode].setRepresentedObject_(mode)
        self.add_item(menu, "Meeting To-Dos").setSubmenu_(meeting_todos_menu)
        self.log_item = self.add_item(menu, "Save Dictations in Log", "toggleLogDictations:")
        self.hotkey_items = {}
        for config_key, title, action in (("hotkey", "Dictation Hotkey", "chooseHotkey:"),
                                          ("hotkey2", "Second Dictation Hotkey", "chooseHotkey2:"),
                                          ("todo_hotkey", "To-Do Hotkey", "chooseTodoHotkey:")):
            hotkey_menu = NSMenu.alloc().init()
            hotkey_menu.setAutoenablesItems_(False)
            names = (["None"] if config_key == "hotkey2" else []) + list(HOTKEYS)
            self.hotkey_items[config_key] = {name: self.add_item(hotkey_menu, name, action) for name in names}
            hotkey_menu.addItem_(NSMenuItem.separatorItem())
            note = self.add_item(hotkey_menu, "For Fn, set Keyboard › “Press 🌐 key to” › Do Nothing")
            note.setEnabled_(False)
            self.add_item(menu, title).setSubmenu_(hotkey_menu)
        self.login_item = self.add_item(menu, "Start at Login", "toggleStartAtLogin:")
        self.login_item.setEnabled_(APP_PATH is not None)
        menu.addItem_(NSMenuItem.separatorItem())

        self.add_item(menu, "Open Log", "openLog:")
        self.add_item(menu, "Open Project Folder", "openFolder:")
        self.add_item(menu, "Restart", "restart:")
        self.add_item(menu, "Quit PersonalWisper", "quit:", "q")
        self.status_item.setMenu_(menu)

    @objc.python_method
    def request_refresh(self):
        AppHelper.callAfter(self.refresh)

    def refresh(self):
        d, meeting = self.dictation, self.meetings
        hotkey = " or ".join(name for name in (self.config["hotkey"], self.config["hotkey2"]) if name)
        if self.missing_permissions:
            icon, status = "exclamationmark.triangle", f"Needs permission: {', '.join(self.missing_permissions)}"
        elif d.problem and not d.loaded:
            icon, status = "exclamationmark.triangle", d.problem
        elif not d.loaded:
            icon, status = "hourglass", d.download_status or d.loading_status
        elif d.holding_mode == "dictate":
            icon, status = "mic.fill", "Listening…"
        elif d.holding_mode == "todo":
            icon, status = "mic.fill", "Listening for a to-do…"
        elif meeting.recording:
            marks = len(meeting.marks)
            icon, status = "record.circle", (f"Recording a meeting since {meeting.started_at:%-I:%M %p}"
                                             + (f" · {marks} mark{'s' * (marks != 1)}" if marks else
                                                f" · tap {self.config['todo_hotkey']} to mark a to-do"))
        elif d.busy:
            icon, status = "waveform", "Transcribing…"
        elif d.download_status:
            icon, status = "arrow.down.circle", d.download_status
        elif meeting.status:
            icon, status = "waveform", meeting.status
        elif d.problem:
            icon, status = "exclamationmark.triangle", d.problem
        elif d.notice:
            icon, status = "mic", d.notice
        else:
            icon, status = "mic", f"Ready. Hold {hotkey} to dictate, {self.config['todo_hotkey']} for a to-do"

        image = NSImage.imageWithSystemSymbolName_accessibilityDescription_(icon, "PersonalWisper")
        image.setTemplate_(True)
        button = self.status_item.button()
        button.setImage_(image)
        button.setContentTintColor_(NSColor.systemRedColor() if d.holding_mode or meeting.recording else None)
        self.status_line.setTitle_(status)
        self.permissions_item.setHidden_(not self.missing_permissions)

        self.cleanup_item.setState_(1 if self.config["cleanup"] else 0)
        self.log_item.setState_(1 if self.config["log_dictations"] else 0)
        for model, item in self.model_items.items():
            item.setState_(1 if model == self.config["llm_model"] else 0)
        for config_key, items in self.hotkey_items.items():
            for name, item in items.items():
                item.setState_(1 if name == (self.config[config_key] or "None") else 0)
        for name, item in self.hotkey_items["hotkey2"].items():  # it can't overlap the other two
            item.setEnabled_(name == "None" or not any(hotkeys_conflict(name, self.config[key])
                                                         for key in ("hotkey", "todo_hotkey")))
        dirty = self.todo.counts()["dirty"]
        self.todo_item.setTitle_(f"Open To-Do List ({dirty} dirty)" if dirty else "Open To-Do List")
        self.todo_item.setEnabled_(self.todo_url is not None)
        unavailable = unavailable_reason()
        if meeting.recording:
            self.meeting_item.setTitle_(f"Stop Meeting Transcript (started {meeting.started_at:%-I:%M %p})")
        elif meeting.status:
            self.meeting_item.setTitle_("Reading the last meeting…")
        else:
            self.meeting_item.setTitle_(unavailable or "Start Meeting Transcript")
        self.meeting_item.setEnabled_(meeting.recording or (d.loaded and not meeting.status and not unavailable))
        for model, item in self.meeting_model_items.items():
            item.setState_(1 if model == self.config["meeting_model"] else 0)
        for model, item in self.meeting_asr_items.items():
            item.setState_(1 if model == self.config["meeting_asr"] else 0)
        for mode, item in self.meeting_todos_items.items():
            item.setState_(1 if mode == self.config["meeting_todos"] else 0)
        self.login_item.setState_(1 if start_at_login_enabled() else 0)

        self.recent_menu.removeAllItems()
        for text in d.history:
            label = text if len(text) <= 60 else text[:57] + "…"
            item = self.add_item(self.recent_menu, label, "copyRecent:")
            item.setRepresentedObject_(text)
        if not d.history:
            self.add_item(self.recent_menu, "Nothing yet").setEnabled_(False)

    # ---- menu actions ----

    def toggleCleanup_(self, sender):
        self.config["cleanup"] = not self.config["cleanup"]
        save_config(self.config)
        self.refresh()

    def toggleLogDictations_(self, sender):
        self.config["log_dictations"] = not self.config["log_dictations"]
        save_config(self.config)
        self.refresh()

    def toggleMeeting_(self, sender):
        try:
            if self.meetings.recording:
                self.meetings.stop()
            else:
                self.meetings.start()
                self.dictation.problem = None
        except Exception as e:
            self.dictation._fail(f"Couldn't start the meeting transcript: {e}")
        self.refresh()

    def chooseMeetingTodos_(self, sender):
        self.config["meeting_todos"] = sender.representedObject()
        save_config(self.config)
        self.refresh()

    def chooseMeetingModel_(self, sender):
        self.config["meeting_model"] = sender.representedObject()
        save_config(self.config)
        self.refresh()

    def chooseMeetingAsr_(self, sender):
        self.config["meeting_asr"] = sender.representedObject()
        save_config(self.config)
        self.refresh()

    def chooseModel_(self, sender):
        model = sender.representedObject()
        if model == self.config["llm_model"]:
            return
        previous = self.config["llm_model"]
        self.config["llm_model"] = model
        save_config(self.config)
        self.refresh()

        def done(future):
            # Keep using (and remembering) the model that's still loaded, unless another was chosen meanwhile.
            if future.exception() and self.config["llm_model"] == model:
                pipe = self.dictation.pipe
                self.config["llm_model"] = pipe.llm_model if pipe else previous
                save_config(self.config)
                self.request_refresh()
        self.dictation.switch_model(model, sender.title()).add_done_callback(done)

    @objc.python_method
    def set_hotkey(self, config_key, name):
        other = "todo_hotkey" if config_key == "hotkey" else "hotkey"
        if hotkeys_conflict(self.config[other], name):  # picking the other hotkey's key swaps them
            previous = self.config[config_key]
            self.config[other] = previous if not hotkeys_conflict(previous, name) else next(
                n for n in HOTKEYS if not hotkeys_conflict(n, name))
        self.config[config_key] = name
        if self.config["hotkey2"] and any(hotkeys_conflict(self.config["hotkey2"], self.config[key])
                                          for key in ("hotkey", "todo_hotkey")):
            self.config["hotkey2"] = None  # it overlaps the hotkey just picked
        save_config(self.config)
        self.refresh()

    def chooseHotkey_(self, sender):
        self.set_hotkey("hotkey", sender.title())

    def chooseHotkey2_(self, sender):
        self.config["hotkey2"] = None if sender.title() == "None" else sender.title()
        save_config(self.config)
        self.refresh()

    def chooseTodoHotkey_(self, sender):
        self.set_hotkey("todo_hotkey", sender.title())

    def openTodo_(self, sender):
        browser = ["-a", str(CHROME_PATH)] if CHROME_PATH.exists() else []
        subprocess.Popen(["/usr/bin/open", *browser, self.todo_url])

    def toggleStartAtLogin_(self, sender):
        set_start_at_login(not start_at_login_enabled())
        self.refresh()

    def copyRecent_(self, sender):
        set_clipboard(sender.representedObject())

    def openPrivacySettings_(self, sender):
        pane = self.missing_permissions[0] if self.missing_permissions else "Accessibility"
        NSWorkspace.sharedWorkspace().openURL_(NSURL.URLWithString_(PRIVACY_URLS[pane]))

    def openLog_(self, sender):
        NSWorkspace.sharedWorkspace().openURL_(NSURL.fileURLWithPath_(str(LOG_PATH)))

    def openFolder_(self, sender):
        NSWorkspace.sharedWorkspace().openURL_(NSURL.fileURLWithPath_(str(ROOT)))

    def restart_(self, sender):
        self.relaunch()

    @objc.python_method
    def relaunch(self, hard=False):
        """Start a new copy and quit. hard: exit at once, since a normal exit could wait on a stuck audio thread."""
        if APP_PATH:
            subprocess.Popen(["/bin/sh", "-c", f'sleep 2; /usr/bin/open "{APP_PATH}"'], start_new_session=True)
            if hard:
                os._exit(0)
            NSApplication.sharedApplication().terminate_(None)
        else:
            os.execv(sys.executable, [sys.executable] + sys.argv)

    @objc.python_method
    def mic_stuck(self):
        """macOS's audio system wedged while starting or stopping the microphone (see Recorder). Only a restart frees
        the mic, so stop taking new recordings and restart once what was already recorded has been typed."""
        d = self.dictation
        d.loaded, d.loading_status = False, "Microphone got stuck. Restarting…"
        self.request_refresh()

        def restart_when_idle():
            time.sleep(1)  # let the recording that was just stopped reach the model thread
            while d.busy or d.running_job or not d.jobs.empty():
                time.sleep(0.2)
            log("Restarting to free the microphone")
            AppHelper.callAfter(self.relaunch, True)
        threading.Thread(target=restart_when_idle, daemon=True).start()

    def quit_(self, sender):
        NSApplication.sharedApplication().terminate_(None)


def main():
    LOG_PATH.parent.mkdir(exist_ok=True)
    # Only one copy may run, or every dictation would be pasted twice.
    lock = open(LOCK_PATH, "w")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        log("PersonalWisper is already running.")
        return

    app = NSApplication.sharedApplication()
    app.setActivationPolicy_(NSApplicationActivationPolicyAccessory)  # menu bar only, no Dock icon
    global CONTROLLER
    CONTROLLER = MenuController.alloc().init()
    app.setDelegate_(CONTROLLER)
    log("Starting PersonalWisper")
    AppHelper.runEventLoop()


if __name__ == "__main__":
    main()
