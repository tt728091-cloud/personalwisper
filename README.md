# PersonalWisper

Local, hold-to-talk dictation for macOS. Hold the hotkey (Fn/Globe by default), speak, release, and the
cleaned-up text is pasted at your cursor. Speech-to-text is NVIDIA Parakeet and cleanup is a Qwen model, both
running on your Mac with MLX. After the one-time model download, your audio and text never leave your Mac.

Needs a Mac with Apple silicon (M1 or later).

## Install

```sh
git clone git@github.com:tt728091-cloud/personalwisper.git ~/personalwisper
cd ~/personalwisper
./setup.sh
open PersonalWisper.app
```

The clone uses SSH, so the Mac's SSH key needs to be added to your GitHub account
(`ssh -T git@github.com` should greet you by name). Without an SSH key, clone with
`https://github.com/tt728091-cloud/personalwisper.git` instead.

`setup.sh` installs everything PersonalWisper needs inside this folder: the Python packages (with
[uv](https://docs.astral.sh/uv/), installed via Homebrew if it's missing), the speech-to-text model and the small
cleanup model (about 4.5 GB), and `PersonalWisper.app`. If Apple's Command Line Tools aren't installed it starts their
installer; run `./setup.sh` again when that's done.

The first time it runs, PersonalWisper asks for three permissions in System Settings › Privacy & Security:
**Microphone**, **Input Monitoring**, and **Accessibility**. Grant them, then choose **Restart** from its menu.
To use Fn as the hotkey, set System Settings › Keyboard › "Press 🌐 key to" › **Do Nothing**.
Keyboard without Fn (a PC keyboard, or one shared through a software KVM like Deskflow)? Pick **Home** under
**Second Dictation Hotkey** in the menu; it works alongside the main hotkey. While it's set, Home dictates
instead of jumping to the start of the line or page.

It lives in the menu bar (no Dock icon). Turn on **Start at Login** in its menu to start it automatically.

### Update

```sh
cd ~/personalwisper
git pull
./setup.sh
```

Then choose **Restart** from the menu. If you move the folder, run `./build_app.sh` (you may need to grant the
permissions again).

## Cleanup models

Pick the model under **Cleanup Model** in the menu. Your choice is remembered.

- **Qwen3 4B** (default): uses about 3 GB of memory. Good for most dictation.
- **Qwen3.6 35B**: noticeably better at long or messy dictation, but uses about 20 GB of memory. It's a 19 GB
  download, fetched the first time you pick it (the menu shows progress, and dictation keeps working with the
  current model meanwhile). Best on a Mac with 32 GB of memory or more.

Turn off **Clean Up Text with AI** to paste exactly what speech-to-text heard.

## Your words

`vocabulary.txt` holds names the cleanup should spell right and fixes for words speech-to-text keeps mishearing
(`personal whisper => PersonalWisper`). `setup.sh` creates it from `vocabulary.example.txt`. Add a fix whenever you
see the same mistake twice, then restart.

## To-do list

Hold the **to-do hotkey** (Option+Z by default: hold Option, then hold Z) anywhere and talk. Whatever you say becomes
a **dirty** item in the to-do list. You'll hear a "morse" sound when it starts listening and a "glass" sound when the
item is saved. Choose **Open To-Do List** from the menu to review items at http://127.0.0.1:47615/ (in Chrome if it's
installed).

- **Dirty**: captured by voice. Edit the notes, or click **Talk to refine** to add details by voice. What you say is
  cleaned up and appended; your earlier notes are never rewritten.
- **Local evaluation**: every dirty, research and work item is evaluated in the background on your Mac by the cleanup
  model you picked: a clearer goal, every detail you gave, what "done" looks like, and one to three questions only
  you can answer. Answer by typing or **Talk to answer**, and it evaluates again until the item is **ready**. It never
  changes your notes, and it pauses whenever you dictate. Put a description of a project in
  `projects/<project name>.md` to give the evaluator background (see `projects/README.md`).
- **Research, work and follow-ups** go through [T3 Code](https://github.com/pingdotgg/t3code) (optional). Without T3, the to-do list
  still captures, refines and evaluates items. With it:
  - **Make research item** / **Start research in T3** creates a read-only (plan mode) thread in the item's T3 project.
    Pick the model and effort next to the button. The choices live in `RESEARCH_MODELS` in `t3client.py`. Nothing is
    automatic: research only starts when you click. The reply is copied back to the item automatically.
  - **Work** items (**Just do it**, **Make it a work item**, **Make work item**) open a regular thread that **can
    change files**, either in a **new worktree** (its own `t3code/…` branch in `~/.t3/worktrees/`) or in the
    **project folder**. Projects listed in `work_worktree_base` in `config.json` (for example
    `{"my-project": "main"}`) default to a new worktree from that branch.
  - **One thread per item**: later steps (research → work, **Follow up**, retries) continue the same thread.
    **Start over in a new thread** is always available.
  - New items go to the project named in `todo_project` in `config.json`, or the first T3 project.
  - Research and work send the item's notes to the model you pick in T3, so only use them for things you're allowed
    to share with that service.
  - T3 access needs a token in `secrets/t3-token`:
    `t3 auth session issue --ttl 365d --label personalwisper --token-only > secrets/t3-token`.
    The integration uses T3's internal orchestration API; if a T3 update changes it, the to-do page shows a banner
    instead of sending anything.

## Meeting transcripts
Choose **Start Meeting Transcript** in the menu when a meeting starts (Google Meet, a Slack huddle, anything that
plays through this Mac) and **Stop Meeting Transcript** when it ends. PersonalWisper records two sides: your
microphone ("Me") and the sound other apps play ("Others"). It records your microphone the whole time, even while
you're muted in the meeting app.

**Mark the to-dos as they come up.** During a meeting the to-do hotkey (Option+Z by default) marks the moment instead
of saving a to-do: tap it when someone asks you for something or you agree to do something (you'll hear a purr), or
hold it and say what the to-do is (mute yourself in the meeting app first if you don't want the others to hear).
The transcript only knows "Me" and "Others", so say who it's from if that matters: "Andreas said create a new
document on PX" becomes "Create a new document on PX (from Andreas)", noted as asked by Andreas. Add the names you
use to `vocabulary.txt` so they're spelled right.

When you stop, it:

1. Transcribes both sides on this Mac and saves the transcript as text in `meetings/` (for example
   `meetings/2026-10-20_1405.txt`, with a time on every line and `>>> MARK <<<` where you pressed the hotkey). The
   audio is only ever in memory and is dropped as soon as it's transcribed; it's never written to disk.
2. For each mark, the **Meeting To-Do Model** (the 35B by default, downloaded the first time) reads the transcript
   from two minutes before it to just after it and writes one **dirty** to-do: what you said there, or else what was
   being discussed just before (usually what someone asked of you). Each item keeps those lines and links the
   transcript, and the transcript ends with the list of to-dos added. No marks, no to-dos.

Choose **Meeting To-Dos › Find Them All** to have it read the whole meeting instead and add everything you said
you'd do and everything others asked of you, as well as your marks. That finds more, including things you didn't
mean as to-dos.

The meeting model is loaded next to the cleanup model just for this (only if there are marks, or with Find Them
All), so dictation keeps working, and it's unloaded once the new items have been evaluated (the 35B needs about
20 GB of memory while it's loaded). Pick the 4B under **Meeting To-Do Model** to use less memory.

- Use headphones. With speakers, your microphone also hears the others; those repeats are dropped, but not perfectly.
- Recording other apps needs macOS 14.2 or later. The first time, macOS asks to let PersonalWisper record system
  audio (System Settings › Privacy & Security › Screen & System Audio Recording, under System Audio Recording Only).
  If that's denied, the others' side comes out silent and the menu says so.
- After updating from a version without meeting transcripts, `setup.sh` rebuilds the app, so macOS may ask for its
  other permissions again.

## Menu bar icon
- 🎙 mic: ready
- red mic: listening
- red record button: recording a meeting
- waveform: transcribing
- arrow: downloading a model
- hourglass: loading models
- ⚠︎ triangle: needs attention; open the menu for details

## Privacy

- Audio and text are processed on your Mac. The only network use is downloading models from Hugging Face the first
  time each one is needed, and T3 research/work when you start it.
- `logs/personalwisper.log` records timings and errors. It records what you dictated only if you turn on
  **Save Dictations in Log** in the menu.
- To-do items are stored in `todo.db` in this folder, and meeting transcripts in `meetings/`. Meeting audio is never
  saved.
- Your settings (`config.json`), vocabulary, project notes, logs, to-dos, meeting transcripts, token and models are
  all ignored by git
  (see `.gitignore`), so they're never part of the repository.

## Files
- `personalwisper.py`: the menu-bar app (hotkeys, recording, paste, model downloads)
- `pipeline.py`: the models and the cleanup instructions and examples (edit these to change cleanup behavior)
- `todo_server.py`, `todos.py`, `t3client.py`, `web/`: the to-do list, its storage (`todo.db`), and the T3 link
- `meetings.py`: meeting transcripts and the to-dos found in them; `app/systemaudio.m`: the helper that records
  other apps' sound (built into the app by `build_app.sh`)
- `setup.sh`: install or update; `build_app.sh`: rebuild `PersonalWisper.app`; `run.sh`: run from a terminal for
  debugging (uses the terminal's permissions instead of the app's)
- `bench.py`: speed and quality test on your own recordings: `.venv/bin/python bench.py recording.wav`

Start at Login writes `~/Library/LaunchAgents/local.personalwisper.plist`. Turning it off deletes that file.

## Credits and license

Built on [MLX](https://github.com/ml-explore/mlx), [mlx-lm](https://github.com/ml-explore/mlx-lm),
[parakeet-mlx](https://github.com/senstella/parakeet-mlx), NVIDIA's
[Parakeet TDT 0.6B v3](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3) (CC-BY-4.0) and
[Qwen](https://huggingface.co/Qwen) models (Apache 2.0), downloaded from the
[mlx-community](https://huggingface.co/mlx-community) conversions.

PersonalWisper is released under the MIT License (see `LICENSE`).
