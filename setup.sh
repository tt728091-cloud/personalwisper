#!/bin/zsh
# Install PersonalWisper, or update it after `git pull`: Python packages, the default models, and the app.
# Safe to run again; it only does what's missing.
set -euo pipefail
cd "$(dirname "$0")"

if [[ "$(uname -m)" != arm64 ]]; then
  echo "PersonalWisper needs a Mac with Apple silicon (M1 or later)."
  exit 1
fi
if ! xcode-select -p >/dev/null 2>&1; then
  echo "Apple's Command Line Tools are needed to build the app. Installing them now;"
  echo "run ./setup.sh again when that finishes."
  xcode-select --install
  exit 1
fi
if ! command -v uv >/dev/null; then
  if command -v brew >/dev/null; then
    echo "Installing uv (Python package manager) with Homebrew..."
    brew install uv
  else
    echo "PersonalWisper needs uv to install its Python packages. Install it with Homebrew (brew install uv)"
    echo "or see https://docs.astral.sh/uv/getting-started/installation/, then run ./setup.sh again."
    exit 1
  fi
fi

echo "Installing Python packages..."
uv sync --locked

mkdir -p logs
[[ -f vocabulary.txt ]] || cp vocabulary.example.txt vocabulary.txt

echo "Downloading speech-to-text and the default cleanup model (about 4.5 GB the first time)..."
.venv/bin/python -c "
from pipeline import ASR_MODEL, LLM_MODEL, model_path
for name in (ASR_MODEL, LLM_MODEL):
    print('  ' + name)
    model_path(name)
"

# Rebuilding the app can make macOS ask for its permissions again, so only build it when it's missing.
if [[ ! -d PersonalWisper.app ]]; then
  ./build_app.sh
fi

if ! command -v ffmpeg >/dev/null; then
  echo "Optional: install ffmpeg (brew install ffmpeg) to dictate for longer than two minutes at a time."
fi
echo
echo "Done. Start it with: open PersonalWisper.app"
