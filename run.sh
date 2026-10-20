#!/bin/zsh
# Start PersonalWisper from a terminal, for debugging. Models that aren't downloaded yet are fetched on first use.
cd "$(dirname "$0")"
exec .venv/bin/python personalwisper.py
