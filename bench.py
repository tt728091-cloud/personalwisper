"""Benchmark the dictation pipeline on audio files.

Usage: .venv/bin/python bench.py recording.wav [more.wav ...]
"""
import os
import sys
import time

import mlx.core as mx

from pipeline import ASR_MODEL, LLM_MODEL, ROOT, Pipeline

RUNS = int(os.environ.get("RUNS", "3"))


def timed(fn):
    start = time.perf_counter()
    result = fn()
    return result, time.perf_counter() - start


def main(paths):
    print(f"Loading {ASR_MODEL} + {LLM_MODEL} ...")
    pipe, load_time = timed(Pipeline)
    print(f"  loaded in {load_time:.1f}s")
    print(f"Memory in use with both models loaded: {mx.get_active_memory() / 1e9:.1f} GB\n")

    for path in paths:
        print("=" * 70)
        print(path)
        asr_times, llm_times = [], []
        for _ in range(RUNS):
            raw, t_asr = timed(lambda: pipe.transcribe_file(path))
            (clean, stats), t_llm = timed(lambda: pipe.cleanup(raw))
            asr_times.append(t_asr)
            llm_times.append(t_llm)

        # First run includes one-time warmup, so report the best of the rest.
        best_asr, best_llm = min(asr_times[1:] or asr_times), min(llm_times[1:] or llm_times)
        print(f"\nRAW (Parakeet):\n  {raw}")
        print(f"\nCLEAN (LLM):\n  {clean}")
        print(f"\nTiming (warm): speech-to-text {best_asr:.2f}s + cleanup {best_llm:.2f}s "
              f"= {best_asr + best_llm:.2f}s after you stop talking")
        print(f"  first run (cold): {asr_times[0]:.2f}s + {llm_times[0]:.2f}s")
        print(f"  LLM: {stats.prompt_tokens} prompt tokens @ {stats.prompt_tps:.0f} tok/s, "
              f"{stats.generation_tokens} output tokens @ {stats.generation_tps:.0f} tok/s\n")

    print(f"Peak memory during run: {mx.get_peak_memory() / 1e9:.1f} GB")


if __name__ == "__main__":
    main(sys.argv[1:] or [str(p) for p in sorted((ROOT / "samples").glob("*.wav"))])
