#!/usr/bin/env python3
"""
Run rag-eval against a batch of prompts and record answers and durations.

Reads an input JSON file that is a list of objects with at least a "prompt"
field, runs `./rag-eval --mode raw --db ./rag.sqlite "<prompt>"` for each,
and writes the results (prompt, answer, duration_seconds) to an output JSON
file.

Usage:
    python run_rag_eval.py input.json output.json
"""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path


RAG_EVAL_CMD = ["./rag-eval", "--mode", "rag", "--db", "./rag.sqlite"]


def run_prompt(prompt: str) -> tuple[str, float]:
    """Run rag-eval for one prompt. Returns (stdout, duration_seconds)."""
    start = time.perf_counter()
    result = subprocess.run(
        RAG_EVAL_CMD + [prompt],
        capture_output=True,
        text=True,
        check=False,
    )
    duration = time.perf_counter() - start

    if result.returncode != 0:
        sys.stderr.write(
            f"warning: rag-eval exited with code {result.returncode} "
            f"for prompt: {prompt!r}\n"
        )
        if result.stderr:
            sys.stderr.write(f"  stderr: {result.stderr.strip()}\n")

    return result.stdout.strip(), duration


def save_results(path: Path, results: list[dict]) -> None:
    """Write results atomically-ish: write to a temp file then replace."""
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    tmp.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run rag-eval over a batch of prompts and record results."
    )
    parser.add_argument("input", type=Path, help="Path to input JSON file with prompts.")
    parser.add_argument("output", type=Path, help="Path to write output JSON results.")
    args = parser.parse_args()

    try:
        with args.input.open("r", encoding="utf-8") as f:
            entries = json.load(f)
    except (OSError, json.JSONDecodeError) as e:
        sys.stderr.write(f"error: failed to read {args.input}: {e}\n")
        return 1

    if not isinstance(entries, list):
        sys.stderr.write("error: input JSON must be a list of objects.\n")
        return 1

    results: list[dict] = []
    total = len(entries)

    try:
        for i, entry in enumerate(entries, start=1):
            if not isinstance(entry, dict):
                sys.stderr.write(f"warning: entry {i} is not an object, skipping.\n")
                continue

            prompt = entry.get("prompt")
            if not isinstance(prompt, str) or not prompt.strip():
                sys.stderr.write(
                    f"warning: entry {i} has no valid 'prompt' field, skipping.\n"
                )
                continue

            preview = prompt if len(prompt) <= 80 else prompt[:77] + "..."
            print(f"[{i}/{total}] {preview}", flush=True)

            answer, duration = run_prompt(prompt)
            print(f"    -> {duration:.2f}s", flush=True)

            results.append({
                "prompt": prompt,
                "answer": answer,
                "duration_seconds": round(duration, 2),
            })

            # Incremental save so we don't lose work on interrupt/crash.
            try:
                save_results(args.output, results)
            except OSError as e:
                sys.stderr.write(f"warning: failed to write incremental results: {e}\n")
    except KeyboardInterrupt:
        sys.stderr.write("\ninterrupted; partial results saved.\n")
        save_results(args.output, results)
        return 130

    save_results(args.output, results)
    print(f"\nDone. Wrote {len(results)} result(s) to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
