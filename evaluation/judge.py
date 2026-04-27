#!/usr/bin/env python3
"""
AI-as-judge test runner.

Reads a JSON file of prompt/answer pairs and asks an OpenAI model to judge
each answer in the context of AI Engineering, producing a pass/fail rating
and a quality score from 1 to 10. Writes the verdicts to an output JSON file.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from openai import OpenAI, OpenAIError


JUDGE_SYSTEM_PROMPT = """You are an expert judge evaluating answers to questions \
in the field of AI Engineering. AI Engineering draws on statistics, probability, \
linear algebra, machine learning, deep learning, large language models, retrieval \
systems, evaluation, and related mathematical and software foundations.

For each answer you evaluate, you must:

1. Assign a pass/fail rating:
   - "pass": the answer contains substantively correct information for the topic \
and does not hallucinate or introduce significant factual errors. Minor omissions \
are acceptable.
   - "fail": the answer contains incorrect information, hallucinated facts, \
fabricated formulas, or is substantively misleading.

2. Assign an integer quality score from 1 to 10 based on the quality of the \
information conveyed (accuracy, completeness, clarity, and usefulness to a \
practitioner):
   - 1-3: poor (inaccurate, confusing, or largely missing the point)
   - 4-6: acceptable (correct but shallow, unclear, or incomplete)
   - 7-8: good (accurate, clear, reasonably thorough)
   - 9-10: excellent (accurate, comprehensive, well-structured, insightful)

A "fail" rating should generally map to a low score, but a "pass" may still \
receive any score from 1 to 10 depending on quality. Reply ONLY with a JSON \
object in this exact shape:

{"pass_fail": "pass" | "fail", "score": <integer 1-10>, "reasoning": "<one or two sentences>"}
"""


def judge_answer(
    client: OpenAI,
    model: str,
    prompt: str,
    answer: str,
) -> dict[str, Any]:
    """Send a single prompt/answer pair to the judge model and parse the verdict."""
    user_message = (
        "Evaluate the following answer to a question, judging it in the context "
        "of AI Engineering.\n\n"
        f"QUESTION:\n{prompt}\n\n"
        f"ANSWER:\n{answer}\n\n"
        "Respond with the JSON object described in the system instructions."
    )

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ],
        response_format={"type": "json_object"},
        temperature=0,
    )
    content = response.choices[0].message.content or ""
    return json.loads(content)


def normalize_verdict(verdict: dict[str, Any]) -> tuple[str, int, str]:
    """Coerce the model's JSON into a clean (pass_fail, score, reasoning) tuple."""
    pass_fail = str(verdict.get("pass_fail", "")).strip().lower()
    if pass_fail not in ("pass", "fail"):
        pass_fail = "fail"

    raw_score = verdict.get("score")
    try:
        score = int(raw_score)
    except (TypeError, ValueError):
        score = 1
    score = max(1, min(10, score))

    reasoning = str(verdict.get("reasoning", "")).strip()
    return pass_fail, score, reasoning


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run AI-as-judge evaluations on prompt/answer pairs using an OpenAI model."
        ),
    )
    parser.add_argument(
        "input_file",
        type=Path,
        help="Path to the input JSON file containing a list of {prompt, answer} objects.",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path("judge_results.json"),
        help="Path to write the output JSON file (default: judge_results.json).",
    )
    parser.add_argument(
        "-m",
        "--model",
        default=None,
        help=(
            "OpenAI model to use as the judge. "
            "Falls back to the OPENAI_MODEL env var, then to gpt-4o-mini."
        ),
    )
    parser.add_argument(
        "--include-reasoning",
        action="store_true",
        help="If set, include the judge's reasoning string in each output record.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    load_dotenv()

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print(
            "Error: OPENAI_API_KEY is not set. Add it to your .env file.",
            file=sys.stderr,
        )
        return 1

    model = args.model or os.environ.get("OPENAI_MODEL") or "gpt-4o-mini"

    if not args.input_file.exists():
        print(f"Error: input file not found: {args.input_file}", file=sys.stderr)
        return 1

    try:
        with args.input_file.open("r", encoding="utf-8") as f:
            entries = json.load(f)
    except json.JSONDecodeError as exc:
        print(f"Error: input file is not valid JSON: {exc}", file=sys.stderr)
        return 1

    if not isinstance(entries, list):
        print("Error: input JSON must be a list of objects.", file=sys.stderr)
        return 1

    client = OpenAI(api_key=api_key)

    print(f"Judging {len(entries)} entries with model '{model}'...\n")
    results: list[dict[str, Any]] = []

    for index, entry in enumerate(entries, start=1):
        if not isinstance(entry, dict):
            print(
                f"[{index}/{len(entries)}] Skipping non-object entry.",
                file=sys.stderr,
            )
            continue

        prompt = entry.get("prompt")
        answer = entry.get("answer")
        if not isinstance(prompt, str) or not isinstance(answer, str):
            print(
                f"[{index}/{len(entries)}] Skipping entry missing 'prompt' or 'answer'.",
                file=sys.stderr,
            )
            continue

        short_prompt = prompt.replace("\n", " ")
        if len(short_prompt) > 80:
            short_prompt = short_prompt[:77] + "..."
        print(f"[{index}/{len(entries)}] {short_prompt}")

        try:
            raw_verdict = judge_answer(client, model, prompt, answer)
        except (OpenAIError, json.JSONDecodeError) as exc:
            print(f"    ERROR: {exc}", file=sys.stderr)
            record: dict[str, Any] = {
                "prompt": prompt,
                "pass_fail": "error",
                "score": 0,
            }
            if args.include_reasoning:
                record["reasoning"] = f"error: {exc}"
            results.append(record)
            continue

        pass_fail, score, reasoning = normalize_verdict(raw_verdict)
        record = {"prompt": prompt, "pass_fail": pass_fail, "score": score}
        if args.include_reasoning:
            record["reasoning"] = reasoning
        results.append(record)
        print(f"    -> {pass_fail.upper()} (score: {score})")

    with args.output.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
        f.write("\n")

    passes = sum(1 for r in results if r.get("pass_fail") == "pass")
    fails = sum(1 for r in results if r.get("pass_fail") == "fail")
    errors = sum(1 for r in results if r.get("pass_fail") == "error")
    print(
        f"\nWrote {len(results)} results to {args.output} "
        f"(pass: {passes}, fail: {fails}, error: {errors})."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
