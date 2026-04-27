#!/usr/bin/env python3
"""
Convert a checklist-style markdown file to JSON.

Markdown format:
    ### SECTION TITLE
    - [ ] [[Unchecked item]]
    - [X] [[Checked item]]

The top-level `#` heading (if any) is ignored. Only `###` section headers
and `- [ ]` / `- [X]` checklist lines are parsed.
"""

import argparse
import json
import re
import sys
import uuid
from pathlib import Path


SECTION_RE = re.compile(r"^###\s+(.+?)\s*$")
# Captures the checkbox state (space, x, or X) and the [[title]] text.
ITEM_RE = re.compile(r"^\s*-\s*\[([ xX])\]\s*\[\[(.+?)\]\]\s*$")


def new_uuid() -> str:
    """Return an uppercase UUID string matching Swift's default format."""
    return str(uuid.uuid4()).upper()


def parse_markdown(md_text: str) -> list[dict]:
    sections: list[dict] = []
    current: dict | None = None

    for line in md_text.splitlines():
        section_match = SECTION_RE.match(line)
        if section_match:
            current = {
                "id": new_uuid(),
                "title": section_match.group(1).strip(),
                "checklists": [],
            }
            sections.append(current)
            continue

        item_match = ITEM_RE.match(line)
        if item_match:
            if current is None:
                # Checklist item appeared before any ### section; skip it.
                continue
            is_complete = item_match.group(1).lower() == "x"
            current["checklists"].append({
                "id": new_uuid(),
                "title": item_match.group(2).strip(),
                "isComplete": is_complete,
            })

    return sections


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Convert a checklist markdown file to JSON."
    )
    parser.add_argument("input", help="Path to the input markdown file")
    parser.add_argument(
        "-o", "--output",
        help="Path to the output JSON file (defaults to <input>.json)",
    )
    parser.add_argument(
        "--indent", type=int, default=2,
        help="JSON indentation (default: 2)",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.is_file():
        print(f"Error: input file not found: {input_path}", file=sys.stderr)
        return 1

    output_path = Path(args.output) if args.output else input_path.with_suffix(".json")

    md_text = input_path.read_text(encoding="utf-8")
    data = parse_markdown(md_text)
    output_path.write_text(
        json.dumps(data, indent=args.indent, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    total_items = sum(len(s["checklists"]) for s in data)
    print(f"Wrote {len(data)} section(s), {total_items} item(s) to {output_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
