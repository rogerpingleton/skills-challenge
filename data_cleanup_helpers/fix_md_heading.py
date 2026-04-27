#!/usr/bin/env python3
"""
fix_md_heading.py

For each markdown file provided, ensures the top-level # heading matches
the filename (without the .md extension).

  - If the file starts with a # heading, its text is replaced with the filename.
  - If the file does not start with a # heading, one is inserted at the top.

Usage:
    python fix_md_heading.py file1.md file2.md ...
    python fix_md_heading.py *.md
"""

import sys
from pathlib import Path


def fix_heading(path: Path) -> None:
    title = path.stem  # filename without extension
    heading_line = f"# {title}\n"

    text = path.read_text(encoding="utf-8")
    lines = text.splitlines(keepends=True)

    # Find the first non-blank line
    first_content_index = next(
        (i for i, line in enumerate(lines) if line.strip()), None
    )

    if first_content_index is not None and lines[first_content_index].startswith("# "):
        # Replace the existing top-level heading
        if lines[first_content_index] == heading_line:
            print(f"  [skip]    {path}  (heading already correct)")
            return
        lines[first_content_index] = heading_line
        path.write_text("".join(lines), encoding="utf-8")
        print(f"  [updated] {path}")
    else:
        # Insert a new heading (plus a blank line) before any existing content
        if first_content_index is None:
            # File is empty or all blank lines
            new_lines = [heading_line]
        elif first_content_index == 0:
            # Content starts at the very top — prepend heading + blank line
            new_lines = [heading_line, "\n"] + lines
        else:
            # There are leading blank lines — insert heading before first content
            new_lines = lines[:first_content_index] + [heading_line, "\n"] + lines[first_content_index:]
        path.write_text("".join(new_lines), encoding="utf-8")
        print(f"  [added]   {path}")


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: python fix_md_heading.py <file.md> [file.md ...]")
        sys.exit(1)

    for arg in sys.argv[1:]:
        path = Path(arg)
        if not path.exists():
            print(f"  [error]   {path}  (file not found)")
            continue
        if path.suffix.lower() != ".md":
            print(f"  [skip]    {path}  (not a .md file)")
            continue
        fix_heading(path)


if __name__ == "__main__":
    main()
