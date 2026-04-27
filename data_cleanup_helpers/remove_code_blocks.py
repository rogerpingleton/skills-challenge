import re
import sys
from pathlib import Path

def remove_code_blocks(text: str) -> str:
    return re.sub(r'```.*?```\n?', '', text, flags=re.DOTALL)

def process_directory(directory: str, dry_run: bool = False) -> None:
    path = Path(directory)

    if not path.is_dir():
        print(f"Error: '{directory}' is not a valid directory.")
        sys.exit(1)

    md_files = list(path.rglob("*.md"))

    if not md_files:
        print("No markdown files found.")
        return

    for file in md_files:
        original = file.read_text(encoding="utf-8")
        cleaned = remove_code_blocks(original)

        if original == cleaned:
            print(f"[unchanged] {file}")
            continue

        if dry_run:
            print(f"[dry run]   {file}")
        else:
            file.write_text(cleaned, encoding="utf-8")
            print(f"[processed] {file}")

if __name__ == "__main__":
    args = sys.argv[1:]

    if not args:
        print("Usage: python solution.py <directory> [--dry-run]")
        sys.exit(1)

    directory = args[0]
    dry_run = "--dry-run" in args

    process_directory(directory, dry_run)
