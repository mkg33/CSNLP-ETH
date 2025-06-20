#!/usr/bin/env python3
"""
Count total number of changes in truth files.
"""

from __future__ import annotations
import argparse, json, sys
from pathlib import Path
from typing import Tuple

DIFFS = ("easy", "medium", "hard")

def count_changes_in_dir(dir_path: Path) -> Tuple[int, int]:
    n_files = 0
    n_changes = 0
    for truth_file in dir_path.glob("truth-problem-*.json"):
        data = json.loads(truth_file.read_text())
        n_changes += len(data["changes"])
        n_files += 1
    return n_files, n_changes

def main() -> None:
    parser = argparse.ArgumentParser("count total changes")
    parser.add_argument("--data-root", default="pan25_data",
                        help="root dir")
    args = parser.parse_args()

    root = Path(args.data_root).expanduser()
    if not root.is_dir():
        sys.exit(f"error: data-root {root} does not exist")

    grand_total = 0
    print("difficulty  #files   #changes")
    for diff in DIFFS:
        train_dir = root / diff / "train"
        if not train_dir.is_dir():
            print(f"error: {train_dir} missing – skipped")
            continue
        n_files, n_changes = count_changes_in_dir(train_dir)
        grand_total += n_changes
        print(f"{diff:<10}  {n_files:>7}  {n_changes:>9}")

    print(f"{'TOTAL':<10}            {grand_total:>9}")

if __name__ == "__main__":
    main()
