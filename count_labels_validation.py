#!/usr/bin/env python3
"""
Count the total number labels.
The dir is (in my case):
    easy/truth-problem-*.json
    ditto for medium and hard.
"""

from __future__ import annotations
import argparse, json, sys
from pathlib import Path
from typing import Tuple

DIFFS = ("easy", "medium", "hard")


def count_changes_in_dir(dir_path: Path) -> Tuple[int, int]:
    n_files = 0
    n_changes = 0
    for jf in dir_path.glob("truth-problem-*.json"):
        data = json.loads(jf.read_text())
        n_changes += len(data["changes"])
        n_files += 1
    return n_files, n_changes


def main() -> None:
    parser = argparse.ArgumentParser("Count labels")
    parser.add_argument("--data-root", default="pan25_data",
                        help="Root for the PAN dir.")
    args = parser.parse_args()

    root = Path(args.data_root).expanduser()
    if not root.is_dir():
        sys.exit(f"root {root} does not exist")

    grand_total = 0
    print(f"{'difficulty':<10} {'#files':>7} {'#changes':>10}")
    print("-" * 28)

    for diff in DIFFS:
        diff_dir = root / diff
        if not diff_dir.is_dir():
            print(f"{diff_dir} missing -> skipped")
            continue
        n_files, n_changes = count_changes_in_dir(diff_dir)
        grand_total += n_changes
        print(f"{diff:<10} {n_files:>7} {n_changes:>10}")

    print("-" * 28)
    print(f"{'TOTAL':<10} {'':>7} {grand_total:>10}")


if __name__ == "__main__":
    main()
