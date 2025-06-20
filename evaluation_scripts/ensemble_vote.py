#!/usr/bin/env python3
"""
Majority-vote ensemble for the easy / medium / hard levels.
"""

from __future__ import annotations
import argparse, json
from pathlib import Path
from typing import Dict, List
from collections import Counter
from tqdm import tqdm

DIFFS = ("easy", "medium", "hard")

def read_changes(folder: Path) -> Dict[str, List[int]]:
    out: Dict[str, List[int]] = {}
    for f in folder.glob("solution-problem-*.json"):
        pid = f.stem[17:]
        out[pid] = json.loads(f.read_text())["changes"]
    return out

def vote(bits: List[int]) -> int:
    """Majority vote with ties -> 1."""
    c = Counter(bits)
    return int(c[1] >= c[0])

def main() -> None:
    p = argparse.ArgumentParser("Majority-vote ensemble: weighted voting possible.")
    p.add_argument("--pred-dir", action="append", required=True,
                   help="Pred dir (repeat the flag)")
    p.add_argument("--output-dir", required=True,
                   help="Root final predictions")
    args = p.parse_args()

    pred_dirs = [Path(d).expanduser() for d in args.pred_dir]
    out_root  = Path(args.output_dir).expanduser()
    out_root.mkdir(parents=True, exist_ok=True)

    for diff in DIFFS:
        diff_data: List[Dict[str, List[int]]] = []
        for d in pred_dirs:
            diff_dir = d / diff
            if not diff_dir.is_dir():
                raise FileNotFoundError(
                    f"sub-dir {diff_dir} not found.")
            diff_data.append(read_changes(diff_dir))


        common = sorted(set.intersection(*(set(d) for d in diff_data)))
        if not common:
            print(f"no common problems for diff '{diff}' -> skipped.")
            continue

        out_dir = out_root / diff
        out_dir.mkdir(parents=True, exist_ok=True)

        for pid in tqdm(common, desc=diff, unit="file"):
            lists = [d[pid] for d in diff_data]

            lens = {len(lst) for lst in lists}
            if len(lens) != 1:
                raise ValueError(f"Len wrong {pid}: {lens}")

            voted = [vote(bits) for bits in zip(*lists)]
            (out_dir / f"solution-problem-{pid}.json").write_text(
                json.dumps({"changes": voted}, indent=2), encoding="utf-8")

        print(f"Done {len(common)} problems for '{diff}'")

    print(f"Done: written to {out_root}")

if __name__ == "__main__":
    main()
