#!/usr/bin/env python3
"""
Majority vote for the StackExchange dataset.
"""

from __future__ import annotations
import argparse, json
from pathlib import Path
from typing import Dict, List
from collections import Counter
from tqdm import tqdm


def read_changes(folder: Path) -> Dict[str, List[int]]:
    out: Dict[str, List[int]] = {}
    for f in folder.glob("solution-problem-*.json"):
        pid = f.stem[17:]
        out[pid] = json.loads(f.read_text())["changes"]
    return out


def vote(bits: List[int]) -> int:
    """Majority vote with ties -> 1"""
    c = Counter(bits)
    return int(c[1] >= c[0])


def main() -> None:
    p = argparse.ArgumentParser("ensemble")
    p.add_argument("--pred-dir", action="append", required=True,
                   help="Pred dir (repeat the flag)")
    p.add_argument("--output-dir", required=True,
                   help="Final dir for solution files")
    args = p.parse_args()

    pred_dirs = [Path(d).expanduser() for d in args.pred_dir]
    out_dir   = Path(args.output_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)

    data: List[Dict[str, List[int]]] = []
    for d in pred_dirs:
        if not d.is_dir():
            raise FileNotFoundError(f"Pred dir {d} not found")
        data.append(read_changes(d))

    common = sorted(set.intersection(*(set(d) for d in data)))
    if not common:
        raise RuntimeError("No common IDs.")

    for pid in tqdm(common, unit="file"):
        lists = [d[pid] for d in data]

        lens = {len(lst) for lst in lists}
        if len(lens) != 1:
            raise ValueError(f"Len mismatch {pid}: {lens}")

        voted = [vote(bits) for bits in zip(*lists)]
        (out_dir / f"solution-problem-{pid}.json").write_text(
            json.dumps({"changes": voted}, indent=2), encoding="utf-8")

    print(f"Done: written to {out_dir} ({len(common)} problems)")


if __name__ == "__main__":
    main()
