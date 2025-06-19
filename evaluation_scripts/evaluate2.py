#!/usr/bin/env python3
"""
Evaluator for the StackExchange task.
"""

from __future__ import annotations
import argparse, glob, json, os
from itertools import chain
from pathlib import Path

import numpy as np
from sklearn.metrics import f1_score

EV_OUT = "evaluation.prototext"


def read_solution_files(folder: str) -> dict[str, dict]:
    """Load solution-problem-*.json in folder into a dict."""
    sols = {}
    for fp in glob.glob(os.path.join(folder, "solution-problem-*.json")):
        pid = Path(fp).stem[9:]
        sols[pid] = json.load(open(fp, "r"))
    return sols


def read_truth_files(folder: str) -> dict[str, dict]:
    """Load truth-problem-*.json in folder into a dict."""
    truth = {}
    for fp in glob.glob(os.path.join(folder, "truth-problem-*.json")):
        pid = Path(fp).stem[6:]
        truth[pid] = json.load(open(fp, "r"))
    return truth


def extract_task(truth: dict, sols: dict, key: str) -> tuple[list, list]:
    gold, pred = [], []
    for pid in sorted(truth):
        if pid not in sols:
            raise KeyError(f"missing solution for problem {pid}")
        if len(truth[pid][key]) != len(sols[pid][key]):
            raise ValueError(f"length mismatch in problem {pid}")
        gold.append(truth[pid][key])
        pred.append(sols[pid][key])
    return list(chain.from_iterable(gold)), list(chain.from_iterable(pred))


def write_result(out_file: str, key: str, value: float) -> None:
    line = f'measure{{\n  key: "{key}"\n  value: "{value:.6f}"\n}}\n'
    print(line)
    Path(out_file).write_text(line)


def main() -> None:
    ap = argparse.ArgumentParser("Task 3 evaluator")
    ap.add_argument("-p", "--predictions", required=True,
                    help="dir with solution-problem-*.json")
    ap.add_argument("-t", "--truth", required=True,
                    help="dir with truth-problem-*.json")
    ap.add_argument("-o", "--output", required=True,
                    help="dir for evaluation.prototext")
    args = ap.parse_args()

    sols   = read_solution_files(args.predictions)
    gold   = read_truth_files(args.truth)
    y_true, y_pred = extract_task(gold, sols, "changes")

    f1 = f1_score(y_true, y_pred, average="macro", labels=[0, 1], zero_division=0)

    out_dir = Path(args.output); out_dir.mkdir(parents=True, exist_ok=True)
    write_result(out_dir / EV_OUT, "task3_f1_score", f1)


if __name__ == "__main__":
    main()
