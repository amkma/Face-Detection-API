#!/usr/bin/env python3
"""Numbers behind the eye-gate charts.

Reads eye_rows.jsonl (built by bench/eye_bench.py) and prints, per corpus:
  - how many rows were measurable
  - the min(EAR) percentiles-ranked distribution
  - for CEW (real label only): the confusion matrix at a threshold band
  - for CelebA (no eye label): the pass rate -- a false-reject cost, not accuracy.
"""
from __future__ import annotations

import json
import pathlib
import sys

import numpy as np


def load(path: str):
    return [json.loads(line) for line in pathlib.Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def ears(rows):
    return np.asarray([r["ear"] for r in rows if r.get("ear") is not None], dtype=np.float32)


def pct(v, p):
    return float(np.percentile(v, p))


def main(path: str) -> int:
    rows = load(path)
    for corpus in sorted({r["corpus"] for r in rows}):
        part = [r for r in rows if r["corpus"] == corpus]
        v = ears(part)
        print("==", corpus, "rows", len(part), "measurable", int(v.size),
              "unmeasurable", len(part) - int(v.size))
        if v.size == 0:
            continue
        print("   min", round(float(v.min()), 3), "p01", round(pct(v, 1), 3),
              "p05", round(pct(v, 5), 3), "median", round(float(np.median(v)), 3),
              "p95", round(pct(v, 95), 3), "max", round(float(v.max()), 3))
        for t in (0.15, 0.18, 0.20, 0.25):
            passed = int((v >= t).sum())
            print("   ear>=", t, "share", round(passed / float(v.size), 4))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1]))
