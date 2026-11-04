#!/usr/bin/env python3
"""Build the eye-gate evidence set:one row per image, for the charts.

Two corpora, two questions, never one number:

  CEW subset (labelled closed_eyes in the HF eye dataset) - real faces, REAL
    eye-state label. Detection: does the gate reject a closed eye, and does it
    pass an open one.
  CelebA shard 0 - real faces, no eye-state label. Behaviour: pass rate (the
    false-reject cost), not accuracy.

The HF "closed-open-eyes" dataset ships CEW faces as closed_eyes and real,
unlabelled faces as open_eyes. Its open_eyes rows are NOT ground truth, so a
combined accuracy over it would be a fabricated. Rows carry
their corpus so a consumer can tell label from proxy before counting.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import pathlib
import sys

import cv2
import numpy as np
import pyarrow.parquet as pq

sys.path.insert(0, "/home/ubuntu/output/fda")

from app import _eyes

CACHE = pathlib.Path("/home/ubuntu/.hermes/cache/scratch/celeba")

MODEL_SHA256 = "64184e229b263107bc2b804c6625db1341ff2bb731874b0bcc2fe6544e0bc9ff"

CLOSED_SOURCES = ["eyes_001", "eyes_131", "p_064", "p_070"]
CELEBA_SOURCES = ["train-00000-of-00132", "train-00001-of-00132"]


def sha256_of(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def gate(image_bgr):
    """The production rule for one image."""
    measured = _eyes.measure(image_bgr)
    ok, reason, measurable = _eyes.verdict(measured)
    if measured is None:
        return {"eyes_ok": False, "measurable": False,            "reason": reason,
                "ear_l": None, "ear_r": None, "corner_px": None}
    return {
        "eyes_ok": bool(ok),
        "measurable": bool(measurable),
        "reason": reason,
        "ear_l": measured["ear_l"],
        "ear_r": measured["ear_r"],
        "corner_px": measured["corner_px"],
    }


def rows_from(path: pathlib.Path, corpus: str, source: str):
    table = pq.read_table(path)
    field = "Image_data" if "Image_data" in table.column_names else "image"
    images = table.column(field).to_pylist()
    labels = table.column("Label").to_pylist() if "Label" in table.column_names else None
    out = []
    for idx, item in enumerate(images):
        field_name = "file" if "file" in item else "bytes"
        raw = item.get(field_name)
        if raw is None:
            continue
        image = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            continue
        decision = gate(image)
        row = {
            "corpus": corpus,
            "source": source,
            "row": idx,
            "label": None if labels is None else labels[idx],
            "sha256": hashlib.sha256(raw).hexdigest()[:16],
            "height": int(image.shape[0]),
            "width": int(image.shape[1]),
            "bytes": len(raw),
        }
        row.update(decision)
        row["ear"] = None
        if decision["ear_l"] is not None and decision["ear_r"] is not None:
            row["ear"] = float(min(decision["ear_l"], decision["ear_r"]))
        out.append(row)
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=str(CACHE / "eye_rows.jsonl"))
    args = parser.parse_args()

    print("model_sha256", sha256_of(pathlib.Path(_eyes.MODEL_PATH)), flush=True)

    everything = []
    for name in CLOSED_SOURCES:
        path = CACHE / f"{name}.parquet"
        if not path.is_file():
            print("skip", name)
            continue
        import time
        t0 = time.time()
        batch = rows_from(path, "cew_subset", name)
        everything.extend(batch)
        print("cew", name, len(batch), "rows", round(time.time() - t0, 1), "s", flush=True)
    for name in CELEBA_SOURCES:
        path = CACHE / f"{name}.parquet"
        if not path.is_file():
            print("skip", name)
            continue
        import time
        t0 = time.time()
        batch = rows_from(path, "celeba+unlabelled", name)
        everything.extend(batch)
        print("celeba", name, len(batch), "rows", round(time.time() - t0, 1), "s", flush=True)

    pathlib.Path(args.out).write_text(
        "".join(json.dumps(row) + "\n" for row in everything), encoding="utf-8"
    )
    print("wrote", args.out, len(everything), "rows")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
