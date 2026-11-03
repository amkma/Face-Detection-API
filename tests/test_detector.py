"""The detector contract: YuNet finds, locates and crops. No recognition weights.

    ~/venvs/face/bin/python -m pytest tests/test_detector.py -q

Each test pins a claim a detection-only build could silently break. The
letterbox/back-map pair matters most: replace the letterbox with a straight
resize and every box on a portrait frame comes out wrong by ~12% while the other
assertions still pass, because nothing else measures where the box landed.
"""

import io
import sys

import cv2
import numpy as np
import pytest
from PIL import Image, ImageOps

sys.path.insert(0, ".")

from pathlib import Path

PORTRAIT = Path(__file__).resolve().parent / "assets" / "portrait.png"
from app import _detector as det
from app.face_service import analyze_image


def png(array: np.ndarray) -> bytes:
    return cv2.imencode(".png", array)[1].tobytes()


def head(width: int = 640, height: int = 464) -> np.ndarray:
    """A face-like head: skin oval, two eyes, a nose, a mouth."""
    canvas = np.full((height, width, 3), 70, np.uint8)
    cx, cy = width // 2, height // 2
    cv2.ellipse(canvas, (cx, cy), (110, 140), 0, 0, 360, (170, 150, 130), -1)
    cv2.ellipse(canvas, (cx - 46, cy - 42), (20, 12), 0, 0, 360, (248, 248, 248), -1)
    cv2.ellipse(canvas, (cx + 46, cy - 42), (20, 12), 0, 0, 360, (248, 248, 248), -1)
    cv2.circle(canvas, (cx - 46, cy - 42), 7, (30, 30, 30), -1)
    cv2.circle(canvas, (cx + 46, cy - 42), 7, (30, 30, 30), -1)
    cv2.ellipse(canvas, (cx, cy + 64), (30, 12), 0, 0, 180, (85, 55, 65), 6)
    return canvas


def test_pinned_blob_is_the_one_on_disk():
    """A swapped weight must fail loudly, not quietly change results."""
    path = det.MODELS_DIR / det.MODEL_FILE
    assert path.is_file(), "run: python -m app.fetch_models"
    import hashlib

    assert hashlib.sha256(path.read_bytes()).hexdigest() == det.MODEL_SHA256


def test_finds_one_face_with_five_points():
    found = det._detect(head())
    assert len(found) == 1
    assert np.asarray(found[0].kps).shape == (5, 2)


def test_landmarks_land_inside_the_box():
    """YuNet's points are its own convention; the contract is that they sit on the face."""
    found = det._detect(head())
    assert found
    x1, y1, x2, y2 = found[0].bbox
    for px, py in np.asarray(found[0].kps):
        assert x1 <= px <= x2 and y1 <= py <= y2


def test_box_is_in_source_pixels_not_detector_pixels():
    """A 640 wide frame letterboxed to 320 must come back near 1x, not at half size."""
    found = det._detect(head(640, 464))
    assert found, "drawn head not detected; the back-map is the suspect"
    x1, y1, x2, y2 = found[0].bbox
    assert 120 < (x2 - x1) < 420
    assert 60 < (y2 - y1) < 340


def test_box_centres_on_the_drawn_head():
    """Catches a missing pad offset in the back-map: the head is drawn dead centre."""
    found = det._detect(head(640, 464))
    assert found
    x1, y1, x2, y2 = found[0].bbox
    assert abs((x1 + x2) / 2 - 320) < 40
    assert abs((y1 + y2) / 2 - 232) < 40


def test_blank_frame_yields_nothing():
    """A detector that invents a face on flat noise is not a gate."""
    assert det._detect(np.full((300, 300, 3), 40, np.uint8)) == []


def test_analyze_image_reports_crop_and_withholds_the_verdict():
    out = analyze_image(png(head()))
    assert out["face_detected"] is True
    assert out["face_count"] == 1
    assert out["facial_area"]["width"] > 100
    assert len(out["landmarks"]) == 5
    assert out["acceptable_for_enrollment"] is False
    assert out["failed_checks"]
    assert "quality" in out


def test_only_the_detector_weight_is_on_disk():
    """Detection and cropping: no recognition graph, no attribute graph."""
    onnx = sorted(path.name for path in det.MODELS_DIR.glob("*.onnx"))
    assert onnx == [det.MODEL_FILE]


def test_landmark_geometry_is_face_shaped():
    """Pin what real pixels support: eyes pair on top, mouth pair lowest, middle between.

    Uses the real portrait, not the drawn head: on a cartoon the five points are
    regressed onto features that were never there.
    """
    frame = cv2.imread(str(PORTRAIT))

    found = det._detect(frame)
    assert found, "no face in the real portrait asset"
    points = np.asarray(found[0].kps, dtype=np.float64)
    x1, y1, x2, y2 = found[0].bbox
    xs = (points[:, 0] - x1) / (x2 - x1)
    ys = (points[:, 1] - y1) / (y2 - y1)

    highest = set(np.argsort(ys)[:2])
    lowest = set(np.argsort(ys)[-2:])
    assert highest == {0, 1}, f"eyes should be the top pair, got {highest}"
    assert abs(ys[0] - ys[1]) < 0.10, "eye pair should be level"
    assert lowest == {3, 4}, f"mouth corners should be the bottom pair, got {lowest}"
    assert xs[0] < xs[1], "image-left eye must be left of image-right eye"
    assert min(xs[3], xs[4]) < xs[2] < max(xs[3], xs[4]) or xs[2] > max(xs[3], xs[4]) - 0.05
    assert ys[0] < ys[2] < min(ys[3], ys[4]), "p2 must sit between eyes and mouth corners"
