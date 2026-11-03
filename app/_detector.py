"""Face detection and crop geometry - detector only, no recognition.

Pipeline
--------
decode  ->  YuNet (OpenCV Zoo, Apache-2.0)  ->  back to source pixels  ->  signal checks  -> verdict

Scope: find every face in the frame, locate it precisely, and cut the crop its
five landmarks define. Nothing here produces or compares embeddings, so no
recognition weights are loaded and no recognition licence applies.

The detector is YuNet. ``cv2.FaceDetectorYN`` returns, per face, the box as
``(x, y, w, h)`` plus five points in template order
[image-left eye, image-right eye, nose, image-left mouth, image-right mouth]
- exactly the layout ``app/_align.py`` maps onto the ArcFace template, so the
crop geometry is unchanged from the SCRFD build. Detector input is the
``DET_SIZE`` canvas; boxes come back in source pixels via ``_scale_face``.

Weights are not vendored. ``python -m app.fetch_models`` pins the blob by
SHA-256 and files its Apache-2.0 licence next to it; ``_get_detector`` refuses
to run on any other bytes. One model, one licence, one hash: an Apache-2.0
detector is cleared for a commercial product, which the previous buffalo_l
detector was not (InsightFace Model Zoo: "ALL models are available for
non-commercial research purposes only").

The verdict is itemised. ``failed_checks`` names every rule that rejected the
image, so the client renders a checklist instead of parsing a prose string.
"""

from __future__ import annotations

import hashlib
import logging
import threading
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import cv2
import numpy as np

LOG = logging.getLogger(__name__)

# ---- detector -------------------------------------------------------------
#
# Weights are NOT vendored and NOT downloaded silently. ``python -m
# app.fetch_models`` writes the blob plus its Apache-2.0 text into models/;
# ``_get_detector`` hashes it on first use and refuses anything but these bytes,
# so "which model is in production" is a hash, not a hope.

MODEL_FILE = "face_detection_yunet_2023mar.onnx"
MODEL_SHA256 = "8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4"
MODELS_DIR = Path(__file__).resolve().parent.parent / "models"

# YuNet's input shape. The blob itself is fully convolutional - declared input is
# 1x3x640x640 and OpenCV builds whatever size you ask for - so this is a recall
# choice, not the graph's limit. 320 could not see a head drawn at 110 px radius
# at all; 640 sees the same drawing at 516x374 and in the 512 portrait asset.
# FaceDetectorYN just needs setInputSize() to agree with the array it is handed.
DET_SIZE = (640, 640)
DET_SCORE_THRESHOLD = 0.5
DET_NMS_THRESHOLD = 0.3
DET_TOP_K = 5000

_DETECTOR: Any | None = None
_DETECTOR_LOCK = threading.Lock()

# ---- thresholds -----------------------------------------------------------

# A detected face must be large and clear enough for a dependable result.
MIN_FACE_WIDTH = 80
MIN_FACE_HEIGHT = 80
MIN_BLUR_VARIANCE = 50.0
BRIGHTNESS_MIN = 35.0
BRIGHTNESS_MAX = 225.0
# A detection below this is a weak localisation and its box may be several
# pixels off in either direction.
MIN_DETECTION_CONFIDENCE = 0.80

# Decoder bounds, enforced on the header before pixels are materialised.
MAX_IMAGE_BYTES = 10 * 1024 * 1024
MAX_SIDE_PX = 16000
MAX_PIXELS = 60_000_000
PNG_MAX_IDAT = MAX_PIXELS * 4

# Face-validity band: the box must not straddle the frame edges, and the
# landmarks must sit where a real face puts them inside the crop.
EDGE_TOLERANCE = 0.02
LANDMARK_BAND = (0.10, 0.90)
MIN_INTER_EYE_RATIO = 0.11
MAX_INTER_EYE_RATIO = 0.90

# Head pose. A passport-style check asks for "face the camera", which for
# roll means both eyes roughly level. Yaw/pitch are reported, never enforced,
# because a five-point crop cannot estimate them without a 3D landmark model.
MAX_ROLL_DEG = 15.0
WARN_YAW_DEG = 25.0
WARN_PITCH_DEG = 20.0

# Human-readable names for the itemised verdict.
CHECK_LABELS = {
    "one_face": "Face count",
    "face_size": "Face size",
    "detection_confidence": "Detection confidence",
    "landmarks_inside": "Landmarks inside the face",
    "landmark_spacing": "Inter-eye spacing",
    "blur": "Sharpness",
    "exposure": "Lighting",
    "head_pose": "Head pose",
}


class FaceAnalysisError(Exception):
    def __init__(self, message: str, status_code: int = 400, code: str = "error"):
        super().__init__(message)
        self.status_code = status_code
        self.code = code


def _get_detector() -> Any:
    """Build the YuNet detector once per process, at its pinned hash.

    Double-checked locking: two requests arriving together must not both build
    and hash the graph.
    """
    global _DETECTOR
    if _DETECTOR is not None:
        return _DETECTOR
    with _DETECTOR_LOCK:
        if _DETECTOR is None:
            path = MODELS_DIR / MODEL_FILE
            if not path.is_file():
                raise FaceAnalysisError(
                    f"{MODEL_FILE} missing; run `python -m app.fetch_models`",
                    500,
                    "model_missing",
                )
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            if digest != MODEL_SHA256:
                raise FaceAnalysisError(
                    "Face detector hash mismatch; re-run app.fetch_models",
                    500,
                    "model_integrity_failed",
                )
            _DETECTOR = cv2.FaceDetectorYN.create(
                str(path),
                "",
                DET_SIZE,
                DET_SCORE_THRESHOLD,
                DET_NMS_THRESHOLD,
                DET_TOP_K,
            )
            LOG.info("YuNet loaded: %s sha256=%s", path, digest)
    return _DETECTOR


def _detector_input(image_bgr: np.ndarray) -> tuple[np.ndarray, float, int, int]:
    """Letterbox ``image_bgr`` into ``DET_SIZE``, centred; return it, its scale, and pads.

    ``FaceDetectorYN`` was exported at one fixed size and does not letterbox.
    Squashing a portrait frame to square would distort faces, so the frame keeps
    its aspect and spare pixels are padded with black. The resized frame is
    centred in the canvas and the pads come back with it: a 1:1 frame fills the
    canvas exactly (pad 0, the old behaviour), while a portrait or landscape
    frame is offset by an amount the caller subtracts back before use.
    """
    height, width = image_bgr.shape[:2]
    det_w, det_h = DET_SIZE
    scale = min(det_w / width, det_h / height)
    new_w = max(1, int(round(width * scale)))
    new_h = max(1, int(round(height * scale)))
    # Centre the resized frame; pad 0 whenever it fills the canvas 1:1.
    off_x = (det_w - new_w) // 2
    off_y = (det_h - new_h) // 2
    resized = cv2.resize(image_bgr, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    canvas = np.zeros((det_h, det_w, 3), dtype=np.uint8)
    canvas[off_y : off_y + new_h, off_x : off_x + new_w] = resized
    return canvas, scale, off_x, off_y


# Landmark order: eyes and mouth corners measured, middle point not.
#
# Measured on tests/assets/portrait.png (real photo), twice, at two detector
# sizes, as fractions of that image box:
#   p0 (0.46, 0.41)  p1 (0.87, 0.45)   two highest points, level, either side of
#                                     centre -> the eyes
#   p3 (0.44, 0.77)  p4 (0.76, 0.80)   two lowest points -> the mouth corners
#   p2 (0.75, 0.65)                   within the eye/mouth band -> the
#                                     middle point (nose tip or mouth centre)
#
# What the pixels support: p0/p1 are the eyes as a pair, p3/p4 are the mouth
# corners as a pair, p2 is the middle point. test_landmark_geometry_is_face_shaped
# pins exactly that, and nothing more.
#
# NOT settled: whether that middle point is the nose tip or the mouth centre.
# Both orderings put p2 in the middle slot; they differ only in the order of the
# two after it. A frontal pose cannot tell them apart, which is why the earlier
# argument ran in circles. The permutation hand is therefore unverified - and
# harmless, because _order_as_arcface is not called anywhere in the app.
#
# Settles it: one real photo with a different head yaw and clean mouth pixels.
# The nose sweeps sideways with yaw; a mouth centre does not.
_YUNET_TO_ARCFACE = np.array([0, 1, 3, 4, 2])


def _order_as_arcface(points: np.ndarray) -> np.ndarray:
    """UNVERIFIED - see the note above. Not called by anything in the app."""
    return points[_YUNET_TO_ARCFACE]



def _scale_face(face: np.ndarray, scale: float, off_x: int = 0, off_y: int = 0) -> Any:
    """One YuNet row (``x, y, w, h`` + 5 points + score) in source-image pixels.

    Exposure is ``(score, bbox_xyxy, kps)`` - the same shape the old
    ``FaceAnalysis`` result gave - so ``_face_payload`` and ``_align`` never
    learn which detector produced a face. YuNet hands back ``w, h``; ``bbox``
    wants ``x2, y2``, hence the two adds.

    ``off_x``/``off_y`` are the letterbox pads. The row is measured on the
    padded canvas, the caller wants it on the source image, so the pad comes
    off the position and not off the size. Both are 0 on a 1:1 frame.
    """
    box_x = (float(face[0]) - off_x) / scale
    box_y = (float(face[1]) - off_y) / scale
    box_w = float(face[2]) / scale
    box_h = float(face[3]) / scale
    points = np.asarray(face[4:14], dtype=np.float32).reshape(5, 2) / scale
    points[:, 0] -= off_x / scale
    points[:, 1] -= off_y / scale
    score = float(face[14])
    return SimpleNamespace(
        bbox=np.array([box_x, box_y, box_x + box_w, box_y + box_h], dtype=np.float64),
        kps=points,
        det_score=score,
    )


def _detect(image_bgr: np.ndarray) -> list[Any]:
    """Every face YuNet finds in ``image_bgr``, located in source-image pixels."""
    detector = _get_detector()
    det_input, input_scale, off_x, off_y = _detector_input(image_bgr)

    # The graph has one fixed input size. A bare ``detect()`` raises
    # "Size does not match" (verified), and so does a bare ``setInputSize``:
    # the input size is what ``setInputSize`` says *and* the call carries the
    # frame to match it. Both together, in that order.
    detector.setInputSize(DET_SIZE)
    _, rows = detector.detect(np.ascontiguousarray(det_input))
    if rows is None:
        return []
    return [_scale_face(row, input_scale, off_x, off_y) for row in rows]

