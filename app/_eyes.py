"""Closed-eye gate: EAR on MediaPipe FaceLandmarker mesh points.

Why a landmarker and not an eye-state classifier: the corpora that label eyes
open or closed (CEW for real faces, the HF eye set for synthetic faces) are
research-use only, so they cannot be the ground truth of a model that ships in
a sold product. A landmarker has no eye-state label to train on; it returns 478
mesh points, and open vs closed is arithmetic on six of them (the Eye Aspect
Ratio). This module is that arithmetic plus two operating points -- no learned
threshold.

Model: mediapipe face_landmarker (float16, 3,758,596 bytes), Apache-2.0.
"""
from __future__ import annotations

import threading
from pathlib import Path

import numpy as np

MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "face_landmarker.task"

# Pinned like YuNet's: a swapped .task is a load failure, never a silent change
# of what the gate measures between.
MODEL_SHA256 = "64184e229b263107bc2b804c6625db1341ff2bb731874b0bcc2fe6544e0bc9ff"

# Eye contour six-point form, frontal face ("image" mesh, 468 or 478 points):
# corner, upper-lid pair, lower-lid pair, corner. Checked by overlay on a known
# frame, not inferred from a landmark diagram.
LEFT_EYE = (33, 133, 159, 145, 158, 153)
RIGHT_EYE = (362, 263, 386, 374, 385, 380)

# Two operating points, not one: one cutoff is the flip-flop -- at a single
# threshold a blink frame and a JPEG-smeared lid edge alternate the verdict.
# The band between them is "hold", not "reject".
OPEN_FLOOR = float("0.20")
CLOSED_CEIL = float("0.18")

# Below this eye-corner distance the lid is one or two pixels thick and EAR is
# noise. There the gate says "not measurable", it does not assert "closed".
MIN_CORNER_PX = 20.0

_LOCK = threading.Lock()
_LANDMARKER = None
_IMPORT_ERROR = None

def _landmarker():
    """Build once, lazily, under the lock the mesh call itself will hold."""
    global _LANDMARKER
    if _LANDMARKER is not None:
        return _LANDMARKER
    with _LOCK:
        if _LANDMARKER is None:
            try:
                import hashlib
                import mediapipe as mp

                digest = hashlib.sha256(MODEL_PATH.read_bytes()).hexdigest()
                if digest != MODEL_SHA256:
                    raise ValueError(
                        f"{MODEL_PATH.name} sha256 {digest} != pinned {MODEL_SHA256}"
                    )

                options = mp.tasks.vision.FaceLandmarkerOptions(
                    base_options=mp.tasks.BaseOptions(model_asset_path=str(MODEL_PATH)),
                    running_mode=mp.tasks.vision.RunningMode.IMAGE,
                    num_faces=1,
                    min_face_detection_confidence=0.3,
                )
                _LANDMARKER = mp.tasks.vision.FaceLandmarker.create_from_options(options)
            except Exception:
                _LANDMARKER = False
        return _LANDMARKER


def _ear(points: np.ndarray, form: tuple[int, ...]) -> float | None:
    corner_a, corner_b, upper_a, lower_a, upper_b, lower_b = form
    width = float(np.linalg.norm(points[corner_a] - points[corner_b]))
    if width <= 1e-6:
        return None
    upper = float(np.linalg.norm(points[upper_a] - points[lower_a]))
    lower = float(np.linalg.norm(points[upper_b] - points[lower_b]))
    return (upper + lower) / (2.0 * width)


def measure(image_bgr: np.ndarray) -> dict | None:
    """Landmark the face in ``image_bgr`` and return the eye geometry.

    None means the eyes were not measurable (no landmarker, no face, no mesh
    points): the caller must report that, not "closed".
    """
    landmarker = _landmarker()
    if not landmarker:
        # False: the model is absent or its digest is wrong. That is a hold
        # ("could not be measured"), never a claim that the eyes are closed --
        # and not an AttributeError on a bool either.
        return None
    import cv2
    import mediapipe as mp

    rgb = np.ascontiguousarray(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))
    height, width = rgb.shape[:2]
    with _LOCK:
        result = landmarker.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))
    if not result.face_landmarks:
        return None
    face = result.face_landmarks[0]
    points = np.asarray([[p.x * width, p.y * height] for p in face], dtype=np.float32)
    ear_l = _ear(points, LEFT_EYE)
    ear_r = _ear(points, RIGHT_EYE)
    if ear_l is None or ear_r is None:
        return None
    width_l = float(np.linalg.norm(points[LEFT_EYE[0]] - points[LEFT_EYE[1]]))
    width_r = float(np.linalg.norm(points[RIGHT_EYE[0]] - points[RIGHT_EYE[1]]))
    return {
        "ear_l": round(ear_l, 4),
        "ear_r": round(ear_r, 4),
        "corner_px": round(max(width_l, width_r), 1),
    }


def landmark_points(image_bgr: np.ndarray) -> np.ndarray | None:
    """All mesh points, in image pixels. None when the frame has no face."""
    landmarker = _landmarker()
    if not landmarker:
        # False: the model is absent or its digest is wrong. That is a hold
        # ("could not be measured"), never a claim that the eyes are closed --
        # and not an AttributeError on a bool either.
        return None
    import cv2
    import mediapipe as mp

    rgb = np.ascontiguousarray(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))
    height, width = rgb.shape[:2]
    with _LOCK:
        result = landmarker.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))
    if not result.face_landmarks:
        return None
    return np.asarray(
        [[p.x * width, p.y * height] for p in result.face_landmarks[0]], dtype=np.float32
    )


def verdict(measured: dict | None) -> tuple[bool, str, bool]:
    """(eyes_ok, reason, measurable) for one frame."""
    if measured is None:
        return False, "Eyes could not be measured in this frame.", False
    low = min(measured["ear_l"], measured["ear_r"])
    if low < CLOSED_CEIL:
        return False, "Eyes appear closed.", True
    if low < OPEN_FLOOR:
        return False, "Your eyes look half closed or blinking; hold them open.", True
    return True, "", True


def median_ear(windows: list) -> float | None:
    """K-frame median for a polling camera; None when no frame measured."""
    values = [w for w in windows if w is not None]
    if not values:
        return None
    return float(np.median([min(v["ear_l"], v["ear_r"]) for v in values]))
