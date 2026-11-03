"""Face detection and clarity checks.

Pipeline
--------
decode  ->  detector (SCRFD)  ->  full-resolution rescale  ->  signal checks  -> verdict

Everything is measured on the photo the user actually uploaded. The detector is
fed a letterboxed copy that fits ``DET_SIZE``, and its boxes and landmarks come
back in that copy's coordinates, so they are divided by the same scale factor
before anything else looks at them. Measuring sharpness on the detector's copy
would score the downsampler, not the photo: Laplacian variance falls ~4x per
2x of downscaling, so a crisp 4K upload would read as "blurry" while a soft
640px one passed.

The verdict is itemised. ``failed_checks`` names every rule that rejected the
image, so the client renders a checklist instead of parsing a prose string.

Signal checks report measurable problems (size, blur, exposure, pose,
one-face-per-frame). They do not guess *what* is covering a face; that is an
attribute question and belongs to the models that answer attribute questions.
"""

from __future__ import annotations

import hashlib
import io
import logging
import math
import threading
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any

import cv2
import numpy as np
from PIL import Image, ImageOps
from app._detector import (  # noqa: F401  re-exported: main.py imports FaceAnalysisError from here
    CHECK_LABELS,
    DET_SIZE,
    MODEL_FILE,
    MODEL_SHA256,
    FaceAnalysisError,
    _detect,
    _get_detector,
)

LOG = logging.getLogger(__name__)

# ---- thresholds -----------------------------------------------------------

# A detected face must be large and clear enough for a dependable result.
MIN_FACE_WIDTH = 80
MIN_FACE_HEIGHT = 80
# Sharpness is measured on the crop resampled to this width, so the cutoff
# means the same thing for a 100 px webcam face and a 2,000 px phone photo.
BLUR_REFERENCE_WIDTH = 112
MIN_BLUR_VARIANCE = 50.0
BRIGHTNESS_MIN = 35.0
BRIGHTNESS_MAX = 225.0
# buffalo_l's detector runs at 0.5; a detection below this is a weak
# localisation and its box may be several pixels off in either direction.
MIN_DETECTION_CONFIDENCE = 0.80

# Detector graph resolution. Must match the shipped det_10g.onnx.
DET_SIZE = (640, 640)

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

@dataclass
class Check:
    name: str
    passed: bool
    value: float | int | str | None = None
    detail: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "label": CHECK_LABELS.get(self.name, self.name),
            "passed": self.passed,
            "value": self.value,
            "detail": self.detail,
        }


@dataclass
class Posteriors:
    """Per-face evidence, so the caller never has to re-run the model."""
    checks: list[Check] = field(default_factory=list)
    bbox_xyxy: tuple[int, int, int, int] = (0, 0, 0, 0)
    landmarks: list[list[float]] = field(default_factory=list)
    detection_confidence: float = 0.0
    pose: dict[str, float] = field(default_factory=dict)
    blur_variance: float = 0.0
    brightness: float = 0.0
    aligned_112: np.ndarray | None = None
    embedding: np.ndarray | None = None

    @property
    def acceptable(self) -> bool:
        return all(check.passed for check in self.checks)

    @property
    def failed(self) -> list[Check]:
        return [check for check in self.checks if not check.passed]


def image_size(file_bytes: bytes) -> dict[str, int] | None:
    """Width and height of the encoded image, or None if it cannot be read.

    Header-only: a rejected request can still report the dimensions it looked
    at without paying for a second full pixel decode.
    """
    try:
        with Image.open(io.BytesIO(file_bytes)) as probe:
            width, height = probe.size
    except Exception:
        return None
    return {"width": width, "height": height}


def _decode(file_bytes: bytes) -> np.ndarray:
    if not file_bytes:
        raise FaceAnalysisError("Empty upload", 400, "empty_upload")
    if len(file_bytes) > MAX_IMAGE_BYTES:
        raise FaceAnalysisError("Image too large (max 10 MB)", 413, "too_large")

    # Header pass: reject oversized rasters before any pixel row is decoded.
    try:
        probe = Image.open(io.BytesIO(file_bytes))
        width, height = probe.size
        probe.verify()
    except Exception as exc:
        raise FaceAnalysisError("Invalid image file", 415, "decode_failed") from exc

    if width > MAX_SIDE_PX or height > MAX_SIDE_PX or width * height > MAX_PIXELS:
        raise FaceAnalysisError(
            f"Image dimensions are too large ({width}x{height}); "
            f"the limit is {MAX_SIDE_PX} px per side and {MAX_PIXELS} total pixels",
            413,
            "dimensions_exceeded",
        )

    # Sparse PNGs (an IDAT that expands to far more than the header claims) are
    # the classic decompression bomb. Count IDAT bytes without inflating them.
    if probe.format == "PNG" and _png_idat_bytes(file_bytes) > PNG_MAX_IDAT:
        raise FaceAnalysisError("Image is too large to decode safely", 413, "decompression_bomb")

    try:
        pil = Image.open(io.BytesIO(file_bytes))
        # Phone cameras store the sensor orientation in EXIF and leave the
        # pixels landscape. Without this the detector receives a sideways face
        # and returns nothing, which reads downstream as "no face in photo".
        pil = ImageOps.exif_transpose(pil)
        pil = pil.convert("RGB")
        rgb = np.asarray(pil, dtype=np.uint8)
    except FaceAnalysisError:
        raise
    except Exception as exc:
        raise FaceAnalysisError("Invalid image file", 415, "decode_failed") from exc

    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def _png_idat_bytes(data: bytes) -> int:
    """Sum chunk payload sizes of the IDAT chunks without inflating them."""
    total = 0
    pos = 8  # after the PNG signature
    while pos + 8 <= len(data):
        length = int.from_bytes(data[pos : pos + 4], "big")
        start = pos + 8
        if start + length > len(data):
            break
        if data[pos + 4 : pos + 8] == b"IDAT":
            total += length
        pos = start + length + 4
        if data[pos - 4 : pos] == b"IEND":
            break
    return total


def _sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _roll_deg(landmarks: np.ndarray) -> float:
    (lx, ly), (rx, ry) = landmarks[0], landmarks[1]
    return float(math.degrees(math.atan2(ry - ly, rx - lx)))


def _roll_deg(landmarks: np.ndarray) -> float:
    (lx, ly), (rx, ry) = landmarks[0], landmarks[1]
    return float(math.degrees(math.atan2(ry - ly, rx - lx)))


def _alignment_checks(
    landmarks: np.ndarray, face_x: int, face_y: int, face_w: int, face_h: int
) -> list[Check]:
    """Geometry that must hold on a five-point detection before it is usable.

    The five points are socket centres, not eye contours, so this can verify
    where they sit and how far apart they are. It cannot tell open from closed.
    """
    inter_eye = float(np.linalg.norm(landmarks[0] - landmarks[1]))
    ratio = inter_eye / float(face_w)

    band_lo, band_hi = LANDMARK_BAND
    inside = all(
        face_x + band_lo * face_w <= landmark[0] < face_x + band_hi * face_w
        and face_y + band_lo * face_h <= landmark[1] < face_y + band_hi * face_h
        for landmark in landmarks
    )

    return [
        Check(
            "landmarks_inside",
            inside,
            round(float(inside), 3),
            ""
            if inside
            else "One or more landmarks fall outside the detected face; the box is "
            "not enclosing the whole face (tight crop, headband, heavy headwear).",
        ),
        Check(
            "landmark_spacing",
            MIN_INTER_EYE_RATIO <= ratio <= MAX_INTER_EYE_RATIO,
            round(ratio, 3),
            ""
            if MIN_INTER_EYE_RATIO <= ratio <= MAX_INTER_EYE_RATIO
            else f"Inter-eye distance is {ratio:.2f} of the face width "
            f"(expected {MIN_INTER_EYE_RATIO:.2f}-{MAX_INTER_EYE_RATIO:.2f}).",
        ),
    ]


def analyze_image(file_bytes: bytes) -> dict[str, Any]:
    image_bgr = _decode(file_bytes)
    img_h, img_w = image_bgr.shape[:2]

    try:
        detected_faces = _detect(image_bgr)
    except FaceAnalysisError:
        raise
    except Exception as exc:
        raise FaceAnalysisError("Face analysis failed", 500, "analysis_failed") from exc

    if not detected_faces:
        return {
            "face_detected": False,
            "face_count": 0,
            "image_size": {"width": img_w, "height": img_h},
            "quality": None,
            "acceptable_for_enrollment": False,
            "failed_checks": ["one_face"],
            "checks": [
                Check(
                    "one_face",
                    False,
                    0,
                    "No face detected. Use a clear front-facing photo with good lighting.",
                ).as_dict()
            ],
            "message": "No face detected. Use a clear front-facing photo with good lighting.",
        }

    per_face = [_face_payload(face, image_bgr) for face in detected_faces]

    payload: dict[str, Any] = {
        "face_detected": True,
        "face_count": len(detected_faces),
        "image_size": {"width": img_w, "height": img_h},
        "faces": per_face,
    }

    # Every face is scored. When more than one is in frame the client still gets
    # a checklist per face (useful on a security camera with bystanders), but the
    # enrollment target is one: the largest face, not "whichever came first".
    # First-in-list would make the verdict depend on detector tie-breaking order.
    target = max(per_face, key=lambda item: item["facial_area"]["width"] * item["facial_area"]["height"])
    count_ok = len(detected_faces) == 1

    payload.update(
        {
            "facial_area": target["facial_area"] if count_ok else None,
            "pose": target["pose"] if count_ok else None,
            "landmarks": target["landmarks"] if count_ok else None,
            "detection_confidence": target["detection_confidence"] if count_ok else None,
            "quality": target["quality"],
            "acceptable_for_enrollment": target["acceptable_for_enrollment"] and count_ok,
            "failed_checks": target["failed_checks"] if count_ok else ["one_face"],
            "checks": target["checks"] if count_ok else [
                Check(
                    "one_face",
                    False,
                    len(detected_faces),
                    f"{len(detected_faces)} faces detected; keep one face in frame",
                ).as_dict()
            ],
            "message": target["message"]
            if count_ok
            else f"{len(detected_faces)} faces detected — exactly one face is required.",
        }
    )

    return payload


def _face_payload(face: Any, image_bgr: np.ndarray) -> dict[str, Any]:
    img_h, img_w = image_bgr.shape[:2]
    values = [int(round(float(value))) for value in face.bbox]
    x1, y1, x2, y2 = values
    x, y = max(0, x1), max(0, y1)
    w, h = min(img_w, x2) - x, min(img_h, y2) - y
    confidence = float(face.det_score)
    landmarks = np.asarray(face.kps, dtype=np.float32)

    size_ok = w >= MIN_FACE_WIDTH and h >= MIN_FACE_HEIGHT
    conf_ok = confidence >= MIN_DETECTION_CONFIDENCE
    checks: list[Check] = [
        Check("one_face", True, 1, ""),
        Check(
            "face_size",
            size_ok,
            f"{w}x{h}",
            ""
            if size_ok
            else f"Face is too small ({w}x{h}px; need at least "
            f"{MIN_FACE_WIDTH}x{MIN_FACE_HEIGHT}px). Move closer to the camera.",
        ),
        Check(
            "detection_confidence",
            conf_ok,
            round(confidence, 3),
            ""
            if conf_ok
            else f"Face detection confidence is too low ({confidence:.2f}; "
            f"need at least {MIN_DETECTION_CONFIDENCE:.2f}).",
        ),
    ]
    checks.extend(_alignment_checks(landmarks, x, y, w, h))

    crop = image_bgr[y : y + h, x : x + w]
    blur_variance = 0.0
    brightness = 0.0
    if crop.size:
        brightness = float(np.mean(cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)))
        # Laplacian variance scales with pixel count: this same face scores
        # 506 at 512 px and 12 at 1600 px. A raw cutoff therefore judges the
        # camera resolution, not focus. Resample to BLUR_REFERENCE_WIDTH first.
        if w > BLUR_REFERENCE_WIDTH:
            height_ref = max(1, int(round(h * BLUR_REFERENCE_WIDTH / float(w))))
            measured = cv2.resize(
                crop, (BLUR_REFERENCE_WIDTH, height_ref), interpolation=cv2.INTER_AREA
            )
        elif w < BLUR_REFERENCE_WIDTH:
            height_ref = max(1, int(round(h * BLUR_REFERENCE_WIDTH / float(w))))
            measured = cv2.resize(
                crop, (BLUR_REFERENCE_WIDTH, height_ref), interpolation=cv2.INTER_CUBIC
            )
        else:
            measured = crop
        gray = cv2.cvtColor(measured, cv2.COLOR_BGR2GRAY)
        blur_variance = float(cv2.Laplacian(gray, cv2.CV_64F).var())

    sharp_ok = blur_variance >= MIN_BLUR_VARIANCE
    checks.append(
        Check(
            "blur",
            sharp_ok,
            round(blur_variance, 2),
            ""
            if sharp_ok
            else f"Face is too blurry (sharpness {blur_variance:.1f}; need at least "
            f"{MIN_BLUR_VARIANCE:.0f}). Hold still.",
        )
    )

    lit_ok = BRIGHTNESS_MIN <= brightness <= BRIGHTNESS_MAX
    if lit_ok:
        exposure_detail = ""
    elif brightness < BRIGHTNESS_MIN:
        exposure_detail = f"Face is too dark (brightness {brightness:.0f}/255). Face a light."
    else:
        exposure_detail = (
            f"Face is overexposed (brightness {brightness:.0f}/255). "
            "Step out of direct light."
        )
    checks.append(Check("exposure", lit_ok, round(brightness, 1), exposure_detail))

    roll = _roll_deg(landmarks)
    roll_ok = abs(roll) <= MAX_ROLL_DEG
    checks.append(
        Check(
            "head_pose",
            roll_ok,
            round(roll, 1),
            "" if roll_ok else f"Head is tilted {roll:.0f}deg; keep it straight.",
        )
    )

    failed = [check.name for check in checks if not check.passed]
    quality_issues = [check.detail for check in checks if not check.passed and check.detail]

    return {
        "facial_area": {"x": x, "y": y, "width": w, "height": h},
        "detection_confidence": round(confidence, 3),
        "landmarks": [
            [round(float(px), 2), round(float(py), 2)] for px, py in landmarks.tolist()
        ],
        "pose": {"roll": round(roll, 1)},
        "quality": {
            "acceptable": not failed,
            "issues": quality_issues,
            "blur_variance": round(blur_variance, 2),
            "brightness": round(brightness, 1),
            "face_size_px": {"width": w, "height": h},
        },
        "failed_checks": failed,
        "checks": [check.as_dict() for check in checks],
        "acceptable_for_enrollment": not failed,
        "message": "One clear face detected."
        if not failed
        else "; ".join(quality_issues),
    }
