"""Simple clarity checks for SCRFD face detections."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

# A detected face must be large and clear enough for a dependable result.
MIN_FACE_WIDTH = 80
MIN_FACE_HEIGHT = 80
MIN_BLUR_VARIANCE = 50.0
BRIGHTNESS_MIN = 35.0
BRIGHTNESS_MAX = 225.0
MIN_DETECTION_CONFIDENCE = 0.80
EYE_CASCADE_PATH = Path(__file__).resolve().parent / "haarcascade_eye.xml"
FULL_FACE_CASCADE_PATH = Path(__file__).resolve().parent / "haarcascade_frontalface_default.xml"
_EYE_CASCADE: cv2.CascadeClassifier | None = None
_FULL_FACE_CASCADE: cv2.CascadeClassifier | None = None


@dataclass
class QualityResult:
    acceptable: bool
    issues: list[str] = field(default_factory=list)
    blur_variance: float = 0.0
    brightness: float = 0.0
    face_width: int = 0
    face_height: int = 0
    detection_confidence: float = 0.0
    visible_eye_count: int = 0
    open_eye_count: int = 0
    full_face_visible: bool = False
    mouth_visible: bool = False


def _eye_cascade() -> cv2.CascadeClassifier | None:
    """Load OpenCV's eye detector used to confirm both eyes are visible."""
    global _EYE_CASCADE
    if _EYE_CASCADE is None and EYE_CASCADE_PATH.is_file():
        cascade = cv2.CascadeClassifier(str(EYE_CASCADE_PATH))
        if not cascade.empty():
            _EYE_CASCADE = cascade
    return _EYE_CASCADE


def _full_face_cascade() -> cv2.CascadeClassifier | None:
    """Load OpenCV's frontal-face cascade for a second full-feature check."""
    global _FULL_FACE_CASCADE
    if _FULL_FACE_CASCADE is None and FULL_FACE_CASCADE_PATH.is_file():
        cascade = cv2.CascadeClassifier(str(FULL_FACE_CASCADE_PATH))
        if not cascade.empty():
            _FULL_FACE_CASCADE = cascade
    return _FULL_FACE_CASCADE


def _visible_eye_count(face_gray: np.ndarray) -> int:
    """Require an eye detection on each side of the detected face."""
    cascade = _eye_cascade()
    if cascade is None:
        # Fail closed: without the visibility model, SCRFD's face box alone
        # cannot tell whether a hand or object covers a large part of the face.
        return 0

    h, w = face_gray.shape[:2]
    detections = cascade.detectMultiScale(
        face_gray,
        scaleFactor=1.08,
        minNeighbors=4,
        minSize=(max(12, w // 12), max(8, h // 16)),
    )
    centers = [
        (x + box_w / 2, y + box_h / 2)
        for x, y, box_w, box_h in detections
        if 0.22 * h <= y + box_h / 2 <= 0.58 * h
        and 0.12 * w <= x + box_w / 2 <= 0.88 * w
    ]
    left_visible = any(cx < 0.5 * w for cx, _ in centers)
    right_visible = any(cx >= 0.5 * w for cx, _ in centers)
    return int(left_visible) + int(right_visible)


def _eye_looks_open(
    image_bgr: np.ndarray,
    center: tuple[float, float] | None,
    face_width: int,
    face_height: int,
) -> bool:
    """Check for a vertically open eye, rather than merely an eye-shaped region."""
    if center is None:
        return False

    img_h, img_w = image_bgr.shape[:2]
    cx, cy = map(int, center)
    half_w = max(8, int(face_width * 0.12))
    half_h = max(6, int(face_height * 0.05))
    x0, x1 = max(0, cx - half_w), min(img_w, cx + half_w)
    y0, y1 = max(0, cy - half_h), min(img_h, cy + half_h)
    patch = image_bgr[y0:y1, x0:x1]
    if patch.shape[0] < 8 or patch.shape[1] < 12:
        return False

    gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    dark = gray < (float(gray.mean()) - max(20.0, float(gray.mean()) * 0.15))
    row_coverage = dark.mean(axis=1)
    vertical_aperture = float(np.count_nonzero(row_coverage >= 0.20)) / gray.shape[0]

    grad_x = np.abs(cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)).mean()
    grad_y = np.abs(cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)).mean()
    edge_ratio = float(grad_x / (grad_y + 1e-6))

    # A closed eyelid tends to be a narrow horizontal line. Require both a
    # vertically spread iris/eye opening and stronger vertical eye structure.
    return vertical_aperture >= 0.32 and edge_ratio >= 0.45


def _full_face_detected(image_gray: np.ndarray, x: int, y: int, w: int, h: int) -> bool:
    """Confirm a frontal face with its internal features visible around SCRFD's box."""
    cascade = _full_face_cascade()
    if cascade is None:
        return False

    img_h, img_w = image_gray.shape[:2]
    pad_x, pad_y = int(w * 0.18), int(h * 0.18)
    x0, y0 = max(0, x - pad_x), max(0, y - pad_y)
    x1, y1 = min(img_w, x + w + pad_x), min(img_h, y + h + pad_y)
    search = image_gray[y0:y1, x0:x1]
    if search.size == 0:
        return False

    min_size = (max(24, int(w * 0.35)), max(24, int(h * 0.35)))
    return len(
        cascade.detectMultiScale(
            search,
            scaleFactor=1.08,
            minNeighbors=4,
            minSize=min_size,
        )
    ) > 0


def _mouth_region_visible(
    image_bgr: np.ndarray,
    mouth_left: tuple[float, float] | None,
    mouth_right: tuple[float, float] | None,
    face_width: int,
    face_height: int,
) -> bool:
    """Check for mouth-like local detail at SCRFD's two mouth landmarks."""
    if mouth_left is None or mouth_right is None:
        return False

    center_x = int(round((mouth_left[0] + mouth_right[0]) / 2))
    center_y = int(round((mouth_left[1] + mouth_right[1]) / 2))
    half_w = max(5, int(face_width * 0.15))
    half_h = max(4, int(face_height * 0.05))
    img_h, img_w = image_bgr.shape[:2]
    x0, x1 = max(0, center_x - half_w), min(img_w, center_x + half_w)
    y0, y1 = max(0, center_y - half_h), min(img_h, center_y + half_h)
    patch = image_bgr[y0:y1, x0:x1]
    if patch.shape[0] < 6 or patch.shape[1] < 10:
        return False

    gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
    # A visible mouth has local contrast and darker lip/inner-mouth detail.
    # A hand or flat object over the mouth usually lacks both cues.
    mean = float(gray.mean())
    detail_fraction = float((gray < mean - max(20.0, mean * 0.15)).mean())
    return float(gray.std()) >= 18.0 and detail_fraction >= 0.04


def assess_face_quality(
    image_bgr: np.ndarray,
    x: int,
    y: int,
    w: int,
    h: int,
    detection_confidence: float,
    left_eye: tuple[float, float] | None = None,
    right_eye: tuple[float, float] | None = None,
    mouth_left: tuple[float, float] | None = None,
    mouth_right: tuple[float, float] | None = None,
) -> QualityResult:
    """Reject a detected face if it is too small, blurry, dark, or washed out."""
    img_h, img_w = image_bgr.shape[:2]
    x0, y0 = max(0, x), max(0, y)
    x1, y1 = min(img_w, x + w), min(img_h, y + h)
    crop = image_bgr[y0:y1, x0:x1]
    if crop.size == 0:
        return QualityResult(
            acceptable=False,
            issues=["Invalid face region"],
            detection_confidence=detection_confidence,
        )

    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    blur_variance = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    brightness = float(np.mean(gray))
    visible_eye_count = _visible_eye_count(gray)
    open_eye_count = sum(
        (
            _eye_looks_open(image_bgr, left_eye, w, h),
            _eye_looks_open(image_bgr, right_eye, w, h),
        )
    )
    full_image_gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    full_face_visible = _full_face_detected(full_image_gray, x0, y0, w, h)
    mouth_visible = _mouth_region_visible(image_bgr, mouth_left, mouth_right, w, h)
    issues: list[str] = []

    if w < MIN_FACE_WIDTH or h < MIN_FACE_HEIGHT:
        issues.append(
            f"Face is too small ({w}x{h}px; need at least {MIN_FACE_WIDTH}x{MIN_FACE_HEIGHT}px)"
        )
    if blur_variance < MIN_BLUR_VARIANCE:
        issues.append(
            f"Face is too blurry (sharpness {blur_variance:.1f}; need at least {MIN_BLUR_VARIANCE:.0f})"
        )
    if brightness < BRIGHTNESS_MIN:
        issues.append(f"Face is too dark (brightness {brightness:.0f}/255)")
    elif brightness > BRIGHTNESS_MAX:
        issues.append(f"Face is overexposed (brightness {brightness:.0f}/255)")
    if detection_confidence < MIN_DETECTION_CONFIDENCE:
        issues.append(
            f"Face detection confidence is too low ({detection_confidence:.2f}; need at least {MIN_DETECTION_CONFIDENCE:.2f})"
        )
    if visible_eye_count < 2:
        issues.append("Full face is not clearly visible. Make sure both eyes are unobstructed and facing the camera.")
    if open_eye_count < 2:
        issues.append("Both eyes must be open and clearly visible. Please open both eyes and face the camera.")
    if not full_face_visible:
        issues.append("Full facial features could not be confirmed. Remove anything covering the face and face the camera.")
    if not mouth_visible:
        issues.append("Mouth and lower face must be fully visible. Remove your hand or any object covering them.")

    return QualityResult(
        acceptable=not issues,
        issues=issues,
        blur_variance=round(blur_variance, 2),
        brightness=round(brightness, 1),
        face_width=w,
        face_height=h,
        detection_confidence=round(detection_confidence, 3),
        visible_eye_count=visible_eye_count,
        open_eye_count=open_eye_count,
        full_face_visible=full_face_visible,
        mouth_visible=mouth_visible,
    )
