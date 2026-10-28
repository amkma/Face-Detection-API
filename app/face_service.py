"""Face detection and clarity checks."""

from __future__ import annotations

import io
from typing import Any

import cv2
import numpy as np
from PIL import Image
from insightface.app import FaceAnalysis

from app.quality import assess_face_quality


_face_analyzer: FaceAnalysis | None = None


def _get_face_analyzer() -> FaceAnalysis:
    """Load InsightFace's SCRFD-10G detector once per process."""
    global _face_analyzer
    if _face_analyzer is None:
        # The buffalo_l model pack uses SCRFD-10GF for face detection. InsightFace
        # downloads the pack on first use into its standard model cache.
        _face_analyzer = FaceAnalysis(
            name="buffalo_l",
            allowed_modules=["detection"],
            providers=["CPUExecutionProvider"],
        )
        # SCRFD scores can vary substantially with input scale. Run both a
        # smaller and standard input and let SCRFD's NMS keep the strongest box.
        _face_analyzer.prepare(ctx_id=-1, det_size=[(320, 320), (640, 640)])
    return _face_analyzer


class FaceAnalysisError(Exception):
    def __init__(self, message: str, status_code: int = 400):
        super().__init__(message)
        self.status_code = status_code


def _read_image(file_bytes: bytes) -> np.ndarray:
    if not file_bytes:
        raise FaceAnalysisError("Empty upload", 400)
    if len(file_bytes) > 10 * 1024 * 1024:
        raise FaceAnalysisError("Image too large (max 10 MB)", 413)

    try:
        pil = Image.open(io.BytesIO(file_bytes))
        pil = pil.convert("RGB")
    except Exception as exc:
        raise FaceAnalysisError("Invalid image file") from exc

    rgb = np.array(pil)
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def analyze_image(file_bytes: bytes) -> dict[str, Any]:
    image_bgr = _read_image(file_bytes)
    img_h, img_w = image_bgr.shape[:2]

    try:
        detected_faces = _get_face_analyzer().get(image_bgr)
    except Exception as exc:
        raise FaceAnalysisError(f"Face analysis failed: {exc}", 500) from exc

    if not detected_faces:
        return {
            "face_detected": False,
            "face_count": 0,
            "image_size": {"width": img_w, "height": img_h},
            "quality": None,
            "acceptable_for_enrollment": False,
            "message": "No face detected. Use a clear front-facing photo with good lighting.",
        }

    if len(detected_faces) > 1:
        return {
            "face_detected": True,
            "face_count": len(detected_faces),
            "image_size": {"width": img_w, "height": img_h},
            "quality": {"acceptable": False, "issues": ["Multiple faces detected (only 1 face allowed)"]},
            "acceptable_for_enrollment": False,
            "message": "Exactly one face is required.",
        }

    face = detected_faces[0]
    x1, y1, x2, y2 = (int(round(float(value))) for value in face.bbox)
    x, y = max(0, x1), max(0, y1)
    w, h = min(img_w, x2) - x, min(img_h, y2) - y
    confidence = float(face.det_score)
    landmarks = face.kps
    left_eye = (float(landmarks[0][0]), float(landmarks[0][1])) if landmarks is not None else None
    right_eye = (float(landmarks[1][0]), float(landmarks[1][1])) if landmarks is not None else None
    mouth_left = (float(landmarks[3][0]), float(landmarks[3][1])) if landmarks is not None else None
    mouth_right = (float(landmarks[4][0]), float(landmarks[4][1])) if landmarks is not None else None
    quality = assess_face_quality(
        image_bgr,
        x,
        y,
        w,
        h,
        confidence,
        left_eye=left_eye,
        right_eye=right_eye,
        mouth_left=mouth_left,
        mouth_right=mouth_right,
    )
    enrollment_ok = quality.acceptable
    message = (
        "One clear face detected."
        if enrollment_ok
        else "A face was detected, but it is not clear enough. " + "; ".join(quality.issues)
    )

    return {
        "face_detected": True,
        "face_count": 1,
        "image_size": {"width": img_w, "height": img_h},
        "facial_area": {"x": x, "y": y, "width": w, "height": h},
        "quality": {
            "acceptable": quality.acceptable,
            "issues": quality.issues,
            "blur_variance": quality.blur_variance,
            "brightness": quality.brightness,
            "face_size_px": {"width": quality.face_width, "height": quality.face_height},
            "detection_confidence": quality.detection_confidence,
            "visible_eye_count": quality.visible_eye_count,
            "open_eye_count": quality.open_eye_count,
            "full_face_visible": quality.full_face_visible,
            "mouth_visible": quality.mouth_visible,
        },
        "acceptable_for_enrollment": enrollment_ok,
        "message": message,
    }
