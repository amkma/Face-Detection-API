"""What a caller needs from the endpoint, which it was not previously getting."""

import base64
import json

import cv2
import numpy as np
from fastapi.testclient import TestClient

from app import main as app_main
from app.face_service import MIN_BLUR_VARIANCE, _alignment_checks

CLIENT = TestClient(app_main.app)


def send(image):
    ok, buffer = cv2.imencode(".png", image)
    payload = {"image": "data:image/png;base64," + base64.b64encode(buffer.tobytes()).decode()}
    return CLIENT.post("/api/verify", json=payload).json()


def portrait(side):
    source = cv2.imread("tests/assets/portrait.png")
    return cv2.resize(source, (side, side), interpolation=cv2.INTER_CUBIC)


def blobs(points):
    return sorted(points, key=lambda point: (round(point[0], 1), round(point[1], 1)))


def test_box_and_landmarks_are_in_the_body():
    body = send(portrait(512))
    assert body["accepted"] is True
    area = body["facial_area"]
    assert area and area["width"] > 100 and area["height"] > 100
    assert body["detection_confidence"] >= 0.8
    assert len(body["landmarks"]) == 5
    for x, y in body["landmarks"]:
        assert area["x"] <= x < area["x"] + area["width"]
        assert area["y"] <= y < area["y"] + area["height"]


def test_sharpness_is_resolution_independent():
    """The same sharp face must not fail for being bigger nor pass for being squashed."""
    scores = [send(portrait(side))["quality"]["blur_variance"] for side in (320, 512, 800, 1024, 1600)]
    assert all(score >= MIN_BLUR_VARIANCE for score in scores), scores
    ratio = float(max(scores)) / float(min(scores))
    assert ratio < 2.0, scores


def test_blurred_face_is_rejected_with_the_reason():
    blurred = cv2.GaussianBlur(portrait(1600), (0, 0), 5.0)
    body = send(blurred)
    assert body["accepted"] is False
    assert "blurry" in body["flaws"]
    assert body["facial_area"]["width"] > 100, "box still reported when rejecting for focus"


def test_rejection_is_never_blank_about_why():
    canvas = np.full((2048, 2048, 3), 255, np.uint8)
    body = send(canvas)
    assert body["accepted"] is False
    assert body["flaws"] and body["flaws"].strip()


def test_landmarks_inside_is_position_aware():
    points = np.asarray([
        [274.08, 266.78], [334.22, 274.18], [315.35, 314.31],
        [271.97, 340.06], [317.74, 346.58],
    ], dtype=np.float32)
    checks = {c.name: c for c in _alignment_checks(points, 210, 182, 142, 205)}
    assert checks["landmarks_inside"].passed, checks["landmarks_inside"].detail

    stray = {c.name: c for c in _alignment_checks(points, 900, 900, 142, 205)}
    assert not stray["landmarks_inside"].passed
