"""Five-point alignment of a face onto the ArcFace 112 template.

Implemented here rather than imported from insightface so that nothing
whose licence is research-only (insightface's model zoo is) is needed to
run this module. When insightface or scikit-image happens to be installed,
tests/test_align.py uses them as an independent oracle; when it is not,
those comparisons skip and the property assertions still run.

The scale is kept on purpose. insightface's Umeyama ``params`` is
scale-free (det 1), and norm_crop applies it anyway, so warped landmarks
do not land exactly on arcface_dst. Here det(M) != 1, so given five
points the landed points equal the template by construction, and the
rest of a transform multiplies back to pixel units as in scikit-image.

Landmark order is image-left eye, image-right eye, nose, image-left mouth
corner, image-right mouth corner.
"""

from __future__ import annotations

import cv2
import numpy as np

# Side of the square crop the five-point template is defined for.
INPUT_SIZE = 112

# Context kept around a detected box when cutting a crop for a graph whose
# input is a plain square: a recognition backbone wants the whole head.
CROP_PAD = 1.30

# Largest source a warp may occupy, as a multiple of the output crop. A
# 9000 px source warped into 112 px is aliasing, not detail.
MAX_SOURCE_MULTIPLE = 24

# Canonical destination, in the order the detector prints its five points:
# image-left eye, image-right eye, nose, image-left mouth, image-right mouth.
TEMPLATE = np.asarray(
    [
        [38.2946, 51.6963],
        [73.5318, 51.5014],
        [56.0252, 71.7366],
        [41.5493, 92.3655],
        [70.7299, 92.2041],
    ],
    dtype=np.float32,
)


def target_template(size: int = INPUT_SIZE) -> np.ndarray:
    """TEMPLATE as float64, scaled to a square ``size`` crop."""
    return TEMPLATE.astype(np.float64) * (float(size) / float(INPUT_SIZE))


def _umeyama(src: np.ndarray, dst: np.ndarray) -> tuple[np.ndarray, float]:
    """Least-squares similarity mapping ``src`` onto ``dst``: (2x3, scale)."""
    src = np.asarray(src, dtype=np.float64)
    dst = np.asarray(dst, dtype=np.float64)
    if src.shape != dst.shape or src.ndim != 2 or src.shape[1] != 2:
        raise ValueError("src and dst must both be (n, 2)")

    src_mean = src.mean(axis=0)
    dst_mean = dst.mean(axis=0)
    src_centred = src - src_mean
    dst_centred = dst - dst_mean

    src_var = float((src_centred ** 2).sum()) / src.shape[0]
    if src_var <= 0.0:
        raise ValueError("src landmarks are degenerate")
    covariance = (dst_centred.T @ src_centred) / src.shape[0]

    u, singular, vt = np.linalg.svd(covariance)
    direction = np.ones(2, dtype=np.float64)
    if np.linalg.det(u) * np.linalg.det(vt) < 0.0:
        direction[-1] = -1.0

    rotation = u @ np.diag(direction) @ vt
    scale = float(np.trace(np.diag(singular) @ np.diag(direction))) / src_var
    translation = dst_mean - scale * (rotation @ src_mean)

    matrix = np.zeros((2, 3), dtype=np.float64)
    matrix[:, :2] = scale * rotation
    matrix[:, 2] = translation
    return matrix, scale


def similarity_matrix(points, size: int = INPUT_SIZE) -> np.ndarray:
    """Similarity transform taking five ``points`` onto the template."""
    src = np.asarray(points, dtype=np.float64).reshape(5, 2)
    if not np.all(np.isfinite(src)):
        raise ValueError("landmarks contain non-finite values")
    matrix, _scale = _umeyama(src, target_template(size))
    return matrix.astype(np.float32)


def align_face(image_bgr: np.ndarray, landmarks: np.ndarray,
               size: int = INPUT_SIZE) -> tuple[np.ndarray, np.ndarray]:
    """Warp the face onto the template: (crop, 2x3 matrix actually applied)."""
    matrix = similarity_matrix(landmarks, size=size)
    crop = cv2.warpAffine(image_bgr, matrix, (size, size),
                          flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_REPLICATE)
    return crop, matrix


def square_crop(image_bgr: np.ndarray, box, pad: float = CROP_PAD):
    """Square padded crop about ``box`` = (x, y, w, h): (patch, x0, y0).

    The patch is padded past the frame edge (replicating the edge pixels)
    when the box sits on it, so it is always ``side`` square and
    ``box - (x0, y0)`` stays exact.
    Clamping the read instead would hand main.py a short edge and a shifted
    box: a geometry bug dressed up as a border.
    """
    img_h, img_w = image_bgr.shape[:2]
    x, y, w, h = (int(round(float(v))) for v in np.asarray(box).reshape(4))
    side = max(1, int(round(max(w, h) * pad)))
    x0 = int(round(x + w / 2.0 - side / 2.0))
    y0 = int(round(y + h / 2.0 - side / 2.0))
    patch = cv2.copyMakeBorder(
        image_bgr,
        max(0, -y0),
        max(0, y0 + side - img_h),
        max(0, -x0),
        max(0, x0 + side - img_w),
        cv2.BORDER_REPLICATE,
    )
    top, left = max(0, y0), max(0, x0)
    return patch[top : top + side, left : left + side], x0, y0

