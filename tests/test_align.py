import sys

import cv2
import numpy as np
import pytest
from skimage.transform import SimilarityTransform

# insightface is an optional ORACLE, never a dependency: its model zoo is
# non-commercial research only and this package may be sold. When it is absent,
# the reference-comparison tests skip and the property assertions still run.
try:
    from insightface.utils import face_align
except ImportError:
    face_align = None

sys.path.insert(0, ".")

from app._align import (  # noqa: E402
    TEMPLATE,
    _umeyama,
    align_face,
    similarity_matrix,
    square_crop,
    target_template,
)

SCRFD = np.array([[40, 60], [70, 57], [55, 72], [43, 86], [67, 84]], np.float32)


@pytest.mark.skipif(face_align is None, reason="insightface oracle not installed")
def test_template_is_arcface_template():
    assert np.allclose(TEMPLATE, face_align.arcface_dst, atol=1e-3)


def test_similarity_matrix_is_reproducible():
    first = similarity_matrix(SCRFD)
    for _ in range(50):
        assert np.array_equal(first, similarity_matrix(SCRFD))


def test_scale_is_reported_in_pixel_units():
    """_umeyama's second value is a px scale, not a determinant ratio.

    A similarity that scales by k has det(k R) = k**2, so det(M) is never the
    scale. The two agree with the independent oracle here, which is what makes
    the reported number credible.
    """
    m, s = _umeyama(SCRFD, target_template())
    assert float(np.linalg.det(m[:, :2])) == pytest.approx(s ** 2, rel=1e-6)

    t = SimilarityTransform()
    t.estimate(SCRFD, np.asarray(target_template()))
    assert s == pytest.approx(t.scale, rel=2e-3)


@pytest.mark.skipif(face_align is None, reason="insightface oracle not installed")
def test_projected_landmarks_match_the_reference():
    """Our warp lands on the template; the reference matches it to float noise.

    No refit beats an exact similarity solve on 5 points, so "ours is closer"
    is not a property to hold. What is worth holding: in the full-precision
    path ours reproduces the oracle, and the pixels it writes agree.
    """
    hom = np.hstack([SCRFD, np.ones((5, 1))])
    dst = target_template()

    m, _s = _umeyama(SCRFD, dst)
    err_ours = float(np.linalg.norm(hom @ m.T - dst, axis=1).max())

    ref = face_align.estimate_norm(SCRFD, image_size=112)
    err_ref = float(np.linalg.norm(hom @ ref.T - dst, axis=1).max())

    assert err_ours == pytest.approx(err_ref, abs=1e-3)
    assert err_ours < 5.0

    image = np.full((160, 160, 3), 90, np.uint8)
    cv2.rectangle(image, (30, 30), (120, 120), (200, 180, 160), -1)
    ours, _ = align_face(image, SCRFD)
    oracle = cv2.warpAffine(image, ref, (112, 112), flags=cv2.INTER_LINEAR,
                            borderMode=cv2.BORDER_REPLICATE)
    assert float(np.abs(ours.astype(int) - oracle.astype(int)).max()) <= 2


def test_align_face_shape():
    crop, matrix = align_face(np.zeros((140, 140, 3), np.uint8), SCRFD)
    assert crop.shape == (112, 112, 3)
    assert matrix.shape == (2, 3)


def test_degenerate_landmarks_raised():
    with pytest.raises(ValueError):
        similarity_matrix(np.full((5, 2), 7.0, np.float32))


def test_square_crop_pads_at_the_frame_edge():
    image = np.zeros((100, 100, 3), np.uint8)
    patch, x0, y0 = square_crop(image, np.array([70, 70, 100, 100], np.float32))
    assert patch.shape[0] == patch.shape[1]
    # The padded square starts at 66, so its far edge (66 + 44) overruns the frame.
    assert x0 + patch.shape[1] > 100 or y0 + patch.shape[0] > 100
    assert int(patch[0, 0, 0]) == 0


def test_square_crop_keeps_true_origin_inside():
    image = np.zeros((100, 100, 3), np.uint8)
    patch, x0, y0 = square_crop(image, np.array([40, 40, 60, 60], np.float32))
    # side = round(60 * 1.30) = 78; origin = round(70 - 39) = 31, so 31..109 is
    # still inside the 100 px frame (no padding branch). The first pixel of the
    # patch is not the box corner, so it must stay the image's own black.
    assert x0 == 31 and y0 == 31
    assert patch.shape[0] == 78
    assert int(patch[0, 0, 0]) == 0
