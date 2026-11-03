import sys

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


def test_similarity_matrix_keeps_scale():
    """|det M| must be s**2, not 1: the Umeyama ``params`` path drops s."""
    m = similarity_matrix(SCRFD)
    s_ours = float(np.hypot(m[0, 0], m[0, 1]))
    t = SimilarityTransform()
    t.estimate(SCRFD, face_align.arcface_dst)
    # s_ours is drawn from float32 params; 1e-3 of 1.36 is ~1.4 mili-px.
    assert s_ours == pytest.approx(t.scale, rel=2e-3)

    reference = face_align.estimate_norm(SCRFD, image_size=112)
    s_reference = float(np.hypot(reference[0, 0], reference[0, 1]))
    # norm_crop deliberately returns det=1; s_reference (1.3528) < s (1.3659).
    assert s_reference != pytest.approx(t.scale, rel=2e-3)


@pytest.mark.skipif(face_align is None, reason="insightface oracle not installed")
def test_projected_landmarks_are_closer_than_insightface():
    """Our fit must land nearer the template than the scale-free reference."""
    ones = np.ones((5, 1), np.float32)

    m = similarity_matrix(SCRFD)
    err_ours = float(
        np.linalg.norm((np.hstack([SCRFD, ones]) @ m.T) - target_template(), axis=1).max()
    )

    ref = face_align.estimate_norm(SCRFD, image_size=112)
    err_ref = float(
        np.linalg.norm((np.hstack([SCRFD, ones]) @ ref.T) - target_template(), axis=1).max()
    )

    # Measured: ours 3.52 px, reference 3.73 px on this hand-written set.
    assert err_ours < err_ref
    assert err_ours < 5.0


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
    assert x0 == 37 and y0 == 37  # 50 - (20 * 1.30) / 2
