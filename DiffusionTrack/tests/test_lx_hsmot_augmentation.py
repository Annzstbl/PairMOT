import pathlib
import random
import sys

import numpy as np


ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from yolox.data.data_augment import (
    _distort_hsmot_lx,
    _mirror_hsmot_lx,
)


def test_lx_distortion_is_seed_reproducible_and_preserves_shape_dtype():
    image = np.arange(4 * 5 * 8, dtype=np.uint8).reshape(4, 5, 8)
    random.seed(8823)
    first = _distort_hsmot_lx(image)
    random.seed(8823)
    second = _distort_hsmot_lx(image)
    assert np.array_equal(first, second)
    assert first.shape == image.shape
    assert first.dtype == image.dtype


def test_lx_mirror_updates_every_qbox_x_coordinate():
    image = np.zeros((4, 10, 8), dtype=np.uint8)
    qboxes = np.array(
        [[1, 1, 3, 1, 3, 2, 1, 2]], dtype=np.float32)
    state = random.getstate()
    try:
        random.seed(0)  # first randrange(2) is 1
        mirrored_image, mirrored = _mirror_hsmot_lx(image, qboxes)
    finally:
        random.setstate(state)
    assert mirrored_image.strides[1] < 0
    np.testing.assert_array_equal(
        mirrored,
        np.array([[9, 1, 7, 1, 7, 2, 9, 2]], dtype=np.float32))
