"""Verify hsmot does not depend on mmcv/mmengine at import time."""

import sys


def test_no_mmcv_mmengine_loaded():
    for name in list(sys.modules):
        assert not name.startswith("mmcv"), f"mmcv module loaded: {name}"
        assert not name.startswith("mmengine"), f"mmengine module loaded: {name}"


def test_core_imports():
    import hsmot  # noqa: F401
    from hsmot.loss.prob_iou_loss import probiou  # noqa: F401
    from hsmot.mmlab import hs_mmcv  # noqa: F401
    from hsmot.util.iou import box_iou_rotated  # noqa: F401

    test_no_mmcv_mmengine_loaded()
