import math
import pathlib
import sys

import torch


ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from diffusion.models.diffusion_losses import (
    HungarianMatcherDynamicK,
    encode_qbox8_l1,
    normalize_qbox8_l1,
)


def test_qbox8_prediction_encoding_matches_lx_corner_order_and_scale():
    boxes = torch.tensor([[50.0, 40.0, 20.0, 10.0, 0.0]])
    image_whwh = torch.tensor([100.0, 80.0, 100.0, 80.0])
    encoded = encode_qbox8_l1(boxes, image_whwh)
    expected = torch.tensor([[
        0.4, 0.4375, 0.6, 0.4375,
        0.6, 0.5625, 0.4, 0.5625,
    ]])
    torch.testing.assert_close(encoded, expected)


def test_qbox8_gt_normalization_preserves_raw_corner_order():
    raw = torch.tensor([[
        60.0, 35.0, 60.0, 45.0,
        40.0, 45.0, 40.0, 35.0,
    ]])
    image_whwh = torch.tensor([100.0, 80.0, 100.0, 80.0])
    encoded = normalize_qbox8_l1(raw, image_whwh)
    expected = torch.tensor([[
        0.6, 0.4375, 0.6, 0.5625,
        0.4, 0.5625, 0.4, 0.4375,
    ]])
    torch.testing.assert_close(encoded, expected)


def test_matcher_defaults_to_le135_and_qbox8_is_opt_in():
    matcher = HungarianMatcherDynamicK(2.0, 5.0, 2.0, True, False)
    assert matcher.matching_l1_representation == "le135"
    matcher.matching_l1_representation = "qbox8"
    assert matcher.matching_l1_representation == "qbox8"
