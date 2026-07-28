import math

import torch

from diffusion.models.diffusion_losses import (encode_le135_l1,
                                                encode_lx_norm5_l1)


def test_equivalent_swapped_edges_have_identical_encoding():
    image_whwh = torch.tensor([1184.0, 896.0, 1184.0, 896.0])
    physical = torch.tensor([[500.0, 400.0, 100.0, 40.0, 0.2]])
    swapped = torch.tensor(
        [[500.0, 400.0, 40.0, 100.0, 0.2 + math.pi / 2]])

    encoded = encode_le135_l1(physical, image_whwh)
    encoded_swapped = encode_le135_l1(swapped, image_whwh)

    torch.testing.assert_close(encoded, encoded_swapped)


def test_spatial_l1_is_image_axis_balanced():
    image_whwh = torch.tensor([1184.0, 896.0, 1184.0, 896.0])
    base = torch.tensor([[500.0, 400.0, 100.0, 40.0, 0.2]])
    shifted_x = base.clone()
    shifted_x[:, 0] += 1
    shifted_y = base.clone()
    shifted_y[:, 1] += 1

    encoded = encode_le135_l1(base, image_whwh)
    encoded_x = encode_le135_l1(shifted_x, image_whwh)
    encoded_y = encode_le135_l1(shifted_y, image_whwh)

    torch.testing.assert_close(
        encoded_x[:, 0] - encoded[:, 0],
        encoded_y[:, 1] - encoded[:, 1])


def test_angle_uses_fixed_weight_without_periodic_shortcut():
    image_whwh = torch.tensor([1184.0, 896.0, 1184.0, 896.0])
    base = torch.tensor([[500.0, 400.0, 100.0, 40.0, 0.2]])
    shifted = base.clone()
    shifted[:, 4] += 0.1

    delta = (
        encode_le135_l1(shifted, image_whwh)
        - encode_le135_l1(base, image_whwh))
    torch.testing.assert_close(
        delta[:, 4], torch.tensor([0.05 * 0.1 / math.pi]))

    low = base.clone()
    high = base.clone()
    low[:, 4] = -math.pi / 4 + 1e-4
    high[:, 4] = 3 * math.pi / 4 - 1e-4
    boundary_cost = (
        encode_le135_l1(high, image_whwh)
        - encode_le135_l1(low, image_whwh)).abs()[0, 4]
    assert 0.049 < boundary_cost < 0.05


def test_encoding_has_finite_spatial_and_angle_gradients():
    image_whwh = torch.tensor([1184.0, 896.0, 1184.0, 896.0])
    boxes = torch.tensor(
        [[500.0, 400.0, 100.0, 40.0, 0.2]], requires_grad=True)
    encode_le135_l1(boxes, image_whwh).sum().backward()

    assert torch.isfinite(boxes.grad).all()
    assert (boxes.grad.abs() > 0).all()


def test_lx_norm5_matches_raw_per_axis_normalization_without_canonicalizing():
    image_whwh = torch.tensor([1200.0, 900.0, 1200.0, 900.0])
    boxes = torch.tensor([[600.0, 450.0, 120.0, 45.0, math.pi / 4]])
    encoded = encode_lx_norm5_l1(
        boxes, image_whwh, angle_weight=1.0)
    torch.testing.assert_close(
        encoded,
        torch.tensor([[0.5, 0.5, 0.1, 0.05, 0.5]]))

    swapped = boxes.clone()
    swapped[:, 2:4] = boxes[:, [3, 2]]
    swapped[:, 4] += math.pi / 2
    assert not torch.equal(
        encoded, encode_lx_norm5_l1(swapped, image_whwh))
