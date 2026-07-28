import math

import torch

from diffusion.models.diffusion_models import RCNNHead


def _small_head():
    return RCNNHead(
        d_model=8,
        num_classes=2,
        pooler_resolution=1,
        dim_feedforward=8,
        nhead=1,
        dropout=0.0,
    )


def test_center_offsets_are_projected_from_box_local_axes():
    head = _small_head()
    boxes = torch.tensor([
        [200.0, 300.0, 100.0, 20.0, math.pi / 2],
        [200.0, 300.0, 100.0, 20.0, math.pi / 2],
    ])
    # bbox_weights=(2,2,1,1,1), so a raw delta of 2 is a unit
    # normalized displacement. At theta=90 degrees the local width axis
    # points along global +y, while the local height axis points along -x.
    deltas = torch.tensor([
        [2.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 2.0, 0.0, 0.0, 0.0],
    ])

    decoded = head.apply_deltas(deltas, boxes)

    torch.testing.assert_close(
        decoded[:, :2],
        torch.tensor([[200.0, 400.0], [180.0, 300.0]]),
        atol=1e-5,
        rtol=0.0,
    )


def test_zero_deltas_remain_identity_under_le135_regularization():
    head = _small_head()
    boxes = torch.tensor([
        [200.0, 300.0, 100.0, 20.0, 0.3],
        [400.0, 100.0, 80.0, 30.0, -0.4],
    ])

    decoded = head.apply_deltas(torch.zeros_like(boxes), boxes)

    torch.testing.assert_close(decoded, boxes)


def test_degree_center_offsets_use_physical_angles():
    head = _small_head()
    head.box_angle_degrees = True
    boxes = torch.tensor([
        [200.0, 300.0, 100.0, 20.0, 90.0],
        [200.0, 300.0, 100.0, 20.0, 90.0],
    ])
    deltas = torch.tensor([
        [2.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 2.0, 0.0, 0.0, 0.0],
    ])

    decoded = head.apply_deltas(deltas, boxes)

    torch.testing.assert_close(
        decoded[:, :2],
        torch.tensor([[200.0, 400.0], [180.0, 300.0]]),
        atol=1e-5,
        rtol=0.0,
    )
    torch.testing.assert_close(decoded[:, 4], torch.tensor([90.0, 90.0]))


def test_degree_le135_wrap_and_long_edge_swap():
    head = _small_head()
    head.box_angle_degrees = True
    boxes = torch.tensor([
        [10.0, 20.0, 10.0, 4.0, 130.0],
        [10.0, 20.0, 4.0, 10.0, 30.0],
    ])
    deltas = torch.tensor([
        [0.0, 0.0, 0.0, 0.0, 10.0],
        [0.0, 0.0, 0.0, 0.0, 0.0],
    ])

    decoded = head.apply_deltas(deltas, boxes)

    torch.testing.assert_close(
        decoded,
        torch.tensor([
            [10.0, 20.0, 10.0, 4.0, -40.0],
            [10.0, 20.0, 10.0, 4.0, 120.0],
        ]),
        atol=1e-5,
        rtol=0.0,
    )


def test_degree_and_radian_geometry_are_equivalent():
    radian_head = _small_head()
    degree_head = _small_head()
    degree_head.box_angle_degrees = True
    boxes_radians = torch.tensor([
        [100.0, 200.0, 20.0, 8.0, math.radians(37.0)],
        [300.0, 150.0, 7.0, 19.0, math.radians(112.0)],
    ])
    boxes_degrees = boxes_radians.clone()
    boxes_degrees[:, 4] = torch.rad2deg(boxes_degrees[:, 4])
    deltas_radians = torch.tensor([
        [0.4, -0.2, math.log(1.2), math.log(0.8), math.radians(11.0)],
        [-0.3, 0.5, math.log(0.7), math.log(1.4), math.radians(-9.0)],
    ])
    deltas_degrees = deltas_radians.clone()
    deltas_degrees[:, 4] = torch.rad2deg(deltas_degrees[:, 4])

    decoded_radians = radian_head.apply_deltas(
        deltas_radians, boxes_radians)
    decoded_degrees = degree_head.apply_deltas(
        deltas_degrees, boxes_degrees)
    decoded_degrees[:, 4] = torch.deg2rad(decoded_degrees[:, 4])

    torch.testing.assert_close(
        decoded_degrees, decoded_radians, atol=2e-5, rtol=1e-6)
