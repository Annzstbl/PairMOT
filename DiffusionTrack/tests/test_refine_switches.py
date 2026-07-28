import math
import pathlib
import sys

import torch


ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from diffusion.models.diffusion_models import RCNNHead


def make_head():
    return RCNNHead(
        d_model=8,
        num_classes=1,
        pooler_resolution=2,
        dim_feedforward=16,
        nhead=1,
        use_focal=True,
    )


def test_proj_xy_rotates_local_center_offset_into_image_coordinates():
    head = make_head()
    boxes = torch.tensor([[100.0, 200.0, 10.0, 4.0, math.pi / 2]])
    deltas = torch.tensor([[2.0, 0.0, 0.0, 0.0, 0.0]])
    projected = head.apply_deltas(deltas, boxes)
    torch.testing.assert_close(
        projected[0, :2], torch.tensor([100.0, 210.0]), atol=1e-5, rtol=0)

    head.proj_xy = False
    global_xy = head.apply_deltas(deltas, boxes)
    torch.testing.assert_close(
        global_xy[0, :2], torch.tensor([110.0, 200.0]), atol=1e-5, rtol=0)


def test_canonicalization_switch_controls_long_edge_swap():
    head = make_head()
    boxes = torch.tensor([[10.0, 20.0, 4.0, 10.0, 0.0]])
    deltas = torch.zeros((1, 5))
    canonical = head.apply_deltas(deltas, boxes)
    torch.testing.assert_close(
        canonical[0, 2:4], torch.tensor([10.0, 4.0]))

    head.canonicalize_refined_boxes = False
    raw = head.apply_deltas(deltas, boxes)
    torch.testing.assert_close(raw[0, 2:4], torch.tensor([4.0, 10.0]))


def test_angle_delta_scale_can_reproduce_lx_degree_sized_step():
    head = make_head()
    boxes = torch.tensor([[10.0, 20.0, 10.0, 4.0, 0.0]])
    deltas = torch.tensor([[0.0, 0.0, 0.0, 0.0, 1.0]])
    radians = head.apply_deltas(deltas, boxes)
    torch.testing.assert_close(radians[0, 4], torch.tensor(1.0))

    head.angle_delta_scale = math.pi / 180.0
    degrees = head.apply_deltas(deltas, boxes)
    torch.testing.assert_close(
        degrees[0, 4], torch.tensor(math.pi / 180.0))


def test_lx_delta_numerics_matches_raw_global_faster_rcnn_formula():
    head = make_head()
    head.proj_xy = False
    head.canonicalize_refined_boxes = False
    head.angle_delta_scale = math.pi / 180.0
    head.lx_delta_numerics = True
    boxes = torch.tensor([[100.0, 200.0, 10.0, 4.0, 0.1]])
    deltas = torch.tensor([[0.2, -0.5, math.log(2), math.log(0.5), 3.0]])

    decoded = head.apply_deltas(deltas, boxes)
    expected = torch.tensor([[
        101.0, 199.0, 20.0, 2.0, 0.1 + 3.0 * math.pi / 180.0]])
    torch.testing.assert_close(decoded, expected)


def test_lx_delta_numerics_keeps_unbounded_negative_log_scale():
    head = make_head()
    head.proj_xy = False
    head.canonicalize_refined_boxes = False
    head.lx_delta_numerics = True
    boxes = torch.tensor([[0.0, 0.0, 10.0, 10.0, 0.0]])
    deltas = torch.tensor([[0.0, 0.0, -20.0, -20.0, 0.0]])
    decoded = head.apply_deltas(deltas, boxes)
    assert decoded[0, 2] < 1e-6
    assert decoded[0, 3] < 1e-6
