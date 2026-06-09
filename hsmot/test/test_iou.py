"""几何旋转框 IoU 单元测试（评测 / 数据管线用，非可微）。"""

import pytest
import torch

from hsmot.util.iou import box_iou_rotated


def test_identical_boxes_iou_one():
    """相同框的几何 IoU 应为 1。"""
    box = torch.tensor([[1.0, 2.0, 3.0, 1.0, 0.5], [10.0, 20.0, 4.0, 2.0, -0.3]])
    result = box_iou_rotated(box, box, aligned=True)
    assert result[0].item() == pytest.approx(1.0, abs=1e-4)
    assert result[1].item() == pytest.approx(1.0, abs=1e-4)


def test_disjoint_boxes_zero_iou():
    """不相交框 IoU 应为 0。"""
    b1 = torch.tensor([[0.0, 0.0, 2.0, 2.0, 0.0]])
    b2 = torch.tensor([[10.0, 10.0, 2.0, 2.0, 0.0]])
    result = box_iou_rotated(b1, b2, aligned=True)
    assert result[0].item() == pytest.approx(0.0, abs=1e-4)


def test_partial_overlap():
    """部分重叠框 IoU 应在 (0, 1)。"""
    b1 = torch.tensor([[0.0, 0.0, 4.0, 4.0, 0.0]])
    b2 = torch.tensor([[2.0, 0.0, 4.0, 4.0, 0.0]])
    result = box_iou_rotated(b1, b2, aligned=True)
    assert 0.0 < result[0].item() < 1.0


def test_rotation_changes_iou():
    """旋转后与非正方形框的 IoU 应随角度变化。"""
    bboxes1 = torch.tensor([[0.5, 0.5, 0.5, 0.2, 0.25], [0.5, 0.5, 0.5, 0.2, 1.25]])
    bboxes2 = torch.tensor([[0.3, 0.3, 0.5, 0.2, 0.5], [0.3, 0.3, 0.5, 0.2, 1.5]])
    result = box_iou_rotated(bboxes1, bboxes2, aligned=True)
    assert result[0].item() != pytest.approx(result[1].item(), abs=1e-4)


def test_scale_invariance():
    """等比例缩放不改变几何 IoU。"""
    bboxes1 = torch.tensor([[0.5, 0.5, 0.3, 0.3, 0.25], [50.0, 50.0, 30.0, 30.0, 0.25]])
    bboxes2 = torch.tensor([[0.3, 0.3, 0.5, 0.5, 0.5], [30.0, 30.0, 50.0, 50.0, 0.5]])
    result = box_iou_rotated(bboxes1, bboxes2, aligned=True)
    assert result[0].item() == pytest.approx(result[1].item(), abs=1e-4)


def test_batch_iou_shape():
    bboxes1 = torch.tensor([[0.5, 0.5, 0.3, 0.3, 0.25], [0.5, 0.5, 0.4, 0.2, 0.1]])
    bboxes2 = torch.tensor([[0.3, 0.3, 0.5, 0.5, 0.5], [0.6, 0.6, 0.3, 0.3, 0.2]])
    result = box_iou_rotated(bboxes1, bboxes2, aligned=False)
    assert result.shape == (2, 2)


def test_iof_mode():
    """iof = intersection / area(bboxes1)。"""
    b1 = torch.tensor([[0.0, 0.0, 4.0, 4.0, 0.0]])
    b2 = torch.tensor([[2.0, 0.0, 4.0, 4.0, 0.0]])
    iou = box_iou_rotated(b1, b2, mode="iou", aligned=True)[0].item()
    iof = box_iou_rotated(b1, b2, mode="iof", aligned=True)[0].item()
    assert iof > iou
