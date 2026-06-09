"""ProbIoU（可微）单元测试 — 仅用于训练 loss，不用于评测匹配。"""

import pytest
import torch

from hsmot.loss.prob_iou_loss import batch_probiou, probiou


def test_rotation_sensitivity():
    """ProbIoU 对非正方形框应随角度变化。"""
    bboxes1 = torch.tensor([[0.5, 0.5, 0.5, 0.2, 0.25], [0.5, 0.5, 0.5, 0.2, 1.25]])
    bboxes2 = torch.tensor([[0.3, 0.3, 0.5, 0.2, 0.5], [0.3, 0.3, 0.5, 0.2, 1.5]])
    result = probiou(bboxes1, bboxes2)
    assert result[0].item() != pytest.approx(result[1].item(), abs=1e-4)


def test_scale_invariance_fixed_ratio():
    """等比例缩放时 ProbIoU 应保持不变。"""
    bboxes1 = torch.tensor([[0.5, 0.5, 0.3, 0.3, 0.25], [50.0, 50.0, 30.0, 30.0, 0.25]])
    bboxes2 = torch.tensor([[0.3, 0.3, 0.5, 0.5, 0.5], [30.0, 30.0, 50.0, 50.0, 0.5]])
    result = probiou(bboxes1, bboxes2)
    assert result[0].item() == pytest.approx(result[1].item(), abs=1e-4)


def test_scale_sensitivity():
    """非等比例缩放时 ProbIoU 应发生变化。"""
    bboxes1 = torch.tensor([[0.5, 0.5, 0.3, 0.3, 0.25], [50.0, 50.0, 60.0, 60.0, 0.25]])
    bboxes2 = torch.tensor([[0.3, 0.3, 0.5, 0.5, 0.5], [30.0, 30.0, 100.0, 100.0, 0.5]])
    result = probiou(bboxes1, bboxes2)
    assert result[0].item() != pytest.approx(result[1].item(), abs=1e-4)


def test_identical_boxes_high_iou():
    """相同框对的 ProbIoU 应接近 1。"""
    box = torch.tensor([[1.0, 2.0, 3.0, 1.0, 0.5], [10.0, 20.0, 4.0, 2.0, -0.3]])
    result = probiou(box, box)
    assert result[0].item() == pytest.approx(1.0, abs=1e-3)
    assert result[1].item() == pytest.approx(1.0, abs=1e-3)


def test_batch_probiou_shape():
    bboxes1 = torch.tensor([[0.5, 0.5, 0.3, 0.3, 0.25], [0.5, 0.5, 0.4, 0.2, 0.1]])
    bboxes2 = torch.tensor([[0.3, 0.3, 0.5, 0.5, 0.5], [0.6, 0.6, 0.3, 0.3, 0.2]])
    result = batch_probiou(bboxes1, bboxes2)
    assert result.shape == (2, 2)
