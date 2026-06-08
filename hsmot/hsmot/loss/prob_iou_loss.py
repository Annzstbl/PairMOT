# Copyright (c) hsmot. ProbIoU implementation extracted from Ultralytics (arxiv.org/abs/2106.06072).

import math
from typing import Union

__all__ = ["get_covariance_matrix", "probiou", "batch_probiou", "probiou_loss"]

import numpy as np
import torch


def get_covariance_matrix(boxes: torch.Tensor):
    """
    从旋转框 (xywhr) 生成对应高斯椭圆的协方差矩阵参数。

    Args:
        boxes: (N, 5)，格式 xywhr [center_x, center_y, width, height, rotation_rad].

    Returns:
        (a, b, c): 2x2 协方差矩阵的三个独立分量，用于后续 Bhattacharyya 距离计算。
    """
    gbbs = torch.cat((boxes[..., 2:4].pow(2) / 12, boxes[..., 4:5]), dim=-1)
    a, b, c = gbbs.split(1, dim=-1)
    cos = c.cos()
    sin = c.sin()
    cos2 = cos.pow(2)
    sin2 = sin.pow(2)
    return a * cos2 + b * sin2, a * sin2 + b * cos2, (a - b) * cos * sin


def probiou(
    obb1: torch.Tensor,
    obb2: torch.Tensor,
    CIoU: bool = False,
    eps: float = 1e-7,
) -> torch.Tensor:
    """
    计算两组旋转框之间的 Probabilistic IoU（逐对，一一对应）。

    论文: https://arxiv.org/pdf/2106.06072v1.pdf

    Args:
        obb1: (N, 5) 格式 xywhr。
        obb2: (N, 5) 格式 xywhr，与 obb1 一一对应。
        CIoU: 若为 True，返回带宽高比惩罚的 CIoU。
        eps: 数值稳定用的小常数。

    Returns:
        (N,) 每个框对的相似度 [0, 1]，越大越相似。
    """
    x1, y1 = obb1[..., :2].split(1, dim=-1)
    x2, y2 = obb2[..., :2].split(1, dim=-1)
    a1, b1, c1 = get_covariance_matrix(obb1)
    a2, b2, c2 = get_covariance_matrix(obb2)

    denom = (a1 + a2) * (b1 + b2) - (c1 + c2).pow(2) + eps
    t1 = (
        ((a1 + a2) * (y1 - y2).pow(2) + (b1 + b2) * (x1 - x2).pow(2)) / denom
    ) * 0.25
    t2 = (((c1 + c2) * (x2 - x1) * (y1 - y2)) / denom) * 0.5
    det1 = (a1 * b1 - c1.pow(2)).clamp(min=0)
    det2 = (a2 * b2 - c2.pow(2)).clamp(min=0)
    t3 = (
        ((a1 + a2) * (b1 + b2) - (c1 + c2).pow(2))
        / (4 * (det1 * det2).sqrt() + eps)
        + eps
    ).log() * 0.5
    bd = (t1 + t2 + t3).clamp(min=eps, max=100.0)
    hd = (1.0 - (-bd).exp() + eps).sqrt()
    iou = 1 - hd

    if CIoU:
        w1, h1 = obb1[..., 2:4].split(1, dim=-1)
        w2, h2 = obb2[..., 2:4].split(1, dim=-1)
        v = (4 / math.pi**2) * ((w2 / h2).atan() - (w1 / h1).atan()).pow(2)
        with torch.no_grad():
            alpha = v / (v - iou + (1 + eps))
        return (iou - v * alpha).squeeze(-1)
    return iou.squeeze(-1)


def batch_probiou(
    obb1: Union[torch.Tensor, np.ndarray],
    obb2: Union[torch.Tensor, np.ndarray],
    eps: float = 1e-7,
) -> torch.Tensor:
    """
    批量计算 N 个框与 M 个框之间的 Prob IoU 矩阵。

    论文: https://arxiv.org/pdf/2106.06072v1.pdf

    Args:
        obb1: (N, 5) 格式 xywhr。
        obb2: (M, 5) 格式 xywhr。
        eps: 数值稳定用的小常数。

    Returns:
        (N, M) 相似度矩阵，可用于 NMS、匹配等。
    """
    if isinstance(obb1, np.ndarray):
        obb1 = torch.from_numpy(obb1).to(obb2.device if isinstance(obb2, torch.Tensor) else "cpu")
    if isinstance(obb2, np.ndarray):
        obb2 = torch.from_numpy(obb2).to(obb1.device)

    x1, y1 = obb1[..., :2].split(1, dim=-1)
    x2, y2 = (x.squeeze(-1)[None] for x in obb2[..., :2].split(1, dim=-1))
    a1, b1, c1 = get_covariance_matrix(obb1)
    a2, b2, c2 = get_covariance_matrix(obb2)
    a2 = a2.squeeze(-1)[None]
    b2 = b2.squeeze(-1)[None]
    c2 = c2.squeeze(-1)[None]

    denom = (a1 + a2) * (b1 + b2) - (c1 + c2).pow(2) + eps
    t1 = (
        ((a1 + a2) * (y1 - y2).pow(2) + (b1 + b2) * (x1 - x2).pow(2)) / denom
    ) * 0.25
    t2 = (((c1 + c2) * (x2 - x1) * (y1 - y2)) / denom) * 0.5
    det1 = (a1 * b1 - c1.pow(2)).clamp(min=0)
    det2 = (a2 * b2 - c2.pow(2)).clamp(min=0)
    t3 = (
        ((a1 + a2) * (b1 + b2) - (c1 + c2).pow(2))
        / (4 * (det1 * det2).sqrt() + eps)
        + eps
    ).log() * 0.5
    bd = (t1 + t2 + t3).clamp(min=eps, max=100.0)
    hd = (1.0 - (-bd).exp() + eps).sqrt()
    return 1 - hd


def probiou_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    CIoU: bool = False,
    eps: float = 1e-7,
    reduction: str = "mean",
) -> torch.Tensor:
    """
    以 Prob IoU 作为相似度时的损失：loss = 1 - probiou（支持 CIoU）。

    适用于训练时对旋转框回归做 IoU 监督。

    Args:
        pred: (N, 5) 预测 xywhr。
        target: (N, 5) 目标 xywhr。
        CIoU: 是否使用 CIoU。
        eps: 数值稳定常数。
        reduction: 'mean' | 'sum' | 'none'。

    Returns:
        标量或 (N,) 的 loss。
    """
    iou = probiou(pred, target, CIoU=CIoU, eps=eps)
    loss = 1 - iou
    if reduction == "mean":
        return loss.mean()
    if reduction == "sum":
        return loss.sum()
    return loss
