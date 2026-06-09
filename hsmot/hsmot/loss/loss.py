import math
from typing import Optional

import torch
from torch import Tensor
from torch.autograd import Function
from torch.autograd.function import once_differentiable

from hsmot.datasets.pipelines.channel import version_index_to_str
from hsmot.loss.prob_iou_loss import probiou
from hsmot.util.dist import l1_dist_rotate


def _convex_giou_not_implemented(*args, **kwargs):
    raise NotImplementedError("convex_giou is not implemented in hsmot; use probiou-based losses instead.")


convex_giou = _convex_giou_not_implemented


def l1_loss_rotate(
    input: Tensor,
    target: Tensor,
    weight: Optional[Tensor] = None,
    angle_cycle: bool = False,
) -> Tensor:
    """
    计算输入张量和目标张量之间的旋转 L1 损失。

    Args:
        input (Tensor): 输入张量，形状为 [B, N, 5] 或 [N, 5]。
        target (Tensor): 目标张量，形状与 input 相同。
        weight (Tensor, optional): 长度为 5 的向量，对最后一维
            各通道赋予权重（如 [w, h, w, h, angle]）。
        angle_cycle (bool, optional): 是否对角度维度进行周期处理。

    Returns:
        Tensor: 旋转 L1 损失。
    """
    assert target.size() == input.size(), "input 和 target 的形状必须一致"

    if weight is not None:
        assert weight.numel() == 5, "weight 必须是长度为 5 的向量"
        weight = weight.view(1, 1, 5)

    if input.dim() == 3:
        batch_size = input.size(0)
        loss = l1_dist_rotate(
            input.flatten(0, 1), target.flatten(0, 1), aligned=True, cal_sum=False, angle_cycle=angle_cycle
        )
        if weight is not None:
            loss = loss.view(batch_size, -1, 5) * weight
            return loss.view(batch_size, -1)
        return loss.view(batch_size, -1)

    if input.dim() == 2:
        loss = l1_dist_rotate(input, target, aligned=True, cal_sum=False, angle_cycle=angle_cycle)
        if weight is not None:
            loss = loss.view(1, -1, 5) * weight
            return loss.view(-1, 5)
        return loss

    raise ValueError("输入的张量必须是 2D 或 3D 张量")


def loss_rotated_iou_norm_bboxes1(
    bboxes1: torch.Tensor,
    bboxes2: torch.Tensor,
    img_shape: torch.Tensor,
    version: str = "le135",
    use_probiou: bool = True,
) -> torch.Tensor:
    """
    计算逐对旋转框 IoU，用于训练 loss（一一对应，无需 batch 版）。

    Args:
        bboxes1: (N, 5) 预测框，归一化 [cx_norm, cy_norm, w_norm, h_norm, angle_norm]。
        bboxes2: (N, 5) GT 框，绝对 xywhr（像素 + 弧度），与 bboxes1 逐对对应。
        img_shape: (h, w)。
        version: 'le135' / 'le90'，角度归一化方式。
        use_probiou: 保留参数以兼容旧调用；训练 loss 使用可微 ProbIoU。

    Returns:
        (N,) 每对的 IoU。
    """
    if not isinstance(version, str):
        version = version_index_to_str(version)
    if version == "oc":
        raise NotImplementedError
    if version == "le135":
        angle_range = 1
        angle_offset = -1 / 4
    elif version == "le90":
        angle_range = 1
        angle_offset = -1 / 2
    else:
        raise ValueError(f"Unsupported version: {version!r}")
    angle_range *= math.pi
    angle_offset *= math.pi
    h, w = img_shape
    bboxes1 = bboxes1 * torch.as_tensor(
        [w, h, w, h, angle_range], dtype=bboxes1.dtype, device=bboxes1.device
    ) + torch.as_tensor([0, 0, 0, 0, angle_offset], dtype=bboxes1.dtype, device=bboxes1.device)

    del use_probiou
    return probiou(bboxes1, bboxes2, CIoU=False, eps=1e-7)


class ConvexGIoULossFuction(Function):
    """Convex GIoU loss (requires external convex_giou implementation)."""

    @staticmethod
    def forward(ctx, pred, target, weight=None, reduction=None, avg_factor=None, loss_weight=1.0):
        ctx.save_for_backward(pred)
        convex_gious, grad = convex_giou(pred, target)

        loss = 1 - convex_gious
        if weight is not None:
            loss = loss * weight
            grad = grad * weight.reshape(-1, 1)
        if reduction == "sum":
            loss = loss.sum()
        elif reduction == "mean":
            loss = loss.mean()

        unvaild_inds = torch.nonzero((grad > 1).sum(1), as_tuple=False)[:, 0]
        grad[unvaild_inds] = 1e-6

        reduce_grad = -grad / grad.size(0) * loss_weight
        ctx.convex_points_grad = reduce_grad
        return loss

    @staticmethod
    @once_differentiable
    def backward(ctx, input=None):
        return ctx.convex_points_grad, None, None, None, None, None


convex_giou_loss = ConvexGIoULossFuction.apply
