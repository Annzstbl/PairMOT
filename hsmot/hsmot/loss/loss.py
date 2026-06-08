from typing import Callable, List, Optional, Tuple
import torch
Tensor = torch.Tensor
from hsmot.util.dist import l1_dist_rotate
import math
from hsmot.datasets.pipelines.channel import version_index_to_str
from hsmot.loss.prob_iou_loss import probiou
from mmcv.ops import diff_iou_rotated_2d
from torch.autograd import Function
from torch.autograd.function import once_differentiable

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
    # 确保输入和目标的形状一致
    assert target.size() == input.size(), "input 和 target 的形状必须一致"

    if weight is not None:
        assert weight.numel() == 5, "weight 必须是长度为 5 的向量"
        weight = weight.view(1, 1, 5)  # 统一为 [1,1,5] 便于广播到结果
    
    # 检查张量是否包含批次维度
    if input.dim() == 3:  # [B, N, 5]
        batch_size = input.size(0)
        loss = l1_dist_rotate(input.flatten(0,1), target.flatten(0,1), aligned=True, cal_sum=False, angle_cycle=angle_cycle)  # 调用 l1_dist_rotate
        if weight is not None:
            loss = loss.view(batch_size, -1, 5) * weight
            return loss.view(batch_size, -1)
        return loss.view(batch_size, -1)
    
    elif input.dim() == 2:  # [N, 5]
        loss = l1_dist_rotate(input, target, aligned=True, cal_sum=False, angle_cycle=angle_cycle)  # 扩展批次维度
        if weight is not None:
            loss = loss.view(1, -1, 5) * weight
            return loss.view(-1, 5)
        return loss
    
    else:
        raise ValueError("输入的张量必须是 2D 或 3D 张量")
    

def loss_rotated_iou_norm_bboxes1(bboxes1: torch.Tensor,
                                bboxes2: torch.Tensor,
                                img_shape: torch.Tensor,
                                version: str = 'le135',
                                use_probiou: bool = True,
                                ) -> torch.Tensor:
    """
    计算逐对旋转框 IoU，用于训练 loss（一一对应，无需 batch 版）。

    Args:
        bboxes1: (N, 5) 预测框，归一化 [cx_norm, cy_norm, w_norm, h_norm, angle_norm]。
        bboxes2: (N, 5) GT 框，绝对 xywhr（像素 + 弧度），与 bboxes1 逐对对应。
        img_shape: (h, w)。
        version: 'le135' / 'le90'，角度归一化方式。
        use_probiou: True 使用 Prob IoU（默认）；False 使用 mmcv diff_iou_rotated_2d。

    Returns:
        (N,) 每对的 IoU。
    """
    if type(version) != str:
        version = version_index_to_str(version)
    if version == 'oc':
        raise NotImplementedError
    elif version == 'le135':
        angle_range = 1
        angle_offset = -1/4
    elif version == 'le90':
        angle_range = 1
        angle_offset = -1/2
    angle_range *= math.pi
    angle_offset *= math.pi
    h, w = img_shape
    bboxes1 = bboxes1 * torch.as_tensor([w, h, w, h, angle_range], dtype=bboxes1.dtype, device=bboxes1.device) + torch.as_tensor([0, 0, 0, 0, angle_offset], dtype=bboxes1.dtype, device=bboxes1.device)

    if use_probiou:
        ious = probiou(bboxes1, bboxes2, CIoU=False, eps=1e-7)
    else:
        ious = diff_iou_rotated_2d(bboxes1.unsqueeze(0), bboxes2.unsqueeze(0)).squeeze(0)
    return ious

class ConvexGIoULossFuction(Function):
    """The function of Convex GIoU loss.
    需要输入的pred是[n,8], target是[n,5]"""

    @staticmethod
    def forward(ctx,
                pred,
                target,
                weight=None,
                reduction=None,
                avg_factor=None,
                loss_weight=1.0):
        """Forward function.

        Args:
            ctx:  {save_for_backward, convex_points_grad}
            pred (torch.Tensor): Predicted convexes.
            target (torch.Tensor): Corresponding gt convexes.
            weight (torch.Tensor, optional): The weight of loss for each
                prediction. Defaults to None.
            reduction (str, optional): The reduction method of the
            loss. Defaults to None.
            avg_factor (int, optional): Average factor that is used to average
                the loss. Defaults to None.
            loss_weight (float, optional): The weight of loss. Defaults to 1.0.
        """
        ctx.save_for_backward(pred)
        convex_gious, grad = convex_giou(pred, target)

        loss = 1 - convex_gious
        if weight is not None:
            loss = loss * weight
            grad = grad * weight.reshape(-1, 1)
        if reduction == 'sum':
            loss = loss.sum()
        elif reduction == 'mean':
            loss = loss.mean()

        unvaild_inds = torch.nonzero((grad > 1).sum(1), as_tuple=False)[:, 0]
        grad[unvaild_inds] = 1e-6

        # _reduce_grad
        reduce_grad = -grad / grad.size(0) * loss_weight
        ctx.convex_points_grad = reduce_grad
        return loss

    @staticmethod
    @once_differentiable
    def backward(ctx, input=None):
        """Backward function."""
        convex_points_grad = ctx.convex_points_grad
        return convex_points_grad, None, None, None, None, None

convex_giou_loss = ConvexGIoULossFuction.apply

# 示 例
# @ROTATED_LOSSES.register_module()
# class ConvexGIoULoss(nn.Module):
#     """Convex GIoU loss.

#     Computing the Convex GIoU loss between a set of predicted convexes and
#     target convexes.
#     Args:
#         reduction (str, optional): The reduction method of the loss. Defaults
#             to 'mean'.
#         loss_weight (float, optional): The weight of loss. Defaults to 1.0.
#     Return:
#         torch.Tensor: Loss tensor.
#     """

#     def __init__(self, reduction='mean', loss_weight=1.0):
#         super(ConvexGIoULoss, self).__init__()
#         self.reduction = reduction
#         self.loss_weight = loss_weight

#     def forward(self,
#                 pred,
#                 target,
#                 weight=None,
#                 avg_factor=None,
#                 reduction_override=None,
#                 **kwargs):
#         """Forward function.

#         Args:
#             pred (torch.Tensor): Predicted convexes.
#             target (torch.Tensor): Corresponding gt convexes.
#             weight (torch.Tensor, optional): The weight of loss for each
#                 prediction. Defaults to None.
#             avg_factor (int, optional): Average factor that is used to average
#                 the loss. Defaults to None.
#             reduction_override (str, optional): The reduction method used to
#                 override the original reduction method of the loss.
#                 Defaults to None.
#         """
#         if weight is not None and not torch.any(weight > 0):
#             return (pred * weight.unsqueeze(-1)).sum()  # 0
#         assert reduction_override in (None, 'none', 'mean', 'sum')
#         reduction = (
#             reduction_override if reduction_override else self.reduction)
#         loss = self.loss_weight * convex_giou_loss(
#             pred, target, weight, reduction, avg_factor, self.loss_weight)
#         return loss
