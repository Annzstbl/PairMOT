"""Rotated-box geometry used by the HSMOT DiffusionTrack adaptation.

Public tensors use long-edge-135 ``(cx, cy, w, h, theta)`` boxes with theta
in ``[-pi/4, 3pi/4)`` radians.  Image-coordinate angles match MMCV's
``clockwise=True`` convention consistently for ROIAlign, IoU and NMS.
"""

import math

import numpy as np
import torch
from mmcv.ops import box_iou_rotated, diff_iou_rotated_2d, nms_rotated


def regularize_rboxes(boxes):
    """Regularize long-edge boxes to the le135 angle interval."""
    result = boxes.clone()
    result[..., 4] = torch.remainder(
        result[..., 4] + math.pi / 4, math.pi) - math.pi / 4
    return result


def qbox_to_rbox(qboxes):
    """Convert ordered quadrilaterals ``[..., 8]`` to le135 rboxes."""
    is_numpy = isinstance(qboxes, np.ndarray)
    boxes = torch.as_tensor(qboxes, dtype=torch.float32)
    shape = boxes.shape[:-1]
    points = boxes.reshape(-1, 4, 2)
    center = points.mean(dim=1)
    edge01 = points[:, 1] - points[:, 0]
    edge12 = points[:, 2] - points[:, 1]
    len01 = torch.linalg.vector_norm(edge01, dim=1)
    len12 = torch.linalg.vector_norm(edge12, dim=1)
    use01 = len01 >= len12
    width = torch.where(use01, len01, len12).clamp_min(1e-6)
    height = torch.where(use01, len12, len01).clamp_min(1e-6)
    direction = torch.where(use01[:, None], edge01, edge12)
    angle = torch.atan2(direction[:, 1], direction[:, 0])
    angle = torch.remainder(angle + math.pi / 4, math.pi) - math.pi / 4
    result = torch.cat(
        [center, width[:, None], height[:, None], angle[:, None]], dim=1)
    result = result.reshape(*shape, 5)
    return result.cpu().numpy() if is_numpy else result.to(qboxes.device)


def rbox_to_qbox(rboxes):
    """Convert rboxes ``[..., 5]`` to four clockwise corner points."""
    is_numpy = isinstance(rboxes, np.ndarray)
    boxes = torch.as_tensor(rboxes, dtype=torch.float32)
    shape = boxes.shape[:-1]
    flat = boxes.reshape(-1, 5)
    cx, cy, w, h, angle = flat.unbind(dim=1)
    local = flat.new_tensor([[-0.5, -0.5], [0.5, -0.5],
                             [0.5, 0.5], [-0.5, 0.5]])
    local = local[None].repeat(len(flat), 1, 1)
    local[..., 0] *= w[:, None]
    local[..., 1] *= h[:, None]
    cos_a, sin_a = torch.cos(angle), torch.sin(angle)
    rotation = torch.stack([cos_a, -sin_a, sin_a, cos_a], dim=1)
    rotation = rotation.reshape(-1, 2, 2)
    points = torch.bmm(local, rotation.transpose(1, 2))
    points += torch.stack([cx, cy], dim=1)[:, None]
    result = points.reshape(*shape, 8)
    return result.cpu().numpy() if is_numpy else result.to(rboxes.device)


def rotated_iou(boxes1, boxes2, aligned=False):
    if boxes1.numel() == 0 or boxes2.numel() == 0:
        shape = (boxes1.size(0),) if aligned else (
            boxes1.size(0), boxes2.size(0))
        return boxes1.new_zeros(shape)
    overlaps = box_iou_rotated(
        boxes1.float(), boxes2.float(), aligned=aligned, clockwise=True)
    # MMCV can emit NaN for degenerate/extreme random diffusion proposals.
    # Such proposals have no useful overlap and must not poison SimOTA costs.
    # The CUDA kernel can return finite values outside the mathematical IoU
    # range for extremely large/degenerate diffusion proposals.  Such values
    # must not create false matches or corrupt Dynamic-K costs.
    return torch.nan_to_num(
        overlaps, nan=0.0, posinf=0.0, neginf=0.0).clamp_(0, 1)


def batched_rotated_nms(boxes, scores, labels, iou_threshold):
    """Class-aware rotated NMS with deterministic score ordering."""
    keep = []
    for label in labels.unique(sorted=True):
        indices = torch.where(labels == label)[0]
        _, local_keep = nms_rotated(
            boxes[indices].float(), scores[indices].float(), iou_threshold,
            clockwise=True)
        keep.append(indices[local_keep])
    if not keep:
        return labels.new_zeros((0,), dtype=torch.long)
    keep = torch.cat(keep)
    return keep[torch.argsort(scores[keep], descending=True)]


def pair_rotated_iou(ref_a, ref_b, cur_a, cur_b):
    """Volume-style IoU for two sets of paired rotated boxes."""
    iou_ref = rotated_iou(ref_a, ref_b)
    iou_cur = rotated_iou(cur_a, cur_b)
    area_ref_a = (ref_a[:, 2] * ref_a[:, 3])[:, None]
    area_ref_b = (ref_b[:, 2] * ref_b[:, 3])[None]
    area_cur_a = (cur_a[:, 2] * cur_a[:, 3])[:, None]
    area_cur_b = (cur_b[:, 2] * cur_b[:, 3])[None]
    inter_ref = iou_ref * (area_ref_a + area_ref_b) / (1 + iou_ref)
    inter_cur = iou_cur * (area_cur_a + area_cur_b) / (1 + iou_cur)
    union_ref = area_ref_a + area_ref_b - inter_ref
    union_cur = area_cur_a + area_cur_b - inter_cur
    result = (inter_ref + inter_cur) / (
        union_ref + union_cur).clamp_min(1e-6)
    return torch.nan_to_num(result, nan=0.0, posinf=0.0, neginf=0.0).clamp_(0, 1)


def aligned_pair_rotated_iou(ref_pred, ref_target, cur_pred, cur_target):
    """Differentiable volume-style IoU for aligned paired le135 boxes."""
    if ref_pred.numel() == 0:
        return ref_pred.new_zeros((0,))
    # diff_iou_rotated_2d uses MMCV's clockwise image-coordinate convention.
    iou_ref = diff_iou_rotated_2d(
        ref_pred.float()[None], ref_target.float()[None])[0]
    iou_cur = diff_iou_rotated_2d(
        cur_pred.float()[None], cur_target.float()[None])[0]
    area_ref_pred = ref_pred[:, 2].float() * ref_pred[:, 3].float()
    area_ref_target = ref_target[:, 2].float() * ref_target[:, 3].float()
    area_cur_pred = cur_pred[:, 2].float() * cur_pred[:, 3].float()
    area_cur_target = cur_target[:, 2].float() * cur_target[:, 3].float()
    inter_ref = iou_ref * (area_ref_pred + area_ref_target) / (1 + iou_ref)
    inter_cur = iou_cur * (area_cur_pred + area_cur_target) / (1 + iou_cur)
    union_ref = area_ref_pred + area_ref_target - inter_ref
    union_cur = area_cur_pred + area_cur_target - inter_cur
    result = (inter_ref + inter_cur) / (
        union_ref + union_cur).clamp_min(1e-6)
    return torch.nan_to_num(result, nan=0.0, posinf=0.0, neginf=0.0).clamp(0, 1)


def pair_cluster_nms_rotated(ref_boxes, cur_boxes, scores,
                             iou_threshold=0.5, top_k=500):
    """Original DiffusionTrack cluster-NMS logic using paired rotated IoU."""
    order = scores.argsort(descending=True)[:top_k]
    pair_iou = pair_rotated_iou(
        ref_boxes[order], ref_boxes[order],
        cur_boxes[order], cur_boxes[order]).triu(diagonal=1)
    propagated = pair_iou
    for _ in range(200):
        previous = propagated
        max_overlap = previous.max(dim=0).values
        eligible = (max_overlap <= iou_threshold).to(previous.dtype)
        propagated = pair_iou * eligible.unsqueeze(1)
        if torch.equal(previous, propagated):
            break
    return order[max_overlap <= iou_threshold]
