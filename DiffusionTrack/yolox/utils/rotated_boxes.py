"""Rotated-box geometry used by the HSMOT DiffusionTrack adaptation.

Public tensors use long-edge-135 ``(cx, cy, w, h, theta)`` boxes with theta
in ``[-pi/4, 3pi/4)`` radians.  Image-coordinate angles match MMCV's
``clockwise=True`` convention consistently for ROIAlign, IoU and NMS.
"""

import math

import numpy as np
import torch
from mmcv.ops import box_iou_rotated, diff_iou_rotated_2d, nms_rotated


# MMCV's rotated CUDA kernels are undefined for zero-area rectangles.  The
# diffusion parameterisation can legitimately produce a width or height of
# 1e-6 after clipping, and box_iou_rotated has been observed to return IoU=1
# for two such boxes hundreds of pixels apart.  Keep those proposals in the
# network (a later refinement head may recover them), but never let their
# undefined geometry participate in matching, loss, or NMS.
_MIN_VALID_SIDE = 1e-4


def valid_rbox_mask(boxes):
    """Return rows on which rotated geometry operators are well-defined."""
    return (torch.isfinite(boxes).all(dim=-1)
            & (boxes[..., 2] > _MIN_VALID_SIDE)
            & (boxes[..., 3] > _MIN_VALID_SIDE))


def _safe_rbox_geometry(boxes):
    """Replace invalid rows before calling an MMCV rotated CUDA kernel."""
    finite = torch.nan_to_num(boxes.float(), nan=0.0,
                              posinf=0.0, neginf=0.0)
    safe = torch.cat(
        [finite[..., :2],
         finite[..., 2:4].clamp_min(_MIN_VALID_SIDE),
         finite[..., 4:5]], dim=-1)
    dummy = safe.new_tensor([0.0, 0.0, 1.0, 1.0, 0.0]).expand_as(safe)
    return torch.where(valid_rbox_mask(boxes)[..., None], safe, dummy)


def regularize_rboxes(boxes):
    """Canonicalize rotated boxes to long-edge-135 representation.

    LE135 has two requirements: the angle lies in ``[-pi/4, 3pi/4)`` and
    ``w >= h``.  Merely wrapping the angle leaves two parameterizations for
    the same rectangle, which is harmless for IoU but invalidates direct L1
    regression on ``(w, h, theta)``.
    """
    center = boxes[..., :2]
    width, height = boxes[..., 2], boxes[..., 3]
    swap = width < height
    long_edge = torch.where(swap, height, width)
    short_edge = torch.where(swap, width, height)
    angle = boxes[..., 4] + swap.to(boxes.dtype) * (math.pi / 2)
    angle = torch.remainder(angle + math.pi / 4, math.pi) - math.pi / 4
    return torch.cat(
        [center, long_edge.unsqueeze(-1), short_edge.unsqueeze(-1),
         angle.unsqueeze(-1)], dim=-1)


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
    valid1 = valid_rbox_mask(boxes1)
    valid2 = valid_rbox_mask(boxes2)
    overlaps = box_iou_rotated(
        _safe_rbox_geometry(boxes1), _safe_rbox_geometry(boxes2),
        aligned=aligned, clockwise=True)
    # MMCV can emit NaN for degenerate/extreme random diffusion proposals.
    # Such proposals have no useful overlap and must not poison SimOTA costs.
    # The CUDA kernel can return finite values outside the mathematical IoU
    # range for extremely large/degenerate diffusion proposals.  Such values
    # must not create false matches or corrupt Dynamic-K costs.
    overlaps = torch.nan_to_num(
        overlaps, nan=0.0, posinf=0.0, neginf=0.0).clamp_(0, 1)
    valid_pairs = (valid1 & valid2 if aligned
                   else valid1[:, None] & valid2[None, :])
    return overlaps.masked_fill(~valid_pairs, 0)


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


def mean_pair_rotated_iou(ref_a, ref_b, cur_a, cur_b):
    """Mean of the two frame-wise rotated IoUs for pair assignment.

    Keep this value in [0, 1] because SimOTA uses it both as an assignment
    cost and to derive Dynamic-K.  Unlike ``pair_rotated_iou``, neither
    frame's union appears in the other frame's IoU term.
    """
    iou_ref = rotated_iou(ref_a, ref_b)
    iou_cur = rotated_iou(cur_a, cur_b)
    return (iou_ref + iou_cur) * 0.5


def aligned_rotated_iou(pred, target):
    """Differentiable ordinary rotated IoU for aligned LE135 boxes."""
    if pred.numel() == 0:
        return pred.new_zeros((0,))
    valid = valid_rbox_mask(pred) & valid_rbox_mask(target)
    safe_pred = _safe_rbox_geometry(pred)
    safe_target = _safe_rbox_geometry(target)
    iou = diff_iou_rotated_2d(safe_pred[None], safe_target[None])[0]
    iou = torch.nan_to_num(
        iou, nan=0.0, posinf=0.0, neginf=0.0).clamp(0, 1)
    # MMCV's differentiable polygon-intersection kernel returns exactly zero
    # when all four edges coincide (and in a very small neighbourhood around
    # that topology), whereas the ordinary rotated IoU is one.  LE135 gives
    # every rectangle a unique parameterization, so parameter-wise
    # coincidence is an unambiguous geometric identity.  Restore the correct
    # optimum and its zero gradient without changing the IoU or gradients for
    # any materially different boxes.
    side_scale = torch.maximum(
        safe_pred[..., 2:4].amax(dim=-1),
        safe_target[..., 2:4].amax(dim=-1)).clamp_min(1.0)
    xywh_close = (
        (safe_pred[..., :4] - safe_target[..., :4]).abs()
        <= (1e-5 * side_scale)[..., None]).all(dim=-1)
    angle_delta = torch.remainder(
        safe_pred[..., 4] - safe_target[..., 4] + math.pi / 2,
        math.pi) - math.pi / 2
    coincident = xywh_close & (angle_delta.abs() <= 1e-5)
    iou = torch.where(coincident, torch.ones_like(iou), iou)
    return torch.where(valid, iou, torch.zeros_like(iou))


def aligned_pair_rotated_iou(ref_pred, ref_target, cur_pred, cur_target):
    """Differentiable volume-style IoU for aligned paired le135 boxes."""
    if ref_pred.numel() == 0:
        return ref_pred.new_zeros((0,))
    valid = (valid_rbox_mask(ref_pred) & valid_rbox_mask(ref_target)
             & valid_rbox_mask(cur_pred) & valid_rbox_mask(cur_target))
    # Feeding degenerate boxes to the differentiable CUDA kernel can create
    # invalid forward values and gradients even if the result is masked
    # afterwards.  Substitute finite dummy geometry for invalid rows first;
    # torch.where then gives those rows exactly zero loss contribution.
    safe_ref_pred = _safe_rbox_geometry(ref_pred)
    safe_ref_target = _safe_rbox_geometry(ref_target)
    safe_cur_pred = _safe_rbox_geometry(cur_pred)
    safe_cur_target = _safe_rbox_geometry(cur_target)
    # diff_iou_rotated_2d uses MMCV's clockwise image-coordinate convention.
    iou_ref = diff_iou_rotated_2d(
        safe_ref_pred[None], safe_ref_target[None])[0]
    iou_cur = diff_iou_rotated_2d(
        safe_cur_pred[None], safe_cur_target[None])[0]
    area_ref_pred = safe_ref_pred[:, 2] * safe_ref_pred[:, 3]
    area_ref_target = safe_ref_target[:, 2] * safe_ref_target[:, 3]
    area_cur_pred = safe_cur_pred[:, 2] * safe_cur_pred[:, 3]
    area_cur_target = safe_cur_target[:, 2] * safe_cur_target[:, 3]
    inter_ref = iou_ref * (area_ref_pred + area_ref_target) / (1 + iou_ref)
    inter_cur = iou_cur * (area_cur_pred + area_cur_target) / (1 + iou_cur)
    union_ref = area_ref_pred + area_ref_target - inter_ref
    union_cur = area_cur_pred + area_cur_target - inter_cur
    result = (inter_ref + inter_cur) / (
        union_ref + union_cur).clamp_min(1e-6)
    result = torch.nan_to_num(
        result, nan=0.0, posinf=0.0, neginf=0.0).clamp(0, 1)
    return torch.where(valid, result, torch.zeros_like(result))


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
