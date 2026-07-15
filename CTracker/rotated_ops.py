"""Tensor utilities for CTracker's paired rotated boxes."""

import math

import torch

from mmcv.ops import box_iou_rotated
from mmrotate.models.losses import GDLoss
from mmrotate.structures.bbox import RotatedBoxes, rbox2qbox


BOX_STDS = (0.1, 0.1, 0.2, 0.2, 1.0)
ANGLE_NORM_FACTOR = 0.5


def regularize_rboxes(boxes):
    if boxes.numel() == 0:
        return boxes.reshape(-1, 5)
    return RotatedBoxes(boxes).regularize_boxes(pattern='le90')


def rboxes_to_hboxes(boxes):
    if boxes.numel() == 0:
        return boxes.new_zeros((0, 4))
    return RotatedBoxes(boxes).convert_to('hbox').tensor


def hbox_iou(anchors, boxes):
    if anchors.numel() == 0 or boxes.numel() == 0:
        return anchors.new_zeros((anchors.size(0), boxes.size(0)))
    area_b = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    iw = (torch.minimum(anchors[:, None, 2], boxes[None, :, 2]) -
          torch.maximum(anchors[:, None, 0], boxes[None, :, 0])).clamp_min(0)
    ih = (torch.minimum(anchors[:, None, 3], boxes[None, :, 3]) -
          torch.maximum(anchors[:, None, 1], boxes[None, :, 1])).clamp_min(0)
    intersection = iw * ih
    area_a = ((anchors[:, 2] - anchors[:, 0]) *
              (anchors[:, 3] - anchors[:, 1]))[:, None]
    return intersection / (area_a + area_b[None] - intersection).clamp_min(1e-8)


def encode_hboxes_to_rboxes(anchors, targets):
    """Encode le90 rotated targets relative to horizontal xyxy anchors."""
    targets = regularize_rboxes(targets)
    px = (anchors[:, 0] + anchors[:, 2]) * 0.5
    py = (anchors[:, 1] + anchors[:, 3]) * 0.5
    pw = (anchors[:, 2] - anchors[:, 0]).clamp_min(1e-6)
    ph = (anchors[:, 3] - anchors[:, 1]).clamp_min(1e-6)
    gx, gy, gw, gh, angle = targets.unbind(dim=-1)
    deltas = torch.stack((
        (gx - px) / pw,
        (gy - py) / ph,
        torch.log(gw.clamp_min(1e-6) / pw),
        torch.log(gh.clamp_min(1e-6) / ph),
        angle / (ANGLE_NORM_FACTOR * math.pi),
    ), dim=-1)
    return deltas / deltas.new_tensor(BOX_STDS)


def decode_hboxes_to_rboxes(anchors, deltas):
    """Decode deltas with arbitrary leading dimensions ending in N x 5."""
    denorm = deltas * deltas.new_tensor(BOX_STDS)
    px = (anchors[..., 0] + anchors[..., 2]) * 0.5
    py = (anchors[..., 1] + anchors[..., 3]) * 0.5
    pw = (anchors[..., 2] - anchors[..., 0]).clamp_min(1e-6)
    ph = (anchors[..., 3] - anchors[..., 1]).clamp_min(1e-6)
    dx, dy, dw, dh, dt = denorm.unbind(dim=-1)
    boxes = torch.stack((
        px + dx * pw,
        py + dy * ph,
        torch.exp(dw.clamp(max=math.log(1000.0 / 16))) * pw,
        torch.exp(dh.clamp(max=math.log(1000.0 / 16))) * ph,
        dt * (ANGLE_NORM_FACTOR * math.pi),
    ), dim=-1)
    flat = regularize_rboxes(boxes.reshape(-1, 5))
    return flat.reshape_as(boxes)


def aligned_rotated_iou(boxes1, boxes2):
    if boxes1.numel() == 0:
        return boxes1.new_zeros((0,))
    return box_iou_rotated(
        boxes1.float(), boxes2.float(), aligned=True).to(boxes1.dtype)


class RotatedKLDLoss(torch.nn.Module):
    def __init__(self, loss_weight=2.0):
        super().__init__()
        self.loss = GDLoss(
            loss_type='kld', fun='log1p', tau=1, sqrt=False,
            loss_weight=loss_weight)

    def forward(self, pred, target):
        if pred.numel() == 0:
            return pred.sum()
        pred = pred.float()
        target = regularize_rboxes(target.float())
        pred = torch.cat((pred[:, :2], pred[:, 2:4].clamp_min(1e-3),
                          pred[:, 4:5]), dim=-1)
        target = torch.cat((target[:, :2], target[:, 2:4].clamp_min(1e-3),
                            target[:, 4:5]), dim=-1)
        return self.loss(pred, target)


def rboxes_to_qboxes(boxes):
    return rbox2qbox(regularize_rboxes(boxes))


def multiclass_rotated_soft_nms(paired_boxes, scores, labels, sigma=0.5,
                                score_threshold=0.001):
    """Gaussian Soft-NMS on the emitted (first) box of each pair."""
    output_boxes = []
    output_scores = []
    output_labels = []
    for class_id in labels.unique(sorted=True):
        mask = labels == class_id
        boxes = paired_boxes[mask].clone()
        class_scores = scores[mask].clone()
        while class_scores.numel():
            max_index = class_scores.argmax()
            best_box = boxes[max_index].clone()
            best_score = class_scores[max_index].clone()
            output_boxes.append(best_box)
            output_scores.append(best_score)
            output_labels.append(class_id)
            keep = torch.ones(
                class_scores.size(0), dtype=torch.bool,
                device=class_scores.device)
            keep[max_index] = False
            boxes = boxes[keep]
            class_scores = class_scores[keep]
            if class_scores.numel() == 0:
                break
            overlaps = box_iou_rotated(
                best_box[None, :5].float(), boxes[:, :5].float()).squeeze(0)
            class_scores *= torch.exp(-(overlaps * overlaps) / sigma)
            keep = class_scores >= score_threshold
            boxes = boxes[keep]
            class_scores = class_scores[keep]

    if not output_scores:
        return (paired_boxes.new_zeros((0, 10)), scores.new_zeros((0,)),
                labels.new_zeros((0,), dtype=torch.long))
    output_boxes = torch.stack(output_boxes)
    output_scores = torch.stack(output_scores)
    output_labels = torch.stack(output_labels).long()
    order = output_scores.argsort(descending=True)
    return output_boxes[order], output_scores[order], output_labels[order]
