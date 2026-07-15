"""Losses for multi-class paired rotated CTracker training."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from rotated_ops import (RotatedKLDLoss, decode_hboxes_to_rboxes,
                         encode_hboxes_to_rboxes, hbox_iou,
                         rboxes_to_hboxes)


def _focal_loss(prediction, targets, alpha=0.25, gamma=2.0):
    prediction = prediction.clamp(1e-4, 1 - 1e-4)
    alpha_factor = torch.full_like(targets, alpha)
    alpha_factor = torch.where(targets == 1, alpha_factor, 1 - alpha_factor)
    focal_weight = torch.where(targets == 1, 1 - prediction, prediction)
    focal_weight = alpha_factor * focal_weight.pow(gamma)
    bce = -(targets * prediction.log() +
            (1 - targets) * (1 - prediction).log())
    loss = focal_weight * bce
    return torch.where(targets != -1, loss, torch.zeros_like(loss))


def _assign_hbox_anchors(anchors, gt_rboxes):
    overlaps = hbox_iou(anchors, rboxes_to_hboxes(gt_rboxes))
    best_overlap, best_index = overlaps.max(dim=1)
    _, best_anchor = overlaps.max(dim=0)
    best_overlap.index_fill_(0, best_anchor, 2)
    for gt_index, anchor_index in enumerate(best_anchor):
        best_index[anchor_index] = gt_index
    return best_overlap, best_index


class RotatedCTrackerLoss(nn.Module):
    def __init__(self, num_classes, delta_weight=5.0, kld_weight=2.0):
        super().__init__()
        self.num_classes = num_classes
        self.delta_weight = delta_weight
        self.kld_loss = RotatedKLDLoss(loss_weight=kld_weight)

    def _association_loss(self, prediction, anchors, target):
        prev_mask = target['valid_prev']
        curr_mask = target['valid_curr']
        if not prev_mask.any() or not curr_mask.any():
            return prediction.sum() * 0

        prev_boxes = target['bboxes_prev'][prev_mask]
        curr_boxes = target['bboxes_curr'][curr_mask]
        prev_ids = target['track_ids'][prev_mask]
        curr_ids = target['track_ids'][curr_mask]
        prev_overlap, prev_index = _assign_hbox_anchors(anchors, prev_boxes)
        curr_overlap, curr_index = _assign_hbox_anchors(anchors, curr_boxes)
        assigned_prev_ids = prev_ids[prev_index]
        assigned_curr_ids = curr_ids[curr_index]

        targets = prediction.new_full(prediction.shape, -1)
        valid = ((prev_overlap >= 0.5) & (curr_overlap >= 0.4)) | (
            (curr_overlap >= 0.5) & (prev_overlap >= 0.4))
        targets[valid, 0] = (
            assigned_prev_ids[valid] == assigned_curr_ids[valid]).float()
        targets[(prev_overlap < 0.4) | (curr_overlap < 0.4), 0] = 0
        ambiguous = ((prev_overlap >= 0.4) & (prev_overlap < 0.5) &
                     (curr_overlap >= 0.4) & (curr_overlap < 0.5))
        targets[ambiguous, 0] = -1
        positives = (targets == 1).sum().clamp_min(1).float()
        return _focal_loss(prediction, targets).sum() / positives

    def forward(self, classifications, regressions, associations, anchors,
                targets):
        anchor = anchors[0]
        cls_losses = []
        delta_losses = []
        kld_losses = []
        association_losses = []

        for batch_index, target in enumerate(targets):
            classification = classifications[batch_index]
            regression = regressions[batch_index]
            association = associations[batch_index]
            target = {
                key: value.to(classification.device)
                for key, value in target.items()
            }
            prev_valid = target['valid_prev']
            if not prev_valid.any():
                zero = classification.sum() * 0
                cls_losses.append(zero)
                delta_losses.append(zero)
                kld_losses.append(zero)
                association_losses.append(zero)
                continue

            prev_boxes = target['bboxes_prev'][prev_valid]
            prev_labels = target['labels'][prev_valid]
            # Pair GT rows stay aligned, so map filtered prev indices back to
            # the union row before looking up current visibility and boxes.
            union_indices = torch.where(prev_valid)[0]
            overlap, assigned_index = _assign_hbox_anchors(anchor, prev_boxes)
            assigned_union = union_indices[assigned_index]
            positive = overlap >= 0.5

            cls_targets = classification.new_full(classification.shape, -1)
            cls_targets[overlap < 0.4] = 0
            assigned_labels = prev_labels[assigned_index]
            cls_targets[positive] = 0
            cls_targets[positive, assigned_labels[positive]] = 1
            positive_count = positive.sum().clamp_min(1).float()
            cls_losses.append(
                _focal_loss(classification, cls_targets).sum() /
                positive_count)
            association_losses.append(
                self._association_loss(association, anchor, target))

            if not positive.any():
                zero = regression.sum() * 0
                delta_losses.append(zero)
                kld_losses.append(zero)
                continue

            positive_anchors = anchor[positive]
            positive_rows = assigned_union[positive]
            prev_targets = target['bboxes_prev'][positive_rows]
            curr_valid = target['valid_curr'][positive_rows]
            curr_targets = target['bboxes_curr'][positive_rows]

            prev_delta = encode_hboxes_to_rboxes(
                positive_anchors, prev_targets)
            delta_loss = F.smooth_l1_loss(
                regression[positive, :5], prev_delta,
                beta=1.0 / 9.0, reduction='mean')
            decoded_prev = decode_hboxes_to_rboxes(
                positive_anchors, regression[positive, :5])
            kld_loss = self.kld_loss(decoded_prev, prev_targets)

            if curr_valid.any():
                curr_anchors = positive_anchors[curr_valid]
                curr_boxes = curr_targets[curr_valid]
                curr_delta = encode_hboxes_to_rboxes(curr_anchors, curr_boxes)
                delta_loss = delta_loss + F.smooth_l1_loss(
                    regression[positive, 5:][curr_valid], curr_delta,
                    beta=1.0 / 9.0, reduction='mean')
                decoded_curr = decode_hboxes_to_rboxes(
                    curr_anchors, regression[positive, 5:][curr_valid])
                kld_loss = kld_loss + self.kld_loss(decoded_curr, curr_boxes)

            delta_losses.append(delta_loss * self.delta_weight)
            kld_losses.append(kld_loss)

        return dict(
            loss_cls=torch.stack(cls_losses).mean(),
            loss_delta=torch.stack(delta_losses).mean(),
            loss_kld=torch.stack(kld_losses).mean(),
            loss_assoc=torch.stack(association_losses).mean(),
        )
