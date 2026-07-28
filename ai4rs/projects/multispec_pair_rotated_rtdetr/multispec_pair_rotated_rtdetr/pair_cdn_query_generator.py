# Copyright (c) AI4RS. All rights reserved.
"""Contrastive denoising queries for paired rotated detection."""

import math
from typing import Dict, List, Optional, Tuple

import torch
from mmdet.models.layers.transformer.utils import inverse_sigmoid
from mmdet.structures import SampleList
from torch import Tensor, nn

from mmrotate.structures.bbox import (RotatedBoxes, qbox2rbox,
                                     rbbox_overlaps)

from .rotated_box_utils import canonicalize_le180_start0


class PairCdnQueryGenerator(nn.Module):
    """Generate DINO-style DN queries from pair-level track unions.

    One DN target represents one track id. It has a shared class query and
    separate noisy references for the previous and current frame. Missing
    sides retain a neutral reference and receive no box supervision. They are
    supervised as absent by either the presence branch or, for a dual-cls head
    without presence branches, the corresponding background class target.
    """

    def __init__(self,
                 num_classes: int,
                 embed_dims: int,
                 num_matching_queries: int,
                 label_noise_scale: float = 0.5,
                 box_noise_scale: float = 1.0,
                 angle_factor: float = torch.pi,
                 angle_cfg: Dict = None,
                 positive_hard_ratio: float = 0.75,
                 positive_hard_min_magnitude: float = 0.5,
                 positive_hard_max_magnitude: float = 1.25,
                 share_pair_noise: bool = True,
                 dn_target_mode: str = 'positive_negative',
                 negative_ratio: float = 0.5,
                 negative_min_magnitude: float = 0.75,
                 negative_max_magnitude: float = 1.5,
                 negative_max_iou: float = 0.4,
                 negative_resample_attempts: int = 4,
                 group_cfg: Dict = None) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.embed_dims = embed_dims
        self.num_matching_queries = num_matching_queries
        self.label_noise_scale = label_noise_scale
        self.box_noise_scale = box_noise_scale
        self.angle_factor = float(angle_factor)
        self.angle_cfg = dict(
            width_longer=True, start_angle=0) if angle_cfg is None else dict(
                angle_cfg)
        if (not self.angle_cfg.get('width_longer', True)
                or float(self.angle_cfg.get('start_angle', 0)) != 0):
            raise ValueError(
                'PairCdnQueryGenerator requires LE180-start0: '
                'width_longer=True, start_angle=0')
        self.positive_hard_ratio = float(positive_hard_ratio)
        self.positive_hard_min_magnitude = float(
            positive_hard_min_magnitude)
        self.positive_hard_max_magnitude = float(
            positive_hard_max_magnitude)
        self.share_pair_noise = bool(share_pair_noise)
        self.dn_target_mode = str(dn_target_mode)
        self.negative_ratio = float(negative_ratio)
        self.negative_min_magnitude = float(negative_min_magnitude)
        self.negative_max_magnitude = float(negative_max_magnitude)
        self.negative_max_iou = float(negative_max_iou)
        self.negative_resample_attempts = int(negative_resample_attempts)
        if not 0 <= self.positive_hard_ratio <= 1:
            raise ValueError('positive_hard_ratio must be in [0, 1]')
        if self.dn_target_mode not in {
                'positive_negative', 'easy_hard_positive'}:
            raise ValueError(
                'dn_target_mode must be positive_negative or '
                'easy_hard_positive')
        if not (0 <= self.positive_hard_min_magnitude
                < self.positive_hard_max_magnitude):
            raise ValueError('Invalid positive-hard magnitude interval')
        if not 0 <= self.negative_ratio <= 1:
            raise ValueError('negative_ratio must be in [0, 1]')
        if not 0 <= self.negative_min_magnitude < self.negative_max_magnitude:
            raise ValueError('Invalid negative magnitude interval')
        if not 0 <= self.negative_max_iou < 1:
            raise ValueError('negative_max_iou must be in [0, 1)')
        if self.negative_resample_attempts < 1:
            raise ValueError('negative_resample_attempts must be positive')
        group_cfg = {} if group_cfg is None else group_cfg
        self.dynamic_dn_groups = group_cfg.get('dynamic', True)
        self.num_dn_queries = group_cfg.get('num_dn_queries', 100)
        self.num_groups = group_cfg.get('num_groups', 1)
        self.label_embedding = nn.Embedding(num_classes, embed_dims)

    def _to_rbox(self, bboxes) -> Tensor:
        if isinstance(bboxes, RotatedBoxes):
            tensor = bboxes.tensor
        else:
            if hasattr(bboxes, 'tensor'):
                bboxes = bboxes.tensor
            tensor = qbox2rbox(bboxes) if bboxes.size(-1) == 8 else bboxes
        boxes = RotatedBoxes(tensor.clone())
        boxes.regularize_boxes(**self.angle_cfg)
        return boxes.tensor

    def _get_num_groups(self, max_num_targets: int) -> int:
        if not self.dynamic_dn_groups:
            return max(1, int(self.num_groups))
        if max_num_targets == 0:
            return 1
        return max(1, int(self.num_dn_queries) // max_num_targets)

    def _num_negative_targets(self, num_targets: int) -> int:
        return min(num_targets, int(math.ceil(num_targets * self.negative_ratio)))

    def _sample_unit_noise(self, num_targets: int, device: torch.device,
                           dtype: torch.dtype, *, negative: bool,
                           positive_hard: Optional[bool] = None) -> Tensor:
        """Sample unitless noise shared by both frames of a pair."""
        if num_targets == 0:
            return torch.zeros((0, 5), device=device, dtype=dtype)
        if negative:
            magnitude = torch.empty(
                num_targets, 5, device=device, dtype=dtype).uniform_(
                    self.negative_min_magnitude,
                    self.negative_max_magnitude)
            sign = torch.randint(
                0, 2, (num_targets, 5), device=device).to(dtype) * 2 - 1
            return sign * magnitude

        normal = ((torch.rand(num_targets, 5, device=device, dtype=dtype) * 2
                   - 1) * torch.rand(
                       num_targets, 5, device=device, dtype=dtype))
        hard_magnitude = torch.empty(
            num_targets, 5, device=device, dtype=dtype).uniform_(
                self.positive_hard_min_magnitude,
                self.positive_hard_max_magnitude)
        hard_sign = torch.randint(
            0, 2, (num_targets, 5), device=device).to(dtype) * 2 - 1
        hard = hard_sign * hard_magnitude
        hard_rows = torch.rand(
            num_targets, 1, device=device) < self.positive_hard_ratio
        if positive_hard is not None:
            return hard if positive_hard else normal
        return torch.where(hard_rows, hard, normal)

    def _apply_noise(self, refs: Tensor, valid: Tensor, unit_noise: Tensor,
                     factor: Tensor) -> Tensor:
        """Apply relative noise and restore canonical LE180-start0 boxes."""
        num_targets = refs.size(0)
        if num_targets == 0:
            return refs.new_zeros((0, 5))
        noise = unit_noise.clone()
        noise[:, :4] *= refs[:, 2:4].repeat(1, 2) * self.box_noise_scale / 2
        noise[:, 4] *= self.box_noise_scale * 0.25
        out = refs + noise
        out = torch.cat([
            out[:, :4].clamp(1e-4, 1 - 1e-4),
            torch.remainder(out[:, 4:5], 1.0),
        ], dim=-1)
        out = canonicalize_le180_start0(out * factor) / factor
        out = torch.cat([
            out[:, :4].clamp(1e-4, 1 - 1e-4),
            torch.remainder(out[:, 4:5], 1.0),
        ], dim=-1)
        out[~valid] = 0.5
        return out

    def _noisy_refs(self, refs: Tensor, valid: Tensor, *, negative: bool,
                    factor: Tensor = None) -> Tensor:
        """Compatibility helper for one-frame DN noise tests/tools."""
        if factor is None:
            factor = refs.new_tensor([1, 1, 1, 1, self.angle_factor])
        unit_noise = self._sample_unit_noise(
            refs.size(0), refs.device, refs.dtype, negative=negative)
        return self._apply_noise(refs, valid, unit_noise, factor)

    def _noisy_pair_refs(self, refs_prev: Tensor, refs_curr: Tensor,
                         valid_prev: Tensor, valid_curr: Tensor, factor: Tensor,
                         *, negative: bool,
                         positive_hard: Optional[bool] = None
                         ) -> Tuple[Tensor, Tensor]:
        unit_noise_prev = self._sample_unit_noise(
            refs_prev.size(0), refs_prev.device, refs_prev.dtype,
            negative=negative, positive_hard=positive_hard)
        if self.share_pair_noise:
            unit_noise_curr = unit_noise_prev
        else:
            unit_noise_curr = self._sample_unit_noise(
                refs_curr.size(0), refs_curr.device, refs_curr.dtype,
                negative=negative, positive_hard=positive_hard)
        return (
            self._apply_noise(
                refs_prev, valid_prev, unit_noise_prev, factor),
            self._apply_noise(
                refs_curr, valid_curr, unit_noise_curr, factor),
        )

    @staticmethod
    def _max_gt_overlap(candidates: Tensor, candidate_valid: Tensor,
                        gt_refs: Tensor, gt_valid: Tensor,
                        factor: Tensor) -> Tensor:
        """Return per-candidate maximum IoU against visible same-frame GT."""
        scores = candidates.new_zeros(candidates.size(0), dtype=torch.float32)
        candidate_inds = torch.nonzero(
            candidate_valid, as_tuple=False).squeeze(-1)
        gt_inds = torch.nonzero(gt_valid, as_tuple=False).squeeze(-1)
        if candidate_inds.numel() == 0 or gt_inds.numel() == 0:
            return scores
        candidate_boxes = (candidates[candidate_inds] * factor).float()
        gt_boxes = (gt_refs[gt_inds] * factor).float()
        overlaps = rbbox_overlaps(candidate_boxes, gt_boxes)
        scores[candidate_inds] = overlaps.max(dim=1).values
        return scores

    def _sample_negative_pair_refs(
            self, source_prev: Tensor, source_curr: Tensor,
            source_valid_prev: Tensor, source_valid_curr: Tensor,
            all_prev: Tensor, all_curr: Tensor, all_valid_prev: Tensor,
            all_valid_curr: Tensor, factor: Tensor) -> Tuple[Tensor, Tensor]:
        """Choose moderate negatives with the least overlap to any real GT."""
        num_targets = source_prev.size(0)
        best_prev = source_prev.new_full((num_targets, 5), 0.5)
        best_curr = source_curr.new_full((num_targets, 5), 0.5)
        best_score = source_prev.new_full(
            (num_targets,), float('inf'), dtype=torch.float32)
        for _ in range(self.negative_resample_attempts):
            cand_prev, cand_curr = self._noisy_pair_refs(
                source_prev, source_curr, source_valid_prev,
                source_valid_curr, factor, negative=True)
            score_prev = self._max_gt_overlap(
                cand_prev, source_valid_prev, all_prev, all_valid_prev, factor)
            score_curr = self._max_gt_overlap(
                cand_curr, source_valid_curr, all_curr, all_valid_curr, factor)
            score = torch.maximum(score_prev, score_curr)
            accepted = best_score <= self.negative_max_iou
            better = (~accepted) & (score < best_score)
            best_prev = torch.where(better[:, None], cand_prev, best_prev)
            best_curr = torch.where(better[:, None], cand_curr, best_curr)
            best_score = torch.minimum(best_score, score)
        return best_prev, best_curr

    def _noisy_labels(self, labels: Tensor) -> Tensor:
        labels = labels.clone()
        if self.label_noise_scale <= 0 or labels.numel() == 0:
            return labels
        noise_mask = torch.rand_like(labels.float()) < (
            self.label_noise_scale * 0.5)
        random_labels = torch.randint(
            0, self.num_classes, labels.shape, device=labels.device)
        return torch.where(noise_mask, random_labels, labels)

    def forward(
            self, batch_data_samples: SampleList
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Dict]:
        device = batch_data_samples[0].pair_gt_instances.labels.device
        first_prev = batch_data_samples[0].pair_gt_instances.bboxes_prev
        dtype = (first_prev.tensor if hasattr(first_prev, 'tensor')
                 else first_prev).dtype
        labels_list: List[Tensor] = []
        refs_prev_list: List[Tensor] = []
        refs_curr_list: List[Tensor] = []
        valid_prev_list: List[Tensor] = []
        valid_curr_list: List[Tensor] = []
        factor_list: List[Tensor] = []
        for sample in batch_data_samples:
            gt = sample.pair_gt_instances
            img_h, img_w = sample.metainfo['img_shape']
            factor = torch.tensor(
                [img_w, img_h, img_w, img_h, self.angle_factor],
                device=device,
                dtype=dtype)
            labels_list.append(gt.labels.to(device=device, dtype=torch.long))
            refs_prev_list.append(self._to_rbox(gt.bboxes_prev).to(
                device=device, dtype=dtype) / factor)
            refs_curr_list.append(self._to_rbox(gt.bboxes_curr).to(
                device=device, dtype=dtype) / factor)
            valid_prev_list.append(torch.as_tensor(
                gt.valid_prev, device=device, dtype=torch.bool))
            valid_curr_list.append(torch.as_tensor(
                gt.valid_curr, device=device, dtype=torch.bool))
            factor_list.append(factor)

        counts = [len(labels) for labels in labels_list]
        max_targets = max(counts, default=0)
        easy_hard_positive = self.dn_target_mode == 'easy_hard_positive'
        max_secondary_targets = (
            max_targets if easy_hard_positive
            else self._num_negative_targets(max_targets))
        num_groups = self._get_num_groups(max_targets)
        group_width = max_targets + max_secondary_targets
        num_dn = group_width * num_groups
        batch_size = len(batch_data_samples)
        dn_query = torch.zeros(batch_size, num_dn, self.embed_dims, device=device,
                               dtype=dtype)
        dn_prev = torch.full((batch_size, num_dn, 5), 0.5, device=device,
                             dtype=dtype)
        dn_curr = torch.full((batch_size, num_dn, 5), 0.5, device=device,
                             dtype=dtype)
        total_queries = num_dn + self.num_matching_queries
        query_key_padding_mask = torch.zeros(
            batch_size, total_queries, device=device, dtype=torch.bool)
        query_key_padding_mask[:, :num_dn] = True
        secondary_counts = []

        for batch_idx, (labels, refs_prev, refs_curr, valid_prev,
                        valid_curr, factor) in enumerate(zip(
                            labels_list, refs_prev_list, refs_curr_list,
                            valid_prev_list, valid_curr_list, factor_list)):
            num_targets = len(labels)
            num_secondary_targets = (
                num_targets if easy_hard_positive
                else self._num_negative_targets(num_targets))
            secondary_counts.append(num_secondary_targets)
            if num_targets == 0:
                continue
            negative_indices = None
            if not easy_hard_positive and num_secondary_targets > 0:
                negative_indices = torch.randperm(
                    num_targets, device=device)[:num_secondary_targets]
            repeated_prev = refs_prev.repeat(num_groups, 1)
            repeated_curr = refs_curr.repeat(num_groups, 1)
            repeated_valid_prev = valid_prev.repeat(num_groups)
            repeated_valid_curr = valid_curr.repeat(num_groups)
            pos_prev_all, pos_curr_all = self._noisy_pair_refs(
                repeated_prev, repeated_curr, repeated_valid_prev,
                repeated_valid_curr, factor, negative=False,
                positive_hard=False if easy_hard_positive else None)
            pos_labels_all = self._noisy_labels(labels.repeat(num_groups))

            if easy_hard_positive:
                secondary_prev_all, secondary_curr_all = (
                    self._noisy_pair_refs(
                        repeated_prev, repeated_curr, repeated_valid_prev,
                        repeated_valid_curr, factor, negative=False,
                        positive_hard=True))
                secondary_labels_all = self._noisy_labels(
                    labels.repeat(num_groups))
            elif num_secondary_targets > 0:
                negative_prev = refs_prev[negative_indices]
                negative_curr = refs_curr[negative_indices]
                negative_valid_prev = valid_prev[negative_indices]
                negative_valid_curr = valid_curr[negative_indices]
                secondary_prev_all, secondary_curr_all = (
                    self._sample_negative_pair_refs(
                        negative_prev.repeat(num_groups, 1),
                        negative_curr.repeat(num_groups, 1),
                        negative_valid_prev.repeat(num_groups),
                        negative_valid_curr.repeat(num_groups), refs_prev,
                        refs_curr, valid_prev, valid_curr, factor))
                secondary_labels_all = self._noisy_labels(
                    labels[negative_indices].repeat(num_groups))
            for group_idx in range(num_groups):
                group_start = group_idx * group_width
                pos_start = group_start
                pos_end = pos_start + num_targets
                pos_source_start = group_idx * num_targets
                pos_source_end = pos_source_start + num_targets
                dn_query[batch_idx, pos_start:pos_end] = self.label_embedding(
                    pos_labels_all[pos_source_start:pos_source_end]).to(dtype)
                dn_prev[batch_idx, pos_start:pos_end] = pos_prev_all[
                    pos_source_start:pos_source_end]
                dn_curr[batch_idx, pos_start:pos_end] = pos_curr_all[
                    pos_source_start:pos_source_end]
                query_key_padding_mask[batch_idx, pos_start:pos_end] = False

                secondary_start = group_start + max_targets
                secondary_end = secondary_start + num_secondary_targets
                if num_secondary_targets > 0:
                    secondary_source_start = group_idx * num_secondary_targets
                    secondary_source_end = (
                        secondary_source_start + num_secondary_targets)
                    dn_query[batch_idx, secondary_start:secondary_end] = (
                        self.label_embedding(
                            secondary_labels_all[
                                secondary_source_start:
                                secondary_source_end]).to(dtype))
                    dn_prev[batch_idx, secondary_start:secondary_end] = (
                        secondary_prev_all[
                            secondary_source_start:secondary_source_end])
                    dn_curr[batch_idx, secondary_start:secondary_end] = (
                        secondary_curr_all[
                            secondary_source_start:secondary_source_end])
                    query_key_padding_mask[
                        batch_idx, secondary_start:secondary_end] = False

        attn_mask = torch.zeros(total_queries, total_queries, device=device,
                                dtype=torch.bool)
        if num_dn > 0:
            attn_mask[num_dn:, :num_dn] = True
            # One contrastive group contains a positive block followed by
            # its negative block. Only different groups are isolated.
            for group_idx in range(num_groups):
                start = group_idx * group_width
                end = start + group_width
                attn_mask[start:end, :start] = True
                attn_mask[start:end, end:num_dn] = True
                if easy_hard_positive:
                    split = start + max_targets
                    attn_mask[start:split, split:end] = True
                    attn_mask[split:end, start:split] = True
        dn_meta = dict(
            num_denoising_queries=num_dn,
            num_denoising_groups=num_groups,
            max_num_dn_targets=max_targets,
            dn_target_mode=self.dn_target_mode)
        if easy_hard_positive:
            dn_meta.update(
                max_num_hard_positive_dn_targets=max_secondary_targets,
                num_hard_positive_dn_targets_per_image=secondary_counts)
        else:
            dn_meta.update(
                max_num_negative_dn_targets=max_secondary_targets,
                num_negative_dn_targets_per_image=secondary_counts)
        return (dn_query, inverse_sigmoid(dn_prev), inverse_sigmoid(dn_curr),
                attn_mask, query_key_padding_mask, dn_meta)
