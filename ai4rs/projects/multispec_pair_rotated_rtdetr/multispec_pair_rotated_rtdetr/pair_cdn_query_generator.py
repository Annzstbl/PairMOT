# Copyright (c) AI4RS. All rights reserved.
"""Contrastive denoising queries for paired rotated detection."""

from typing import Dict, List, Tuple

import torch
from mmdet.models.layers.transformer.utils import inverse_sigmoid
from mmdet.structures import SampleList
from torch import Tensor, nn

from mmrotate.structures.bbox import RotatedBoxes, qbox2rbox


class PairCdnQueryGenerator(nn.Module):
    """Generate DINO-style DN queries from pair-level track unions.

    One DN target represents one track id.  It has a shared class query and
    separate noisy references for the previous and current frame.  Missing
    sides retain a neutral reference and are supervised only through presence.
    """

    def __init__(self,
                 num_classes: int,
                 embed_dims: int,
                 num_matching_queries: int,
                 label_noise_scale: float = 0.5,
                 box_noise_scale: float = 1.0,
                 angle_factor: float = torch.pi,
                 group_cfg: Dict = None) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.embed_dims = embed_dims
        self.num_matching_queries = num_matching_queries
        self.label_noise_scale = label_noise_scale
        self.box_noise_scale = box_noise_scale
        self.angle_factor = float(angle_factor)
        group_cfg = {} if group_cfg is None else group_cfg
        self.dynamic_dn_groups = group_cfg.get('dynamic', True)
        self.num_dn_queries = group_cfg.get('num_dn_queries', 100)
        self.num_groups = group_cfg.get('num_groups', 1)
        self.label_embedding = nn.Embedding(num_classes, embed_dims)

    @staticmethod
    def _to_rbox(bboxes) -> Tensor:
        if isinstance(bboxes, RotatedBoxes):
            return bboxes.tensor
        if hasattr(bboxes, 'tensor'):
            bboxes = bboxes.tensor
        if bboxes.size(-1) == 8:
            return qbox2rbox(bboxes)
        return bboxes

    def _get_num_groups(self, max_num_targets: int) -> int:
        if not self.dynamic_dn_groups:
            return max(1, int(self.num_groups))
        if max_num_targets == 0:
            return 1
        return max(1, int(self.num_dn_queries) // max_num_targets)

    def _noisy_refs(self, refs: Tensor, valid: Tensor, *, negative: bool) -> Tensor:
        """Perturb normalized rboxes with separate positive/negative ranges.

        Positive DN copies stay within half a box size.  Negative copies use
        the outer [1, 2) half-box band, matching DINO's contrastive DN design.
        """
        num_targets = refs.size(0)
        if num_targets == 0:
            return refs.new_zeros((0, 5))
        magnitude = torch.rand_like(refs)
        if negative:
            magnitude += 1.0
        noise = (torch.rand_like(refs) * 2 - 1) * magnitude
        noise[:, :4] *= refs[:, 2:4].repeat(1, 2) * self.box_noise_scale / 2
        noise[:, 4] *= self.box_noise_scale * 0.25
        out = (refs + noise).clamp(1e-4, 1 - 1e-4)
        out[~valid] = 0.5
        return out

    def forward(self, batch_data_samples: SampleList) -> Tuple[Tensor, Tensor,
                                                                  Tensor, Tensor,
                                                                  Dict[str, int]]:
        device = batch_data_samples[0].pair_gt_instances.labels.device
        dtype = batch_data_samples[0].pair_gt_instances.bboxes_prev.tensor.dtype
        labels_list: List[Tensor] = []
        refs_prev_list: List[Tensor] = []
        refs_curr_list: List[Tensor] = []
        valid_prev_list: List[Tensor] = []
        valid_curr_list: List[Tensor] = []
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

        counts = [len(labels) for labels in labels_list]
        max_targets = max(counts, default=0)
        num_groups = self._get_num_groups(max_targets)
        num_dn = max_targets * 2 * num_groups
        batch_size = len(batch_data_samples)
        dn_query = torch.zeros(batch_size, num_dn, self.embed_dims, device=device,
                               dtype=dtype)
        dn_prev = torch.full((batch_size, num_dn, 5), 0.5, device=device,
                             dtype=dtype)
        dn_curr = torch.full((batch_size, num_dn, 5), 0.5, device=device,
                             dtype=dtype)

        for batch_idx, (labels, refs_prev, refs_curr, valid_prev,
                        valid_curr) in enumerate(zip(
                            labels_list, refs_prev_list, refs_curr_list,
                            valid_prev_list, valid_curr_list)):
            num_targets = len(labels)
            if num_targets == 0:
                continue
            expanded_labels = labels.repeat(2 * num_groups)
            if self.label_noise_scale > 0:
                noise_mask = torch.rand_like(expanded_labels.float()) < (
                    self.label_noise_scale * 0.5)
                random_labels = torch.randint(
                    0, self.num_classes, expanded_labels.shape, device=device)
                expanded_labels = torch.where(noise_mask, random_labels,
                                              expanded_labels)
            for group_idx in range(2 * num_groups):
                start = group_idx * max_targets
                end = start + num_targets
                dn_query[batch_idx, start:end] = self.label_embedding(
                    expanded_labels[group_idx * num_targets:(group_idx + 1) *
                                    num_targets]).to(dtype)
                dn_prev[batch_idx, start:end] = self._noisy_refs(
                    refs_prev, valid_prev, negative=bool(group_idx % 2))
                dn_curr[batch_idx, start:end] = self._noisy_refs(
                    refs_curr, valid_curr, negative=bool(group_idx % 2))

        total_queries = num_dn + self.num_matching_queries
        attn_mask = torch.zeros(total_queries, total_queries, device=device,
                                dtype=torch.bool)
        if num_dn > 0:
            attn_mask[num_dn:, :num_dn] = True
            for group_idx in range(2 * num_groups):
                start = group_idx * max_targets
                end = start + max_targets
                attn_mask[start:end, :start] = True
                attn_mask[start:end, end:num_dn] = True
        dn_meta = dict(
            num_denoising_queries=num_dn,
            num_denoising_groups=num_groups,
            max_num_dn_targets=max_targets)
        return dn_query, inverse_sigmoid(dn_prev), inverse_sigmoid(dn_curr), attn_mask, dn_meta
