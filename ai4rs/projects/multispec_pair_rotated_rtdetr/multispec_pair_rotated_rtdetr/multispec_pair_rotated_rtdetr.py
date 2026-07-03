# Copyright (c) AI4RS. All rights reserved.
"""Pair-frame multispec Rotated RT-DETR."""

from typing import Dict, List, Literal, Optional, Tuple

import torch
import torch.nn.functional as F
from mmengine.logging import print_log
from mmdet.structures import OptSampleList, SampleList
from mmrotate.registry import MODELS
from mmrotate.structures.bbox import RotatedBoxes, qbox2rbox
from torch import Tensor, nn

from projects.rotated_rtdetr.rotated_rtdetr import RotatedRTDETR
from projects.rotated_rtdetr.rotated_rtdetr.rtdetr_layers import RTDETRHybridEncoder

from .pair_rotated_rtdetr_layers import PairRotatedRTDETRTransformerDecoder
from .pair_cdn_query_generator import PairCdnQueryGenerator
from .component_timer import CudaComponentTimer

QueryInitMode = Literal['learned', 'gt_noised', 'dual_topk',
                        'pair_topk_v1', 'pair_topk_sameidx_v1',
                        'pair_topk_v2', 'typed_pair_topk_v1']

QUERY_TYPE_SURVIVAL = 0
QUERY_TYPE_CURR_ONLY = 1
QUERY_TYPE_PREV_ONLY = 2


@MODELS.register_module()
class MultispecPairRotatedRTDETR(RotatedRTDETR):
    """Rotated RT-DETR for HSMOT image pairs.

    ``pair_mode=False`` (default): M2 dual independent decoders / heads.
    ``pair_mode=True``: shared Pair decoder + ``PairRotatedRTDETRHead``.
    """

    PAIR_NUM_FRAMES = 2

    def __init__(self,
                 *args,
                 debug_shapes: bool = False,
                 pair_mode: bool = False,
                 query_init: QueryInitMode = 'learned',
                 gt_ref_noise_scale: float = 0.02,
                 pair_dn_cfg: Optional[Dict] = None,
                 pair_proposal_cfg: Optional[Dict] = None,
                 **kwargs) -> None:
        if query_init not in ('learned', 'gt_noised', 'dual_topk',
                              'pair_topk_v1', 'pair_topk_sameidx_v1',
                              'pair_topk_v2', 'typed_pair_topk_v1'):
            raise ValueError(
                f'query_init must be learned, gt_noised, dual_topk, '
                f'pair_topk_v1, pair_topk_sameidx_v1, pair_topk_v2, '
                f'or typed_pair_topk_v1, '
                f'got {query_init!r}')
        self.pair_mode = pair_mode
        self.query_init = query_init
        self.gt_ref_noise_scale = gt_ref_noise_scale
        self.pair_dn_cfg = pair_dn_cfg
        self.pair_proposal_cfg = pair_proposal_cfg or {}
        super().__init__(*args, **kwargs)
        self.debug_shapes = debug_shapes
        self.pair_dn_query_generator = None
        if self.pair_mode and pair_dn_cfg is not None:
            self.pair_dn_query_generator = PairCdnQueryGenerator(
                num_classes=self.bbox_head.num_classes,
                embed_dims=self.decoder.embed_dims,
                num_matching_queries=self.num_queries,
                angle_factor=self.decoder.angle_factor,
                **pair_dn_cfg)
        if self.pair_mode and self.query_init == 'learned':
            self._freeze_unused_learned_pair_params()

    def _freeze_unused_learned_pair_params(self) -> None:
        """Freeze params skipped by pair ``learned`` init (no encoder top-k)."""
        for param in self.memory_trans_fc.parameters():
            param.requires_grad = False
        for param in self.memory_trans_norm.parameters():
            param.requires_grad = False
        enc_idx = self.decoder.num_layers
        for branch in (
            self.bbox_head.cls_branches[enc_idx],
            self.bbox_head.reg_branches[enc_idx],
            self.bbox_head.reg_branches_curr[enc_idx],
        ):
            for param in branch.parameters():
                param.requires_grad = False
        if hasattr(self.bbox_head, 'cls_branches_curr'):
            for param in self.bbox_head.cls_branches_curr[enc_idx].parameters():
                param.requires_grad = False
        if hasattr(self.bbox_head, 'presence_prev_branches'):
            for branch in (
                self.bbox_head.presence_prev_branches[enc_idx],
                self.bbox_head.presence_curr_branches[enc_idx],
            ):
                for param in branch.parameters():
                    param.requires_grad = False

    def _init_layers(self) -> None:
        """Initialize encoder/decoder; pair mode swaps in Pair decoder."""
        self.encoder = RTDETRHybridEncoder(**self.encoder)
        if self.pair_mode:
            self.decoder = PairRotatedRTDETRTransformerDecoder(**self.decoder)
        else:
            from projects.rotated_rtdetr.rotated_rtdetr.rotated_rtdetr_layers import (
                RotatedRTDETRTransformerDecoder)
            self.decoder = RotatedRTDETRTransformerDecoder(**self.decoder)
        self.embed_dims = self.decoder.embed_dims
        self.memory_trans_fc = nn.Linear(self.embed_dims, self.embed_dims)
        self.memory_trans_norm = nn.LayerNorm(self.embed_dims)
        if self.pair_mode:
            self.pair_query_fusion = nn.Linear(
                self.embed_dims * 2, self.embed_dims)
            nn.init.zeros_(self.pair_query_fusion.weight)
            nn.init.zeros_(self.pair_query_fusion.bias)
            with torch.no_grad():
                eye = torch.eye(self.embed_dims)
                self.pair_query_fusion.weight[:, :self.embed_dims].copy_(
                    0.5 * eye)
                self.pair_query_fusion.weight[:, self.embed_dims:].copy_(
                    0.5 * eye)
            if self.query_init in ('pair_topk_v2', 'typed_pair_topk_v1'):
                self.pair_quality_predictor = nn.Sequential(
                    nn.Linear(self.embed_dims, self.embed_dims),
                    nn.ReLU(inplace=True),
                    nn.Linear(self.embed_dims, 1),
                )
            if self.query_init == 'typed_pair_topk_v1':
                self.null_prev_query = nn.Parameter(torch.zeros(self.embed_dims))
                self.null_curr_query = nn.Parameter(torch.zeros(self.embed_dims))
                null_ref = torch.tensor(
                    [0.5, 0.5, 0.05, 0.05, 0.5], dtype=torch.float32)
                null_ref_unact = torch.logit(null_ref.clamp(1e-4, 1 - 1e-4))
                self.null_prev_ref_unact = nn.Parameter(null_ref_unact.clone())
                self.null_curr_ref_unact = nn.Parameter(null_ref_unact.clone())

    def _log_shape(self, name: str, tensor: Tensor) -> None:
        if self.debug_shapes:
            print_log(
                f'[MultispecPairRTDETR] {name}: {tuple(tensor.shape)}',
                logger='current')

    @staticmethod
    def _split_pair_batch(flat_batch: int) -> Tuple[int, int]:
        if flat_batch % MultispecPairRotatedRTDETR.PAIR_NUM_FRAMES != 0:
            raise ValueError(
                f'Flat batch size {flat_batch} is not divisible by '
                f'{MultispecPairRotatedRTDETR.PAIR_NUM_FRAMES}')
        pair_batch = flat_batch // MultispecPairRotatedRTDETR.PAIR_NUM_FRAMES
        return pair_batch, flat_batch

    def extract_feat(self, batch_inputs: Tensor) -> Tuple[Tensor, ...]:
        """Extract features from pair or single-frame inputs."""
        if batch_inputs.dim() == 5:
            b, num_frames, c, h, w = batch_inputs.shape
            if num_frames != self.PAIR_NUM_FRAMES:
                raise ValueError(
                    f'Expected {self.PAIR_NUM_FRAMES} frames, got {num_frames}')
            self._log_shape('input_pair', batch_inputs)
            # Keep all previous frames before all current frames.  The pair
            # transformer later splits encoder memory as ``[:B]`` and ``[B:]``.
            batch_inputs = batch_inputs.transpose(0, 1).reshape(
                b * num_frames, c, h, w)
        elif batch_inputs.dim() != 4:
            raise ValueError(
                'MultispecPairRotatedRTDETR expects input shape '
                f'(B, 2, C, H, W) or (B, C, H, W), got {batch_inputs.shape}')

        self._log_shape('input_flat', batch_inputs)
        x = self.backbone(batch_inputs)
        if self.with_neck:
            x = self.neck(x)
        for lvl, feat in enumerate(x):
            self._log_shape(f'neck_level{lvl}_feat', feat)
        return x

    def _topk_pair_queries(
        self,
        memory_prev: Tensor,
        memory_curr: Tensor,
        memory_mask: Optional[Tensor],
        spatial_shapes: Tensor,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """Top-K pair queries with aligned prev/curr references.

        A pair query is one object hypothesis shared by both frames, so both
        box references must be anchored to the same top-k proposal.  Using an
        independent curr top-k order mixes a prev query with an unrelated curr
        reference and makes the curr branch harder to overfit.
        """
        bs, _, c = memory_prev.shape
        num_layers = self.decoder.num_layers

        output_memory_p, output_proposals_p = self.gen_encoder_output_proposals(
            memory_prev, memory_mask, spatial_shapes)
        enc_cls_p = self.bbox_head.cls_branches[num_layers](output_memory_p)
        topk_idx_p = torch.topk(
            enc_cls_p.max(-1)[0], k=self.num_queries, dim=1)[1]
        query = torch.gather(
            output_memory_p, 1,
            topk_idx_p.unsqueeze(-1).repeat(1, 1, c))
        topk_props_p = torch.gather(
            output_proposals_p, 1,
            topk_idx_p.unsqueeze(-1).repeat(1, 1, 5))
        ref_prev_unact = self.bbox_head.reg_branches[num_layers](
            query) + topk_props_p

        output_memory_c, output_proposals_c = self.gen_encoder_output_proposals(
            memory_curr, memory_mask, spatial_shapes)
        topk_props_c = torch.gather(
            output_proposals_c, 1,
            topk_idx_p.unsqueeze(-1).repeat(1, 1, 5))
        ref_curr_unact = self.bbox_head.reg_branches_curr[num_layers](
            query) + topk_props_c

        reference_prev = ref_prev_unact.sigmoid()
        reference_curr = ref_curr_unact.sigmoid()
        return query, reference_prev, reference_curr

    def _single_frame_topk_proposals(
        self,
        memory: Tensor,
        memory_mask: Optional[Tensor],
        spatial_shapes: Tensor,
        reg_branch: nn.Module,
        pre_topk: int,
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        """Generate independent per-frame encoder top-k proposals."""
        bs, num_values, c = memory.shape
        k = min(pre_topk, num_values)
        num_layers = self.decoder.num_layers
        output_memory, output_proposals = self.gen_encoder_output_proposals(
            memory, memory_mask, spatial_shapes)
        enc_cls = self.bbox_head.cls_branches[num_layers](output_memory)
        scores = enc_cls.sigmoid().max(-1)[0]
        topk_idx = torch.topk(scores, k=k, dim=1)[1]
        query = torch.gather(
            output_memory, 1,
            topk_idx.unsqueeze(-1).repeat(1, 1, c))
        props = torch.gather(
            output_proposals, 1,
            topk_idx.unsqueeze(-1).repeat(1, 1, 5))
        refs = (reg_branch(query) + props).sigmoid()
        topk_scores = torch.gather(scores, 1, topk_idx)
        return query, refs, topk_scores, topk_idx

    def _single_frame_topk_proposals_v2(
        self,
        memory: Tensor,
        memory_mask: Optional[Tensor],
        spatial_shapes: Tensor,
        cls_branch: nn.Module,
        reg_branch: nn.Module,
        pre_topk: int,
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        """Generate per-frame proposals with logits and labels for pairing."""
        bs, num_values, c = memory.shape
        k = min(pre_topk, num_values)
        output_memory, output_proposals = self.gen_encoder_output_proposals(
            memory, memory_mask, spatial_shapes)
        enc_cls = cls_branch(output_memory)
        probs = enc_cls.sigmoid()
        scores, labels = probs.max(dim=-1)
        topk_idx = torch.topk(scores, k=k, dim=1)[1]
        query = torch.gather(
            output_memory, 1,
            topk_idx.unsqueeze(-1).repeat(1, 1, c))
        props = torch.gather(
            output_proposals, 1,
            topk_idx.unsqueeze(-1).repeat(1, 1, 5))
        refs = (reg_branch(query) + props).sigmoid()
        topk_scores = torch.gather(scores, 1, topk_idx)
        topk_labels = torch.gather(labels, 1, topk_idx)
        topk_logits = torch.gather(
            enc_cls, 1,
            topk_idx.unsqueeze(-1).repeat(1, 1, enc_cls.size(-1)))
        return query, refs, topk_scores, topk_labels, topk_logits, topk_idx

    def _topk_pair_queries_sameidx_v1(
        self,
        memory_prev: Tensor,
        memory_curr: Tensor,
        memory_mask: Optional[Tensor],
        spatial_shapes: Tensor,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """Dual-frame proposal scoring while preserving aligned proposal index.

        This is a conservative two-frame proposal init: both frames build their
        encoder proposals independently, but top-k is selected over the same
        flattened spatial index. It avoids the free proposal rematching used in
        ``pair_topk_v1`` so pretrained RT-DETR proposal ordering remains stable.
        """
        cfg = self.pair_proposal_cfg
        bs, num_values, c = memory_prev.shape
        k = min(self.num_queries, num_values)
        num_layers = self.decoder.num_layers

        output_memory_p, output_proposals_p = self.gen_encoder_output_proposals(
            memory_prev, memory_mask, spatial_shapes)
        output_memory_c, output_proposals_c = self.gen_encoder_output_proposals(
            memory_curr, memory_mask, spatial_shapes)

        enc_cls_p = self.bbox_head.cls_branches[num_layers](output_memory_p)
        enc_cls_c = self.bbox_head.cls_branches[num_layers](output_memory_c)
        score_p = enc_cls_p.sigmoid().max(-1)[0]
        score_c = enc_cls_c.sigmoid().max(-1)[0]

        score_mode = str(cfg.get('sameidx_score_mode', 'sqrt'))
        if score_mode == 'prev':
            joint_score = score_p
        elif score_mode == 'mean':
            joint_score = 0.5 * (score_p + score_c)
        elif score_mode == 'min':
            joint_score = torch.minimum(score_p, score_c)
        else:
            joint_score = torch.sqrt(
                score_p.clamp(min=1e-6) * score_c.clamp(min=1e-6))

        topk_idx = torch.topk(joint_score, k=k, dim=1)[1]
        query_p = torch.gather(
            output_memory_p, 1,
            topk_idx.unsqueeze(-1).repeat(1, 1, c))
        query_c = torch.gather(
            output_memory_c, 1,
            topk_idx.unsqueeze(-1).repeat(1, 1, c))
        query = self.pair_query_fusion(torch.cat([query_p, query_c], dim=-1))

        props_p = torch.gather(
            output_proposals_p, 1,
            topk_idx.unsqueeze(-1).repeat(1, 1, 5))
        props_c = torch.gather(
            output_proposals_c, 1,
            topk_idx.unsqueeze(-1).repeat(1, 1, 5))

        ref_source = str(cfg.get('sameidx_ref_source', 'frame'))
        if ref_source == 'fused':
            ref_prev_unact = self.bbox_head.reg_branches[num_layers](
                query) + props_p
            ref_curr_unact = self.bbox_head.reg_branches_curr[num_layers](
                query) + props_c
        else:
            ref_prev_unact = self.bbox_head.reg_branches[num_layers](
                query_p) + props_p
            ref_curr_unact = self.bbox_head.reg_branches_curr[num_layers](
                query_c) + props_c

        reference_prev = ref_prev_unact.sigmoid()
        reference_curr = ref_curr_unact.sigmoid()

        if k < self.num_queries:
            pad = self.num_queries - k
            learned_query = self.decoder.query_embedding.weight.to(
                device=memory_prev.device, dtype=memory_prev.dtype)
            learned_prev = self.decoder.ref_prev_embedding.weight.sigmoid().to(
                device=memory_prev.device, dtype=memory_prev.dtype)
            learned_curr = self.decoder.ref_curr_embedding.weight.sigmoid().to(
                device=memory_prev.device, dtype=memory_prev.dtype)
            query = torch.cat([
                query,
                learned_query[:pad].unsqueeze(0).expand(bs, -1, -1)
            ], dim=1)
            reference_prev = torch.cat([
                reference_prev,
                learned_prev[:pad].unsqueeze(0).expand(bs, -1, -1)
            ], dim=1)
            reference_curr = torch.cat([
                reference_curr,
                learned_curr[:pad].unsqueeze(0).expand(bs, -1, -1)
            ], dim=1)

        return query, reference_prev, reference_curr

    def _pair_match_score(self, query_prev: Tensor, query_curr: Tensor,
                          ref_prev: Tensor, ref_curr: Tensor,
                          score_prev: Tensor, score_curr: Tensor) -> Tensor:
        """Build proposal pair score from appearance, geometry, and cls prior."""
        cfg = self.pair_proposal_cfg
        sim_weight = float(cfg.get('sim_weight', 1.0))
        geom_weight = float(cfg.get('geom_weight', 1.0))
        score_weight = float(cfg.get('score_weight', 1.0))
        geom_sigma = float(cfg.get('geom_sigma', 0.08))
        max_center_dist = float(cfg.get('max_center_dist', 0.35))
        max_log_scale = float(cfg.get('max_log_scale', 1.2))

        q_prev = F.normalize(query_prev, dim=-1)
        q_curr = F.normalize(query_curr, dim=-1)
        sim = torch.matmul(q_prev, q_curr.transpose(0, 1))
        sim = (sim + 1.0) * 0.5

        center_delta = ref_prev[:, None, :2] - ref_curr[None, :, :2]
        center_dist = center_delta.norm(dim=-1)
        center_score = torch.exp(-center_dist / max(geom_sigma, 1e-6))

        wh_prev = ref_prev[:, None, 2:4].clamp(min=1e-4)
        wh_curr = ref_curr[None, :, 2:4].clamp(min=1e-4)
        log_scale = (wh_prev.log() - wh_curr.log()).abs().amax(dim=-1)
        scale_score = torch.exp(-log_scale / max(max_log_scale, 1e-6))
        geom = center_score * scale_score

        cls_prior = torch.sqrt(
            score_prev[:, None].clamp(min=1e-6) *
            score_curr[None, :].clamp(min=1e-6))
        match_score = (
            sim_weight * sim + geom_weight * geom +
            score_weight * cls_prior)
        invalid = center_dist > max_center_dist
        match_score = match_score.masked_fill(invalid, -1e6)
        return match_score

    @staticmethod
    def _as_gmc_tensor(batch_data_samples: OptSampleList, batch_idx: int,
                       device: torch.device, dtype: torch.dtype) -> Tensor:
        if batch_data_samples is None:
            return torch.eye(3, device=device, dtype=dtype)
        meta = batch_data_samples[batch_idx].metainfo
        matrix = meta.get('gmc_matrix', None)
        if matrix is None:
            return torch.eye(3, device=device, dtype=dtype)
        return torch.as_tensor(matrix, device=device, dtype=dtype).reshape(3, 3)

    @staticmethod
    def _warp_ref_centers_by_gmc(refs: Tensor, gmc: Tensor,
                                 img_shape: Tuple[int, int]) -> Tensor:
        """Warp normalized proposal centers by prev->curr GMC matrix."""
        img_h, img_w = img_shape
        xy = refs[:, :2].clone()
        xy_px = xy * refs.new_tensor([img_w, img_h])
        ones = torch.ones((xy_px.size(0), 1), device=xy_px.device,
                          dtype=xy_px.dtype)
        homo = torch.cat([xy_px, ones], dim=-1)
        warped = homo @ gmc.transpose(0, 1)
        denom = warped[:, 2:3].clamp(min=1e-6)
        warped_xy = warped[:, :2] / denom
        return (warped_xy / refs.new_tensor([img_w, img_h])).clamp(0.0, 1.0)

    def _pair_affinity_score_v2(self, query_prev: Tensor, query_curr: Tensor,
                                ref_prev: Tensor, ref_curr: Tensor,
                                score_prev: Tensor, score_curr: Tensor,
                                label_prev: Tensor, label_curr: Tensor,
                                gmc: Tensor, img_shape: Tuple[int, int]) -> Tensor:
        """Pairing affinity used as a constraint, not the primary top-k rank."""
        cfg = self.pair_proposal_cfg
        sim_weight = float(cfg.get('sim_weight', 0.25))
        geom_weight = float(cfg.get('geom_weight', 0.5))
        score_weight = float(cfg.get('score_weight', 0.25))
        class_mismatch_penalty = float(cfg.get('class_mismatch_penalty', 0.5))
        class_aware = bool(cfg.get('class_aware', True))
        geom_sigma = float(cfg.get('geom_sigma', 0.08))
        max_center_dist = float(cfg.get('max_center_dist', 0.20))
        max_log_scale = float(cfg.get('max_log_scale', 1.0))

        q_prev = F.normalize(query_prev, dim=-1)
        q_curr = F.normalize(query_curr, dim=-1)
        sim = (torch.matmul(q_prev, q_curr.transpose(0, 1)) + 1.0) * 0.5

        warped_prev_xy = self._warp_ref_centers_by_gmc(
            ref_prev, gmc, img_shape)
        center_delta = warped_prev_xy[:, None, :] - ref_curr[None, :, :2]
        center_dist = center_delta.norm(dim=-1)
        center_score = torch.exp(-center_dist / max(geom_sigma, 1e-6))

        wh_prev = ref_prev[:, None, 2:4].clamp(min=1e-4)
        wh_curr = ref_curr[None, :, 2:4].clamp(min=1e-4)
        log_scale = (wh_prev.log() - wh_curr.log()).abs().amax(dim=-1)
        scale_score = torch.exp(-log_scale / max(max_log_scale, 1e-6))
        geom = center_score * scale_score

        cls_prior = torch.sqrt(
            score_prev[:, None].clamp(min=1e-6) *
            score_curr[None, :].clamp(min=1e-6))
        affinity = (
            sim_weight * sim + geom_weight * geom +
            score_weight * cls_prior)

        if class_aware:
            same_cls = label_prev[:, None] == label_curr[None, :]
            affinity = torch.where(
                same_cls, affinity, affinity - class_mismatch_penalty)

        invalid = center_dist > max_center_dist
        affinity = affinity.masked_fill(invalid, -1e6)
        return affinity.clamp(max=1.0)

    @staticmethod
    def _hungarian_affinity_pairs(
        affinity: Tensor,
        affinity_thr: float,
        candidate_topk: int,
    ) -> Tuple[Tensor, Tensor]:
        """Select one-to-one proposal pairs by maximizing affinity."""
        try:
            from scipy.optimize import linear_sum_assignment
        except ImportError as exc:
            raise RuntimeError(
                'pair_selection_mode="hungarian_affinity" requires scipy.'
            ) from exc

        valid = affinity > affinity_thr
        if valid.any():
            cost = torch.where(valid, -affinity, affinity.new_full((), 1e6))
        else:
            flat_scores, flat_idx = torch.topk(
                affinity.reshape(-1),
                k=min(candidate_topk, affinity.numel()))
            keep = flat_scores > -1e5
            return flat_idx[keep] // affinity.size(1), flat_idx[keep] % affinity.size(1)

        row_ind, col_ind = linear_sum_assignment(
            cost.detach().cpu().float().numpy())
        rows = torch.as_tensor(row_ind, device=affinity.device, dtype=torch.long)
        cols = torch.as_tensor(col_ind, device=affinity.device, dtype=torch.long)
        keep = valid[rows, cols]
        return rows[keep], cols[keep]

    def _unique_pair_order(self, order: Tensor, pair_i: Tensor,
                           pair_j: Tensor) -> Tensor:
        """Apply the existing greedy 1v1 preference with one host sync.

        Greedy uniqueness is inherently sequential. Keeping it in Python
        preserves the exact selection order, but converting candidate tensors to
        lists once avoids a GPU scalar synchronization for every candidate.
        """
        order_list = order.detach().cpu().tolist()
        pair_i_list = pair_i.detach().cpu().tolist()
        pair_j_list = pair_j.detach().cpu().tolist()

        selected = []
        used_prev = set()
        used_curr = set()
        for idx in order_list:
            pi = pair_i_list[idx]
            pj = pair_j_list[idx]
            if pi in used_prev or pj in used_curr:
                continue
            selected.append(idx)
            used_prev.add(pi)
            used_curr.add(pj)
            if len(selected) >= self.num_queries:
                break
        if len(selected) < self.num_queries:
            selected_set = set(selected)
            for idx in order_list:
                if idx not in selected_set:
                    selected.append(idx)
                    if len(selected) >= self.num_queries:
                        break
        return order.new_tensor(selected)

    @staticmethod
    def _topk_unused_indices(scores: Tensor, used: Tensor, k: int) -> Tensor:
        if k <= 0 or scores.numel() == 0:
            return scores.new_zeros((0,), dtype=torch.long)
        masked = scores.masked_fill(used, -1.0)
        k = min(k, scores.numel())
        vals, inds = torch.topk(masked, k=k)
        return inds[vals > -0.5]

    def _typed_pair_queries_v1(
        self,
        memory_prev: Tensor,
        memory_curr: Tensor,
        memory_mask: Optional[Tensor],
        spatial_shapes: Tensor,
        batch_data_samples: OptSampleList = None,
    ) -> Tuple[Tensor, Tensor, Tensor, Optional[Tensor], Optional[Tensor],
               Optional[Tensor], Optional[Tensor], Tensor]:
        """Typed survival / curr-only / prev-only proposal initialization."""
        cfg = self.pair_proposal_cfg
        bs, _, c = memory_prev.shape
        num_survival = int(cfg.get('num_survival_queries', 300))
        num_curr_only = int(cfg.get('num_curr_only_queries', 30))
        num_prev_only = int(cfg.get('num_prev_only_queries', 30))
        total_queries = num_survival + num_curr_only + num_prev_only
        if total_queries != self.num_queries:
            raise ValueError(
                f'typed_pair_topk_v1 expects decoder.num_queries='
                f'{total_queries}, got {self.num_queries}')

        pre_topk = int(cfg.get('pre_topk', max(self.num_queries * 3, 900)))
        candidate_topk = int(cfg.get('candidate_topk', self.num_queries * 6))
        affinity_thr = float(cfg.get('affinity_thr', 0.25))
        proposal_quality_weight = float(cfg.get('proposal_quality_weight', 0.70))
        learned_quality_weight = float(cfg.get('learned_quality_weight', 0.20))
        affinity_rank_weight = float(cfg.get('affinity_rank_weight', 0.10))
        unique_pair_selection = bool(cfg.get('unique_pair_selection', True))
        pair_selection_mode = str(cfg.get('pair_selection_mode', 'rank'))
        num_layers = self.decoder.num_layers
        cls_prev_branch = self.bbox_head.cls_branches[num_layers]
        cls_curr_branch = getattr(self.bbox_head, 'cls_branches_curr',
                                  self.bbox_head.cls_branches)[num_layers]

        query_p, ref_p, score_p, label_p, logits_p, _ = (
            self._single_frame_topk_proposals_v2(
                memory_prev, memory_mask, spatial_shapes, cls_prev_branch,
                self.bbox_head.reg_branches[num_layers], pre_topk))
        query_c, ref_c, score_c, label_c, logits_c, _ = (
            self._single_frame_topk_proposals_v2(
                memory_curr, memory_mask, spatial_shapes, cls_curr_branch,
                self.bbox_head.reg_branches_curr[num_layers], pre_topk))

        learned_query = self.decoder.query_embedding.weight.to(
            device=memory_prev.device, dtype=memory_prev.dtype)
        learned_prev = self.decoder.ref_prev_embedding.weight.sigmoid().to(
            device=memory_prev.device, dtype=memory_prev.dtype)
        learned_curr = self.decoder.ref_curr_embedding.weight.sigmoid().to(
            device=memory_prev.device, dtype=memory_prev.dtype)
        null_prev_query = self.null_prev_query.to(
            device=memory_prev.device, dtype=memory_prev.dtype)
        null_curr_query = self.null_curr_query.to(
            device=memory_prev.device, dtype=memory_prev.dtype)
        null_prev_ref = self.null_prev_ref_unact.to(
            device=memory_prev.device, dtype=memory_prev.dtype).sigmoid()
        null_curr_ref = self.null_curr_ref_unact.to(
            device=memory_prev.device, dtype=memory_prev.dtype).sigmoid()

        batch_queries: List[Tensor] = []
        batch_ref_prev: List[Tensor] = []
        batch_ref_curr: List[Tensor] = []
        batch_enc_cls_prev: List[Tensor] = []
        batch_enc_cls_curr: List[Tensor] = []
        batch_enc_bbox_prev: List[Tensor] = []
        batch_enc_bbox_curr: List[Tensor] = []
        batch_query_types: List[Tensor] = []

        for b in range(bs):
            img_shape = (batch_data_samples[b].metainfo['img_shape']
                         if batch_data_samples is not None else (1, 1))
            gmc = self._as_gmc_tensor(
                batch_data_samples, b, memory_prev.device, memory_prev.dtype)
            affinity = self._pair_affinity_score_v2(
                query_p[b], query_c[b], ref_p[b], ref_c[b], score_p[b],
                score_c[b], label_p[b], label_c[b], gmc, img_shape)
            if pair_selection_mode == 'hungarian_affinity':
                pair_i, pair_j = self._hungarian_affinity_pairs(
                    affinity, affinity_thr, candidate_topk)
            else:
                valid = affinity > affinity_thr
                pair_i, pair_j = torch.nonzero(valid, as_tuple=True)
                if pair_i.numel() == 0:
                    flat_scores, flat_idx = torch.topk(
                        affinity.reshape(-1),
                        k=min(candidate_topk, affinity.numel()))
                    keep = flat_scores > -1e5
                    flat_idx = flat_idx[keep]
                    pair_i = flat_idx // affinity.size(1)
                    pair_j = flat_idx % affinity.size(1)
                if pair_i.numel() > candidate_topk:
                    flat_valid_scores = affinity[pair_i, pair_j]
                    keep = torch.topk(
                        flat_valid_scores, k=candidate_topk).indices
                    pair_i = pair_i[keep]
                    pair_j = pair_j[keep]

            if pair_i.numel() > 0:
                prop_quality = torch.sqrt(
                    score_p[b, pair_i].clamp(min=1e-6) *
                    score_c[b, pair_j].clamp(min=1e-6))
                aff = affinity[pair_i, pair_j].clamp(min=0.0)
                rank_score = proposal_quality_weight * prop_quality
                if learned_quality_weight != 0.0:
                    with torch.no_grad():
                        quality_fused = self.pair_query_fusion(
                            torch.cat([
                                query_p[b, pair_i].detach(),
                                query_c[b, pair_j].detach()
                            ], dim=-1))
                        learned_quality = self.pair_quality_predictor(
                            quality_fused).squeeze(-1).sigmoid()
                    rank_score = (
                        rank_score + learned_quality_weight * learned_quality)
                if affinity_rank_weight != 0.0:
                    rank_score = rank_score + affinity_rank_weight * aff
                order = torch.argsort(rank_score, descending=True)
                if (pair_selection_mode != 'hungarian_affinity'
                        and unique_pair_selection
                        and order.numel() > num_survival):
                    order = self._unique_pair_order(order, pair_i, pair_j)
                pair_i = pair_i[order][:num_survival]
                pair_j = pair_j[order][:num_survival]
                q_surv = self.pair_query_fusion(
                    torch.cat([query_p[b, pair_i], query_c[b, pair_j]],
                              dim=-1))
                rp_surv = ref_p[b, pair_i]
                rc_surv = ref_c[b, pair_j]
                ep_surv = logits_p[b, pair_i]
                ec_surv = logits_c[b, pair_j]
            else:
                pair_i = score_p.new_zeros((0,), dtype=torch.long)
                pair_j = score_c.new_zeros((0,), dtype=torch.long)
                q_surv = query_p.new_zeros((0, c))
                rp_surv = ref_p.new_zeros((0, 5))
                rc_surv = ref_c.new_zeros((0, 5))
                ep_surv = logits_p.new_zeros((0, logits_p.size(-1)))
                ec_surv = logits_c.new_zeros((0, logits_c.size(-1)))

            used_prev = torch.zeros(
                query_p.size(1), dtype=torch.bool, device=query_p.device)
            used_curr = torch.zeros(
                query_c.size(1), dtype=torch.bool, device=query_c.device)
            if pair_i.numel() > 0:
                used_prev[pair_i] = True
                used_curr[pair_j] = True
            curr_idx = self._topk_unused_indices(
                score_c[b], used_curr, num_curr_only)
            prev_idx = self._topk_unused_indices(
                score_p[b], used_prev, num_prev_only)

            if curr_idx.numel() > 0:
                q_prev_null = null_prev_query.expand(curr_idx.numel(), -1)
                q_curr = query_c[b, curr_idx]
                q_curr_only = self.pair_query_fusion(
                    torch.cat([q_prev_null, q_curr], dim=-1))
                rp_curr_only = null_prev_ref.expand(curr_idx.numel(), -1)
                rc_curr_only = ref_c[b, curr_idx]
                ep_curr_only = logits_p.new_zeros(
                    (curr_idx.numel(), logits_p.size(-1)))
                ec_curr_only = logits_c[b, curr_idx]
            else:
                q_curr_only = query_c.new_zeros((0, c))
                rp_curr_only = ref_p.new_zeros((0, 5))
                rc_curr_only = ref_c.new_zeros((0, 5))
                ep_curr_only = logits_p.new_zeros((0, logits_p.size(-1)))
                ec_curr_only = logits_c.new_zeros((0, logits_c.size(-1)))

            if prev_idx.numel() > 0:
                q_prev = query_p[b, prev_idx]
                q_curr_null = null_curr_query.expand(prev_idx.numel(), -1)
                q_prev_only = self.pair_query_fusion(
                    torch.cat([q_prev, q_curr_null], dim=-1))
                rp_prev_only = ref_p[b, prev_idx]
                rc_prev_only = null_curr_ref.expand(prev_idx.numel(), -1)
                ep_prev_only = logits_p[b, prev_idx]
                ec_prev_only = logits_c.new_zeros(
                    (prev_idx.numel(), logits_c.size(-1)))
            else:
                q_prev_only = query_p.new_zeros((0, c))
                rp_prev_only = ref_p.new_zeros((0, 5))
                rc_prev_only = ref_c.new_zeros((0, 5))
                ep_prev_only = logits_p.new_zeros((0, logits_p.size(-1)))
                ec_prev_only = logits_c.new_zeros((0, logits_c.size(-1)))

            groups = [
                (q_surv, rp_surv, rc_surv, ep_surv, ec_surv, num_survival,
                 QUERY_TYPE_SURVIVAL, 0),
                (q_curr_only, rp_curr_only, rc_curr_only, ep_curr_only,
                 ec_curr_only, num_curr_only, QUERY_TYPE_CURR_ONLY,
                 num_survival),
                (q_prev_only, rp_prev_only, rc_prev_only, ep_prev_only,
                 ec_prev_only, num_prev_only, QUERY_TYPE_PREV_ONLY,
                 num_survival + num_curr_only),
            ]
            q_parts = []
            rp_parts = []
            rc_parts = []
            ep_parts = []
            ec_parts = []
            type_parts = []
            for q, rp, rc, ep, ec, target, qtype, learned_offset in groups:
                pad = target - q.size(0)
                if pad > 0:
                    start = learned_offset + q.size(0)
                    end = start + pad
                    q = torch.cat([q, learned_query[start:end]], dim=0)
                    rp = torch.cat([rp, learned_prev[start:end]], dim=0)
                    rc = torch.cat([rc, learned_curr[start:end]], dim=0)
                    ep = torch.cat([
                        ep,
                        logits_p.new_zeros((pad, logits_p.size(-1)))
                    ], dim=0)
                    ec = torch.cat([
                        ec,
                        logits_c.new_zeros((pad, logits_c.size(-1)))
                    ], dim=0)
                q_parts.append(q[:target])
                rp_parts.append(rp[:target])
                rc_parts.append(rc[:target])
                ep_parts.append(ep[:target])
                ec_parts.append(ec[:target])
                type_parts.append(torch.full(
                    (target,), qtype, dtype=torch.long, device=memory_prev.device))

            q = torch.cat(q_parts, dim=0)
            rp = torch.cat(rp_parts, dim=0)
            rc = torch.cat(rc_parts, dim=0)
            ep = torch.cat(ep_parts, dim=0)
            ec = torch.cat(ec_parts, dim=0)
            query_types = torch.cat(type_parts, dim=0)

            batch_queries.append(q)
            batch_ref_prev.append(rp)
            batch_ref_curr.append(rc)
            batch_enc_cls_prev.append(ep)
            batch_enc_cls_curr.append(ec)
            batch_enc_bbox_prev.append(rp)
            batch_enc_bbox_curr.append(rc)
            batch_query_types.append(query_types)

        return (
            torch.stack(batch_queries, dim=0),
            torch.stack(batch_ref_prev, dim=0),
            torch.stack(batch_ref_curr, dim=0),
            torch.stack(batch_enc_cls_prev, dim=0),
            torch.stack(batch_enc_cls_curr, dim=0),
            torch.stack(batch_enc_bbox_prev, dim=0),
            torch.stack(batch_enc_bbox_curr, dim=0),
            torch.stack(batch_query_types, dim=0),
        )

    def _topk_pair_queries_v2(
        self,
        memory_prev: Tensor,
        memory_curr: Tensor,
        memory_mask: Optional[Tensor],
        spatial_shapes: Tensor,
        batch_data_samples: OptSampleList = None,
    ) -> Tuple[Tensor, Tensor, Tensor, Optional[Tensor], Optional[Tensor],
               Optional[Tensor], Optional[Tensor]]:
        """Proposal-quality-first pair proposals with GMC-constrained affinity."""
        cfg = self.pair_proposal_cfg
        bs, _, c = memory_prev.shape
        pre_topk = int(cfg.get('pre_topk', self.num_queries * 3))
        candidate_topk = int(cfg.get('candidate_topk', self.num_queries * 6))
        affinity_thr = float(cfg.get('affinity_thr', 0.25))
        proposal_quality_weight = float(cfg.get('proposal_quality_weight', 0.70))
        learned_quality_weight = float(cfg.get('learned_quality_weight', 0.20))
        affinity_rank_weight = float(cfg.get('affinity_rank_weight', 0.10))
        unique_pair_selection = bool(cfg.get('unique_pair_selection', False))
        pair_selection_mode = str(cfg.get('pair_selection_mode', 'rank'))
        num_layers = self.decoder.num_layers
        cls_prev_branch = self.bbox_head.cls_branches[num_layers]
        cls_curr_branch = getattr(self.bbox_head, 'cls_branches_curr',
                                  self.bbox_head.cls_branches)[num_layers]

        query_p, ref_p, score_p, label_p, logits_p, _ = (
            self._single_frame_topk_proposals_v2(
                memory_prev, memory_mask, spatial_shapes, cls_prev_branch,
                self.bbox_head.reg_branches[num_layers], pre_topk))
        query_c, ref_c, score_c, label_c, logits_c, _ = (
            self._single_frame_topk_proposals_v2(
                memory_curr, memory_mask, spatial_shapes, cls_curr_branch,
                self.bbox_head.reg_branches_curr[num_layers], pre_topk))

        learned_query = self.decoder.query_embedding.weight.to(
            device=memory_prev.device, dtype=memory_prev.dtype)
        learned_prev = self.decoder.ref_prev_embedding.weight.sigmoid().to(
            device=memory_prev.device, dtype=memory_prev.dtype)
        learned_curr = self.decoder.ref_curr_embedding.weight.sigmoid().to(
            device=memory_prev.device, dtype=memory_prev.dtype)

        batch_queries: List[Tensor] = []
        batch_ref_prev: List[Tensor] = []
        batch_ref_curr: List[Tensor] = []
        batch_enc_cls_prev: List[Tensor] = []
        batch_enc_cls_curr: List[Tensor] = []
        batch_enc_bbox_prev: List[Tensor] = []
        batch_enc_bbox_curr: List[Tensor] = []

        for b in range(bs):
            img_shape = (batch_data_samples[b].metainfo['img_shape']
                         if batch_data_samples is not None else (1, 1))
            gmc = self._as_gmc_tensor(
                batch_data_samples, b, memory_prev.device, memory_prev.dtype)
            affinity = self._pair_affinity_score_v2(
                query_p[b], query_c[b], ref_p[b], ref_c[b], score_p[b],
                score_c[b], label_p[b], label_c[b], gmc, img_shape)
            if pair_selection_mode == 'hungarian_affinity':
                pair_i, pair_j = self._hungarian_affinity_pairs(
                    affinity, affinity_thr, candidate_topk)
            else:
                valid = affinity > affinity_thr
                pair_i, pair_j = torch.nonzero(valid, as_tuple=True)
                if pair_i.numel() == 0:
                    flat_scores, flat_idx = torch.topk(
                        affinity.reshape(-1),
                        k=min(candidate_topk, affinity.numel()))
                    keep = flat_scores > -1e5
                    flat_idx = flat_idx[keep]
                    pair_i = flat_idx // affinity.size(1)
                    pair_j = flat_idx % affinity.size(1)
                if pair_i.numel() > candidate_topk:
                    flat_valid_scores = affinity[pair_i, pair_j]
                    keep = torch.topk(
                        flat_valid_scores, k=candidate_topk).indices
                    pair_i = pair_i[keep]
                    pair_j = pair_j[keep]

            if pair_i.numel() > 0:
                prop_quality = torch.sqrt(
                    score_p[b, pair_i].clamp(min=1e-6) *
                    score_c[b, pair_j].clamp(min=1e-6))
                aff = affinity[pair_i, pair_j].clamp(min=0.0)
                rank_score = proposal_quality_weight * prop_quality
                if learned_quality_weight != 0.0:
                    with torch.no_grad():
                        quality_fused = self.pair_query_fusion(
                            torch.cat([
                                query_p[b, pair_i].detach(),
                                query_c[b, pair_j].detach()
                            ], dim=-1))
                        learned_quality = self.pair_quality_predictor(
                            quality_fused).squeeze(-1).sigmoid()
                    rank_score = (
                        rank_score + learned_quality_weight * learned_quality)
                if affinity_rank_weight != 0.0:
                    rank_score = rank_score + affinity_rank_weight * aff
                order = torch.argsort(rank_score, descending=True)
                if (pair_selection_mode != 'hungarian_affinity'
                        and unique_pair_selection
                        and order.numel() > self.num_queries):
                    order = self._unique_pair_order(order, pair_i, pair_j)
                pair_i = pair_i[order]
                pair_j = pair_j[order]
                pair_i = pair_i[:self.num_queries]
                pair_j = pair_j[:self.num_queries]
                q = self.pair_query_fusion(
                    torch.cat([query_p[b, pair_i], query_c[b, pair_j]],
                              dim=-1))
                rp = ref_p[b, pair_i]
                rc = ref_c[b, pair_j]
                ep = logits_p[b, pair_i]
                ec = logits_c[b, pair_j]
            else:
                q = query_p.new_zeros((0, c))
                rp = ref_p.new_zeros((0, 5))
                rc = ref_c.new_zeros((0, 5))
                ep = logits_p.new_zeros((0, logits_p.size(-1)))
                ec = logits_c.new_zeros((0, logits_c.size(-1)))

            pad = self.num_queries - q.size(0)
            if pad > 0:
                q = torch.cat([q, learned_query[:pad]], dim=0)
                rp = torch.cat([rp, learned_prev[:pad]], dim=0)
                rc = torch.cat([rc, learned_curr[:pad]], dim=0)
                ep = torch.cat([
                    ep,
                    logits_p.new_zeros((pad, logits_p.size(-1)))
                ], dim=0)
                ec = torch.cat([
                    ec,
                    logits_c.new_zeros((pad, logits_c.size(-1)))
                ], dim=0)

            batch_queries.append(q[:self.num_queries])
            batch_ref_prev.append(rp[:self.num_queries])
            batch_ref_curr.append(rc[:self.num_queries])
            batch_enc_cls_prev.append(ep[:self.num_queries])
            batch_enc_cls_curr.append(ec[:self.num_queries])
            batch_enc_bbox_prev.append(rp[:self.num_queries])
            batch_enc_bbox_curr.append(rc[:self.num_queries])

        return (
            torch.stack(batch_queries, dim=0),
            torch.stack(batch_ref_prev, dim=0),
            torch.stack(batch_ref_curr, dim=0),
            torch.stack(batch_enc_cls_prev, dim=0),
            torch.stack(batch_enc_cls_curr, dim=0),
            torch.stack(batch_enc_bbox_prev, dim=0),
            torch.stack(batch_enc_bbox_curr, dim=0),
        )

    def _topk_pair_queries_v1(
        self,
        memory_prev: Tensor,
        memory_curr: Tensor,
        memory_mask: Optional[Tensor],
        spatial_shapes: Tensor,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """Independent frame proposals + greedy pair matching + query fusion."""
        cfg = self.pair_proposal_cfg
        bs, _, c = memory_prev.shape
        pre_topk = int(cfg.get('pre_topk', self.num_queries * 2))
        match_score_thr = float(cfg.get('match_score_thr', 0.0))
        birth_score_thr = float(cfg.get('birth_score_thr', 0.35))
        death_score_thr = float(cfg.get('death_score_thr', 0.35))
        enable_birth = bool(cfg.get('enable_birth', True))
        enable_death = bool(cfg.get('enable_death', True))

        num_layers = self.decoder.num_layers
        query_p, ref_p, score_p, _ = self._single_frame_topk_proposals(
            memory_prev, memory_mask, spatial_shapes,
            self.bbox_head.reg_branches[num_layers], pre_topk)
        query_c, ref_c, score_c, _ = self._single_frame_topk_proposals(
            memory_curr, memory_mask, spatial_shapes,
            self.bbox_head.reg_branches_curr[num_layers], pre_topk)

        learned_query = self.decoder.query_embedding.weight.to(
            device=memory_prev.device, dtype=memory_prev.dtype)
        learned_prev = self.decoder.ref_prev_embedding.weight.sigmoid().to(
            device=memory_prev.device, dtype=memory_prev.dtype)
        learned_curr = self.decoder.ref_curr_embedding.weight.sigmoid().to(
            device=memory_prev.device, dtype=memory_prev.dtype)

        batch_queries: List[Tensor] = []
        batch_ref_prev: List[Tensor] = []
        batch_ref_curr: List[Tensor] = []
        for b in range(bs):
            pair_scores = self._pair_match_score(
                query_p[b], query_c[b], ref_p[b], ref_c[b], score_p[b],
                score_c[b])
            best_scores, best_curr = pair_scores.max(dim=1)
            order = torch.argsort(best_scores, descending=True)
            used_prev = torch.zeros(
                query_p.size(1), dtype=torch.bool, device=query_p.device)
            used_curr = torch.zeros(
                query_c.size(1), dtype=torch.bool, device=query_c.device)
            cand_q: List[Tensor] = []
            cand_prev: List[Tensor] = []
            cand_curr: List[Tensor] = []
            cand_score: List[Tensor] = []

            for pi in order:
                score = best_scores[pi]
                if score <= match_score_thr or score < -1e5:
                    break
                ci = best_curr[pi]
                if used_prev[pi] or used_curr[ci]:
                    continue
                fused = self.pair_query_fusion(
                    torch.cat([query_p[b, pi], query_c[b, ci]], dim=-1))
                cand_q.append(fused)
                cand_prev.append(ref_p[b, pi])
                cand_curr.append(ref_c[b, ci])
                cand_score.append(score)
                used_prev[pi] = True
                used_curr[ci] = True
                if len(cand_q) >= self.num_queries:
                    break

            if enable_birth and len(cand_q) < self.num_queries:
                birth_order = torch.argsort(score_c[b], descending=True)
                for ci in birth_order:
                    if used_curr[ci] or score_c[b, ci] < birth_score_thr:
                        continue
                    fused = self.pair_query_fusion(
                        torch.cat([query_c[b, ci], query_c[b, ci]], dim=-1))
                    cand_q.append(fused)
                    cand_prev.append(learned_prev[len(cand_q) - 1])
                    cand_curr.append(ref_c[b, ci])
                    cand_score.append(score_c[b, ci])
                    used_curr[ci] = True
                    if len(cand_q) >= self.num_queries:
                        break

            if enable_death and len(cand_q) < self.num_queries:
                death_order = torch.argsort(score_p[b], descending=True)
                for pi in death_order:
                    if used_prev[pi] or score_p[b, pi] < death_score_thr:
                        continue
                    fused = self.pair_query_fusion(
                        torch.cat([query_p[b, pi], query_p[b, pi]], dim=-1))
                    cand_q.append(fused)
                    cand_prev.append(ref_p[b, pi])
                    cand_curr.append(learned_curr[len(cand_q) - 1])
                    cand_score.append(score_p[b, pi])
                    used_prev[pi] = True
                    if len(cand_q) >= self.num_queries:
                        break

            if cand_q:
                q = torch.stack(cand_q)
                rp = torch.stack(cand_prev)
                rc = torch.stack(cand_curr)
                scores = torch.stack(cand_score)
                if q.size(0) > self.num_queries:
                    keep = torch.topk(scores, k=self.num_queries).indices
                    q, rp, rc = q[keep], rp[keep], rc[keep]
            else:
                q = query_p.new_zeros((0, c))
                rp = ref_p.new_zeros((0, 5))
                rc = ref_c.new_zeros((0, 5))

            pad = self.num_queries - q.size(0)
            if pad > 0:
                q = torch.cat([q, learned_query[:pad]], dim=0)
                rp = torch.cat([rp, learned_prev[:pad]], dim=0)
                rc = torch.cat([rc, learned_curr[:pad]], dim=0)

            batch_queries.append(q[:self.num_queries])
            batch_ref_prev.append(rp[:self.num_queries])
            batch_ref_curr.append(rc[:self.num_queries])

        return (
            torch.stack(batch_queries, dim=0),
            torch.stack(batch_ref_prev, dim=0),
            torch.stack(batch_ref_curr, dim=0),
        )

    @staticmethod
    def _to_pair_rbox(bboxes) -> Tensor:
        """Convert pair GT boxes to ``(N, 5)`` rbox tensor."""
        if isinstance(bboxes, RotatedBoxes):
            return bboxes.tensor
        if hasattr(bboxes, 'tensor'):
            bboxes = bboxes.tensor
        if bboxes.size(-1) == 8:
            return qbox2rbox(bboxes)
        return bboxes

    def _gt_noised_pair_queries(
        self,
        batch_data_samples: OptSampleList,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Tuple[Optional[Tensor], Tensor, Tensor]:
        """Debug init: learnable query + GT pair refs with small noise."""
        if batch_data_samples is None:
            raise ValueError(
                'query_init="gt_noised" requires batch_data_samples with '
                'pair_gt_instances')

        angle_factor = self.decoder.angle_factor
        ref_prev_learned = self.decoder.ref_prev_embedding.weight.sigmoid()
        ref_curr_learned = self.decoder.ref_curr_embedding.weight.sigmoid()
        ref_prev_list: List[Tensor] = []
        ref_curr_list: List[Tensor] = []

        for sample in batch_data_samples:
            pair_gt = sample.pair_gt_instances
            img_h, img_w = sample.metainfo['img_shape']
            factor = torch.tensor(
                [img_w, img_h, img_w, img_h, angle_factor],
                device=device,
                dtype=dtype)
            gt_prev = self._to_pair_rbox(pair_gt.bboxes_prev).to(
                device=device, dtype=dtype)
            gt_curr = self._to_pair_rbox(pair_gt.bboxes_curr).to(
                device=device, dtype=dtype)
            valid_prev = torch.as_tensor(
                pair_gt.valid_prev, device=device, dtype=torch.bool)
            valid_curr = torch.as_tensor(
                pair_gt.valid_curr, device=device, dtype=torch.bool)

            num_gt = gt_prev.size(0)
            ref_prev = ref_prev_learned.clone()
            ref_curr = ref_curr_learned.clone()
            if num_gt > 0:
                norm_prev = gt_prev / factor.unsqueeze(0)
                norm_curr = gt_curr / factor.unsqueeze(0)
                noise_prev = torch.randn_like(norm_prev) * self.gt_ref_noise_scale
                noise_curr = torch.randn_like(norm_curr) * self.gt_ref_noise_scale
                norm_prev = (norm_prev + noise_prev).clamp(1e-4, 1 - 1e-4)
                norm_curr = (norm_curr + noise_curr).clamp(1e-4, 1 - 1e-4)
                fill_n = min(num_gt, self.num_queries)
                ref_prev[:fill_n] = norm_prev[:fill_n]
                ref_curr[:fill_n] = norm_curr[:fill_n]
                for idx in range(fill_n):
                    if not valid_prev[idx]:
                        ref_prev[idx] = ref_prev_learned[idx]
                    if not valid_curr[idx]:
                        ref_curr[idx] = ref_curr_learned[idx]

            ref_prev_list.append(ref_prev.unsqueeze(0))
            ref_curr_list.append(ref_curr.unsqueeze(0))

        reference_prev = torch.cat(ref_prev_list, dim=0)
        reference_curr = torch.cat(ref_curr_list, dim=0)
        # query=None → decoder uses learnable ``query_embedding``
        return None, reference_prev, reference_curr

    def _init_pair_decoder_queries(
        self,
        memory_prev: Tensor,
        memory_curr: Tensor,
        memory_mask: Optional[Tensor],
        spatial_shapes: Tensor,
        batch_data_samples: OptSampleList = None,
    ) -> Tuple[Optional[Tensor], Optional[Tensor], Optional[Tensor],
               Optional[Tensor], Optional[Dict], Optional[Tensor],
               Optional[Tensor], Optional[Tensor], Optional[Tensor]]:
        """Select pair query / dual reference init per ``query_init``."""
        enc_cls_prev = enc_cls_curr = None
        enc_bbox_prev = enc_bbox_curr = None
        query_types = None
        if self.query_init == 'learned':
            query, reference_prev, reference_curr = None, None, None
        elif self.query_init == 'gt_noised':
            query, reference_prev, reference_curr = self._gt_noised_pair_queries(
                batch_data_samples,
                device=memory_prev.device,
                dtype=memory_prev.dtype,
            )
        elif self.query_init == 'dual_topk':
            query, reference_prev, reference_curr = self._topk_pair_queries(
                memory_prev,
                memory_curr,
                memory_mask,
                spatial_shapes,
            )
        elif self.query_init == 'pair_topk_v1':
            query, reference_prev, reference_curr = self._topk_pair_queries_v1(
                memory_prev,
                memory_curr,
                memory_mask,
                spatial_shapes,
            )
        elif self.query_init == 'pair_topk_sameidx_v1':
            query, reference_prev, reference_curr = (
                self._topk_pair_queries_sameidx_v1(
                    memory_prev,
                    memory_curr,
                    memory_mask,
                    spatial_shapes,
                ))
        elif self.query_init == 'pair_topk_v2':
            (query, reference_prev, reference_curr, enc_cls_prev,
             enc_cls_curr, enc_bbox_prev, enc_bbox_curr) = (
                 self._topk_pair_queries_v2(
                     memory_prev,
                     memory_curr,
                     memory_mask,
                     spatial_shapes,
                     batch_data_samples=batch_data_samples,
                 ))
        elif self.query_init == 'typed_pair_topk_v1':
            (query, reference_prev, reference_curr, enc_cls_prev,
             enc_cls_curr, enc_bbox_prev, enc_bbox_curr, query_types) = (
                 self._typed_pair_queries_v1(
                     memory_prev,
                     memory_curr,
                     memory_mask,
                     spatial_shapes,
                     batch_data_samples=batch_data_samples,
                 ))
            if batch_data_samples is not None:
                for sample, sample_query_types in zip(
                        batch_data_samples, query_types):
                    sample.set_metainfo(dict(
                        pair_query_types=sample_query_types.detach()))
        else:
            raise RuntimeError(f'Unsupported query_init: {self.query_init!r}')

        if (not self.training or self.pair_dn_query_generator is None
                or batch_data_samples is None):
            return (query, reference_prev, reference_curr, None, None,
                    enc_cls_prev, enc_cls_curr, enc_bbox_prev, enc_bbox_curr)

        (dn_query, dn_prev_unact, dn_curr_unact, self_attn_mask,
         dn_meta) = self.pair_dn_query_generator(batch_data_samples)
        if query is None:
            query = self.decoder.query_embedding.weight.unsqueeze(0).expand(
                dn_query.size(0), -1, -1).to(dtype=dn_query.dtype)
        if reference_prev is None:
            reference_prev = self.decoder.ref_prev_embedding.weight.sigmoid(
            ).unsqueeze(0).expand(dn_query.size(0), -1, -1)
        if reference_curr is None:
            reference_curr = self.decoder.ref_curr_embedding.weight.sigmoid(
            ).unsqueeze(0).expand(dn_query.size(0), -1, -1)
        query = torch.cat([dn_query, query], dim=1)
        reference_prev = torch.cat(
            [dn_prev_unact, torch.logit(reference_prev.clamp(1e-4, 1 - 1e-4))],
            dim=1).sigmoid()
        reference_curr = torch.cat(
            [dn_curr_unact, torch.logit(reference_curr.clamp(1e-4, 1 - 1e-4))],
            dim=1).sigmoid()
        return (query, reference_prev, reference_curr, self_attn_mask, dn_meta,
                enc_cls_prev, enc_cls_curr, enc_bbox_prev, enc_bbox_curr)

    def _pair_decoder_reg_branches(
        self,
    ) -> Tuple[nn.ModuleList, nn.ModuleList]:
        """Slice head regression branches for Pair decoder layers."""
        num_layers = self.decoder.num_layers
        return (
            self.bbox_head.reg_branches[:num_layers],
            self.bbox_head.reg_branches_curr[:num_layers],
        )

    def forward_decoder_pair(
        self,
        query: Tensor,
        memory_prev: Tensor,
        memory_curr: Tensor,
        reference_prev: Tensor,
        reference_curr: Tensor,
        spatial_shapes: Tensor,
        level_start_index: Tensor,
        self_attn_mask: Optional[Tensor] = None,
    ) -> Dict:
        """Run ``PairRotatedRTDETRTransformerDecoder``."""
        reg_branches_prev, reg_branches_curr = self._pair_decoder_reg_branches()
        hidden_states, references_prev, references_curr = self.decoder(
            memory_prev=memory_prev,
            memory_curr=memory_curr,
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index,
            reg_branches_prev=reg_branches_prev,
            reg_branches_curr=reg_branches_curr,
            query=query,
            reference_prev=reference_prev,
            reference_curr=reference_curr,
            self_attn_mask=self_attn_mask,
        )
        return dict(
            hidden_states=torch.stack(hidden_states),
            references_prev=references_prev,
            references_curr=references_curr,
        )

    def _forward_single_frame(
        self,
        memory: Tensor,
        encoder_outputs_dict: Dict,
        decoder_inputs_dict: Dict,
        batch_data_samples: OptSampleList = None,
    ) -> Dict:
        """Run ``pre_decoder`` → ``forward_decoder`` for one frame memory."""
        enc_slice = dict(
            memory=memory,
            memory_mask=encoder_outputs_dict['memory_mask'],
            spatial_shapes=encoder_outputs_dict['spatial_shapes'],
        )
        tmp_dec_in, head_inputs_dict = self.pre_decoder(
            batch_data_samples=batch_data_samples, **enc_slice)
        dec_in = dict(decoder_inputs_dict)
        dec_in.update(tmp_dec_in)
        decoder_outputs_dict = self.forward_decoder(**dec_in)
        head_inputs_dict.update(decoder_outputs_dict)
        return head_inputs_dict

    def _log_head_shapes(self, branch: str, head_inputs_dict: Dict) -> None:
        hidden_states = head_inputs_dict['hidden_states']
        references = head_inputs_dict['references']
        self._log_shape(f'{branch}_hidden_states[0]', hidden_states[0])
        self._log_shape(f'{branch}_references_cls[0]', references[0][0])
        self._log_shape(f'{branch}_references_coord[0]', references[1][0])

    def forward_transformer(
        self,
        img_feats: Tuple[Tensor, ...],
        batch_data_samples: OptSampleList = None,
    ) -> Dict:
        """Shared encoder; M2 dual decoders or M5 pair decoder."""
        timer: Optional[CudaComponentTimer] = getattr(
            self, '_active_timer', None)
        flat_batch = img_feats[0].shape[0]
        pair_batch, _ = self._split_pair_batch(flat_batch)

        if timer is not None:
            encoder_inputs_dict, decoder_inputs_dict = timer.record(
                'pre_transformer',
                lambda: self.pre_transformer(img_feats, batch_data_samples))
            encoder_outputs_dict = timer.record(
                'encoder',
                lambda: self.forward_encoder(**encoder_inputs_dict))
        else:
            encoder_inputs_dict, decoder_inputs_dict = self.pre_transformer(
                img_feats, batch_data_samples)
            encoder_outputs_dict = self.forward_encoder(**encoder_inputs_dict)

        memory = encoder_outputs_dict['memory']
        memory_prev = memory[:pair_batch]
        memory_curr = memory[pair_batch:]
        self._log_shape('memory_prev', memory_prev)
        self._log_shape('memory_curr', memory_curr)

        if not self.pair_mode:
            if timer is not None:
                head_prev = timer.record(
                    'decoder_prev',
                    lambda: self._forward_single_frame(
                        memory_prev,
                        encoder_outputs_dict,
                        decoder_inputs_dict,
                        batch_data_samples=batch_data_samples))
                head_curr = timer.record(
                    'decoder_curr',
                    lambda: self._forward_single_frame(
                        memory_curr,
                        encoder_outputs_dict,
                        decoder_inputs_dict,
                        batch_data_samples=batch_data_samples))
            else:
                head_prev = self._forward_single_frame(
                    memory_prev,
                    encoder_outputs_dict,
                    decoder_inputs_dict,
                    batch_data_samples=batch_data_samples)
                head_curr = self._forward_single_frame(
                    memory_curr,
                    encoder_outputs_dict,
                    decoder_inputs_dict,
                    batch_data_samples=batch_data_samples)
            self._log_head_shapes('prev', head_prev)
            self._log_head_shapes('curr', head_curr)
            return dict(prev=head_prev, curr=head_curr)

        if timer is not None:
            (query, reference_prev, reference_curr, self_attn_mask, dn_meta,
             enc_cls_prev, enc_cls_curr, enc_bbox_prev,
             enc_bbox_curr) = timer.record(
                'query_init',
                lambda: self._init_pair_decoder_queries(
                    memory_prev,
                    memory_curr,
                    encoder_outputs_dict['memory_mask'],
                    encoder_outputs_dict['spatial_shapes'],
                    batch_data_samples=batch_data_samples,
                ))
            pair_decoder_out = timer.record(
                'decoder',
                lambda: self.forward_decoder_pair(
                    query=query,
                    memory_prev=memory_prev,
                    memory_curr=memory_curr,
                    reference_prev=reference_prev,
                    reference_curr=reference_curr,
                    spatial_shapes=decoder_inputs_dict['spatial_shapes'],
                    level_start_index=decoder_inputs_dict['level_start_index'],
                    self_attn_mask=self_attn_mask,
                ))
        else:
            (query, reference_prev, reference_curr, self_attn_mask, dn_meta,
             enc_cls_prev, enc_cls_curr, enc_bbox_prev,
             enc_bbox_curr) = self._init_pair_decoder_queries(
                 memory_prev,
                 memory_curr,
                 encoder_outputs_dict['memory_mask'],
                 encoder_outputs_dict['spatial_shapes'],
                 batch_data_samples=batch_data_samples,
             )
            pair_decoder_out = self.forward_decoder_pair(
                query=query,
                memory_prev=memory_prev,
                memory_curr=memory_curr,
                reference_prev=reference_prev,
                reference_curr=reference_curr,
                spatial_shapes=decoder_inputs_dict['spatial_shapes'],
                level_start_index=decoder_inputs_dict['level_start_index'],
                self_attn_mask=self_attn_mask,
            )
        pair_decoder_out['dn_meta'] = dn_meta
        pair_decoder_out['enc_outputs_class_prev'] = enc_cls_prev
        pair_decoder_out['enc_outputs_class_curr'] = enc_cls_curr
        pair_decoder_out['enc_outputs_coord_prev'] = enc_bbox_prev
        pair_decoder_out['enc_outputs_coord_curr'] = enc_bbox_curr
        return pair_decoder_out

    def loss(self, batch_inputs: Tensor,
             batch_data_samples: OptSampleList) -> Dict:
        """Forward + loss with per-component CUDA timing."""
        timer = CudaComponentTimer()
        self._active_timer = timer
        try:
            img_feats = timer.record(
                'backbone_neck',
                lambda: self.extract_feat(batch_inputs))
            head_inputs_dict = self.forward_transformer(
                img_feats, batch_data_samples)
            losses = timer.record(
                'head_loss',
                lambda: self.bbox_head.loss(
                    **head_inputs_dict,
                    batch_data_samples=batch_data_samples))
        finally:
            self._active_timer = None
        timings = timer.get_durations()
        assigner = getattr(self.bbox_head, 'assigner', None)
        if assigner is not None and hasattr(assigner, 'pop_timings'):
            timings.update(assigner.pop_timings())
        self._last_component_timings = timings
        return losses

    def _forward(
        self,
        batch_inputs: Tensor,
        batch_data_samples: OptSampleList = None,
    ):
        """Return head outputs (pair mode) or dual-branch outputs (M2)."""
        img_feats = self.extract_feat(batch_inputs)
        head_inputs = self.forward_transformer(img_feats, batch_data_samples)
        if not self.pair_mode:
            prev_out = self.bbox_head.forward(**head_inputs['prev'])
            curr_out = self.bbox_head.forward(**head_inputs['curr'])
            self._log_shape('prev_output_cls[0]', prev_out[0][0])
            self._log_shape('prev_output_coord[0]', prev_out[1][0])
            self._log_shape('curr_output_cls[0]', curr_out[0][0])
            self._log_shape('curr_output_coord[0]', curr_out[1][0])
            return dict(prev=prev_out, curr=curr_out)
        return self.bbox_head.forward(
            head_inputs['hidden_states'],
            head_inputs['references_prev'],
            head_inputs['references_curr'],
        )

    def predict(
        self,
        batch_inputs: Tensor,
        batch_data_samples: SampleList,
        rescale: bool = True,
    ) -> SampleList:
        """Predict pair instances or independent prev/curr detections."""
        if isinstance(self.test_cfg, dict):
            rescale = self.test_cfg.get('rescale', rescale)

        img_feats = self.extract_feat(batch_inputs)
        head_inputs = self.forward_transformer(img_feats, batch_data_samples)

        if not self.pair_mode:
            prev_results = self.bbox_head.predict(
                **head_inputs['prev'],
                rescale=rescale,
                batch_data_samples=batch_data_samples)
            curr_results = self.bbox_head.predict(
                **head_inputs['curr'],
                rescale=rescale,
                batch_data_samples=batch_data_samples)
            for sample, prev_inst, curr_inst in zip(batch_data_samples,
                                                    prev_results, curr_results):
                sample.pred_instances = prev_inst
                sample.set_field(curr_inst, 'pred_instances_curr')
            return batch_data_samples

        results_list = self.bbox_head.predict(
            **head_inputs,
            batch_data_samples=batch_data_samples,
            rescale=rescale)
        for sample, pred in zip(batch_data_samples, results_list):
            sample.set_field(pred, 'pred_pair_instances')
        return batch_data_samples
