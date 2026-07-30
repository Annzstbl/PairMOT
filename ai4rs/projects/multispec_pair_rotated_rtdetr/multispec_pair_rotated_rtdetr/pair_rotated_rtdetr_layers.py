# Copyright (c) AI4RS. All rights reserved.
"""Pair RT-DETR transformer decoder (M3j).

One shared content query per pair with dual 5D oriented references; each layer
runs self-attention once, dual rotated deformable cross-attention, fusion, and
separate reference refinement (O2-RTDETR angle convention).
"""

from __future__ import annotations

import copy
import math
from typing import List, Optional, Tuple, Union

import torch
from mmcv.cnn import build_norm_layer
from mmcv.cnn.bricks.transformer import FFN, MultiheadAttention
from mmengine.model import ModuleList
from mmdet.models.layers.transformer import DetrTransformerDecoderLayer
from mmdet.models.layers.transformer.dino_layers import DinoTransformerDecoder
from mmdet.models.layers.transformer.utils import inverse_sigmoid
from projects.rotated_dino.rotated_dino.rotated_attention import (
    RotatedMultiScaleDeformableAttention,
)
from projects.rotated_rtdetr.rotated_rtdetr.utils import MLP
from torch import Tensor, nn


class PairRotatedRTDETRTransformerDecoderLayer(DetrTransformerDecoderLayer):
    """Decoder layer for pair states.

    The default path is the original shared-query decoder.  When
    ``tristate=True``, the layer maintains a pair ``pointer`` plus two frame
    queries.  The pointer handles pair-level self-attention; frame queries do
    frame-specific cross-attention and are used by frame heads.
    """

    def __init__(self,
                 *args,
                 tristate: bool = False,
                 tristate_separate_ffn: bool = False,
                 **kwargs) -> None:
        self.tristate = bool(tristate)
        self.tristate_separate_ffn = bool(tristate_separate_ffn)
        super().__init__(*args, **kwargs)

    @staticmethod
    def _init_pair_average_fusion(linear: nn.Linear) -> None:
        """Initialize ``[prev, curr] -> shared`` fusion as 0.5 identity sum."""
        nn.init.zeros_(linear.weight)
        nn.init.zeros_(linear.bias)
        out_dim = linear.out_features
        with torch.no_grad():
            eye = torch.eye(out_dim, device=linear.weight.device)
            linear.weight[:, :out_dim].copy_(0.5 * eye)
            linear.weight[:, out_dim:out_dim * 2].copy_(0.5 * eye)

    def _init_layers(self) -> None:
        self.self_attn = MultiheadAttention(**self.self_attn_cfg)
        self.cross_attn_prev = RotatedMultiScaleDeformableAttention(
            **self.cross_attn_cfg)
        self.cross_attn_curr = RotatedMultiScaleDeformableAttention(
            **self.cross_attn_cfg)
        self.embed_dims = self.self_attn.embed_dims
        # fuse (bs, num_queries, 2*D) -> (bs, num_queries, D)
        self.cross_fusion = nn.Linear(self.embed_dims * 2, self.embed_dims)
        self._init_pair_average_fusion(self.cross_fusion)
        if self.tristate:
            # Tri-state decoding consumes the two frame outputs directly.
            self.cross_fusion.requires_grad_(False)
            self.pointer_to_prev_gate = nn.Sequential(
                nn.Linear(self.embed_dims, self.embed_dims),
                nn.Sigmoid(),
            )
            self.pointer_to_curr_gate = nn.Sequential(
                nn.Linear(self.embed_dims, self.embed_dims),
                nn.Sigmoid(),
            )
            self.pointer_to_prev = nn.Linear(self.embed_dims, self.embed_dims)
            self.pointer_to_curr = nn.Linear(self.embed_dims, self.embed_dims)
            self.pointer_update = nn.Linear(
                self.embed_dims * 2 + 12, self.embed_dims)
        self.ffn = FFN(**self.ffn_cfg)
        if self.tristate and self.tristate_separate_ffn:
            self.ffn_prev = FFN(**copy.deepcopy(self.ffn_cfg))
            self.ffn_curr = FFN(**copy.deepcopy(self.ffn_cfg))
        self.norms = ModuleList([
            build_norm_layer(self.norm_cfg, self.embed_dims)[1]
            for _ in range(6 if self.tristate else 3)
        ])

    def forward(
        self,
        query: Tensor,
        value_prev: Tensor,
        value_curr: Tensor,
        query_pos: Tensor,
        query_pos_prev: Tensor,
        query_pos_curr: Tensor,
        key_padding_mask: Optional[Tensor] = None,
        query_key_padding_mask: Optional[Tensor] = None,
        self_attn_mask: Optional[Tensor] = None,
        spatial_shapes: Optional[Tensor] = None,
        level_start_index: Optional[Tensor] = None,
        reference_points_prev: Optional[Tensor] = None,
        reference_points_curr: Optional[Tensor] = None,
        return_frame_evidence: bool = False,
        **kwargs,
    ) -> Union[Tensor, Tuple[Tensor, Tensor, Tensor]]:
        """Forward one pair decoder layer.

        Args:
            query (Tensor): Shared pair queries, shape (bs, num_queries, D).
            value_prev (Tensor): Prev-frame memory, shape (bs, num_value, D).
            value_curr (Tensor): Curr-frame memory, shape (bs, num_value, D).
            query_pos (Tensor): Self-attn position encoding, (bs, num_queries, D).
            query_pos_prev (Tensor): Prev cross-attn pos, (bs, num_queries, D).
            query_pos_curr (Tensor): Curr cross-attn pos, (bs, num_queries, D).
            reference_points_prev (Tensor): (bs, num_queries, num_levels, 5).
            reference_points_curr (Tensor): (bs, num_queries, num_levels, 5).

        Returns:
            Tensor: Updated queries, shape (bs, num_queries, D).
        """
        query = self.self_attn(
            query=query,
            key=query,
            value=query,
            query_pos=query_pos,
            key_pos=query_pos,
            attn_mask=self_attn_mask,
            key_padding_mask=query_key_padding_mask,
            **kwargs)
        query = self.norms[0](query)

        out_prev = self.cross_attn_prev(
            query=query,
            value=value_prev,
            query_pos=query_pos_prev,
            key_padding_mask=key_padding_mask,
            reference_points=reference_points_prev,
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index,
            **kwargs)
        out_curr = self.cross_attn_curr(
            query=query,
            value=value_curr,
            query_pos=query_pos_curr,
            key_padding_mask=key_padding_mask,
            reference_points=reference_points_curr,
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index,
            **kwargs)
        # cat on embed dim: (bs, num_queries, 2*D) -> (bs, num_queries, D)
        query = self.cross_fusion(torch.cat([out_prev, out_curr], dim=-1))
        query = self.norms[1](query)
        query = self.ffn(query)
        query = self.norms[2](query)
        if return_frame_evidence:
            return query, out_prev, out_curr
        return query

    def forward_tristate(
        self,
        pointer: Tensor,
        query_prev: Tensor,
        query_curr: Tensor,
        value_prev: Tensor,
        value_curr: Tensor,
        query_pos_pointer: Tensor,
        query_pos_prev: Tensor,
        query_pos_curr: Tensor,
        key_padding_mask: Optional[Tensor] = None,
        query_key_padding_mask: Optional[Tensor] = None,
        self_attn_mask: Optional[Tensor] = None,
        spatial_shapes: Optional[Tensor] = None,
        level_start_index: Optional[Tensor] = None,
        reference_points_prev: Optional[Tensor] = None,
        reference_points_curr: Optional[Tensor] = None,
        pointer_state: Optional[Tensor] = None,
        **kwargs,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """Forward one tri-state pair decoder layer."""
        pointer = self.self_attn(
            query=pointer,
            key=pointer,
            value=pointer,
            query_pos=query_pos_pointer,
            key_pos=query_pos_pointer,
            attn_mask=self_attn_mask,
            key_padding_mask=query_key_padding_mask,
            **kwargs)
        pointer = self.norms[0](pointer)

        query_prev = query_prev + (
            self.pointer_to_prev_gate(pointer) * self.pointer_to_prev(pointer))
        query_curr = query_curr + (
            self.pointer_to_curr_gate(pointer) * self.pointer_to_curr(pointer))

        query_prev = self.cross_attn_prev(
            query=query_prev,
            value=value_prev,
            query_pos=query_pos_prev,
            key_padding_mask=key_padding_mask,
            reference_points=reference_points_prev,
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index,
            **kwargs)
        query_prev = self.norms[1](query_prev)
        query_prev = (self.ffn_prev
                      if self.tristate_separate_ffn else self.ffn)(query_prev)
        query_prev = self.norms[2](query_prev)

        query_curr = self.cross_attn_curr(
            query=query_curr,
            value=value_curr,
            query_pos=query_pos_curr,
            key_padding_mask=key_padding_mask,
            reference_points=reference_points_curr,
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index,
            **kwargs)
        query_curr = self.norms[3](query_curr)
        query_curr = (self.ffn_curr
                      if self.tristate_separate_ffn else self.ffn)(query_curr)
        query_curr = self.norms[4](query_curr)

        if pointer_state is not None:
            pointer_delta = self.pointer_update(
                torch.cat([query_prev, query_curr, pointer_state], dim=-1))
            pointer = self.norms[5](pointer + pointer_delta)
        return pointer, query_prev, query_curr


class PairRotatedRTDETRTransformerDecoder(DinoTransformerDecoder):
    """Pair RT-DETR decoder with shared query and dual oriented references."""

    def __init__(self,
                 *args,
                 num_queries: int = 300,
                 angle_factor: float = math.pi,
                 tristate_decoder: bool = False,
                 tristate_separate_ffn: bool = False,
                 tristate_zero_init_coupling: bool = False,
                 dual_output_adapter: bool = False,
                 dual_output_cls_scale: float = 1.0,
                 dual_output_reg_scale: float = 1.0,
                 dual_output_detach_adapter_input: bool = False,
                 common_motion_decoder: bool = False,
                 shared_evidence_decoder: bool = False,
                 **kwargs) -> None:
        self.num_queries = num_queries
        self.angle_factor = angle_factor
        self.tristate_decoder = bool(tristate_decoder)
        self.tristate_separate_ffn = bool(tristate_separate_ffn)
        self.tristate_zero_init_coupling = bool(tristate_zero_init_coupling)
        self.dual_output_adapter = bool(dual_output_adapter)
        self.dual_output_cls_scale = float(dual_output_cls_scale)
        self.dual_output_reg_scale = float(dual_output_reg_scale)
        self.dual_output_detach_adapter_input = bool(
            dual_output_detach_adapter_input)
        self.common_motion_decoder = bool(common_motion_decoder)
        self.shared_evidence_decoder = bool(shared_evidence_decoder)
        if self.dual_output_cls_scale < 0:
            raise ValueError('dual_output_cls_scale must be non-negative')
        if self.dual_output_reg_scale < 0:
            raise ValueError('dual_output_reg_scale must be non-negative')
        if sum((
                self.tristate_decoder,
                self.dual_output_adapter,
                self.common_motion_decoder,
        )) > 1:
            raise ValueError(
                'tristate_decoder, dual_output_adapter, and '
                'common_motion_decoder are mutually exclusive')
        if self.shared_evidence_decoder and (
                self.tristate_decoder or self.dual_output_adapter):
            raise ValueError(
                'shared_evidence_decoder is incompatible with tristate_decoder '
                'and dual_output_adapter')
        super().__init__(*args, **kwargs)

    def _init_layers(self) -> None:
        self.layers = ModuleList([
            PairRotatedRTDETRTransformerDecoderLayer(
                **self.layer_cfg,
                tristate=self.tristate_decoder,
                tristate_separate_ffn=self.tristate_separate_ffn)
            for _ in range(self.num_layers)
        ])
        if self.tristate_decoder:
            # A post-frame pointer update only seeds the next layer. The final
            # update has no loss consumer and remains the historical no-op.
            self.layers[-1].pointer_update.requires_grad_(False)
            self.layers[-1].norms[5].requires_grad_(False)
        self.embed_dims = self.layers[0].embed_dims
        if self.dual_output_adapter:
            self.dual_output_prev_adapters = ModuleList([
                nn.Linear(self.embed_dims, self.embed_dims)
                for _ in range(self.num_layers)
            ])
            self.dual_output_curr_adapters = ModuleList([
                nn.Linear(self.embed_dims, self.embed_dims)
                for _ in range(self.num_layers)
            ])
        if self.common_motion_decoder:
            self.common_motion_adapters = ModuleList([
                nn.Linear(self.embed_dims + 5, 5, bias=False)
                for _ in range(self.num_layers)
            ])
        if self.shared_evidence_decoder:
            self.shared_evidence_adapters = ModuleList([
                nn.Linear(self.embed_dims, self.embed_dims, bias=False)
                for _ in range(self.num_layers)
            ])
        if self.post_norm_cfg is not None:
            raise ValueError(f'There is not post_norm in {self._get_name()}')
        # O2-RTDETR: MLP(5 -> D) on sigmoid 5D refs (not sine encoding)
        self.ref_point_head = MLP(5, self.embed_dims * 2, self.embed_dims, 2)
        # Ordered prev→curr fusion for self-attn position encoding (2C -> C)
        self.pair_pos_fusion = nn.Linear(self.embed_dims * 2, self.embed_dims)
        PairRotatedRTDETRTransformerDecoderLayer._init_pair_average_fusion(
            self.pair_pos_fusion)
        self.norm = nn.Identity()
        # Content-independent learnable pair query / dual references (M3-1)
        self.query_embedding = nn.Embedding(self.num_queries, self.embed_dims)
        if self.tristate_decoder:
            self.query_to_prev = nn.Linear(self.embed_dims, self.embed_dims)
            self.query_to_curr = nn.Linear(self.embed_dims, self.embed_dims)
            self.query_to_pointer = nn.Linear(self.embed_dims, self.embed_dims)
            self.pointer_init_fusion = nn.Linear(
                self.embed_dims * 3 + 2, self.embed_dims)
        init_ref = torch.rand(self.num_queries, 5)
        init_ref[..., 4] = 0.5  # default angle in sigmoid space
        init_ref_unact = inverse_sigmoid(init_ref.clamp(1e-4, 1 - 1e-4))
        self.ref_prev_embedding = nn.Embedding(self.num_queries, 5)
        self.ref_curr_embedding = nn.Embedding(self.num_queries, 5)
        # Same init values, independent parameters (M3-2)
        self.ref_prev_embedding.weight.data.copy_(init_ref_unact)
        self.ref_curr_embedding.weight.data.copy_(init_ref_unact)

    @staticmethod
    def _prepare_reference_input(reference_points: Tensor, num_levels: int,
                                 angle_factor: float) -> Tensor:
        """Expand sigmoid 5D refs for multi-level deformable cross-attn.

        Args:
            reference_points (Tensor): (bs, num_queries, 5) in sigmoid space.
            num_levels (int): Number of FPN levels.
            angle_factor (float): O2-RTDETR angle scaling factor.

        Returns:
            Tensor: (bs, num_queries, num_levels, 5) with scaled angle dim.
        """
        # (bs, num_queries, 1, 5) -> (bs, num_queries, num_levels, 5)
        ref_input = reference_points.unsqueeze(2).repeat(1, 1, num_levels, 1)
        ref_input[..., -1] *= angle_factor
        return ref_input

    def _init_pair_queries(
        self,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
        query: Optional[Tensor] = None,
        reference_prev: Optional[Tensor] = None,
        reference_curr: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """Build or expand pair query and dual references for a batch."""
        if query is None:
            # (num_queries, D) -> (bs, num_queries, D)
            query = self.query_embedding.weight.unsqueeze(0).expand(
                batch_size, -1, -1)
        if reference_prev is None:
            # embedding stores unactivated refs; apply sigmoid like pre_decoder
            reference_prev = self.ref_prev_embedding.weight.sigmoid().unsqueeze(
                0).expand(batch_size, -1, -1)
        if reference_curr is None:
            reference_curr = self.ref_curr_embedding.weight.sigmoid().unsqueeze(
                0).expand(batch_size, -1, -1)
        return (
            query.to(device=device, dtype=dtype),
            reference_prev.to(device=device, dtype=dtype),
            reference_curr.to(device=device, dtype=dtype),
        )

    @staticmethod
    def _normalized_motion_evidence(
        out_prev: Tensor,
        out_curr: Tensor,
    ) -> Tensor:
        """Build scale-stable, classification-isolated signed evidence."""
        evidence = 0.5 * (out_curr - out_prev)
        rms = evidence.square().mean(dim=-1, keepdim=True).add(1e-6).sqrt()
        return (evidence / rms).detach()

    @staticmethod
    def _reference_motion(
        reference_prev: Tensor,
        reference_curr: Tensor,
    ) -> Tensor:
        """Encode signed prev-to-curr displacement with periodic angle."""
        linear_motion = (
            inverse_sigmoid(reference_curr[..., :4], eps=1e-3)
            - inverse_sigmoid(reference_prev[..., :4], eps=1e-3)
        ).tanh()
        angle_motion = torch.remainder(
            reference_curr[..., 4:5] - reference_prev[..., 4:5] + 0.5,
            1.0) - 0.5
        return torch.cat([linear_motion, 2.0 * angle_motion], dim=-1).detach()

    def _common_motion_correction(
        self,
        lid: int,
        out_prev: Tensor,
        out_curr: Tensor,
        reference_prev: Tensor,
        reference_curr: Tensor,
    ) -> Tensor:
        evidence = self._normalized_motion_evidence(out_prev, out_curr)
        reference_motion = self._reference_motion(
            reference_prev, reference_curr)
        return self.common_motion_adapters[lid](
            torch.cat([evidence, reference_motion], dim=-1))

    @staticmethod
    def _normalized_shared_evidence(
        out_prev: Tensor,
        out_curr: Tensor,
    ) -> Tensor:
        """Return swap-invariant relative disagreement between two frames."""
        difference = 0.5 * (out_curr - out_prev)
        frame_rms = (
            0.5 * (out_prev.square() + out_curr.square())
        ).mean(dim=-1, keepdim=True).add(1e-6).sqrt()
        return (difference.abs() / frame_rms).detach()

    def _shared_evidence_correction(
        self,
        lid: int,
        out_prev: Tensor,
        out_curr: Tensor,
    ) -> Tensor:
        evidence = self._normalized_shared_evidence(out_prev, out_curr)
        return self.shared_evidence_adapters[lid](evidence)

    @staticmethod
    def _init_identity_linear(linear: nn.Linear) -> None:
        nn.init.zeros_(linear.weight)
        nn.init.zeros_(linear.bias)
        with torch.no_grad():
            eye = torch.eye(linear.out_features, device=linear.weight.device)
            linear.weight[:, :linear.out_features].copy_(eye)

    def init_weights(self) -> None:
        super().init_weights()
        self.init_pair_structural_weights()

    def init_pair_structural_weights(self) -> None:
        """Restore Pair decoder invariants after detector-level init.

        ``RotatedRTDETR.init_weights`` applies Xavier initialization to every
        decoder matrix after the decoder's own initialization hook.  Keep the
        Pair-specific average/identity/zero initialization in a separate
        idempotent hook so the detector can reapply it at the end of that
        chain.
        """
        PairRotatedRTDETRTransformerDecoderLayer._init_pair_average_fusion(
            self.pair_pos_fusion)
        for layer in self.layers:
            layer._init_pair_average_fusion(layer.cross_fusion)
        if self.dual_output_adapter:
            for adapter in (
                    list(self.dual_output_prev_adapters)
                    + list(self.dual_output_curr_adapters)):
                nn.init.zeros_(adapter.weight)
                nn.init.zeros_(adapter.bias)
        if self.common_motion_decoder:
            for adapter in self.common_motion_adapters:
                nn.init.zeros_(adapter.weight)
        if self.shared_evidence_decoder:
            for adapter in self.shared_evidence_adapters:
                nn.init.zeros_(adapter.weight)
        if not self.tristate_decoder:
            return
        self._init_identity_linear(self.query_to_prev)
        self._init_identity_linear(self.query_to_curr)
        self._init_identity_linear(self.query_to_pointer)
        nn.init.zeros_(self.pointer_init_fusion.weight)
        nn.init.zeros_(self.pointer_init_fusion.bias)
        with torch.no_grad():
            out_dim = self.pointer_init_fusion.out_features
            eye = torch.eye(out_dim, device=self.pointer_init_fusion.weight.device)
            self.pointer_init_fusion.weight[:, :out_dim].copy_(eye)
        if self.tristate_zero_init_coupling:
            for layer in self.layers:
                nn.init.zeros_(layer.pointer_to_prev.weight)
                nn.init.zeros_(layer.pointer_to_prev.bias)
                nn.init.zeros_(layer.pointer_to_curr.weight)
                nn.init.zeros_(layer.pointer_to_curr.bias)
                nn.init.zeros_(layer.pointer_update.weight)
                nn.init.zeros_(layer.pointer_update.bias)

    def forward(
        self,
        memory_prev: Tensor,
        memory_curr: Tensor,
        spatial_shapes: Tensor,
        level_start_index: Tensor,
        reg_branches_prev: nn.ModuleList,
        reg_branches_curr: nn.ModuleList,
        cls_branches_prev: Optional[nn.ModuleList] = None,
        cls_branches_curr: Optional[nn.ModuleList] = None,
        initial_cls_prev: Optional[Tensor] = None,
        initial_cls_curr: Optional[Tensor] = None,
        key_padding_mask: Optional[Tensor] = None,
        query_key_padding_mask: Optional[Tensor] = None,
        self_attn_mask: Optional[Tensor] = None,
        valid_ratios: Optional[Tensor] = None,
        query: Optional[Tensor] = None,
        reference_prev: Optional[Tensor] = None,
        reference_curr: Optional[Tensor] = None,
        **kwargs,
    ) -> Tuple:
        """Run pair decoder layers.

        Args:
            memory_prev (Tensor): (bs, num_value, embed_dims).
            memory_curr (Tensor): (bs, num_value, embed_dims).
            spatial_shapes (Tensor): (num_levels, 2).
            level_start_index (Tensor): (num_levels,).
            reg_branches_prev (nn.ModuleList): Per-layer prev 5D box refine.
            reg_branches_curr (nn.ModuleList): Per-layer curr 5D box refine.
            query (Tensor, optional): Override learned content query
                (bs, num_queries, embed_dims).
            reference_prev (Tensor, optional): Sigmoid 5D refs
                (bs, num_queries, 5).
            reference_curr (Tensor, optional): Sigmoid 5D refs
                (bs, num_queries, 5).

        Returns:
            tuple:
                hidden_states: list of (bs, num_queries, embed_dims).
                references_prev: list of (bs, num_queries, 5) per-layer preds.
                references_curr: list of (bs, num_queries, 5) per-layer preds.
        """
        del valid_ratios  # RT-DETR path; kept for API compatibility
        assert self.return_intermediate
        assert reg_branches_prev is not None
        assert reg_branches_curr is not None
        assert len(reg_branches_prev) == self.num_layers
        assert len(reg_branches_curr) == self.num_layers
        if self.tristate_decoder:
            assert cls_branches_prev is not None
            assert cls_branches_curr is not None
            assert len(cls_branches_prev) >= self.num_layers
            assert len(cls_branches_curr) >= self.num_layers

        batch_size = memory_prev.shape[0]
        query, reference_prev, reference_curr = self._init_pair_queries(
            batch_size,
            memory_prev.device,
            memory_prev.dtype,
            query=query,
            reference_prev=reference_prev,
            reference_curr=reference_curr,
        )

        hidden_states: List[Tensor] = []
        hidden_states_prev: List[Tensor] = []
        hidden_states_curr: List[Tensor] = []
        references_prev: List[Tensor] = []
        references_curr: List[Tensor] = []

        if self.tristate_decoder:
            query_pos_prev = self.ref_point_head(reference_prev)
            query_pos_curr = self.ref_point_head(reference_curr)
            query_pos = self.pair_pos_fusion(
                torch.cat([query_pos_prev, query_pos_curr], dim=-1))
            if initial_cls_prev is not None and initial_cls_curr is not None:
                cls_prev_init = initial_cls_prev.detach().sigmoid().amax(
                    dim=-1, keepdim=True)
                cls_curr_init = initial_cls_curr.detach().sigmoid().amax(
                    dim=-1, keepdim=True)
                if cls_prev_init.size(1) < query.size(1):
                    pad_n = query.size(1) - cls_prev_init.size(1)
                    pad = cls_prev_init.new_zeros(
                        cls_prev_init.size(0), pad_n, 1)
                    cls_prev_init = torch.cat([pad, cls_prev_init], dim=1)
                    cls_curr_init = torch.cat([pad, cls_curr_init], dim=1)
                cls_prev_init = cls_prev_init[:, :query.size(1)]
                cls_curr_init = cls_curr_init[:, :query.size(1)]
            else:
                cls_prev_init = query.new_zeros((*query.shape[:2], 1))
                cls_curr_init = query.new_zeros((*query.shape[:2], 1))
            pointer = self.pointer_init_fusion(
                torch.cat([
                    self.query_to_pointer(query),
                    query_pos.detach(),
                    0.5 * (query_pos_prev.detach() + query_pos_curr.detach()),
                    cls_prev_init,
                    cls_curr_init,
                ], dim=-1))
            query_prev = self.query_to_prev(query)
            query_curr = self.query_to_curr(query)

        for lid, layer in enumerate(self.layers):
            num_levels = layer.cross_attn_prev.num_levels
            ref_prev_input = self._prepare_reference_input(
                reference_prev, num_levels, self.angle_factor)
            ref_curr_input = self._prepare_reference_input(
                reference_curr, num_levels, self.angle_factor)
            query_pos_prev = self.ref_point_head(reference_prev)
            query_pos_curr = self.ref_point_head(reference_curr)
            query_pos = self.pair_pos_fusion(
                torch.cat([query_pos_prev, query_pos_curr], dim=-1))

            if self.tristate_decoder:
                pointer, query_prev, query_curr = layer.forward_tristate(
                    pointer=pointer,
                    query_prev=query_prev,
                    query_curr=query_curr,
                    value_prev=memory_prev,
                    value_curr=memory_curr,
                    query_pos_pointer=query_pos,
                    query_pos_prev=query_pos_prev,
                    query_pos_curr=query_pos_curr,
                    key_padding_mask=key_padding_mask,
                    query_key_padding_mask=query_key_padding_mask,
                    self_attn_mask=self_attn_mask,
                    spatial_shapes=spatial_shapes,
                    level_start_index=level_start_index,
                    reference_points_prev=ref_prev_input,
                    reference_points_curr=ref_curr_input,
                    **kwargs)

                layer_output_prev = self.norm(query_prev)
                layer_output_curr = self.norm(query_curr)
                tmp_prev = reg_branches_prev[lid](layer_output_prev)
                tmp_curr = reg_branches_curr[lid](layer_output_curr)
            else:
                needs_frame_evidence = (
                    self.common_motion_decoder
                    or self.shared_evidence_decoder)
                layer_result = layer(
                    query=query,
                    value_prev=memory_prev,
                    value_curr=memory_curr,
                    query_pos=query_pos,
                    query_pos_prev=query_pos_prev,
                    query_pos_curr=query_pos_curr,
                    key_padding_mask=key_padding_mask,
                    query_key_padding_mask=query_key_padding_mask,
                    self_attn_mask=self_attn_mask,
                    spatial_shapes=spatial_shapes,
                    level_start_index=level_start_index,
                    reference_points_prev=ref_prev_input,
                    reference_points_curr=ref_curr_input,
                    return_frame_evidence=needs_frame_evidence,
                    **kwargs)
                if needs_frame_evidence:
                    query, frame_evidence_prev, frame_evidence_curr = (
                        layer_result)
                else:
                    query = layer_result
                if self.shared_evidence_decoder:
                    query = query + self._shared_evidence_correction(
                        lid, frame_evidence_prev, frame_evidence_curr)

                layer_output = self.norm(query)
                if self.dual_output_adapter:
                    adapter_input = (
                        layer_output.detach()
                        if self.dual_output_detach_adapter_input
                        else layer_output)
                    adapter_prev = self.dual_output_prev_adapters[lid](
                        adapter_input)
                    adapter_curr = self.dual_output_curr_adapters[lid](
                        adapter_input)
                    # The frame-specific box residual can be kept out of the
                    # classification path and, optionally, prevented from
                    # adding a second gradient path into the shared decoder.
                    reg_output_prev = layer_output + (
                        self.dual_output_reg_scale * adapter_prev)
                    reg_output_curr = layer_output + (
                        self.dual_output_reg_scale * adapter_curr)
                    layer_output_prev = layer_output + (
                        self.dual_output_cls_scale * adapter_prev)
                    layer_output_curr = layer_output + (
                        self.dual_output_cls_scale * adapter_curr)
                    tmp_prev = reg_branches_prev[lid](reg_output_prev)
                    tmp_curr = reg_branches_curr[lid](reg_output_curr)
                elif self.common_motion_decoder:
                    tmp_prev = reg_branches_prev[lid](layer_output)
                    tmp_curr = reg_branches_curr[lid](layer_output)
                    common_motion = self._common_motion_correction(
                        lid,
                        frame_evidence_prev,
                        frame_evidence_curr,
                        reference_prev,
                        reference_curr)
                    tmp_prev = tmp_prev - common_motion
                    tmp_curr = tmp_curr + common_motion
                else:
                    tmp_prev = reg_branches_prev[lid](layer_output)
                    tmp_curr = reg_branches_curr[lid](layer_output)

            new_reference_prev = tmp_prev + inverse_sigmoid(
                reference_prev, eps=1e-3)
            new_reference_prev = new_reference_prev.sigmoid()
            new_reference_curr = tmp_curr + inverse_sigmoid(
                reference_curr, eps=1e-3)
            new_reference_curr = new_reference_curr.sigmoid()
            if self.tristate_decoder:
                cls_prev = cls_branches_prev[lid](layer_output_prev).detach()
                cls_curr = cls_branches_curr[lid](layer_output_curr).detach()
                cls_prev_score = cls_prev.sigmoid().amax(dim=-1, keepdim=True)
                cls_curr_score = cls_curr.sigmoid().amax(dim=-1, keepdim=True)
                pointer_state = torch.cat([
                    new_reference_prev.detach(),
                    new_reference_curr.detach(),
                    cls_prev_score,
                    cls_curr_score,
                ], dim=-1)
                pointer = layer.norms[5](pointer + layer.pointer_update(
                    torch.cat([
                        layer_output_prev,
                        layer_output_curr,
                        pointer_state,
                    ], dim=-1)))
            reference_prev = new_reference_prev.detach()
            reference_curr = new_reference_curr.detach()

            if self.tristate_decoder:
                hidden_states.append(pointer)
                hidden_states_prev.append(layer_output_prev)
                hidden_states_curr.append(layer_output_curr)
            else:
                hidden_states.append(layer_output)
                if self.dual_output_adapter:
                    hidden_states_prev.append(layer_output_prev)
                    hidden_states_curr.append(layer_output_curr)
            references_prev.append(new_reference_prev)
            references_curr.append(new_reference_curr)

        if self.tristate_decoder or self.dual_output_adapter:
            return (hidden_states, references_prev, references_curr,
                    hidden_states_prev, hidden_states_curr)
        return hidden_states, references_prev, references_curr
