# Copyright (c) AI4RS. All rights reserved.
"""Pair RT-DETR transformer decoder (M3j).

One shared content query per pair with dual 5D oriented references; each layer
runs self-attention once, dual rotated deformable cross-attention, fusion, and
separate reference refinement (O2-RTDETR angle convention).
"""

from __future__ import annotations

import math
from typing import List, Optional, Tuple

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
    """Decoder layer: one self-attn + dual rotated cross-attn + linear fusion."""

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
        self.ffn = FFN(**self.ffn_cfg)
        self.norms = ModuleList([
            build_norm_layer(self.norm_cfg, self.embed_dims)[1]
            for _ in range(3)
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
        self_attn_mask: Optional[Tensor] = None,
        spatial_shapes: Optional[Tensor] = None,
        level_start_index: Optional[Tensor] = None,
        reference_points_prev: Optional[Tensor] = None,
        reference_points_curr: Optional[Tensor] = None,
        **kwargs,
    ) -> Tensor:
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
        return query


class PairRotatedRTDETRTransformerDecoder(DinoTransformerDecoder):
    """Pair RT-DETR decoder with shared query and dual oriented references."""

    def __init__(self,
                 *args,
                 num_queries: int = 300,
                 angle_factor: float = math.pi,
                 **kwargs) -> None:
        self.num_queries = num_queries
        self.angle_factor = angle_factor
        super().__init__(*args, **kwargs)

    def _init_layers(self) -> None:
        self.layers = ModuleList([
            PairRotatedRTDETRTransformerDecoderLayer(**self.layer_cfg)
            for _ in range(self.num_layers)
        ])
        self.embed_dims = self.layers[0].embed_dims
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

    def forward(
        self,
        memory_prev: Tensor,
        memory_curr: Tensor,
        spatial_shapes: Tensor,
        level_start_index: Tensor,
        reg_branches_prev: nn.ModuleList,
        reg_branches_curr: nn.ModuleList,
        key_padding_mask: Optional[Tensor] = None,
        self_attn_mask: Optional[Tensor] = None,
        valid_ratios: Optional[Tensor] = None,
        query: Optional[Tensor] = None,
        reference_prev: Optional[Tensor] = None,
        reference_curr: Optional[Tensor] = None,
        **kwargs,
    ) -> Tuple[List[Tensor], List[Tensor], List[Tensor]]:
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
        references_prev: List[Tensor] = []
        references_curr: List[Tensor] = []

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

            query = layer(
                query=query,
                value_prev=memory_prev,
                value_curr=memory_curr,
                query_pos=query_pos,
                query_pos_prev=query_pos_prev,
                query_pos_curr=query_pos_curr,
                key_padding_mask=key_padding_mask,
                self_attn_mask=self_attn_mask,
                spatial_shapes=spatial_shapes,
                level_start_index=level_start_index,
                reference_points_prev=ref_prev_input,
                reference_points_curr=ref_curr_input,
                **kwargs)

            layer_output = self.norm(query)
            tmp_prev = reg_branches_prev[lid](layer_output)
            tmp_curr = reg_branches_curr[lid](layer_output)

            new_reference_prev = tmp_prev + inverse_sigmoid(
                reference_prev, eps=1e-3)
            new_reference_prev = new_reference_prev.sigmoid()
            new_reference_curr = tmp_curr + inverse_sigmoid(
                reference_curr, eps=1e-3)
            new_reference_curr = new_reference_curr.sigmoid()
            reference_prev = new_reference_prev.detach()
            reference_curr = new_reference_curr.detach()

            hidden_states.append(layer_output)
            references_prev.append(new_reference_prev)
            references_curr.append(new_reference_curr)

        return hidden_states, references_prev, references_curr
