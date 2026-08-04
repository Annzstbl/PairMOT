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
                 symmetric_pair_decoder: bool = False,
                 symmetric_feature_decoder: bool = False,
                 residual_preserving_fusion_decoder: bool = False,
                 **kwargs) -> None:
        self.tristate = bool(tristate)
        self.tristate_separate_ffn = bool(tristate_separate_ffn)
        self.symmetric_pair_decoder = bool(symmetric_pair_decoder)
        self.symmetric_feature_decoder = bool(symmetric_feature_decoder)
        self.residual_preserving_fusion_decoder = bool(
            residual_preserving_fusion_decoder)
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
        query = self._fuse_frame_features(
            out_prev, out_curr, shared_query=query)
        query = self.norms[1](query)
        query = self.ffn(query)
        query = self.norms[2](query)
        if return_frame_evidence:
            return query, out_prev, out_curr
        return query

    def _fuse_frame_features(
        self,
        out_prev: Tensor,
        out_curr: Tensor,
        shared_query: Optional[Tensor] = None,
    ) -> Tensor:
        """Fuse frame evidence into the shared recurrent query.

        The feature-only symmetric mode removes concatenation-order bias with
        one invocation of the existing fusion layer. It preserves independent
        frame cross-attention, the ordered pair-position path, parameter count,
        and matrix-multiplication count.
        """
        if self.symmetric_pair_decoder:
            # The common query should not depend on the concatenation order at
            # this fusion site.  Averaging both orders enforces that invariant
            # while retaining the existing Linear parameterization and its
            # pair-average initialization.
            return 0.5 * (
                self.cross_fusion(torch.cat([out_prev, out_curr], dim=-1))
                + self.cross_fusion(
                    torch.cat([out_curr, out_prev], dim=-1)))
        if self.symmetric_feature_decoder:
            pair_mean = 0.5 * (out_prev + out_curr)
            return self.cross_fusion(
                torch.cat([pair_mean, pair_mean], dim=-1))
        if self.residual_preserving_fusion_decoder:
            if shared_query is None:
                raise ValueError(
                    'shared_query is required for residual-preserving '
                    'feature fusion')
            innovation_prev = out_prev - shared_query
            innovation_curr = out_curr - shared_query
            return shared_query + self.cross_fusion(torch.cat(
                [innovation_prev, innovation_curr], dim=-1))
        return self.cross_fusion(torch.cat([out_prev, out_curr], dim=-1))

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
                 competitive_evidence_decoder: bool = False,
                 motion_trust_decoder: bool = False,
                 symmetric_pair_decoder: bool = False,
                 symmetric_position_decoder: bool = False,
                 symmetric_feature_decoder: bool = False,
                  residual_preserving_fusion_decoder: bool = False,
                  pair_shared_shape_refinement_decoder: bool = False,
                   pair_shared_angle_refinement_decoder: bool = False,
                   pair_shared_periodic_angle_refinement_decoder: bool = False,
                   pair_shared_log_size_periodic_angle_refinement_decoder:
                   bool = False,
                   pair_shared_log_area_periodic_angle_refinement_decoder:
                   bool = False,
                   pair_shared_late_log_size_periodic_angle_refinement_decoder:
                   bool = False,
                   pair_shared_terminal_log_size_periodic_angle_refinement_decoder:
                   bool = False,
                   pair_shared_terminal_log_area_periodic_angle_refinement_decoder:
                   bool = False,
                   pair_shared_terminal_periodic_angle_refinement_decoder:
                   bool = False,
                   pair_shared_terminal_log_size_refinement_decoder:
                   bool = False,
                    pair_shared_terminal_normalized_center_refinement_decoder:
                    bool = False,
                    pair_shared_terminal_full_tangent_refinement_decoder:
                    bool = False,
                   pair_shared_terminal_transport_center_tangent_refinement_decoder:
                   bool = False,
                   pair_shared_terminal_transport_shape_tangent_refinement_decoder:
                   bool = False,
                   pair_shared_terminal_transport_product_tangent_refinement_decoder:
                   bool = False,
                   pair_shared_terminal_transport_shared_metric_product_tangent_refinement_decoder:
                   bool = False,
                   pair_shared_terminal_transport_body_frame_product_tangent_refinement_decoder:
                   bool = False,
                   pair_shared_terminal_transport_se2_product_tangent_refinement_decoder:
                   bool = False,
                   pair_shared_terminal_transport_frenet_product_tangent_refinement_decoder:
                   bool = False,
                   pair_shared_terminal_transport_axis_frenet_product_tangent_refinement_decoder:
                   bool = False,
                   pair_shared_terminal_transport_tangent_refinement_decoder:
                   bool = False,
                   pair_shared_terminal_transport_plane_refinement_decoder:
                   bool = False,
                   terminal_position_tangent_product_decoder: bool = False,
                   terminal_position_tangent_transport_decoder: bool = False,
                   terminal_position_tangent_plane_decoder: bool = False,
                   pair_shared_progressive_log_shape_periodic_angle_refinement_decoder:
                   bool = False,
                   pair_shared_normalized_center_refinement_decoder:
                   bool = False,
                   frame_evidence_cls_decoder: bool = False,
                   frame_detail_cls_decoder: bool = False,
                   shared_routing_decoder: bool = False,
                 shared_attention_decoder: bool = False,
                 antisymmetric_detail_decoder: bool = False,
                 enveloped_detail_decoder: bool = False,
                 regression_enveloped_detail_decoder: bool = False,
                 midpoint_regression_enveloped_detail_decoder: bool = False,
                 classification_enveloped_detail_decoder: bool = False,
                 terminal_enveloped_detail_decoder: bool = False,
                 terminal_midpoint_enveloped_detail_decoder: bool = False,
                 terminal_regression_enveloped_detail_decoder: bool = False,
                 terminal_midpoint_regression_enveloped_detail_decoder:
                 bool = False,
                 common_evidence_bypass_decoder: bool = False,
                 terminal_common_evidence_bypass_decoder: bool = False,
                 terminal_classification_common_evidence_decoder:
                 bool = False,
                 terminal_factorized_evidence_decoder: bool = False,
                 terminal_factorized_confidence: str = 'none',
                 terminal_factorized_diagonal_gates: bool = False,
                 terminal_factorized_coupled_gate: bool = False,
                 terminal_factorized_center_motion_only: bool = False,
                 terminal_factorized_detail_only: bool = False,
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
        self.competitive_evidence_decoder = bool(
            competitive_evidence_decoder)
        self.motion_trust_decoder = bool(motion_trust_decoder)
        self.symmetric_pair_decoder = bool(symmetric_pair_decoder)
        self.symmetric_position_decoder = bool(symmetric_position_decoder)
        self.symmetric_feature_decoder = bool(symmetric_feature_decoder)
        self.residual_preserving_fusion_decoder = bool(
            residual_preserving_fusion_decoder)
        self.pair_shared_shape_refinement_decoder = bool(
            pair_shared_shape_refinement_decoder)
        self.pair_shared_angle_refinement_decoder = bool(
            pair_shared_angle_refinement_decoder)
        self.pair_shared_periodic_angle_refinement_decoder = bool(
            pair_shared_periodic_angle_refinement_decoder)
        self.pair_shared_log_size_periodic_angle_refinement_decoder = bool(
            pair_shared_log_size_periodic_angle_refinement_decoder)
        self.pair_shared_log_area_periodic_angle_refinement_decoder = bool(
            pair_shared_log_area_periodic_angle_refinement_decoder)
        self.pair_shared_late_log_size_periodic_angle_refinement_decoder = (
            bool(pair_shared_late_log_size_periodic_angle_refinement_decoder))
        self.pair_shared_terminal_log_size_periodic_angle_refinement_decoder = (
            bool(
                pair_shared_terminal_log_size_periodic_angle_refinement_decoder))
        self.pair_shared_terminal_log_area_periodic_angle_refinement_decoder = (
            bool(
                pair_shared_terminal_log_area_periodic_angle_refinement_decoder))
        self.pair_shared_terminal_periodic_angle_refinement_decoder = bool(
            pair_shared_terminal_periodic_angle_refinement_decoder)
        self.pair_shared_terminal_log_size_refinement_decoder = bool(
            pair_shared_terminal_log_size_refinement_decoder)
        self.pair_shared_terminal_normalized_center_refinement_decoder = bool(
            pair_shared_terminal_normalized_center_refinement_decoder)
        self.pair_shared_terminal_full_tangent_refinement_decoder = bool(
            pair_shared_terminal_full_tangent_refinement_decoder)
        self.pair_shared_terminal_transport_center_tangent_refinement_decoder = (
            bool(
                pair_shared_terminal_transport_center_tangent_refinement_decoder))
        self.pair_shared_terminal_transport_shape_tangent_refinement_decoder = (
            bool(
                pair_shared_terminal_transport_shape_tangent_refinement_decoder))
        self.pair_shared_terminal_transport_product_tangent_refinement_decoder = (
            bool(
                pair_shared_terminal_transport_product_tangent_refinement_decoder))
        self.pair_shared_terminal_transport_shared_metric_product_tangent_refinement_decoder = (
            bool(
                pair_shared_terminal_transport_shared_metric_product_tangent_refinement_decoder))
        self.pair_shared_terminal_transport_body_frame_product_tangent_refinement_decoder = (
            bool(
                pair_shared_terminal_transport_body_frame_product_tangent_refinement_decoder))
        self.pair_shared_terminal_transport_se2_product_tangent_refinement_decoder = (
            bool(
                pair_shared_terminal_transport_se2_product_tangent_refinement_decoder))
        self.pair_shared_terminal_transport_frenet_product_tangent_refinement_decoder = (
            bool(
                pair_shared_terminal_transport_frenet_product_tangent_refinement_decoder))
        self.pair_shared_terminal_transport_axis_frenet_product_tangent_refinement_decoder = (
            bool(
                pair_shared_terminal_transport_axis_frenet_product_tangent_refinement_decoder))
        self.pair_shared_terminal_transport_tangent_refinement_decoder = bool(
            pair_shared_terminal_transport_tangent_refinement_decoder)
        self.pair_shared_terminal_transport_plane_refinement_decoder = bool(
            pair_shared_terminal_transport_plane_refinement_decoder)
        self.terminal_position_tangent_product_decoder = bool(
            terminal_position_tangent_product_decoder)
        self.terminal_position_tangent_transport_decoder = bool(
            terminal_position_tangent_transport_decoder)
        self.terminal_position_tangent_plane_decoder = bool(
            terminal_position_tangent_plane_decoder)
        self.pair_shared_progressive_log_shape_periodic_angle_refinement_decoder = (
            bool(
                pair_shared_progressive_log_shape_periodic_angle_refinement_decoder))
        self.pair_shared_normalized_center_refinement_decoder = bool(
            pair_shared_normalized_center_refinement_decoder)
        self.frame_evidence_cls_decoder = bool(
            frame_evidence_cls_decoder)
        self.frame_detail_cls_decoder = bool(frame_detail_cls_decoder)
        self.shared_routing_decoder = bool(shared_routing_decoder)
        self.shared_attention_decoder = bool(shared_attention_decoder)
        self.antisymmetric_detail_decoder = bool(
            antisymmetric_detail_decoder)
        self.enveloped_detail_decoder = bool(enveloped_detail_decoder)
        self.regression_enveloped_detail_decoder = bool(
            regression_enveloped_detail_decoder)
        self.midpoint_regression_enveloped_detail_decoder = bool(
            midpoint_regression_enveloped_detail_decoder)
        self.classification_enveloped_detail_decoder = bool(
            classification_enveloped_detail_decoder)
        self.terminal_enveloped_detail_decoder = bool(
            terminal_enveloped_detail_decoder)
        self.terminal_midpoint_enveloped_detail_decoder = bool(
            terminal_midpoint_enveloped_detail_decoder)
        self.terminal_regression_enveloped_detail_decoder = bool(
            terminal_regression_enveloped_detail_decoder)
        self.terminal_midpoint_regression_enveloped_detail_decoder = bool(
            terminal_midpoint_regression_enveloped_detail_decoder)
        self.common_evidence_bypass_decoder = bool(
            common_evidence_bypass_decoder)
        self.terminal_common_evidence_bypass_decoder = bool(
            terminal_common_evidence_bypass_decoder)
        self.terminal_classification_common_evidence_decoder = bool(
            terminal_classification_common_evidence_decoder)
        self.terminal_factorized_evidence_decoder = bool(
            terminal_factorized_evidence_decoder)
        self.terminal_factorized_confidence = str(
            terminal_factorized_confidence)
        self.terminal_factorized_diagonal_gates = bool(
            terminal_factorized_diagonal_gates)
        self.terminal_factorized_coupled_gate = bool(
            terminal_factorized_coupled_gate)
        self.terminal_factorized_center_motion_only = bool(
            terminal_factorized_center_motion_only)
        self.terminal_factorized_detail_only = bool(
            terminal_factorized_detail_only)
        if sum((
                self.pair_shared_shape_refinement_decoder,
                self.pair_shared_angle_refinement_decoder,
                self.pair_shared_periodic_angle_refinement_decoder,
                self.pair_shared_log_size_periodic_angle_refinement_decoder,
                self.pair_shared_log_area_periodic_angle_refinement_decoder,
                self.
                pair_shared_late_log_size_periodic_angle_refinement_decoder,
                self.
                pair_shared_terminal_log_size_periodic_angle_refinement_decoder,
                self.
                pair_shared_terminal_log_area_periodic_angle_refinement_decoder,
                self.
                pair_shared_terminal_periodic_angle_refinement_decoder,
                self.
                pair_shared_terminal_log_size_refinement_decoder,
                self.
                pair_shared_terminal_normalized_center_refinement_decoder,
                self.
                pair_shared_terminal_full_tangent_refinement_decoder,
                self.
                pair_shared_terminal_transport_center_tangent_refinement_decoder,
                self.
                pair_shared_terminal_transport_shape_tangent_refinement_decoder,
                self.
                pair_shared_terminal_transport_product_tangent_refinement_decoder,
                self.
                pair_shared_terminal_transport_shared_metric_product_tangent_refinement_decoder,
                self.
                pair_shared_terminal_transport_body_frame_product_tangent_refinement_decoder,
                self.
                pair_shared_terminal_transport_se2_product_tangent_refinement_decoder,
                self.
                pair_shared_terminal_transport_frenet_product_tangent_refinement_decoder,
                self.
                pair_shared_terminal_transport_axis_frenet_product_tangent_refinement_decoder,
                self.
                pair_shared_terminal_transport_tangent_refinement_decoder,
                self.
                pair_shared_terminal_transport_plane_refinement_decoder,
                self.terminal_position_tangent_product_decoder,
                self.terminal_position_tangent_transport_decoder,
                self.terminal_position_tangent_plane_decoder,
                self.
                pair_shared_progressive_log_shape_periodic_angle_refinement_decoder,
                self.pair_shared_normalized_center_refinement_decoder,
        )) > 1:
            raise ValueError(
                'pair-shared shape, residual-angle, periodic-angle, '
                'log-size-periodic-angle, log-area-periodic-angle, '
                'late-log-size-periodic-angle, terminal-log-size-periodic-'
                'angle, terminal-log-area-periodic-angle, terminal-periodic-'
                'angle, terminal-log-size, terminal-normalized-center, '
                'terminal-full-tangent, '
                'terminal-transport-center-tangent, '
                'terminal-transport-shape-tangent, terminal-transport-product-'
                'tangent, terminal-transport-shared-metric-product-tangent, '
                'terminal-transport-body-frame-product-tangent, '
                'terminal-transport-SE2-product-tangent, '
                'terminal-transport-Frenet-product-tangent, '
                'terminal-transport-axis-Frenet-product-tangent, '
                'terminal-transport-tangent, '
                'terminal-transport-plane, '
                'tangent-product, terminal-position-tangent-transport, '
                'terminal-position-tangent-plane, '
                'progressive-log-shape-'
                'periodic-angle, and '
                'normalized-center '
                'refinement decoders are mutually exclusive')
        if (self.frame_evidence_cls_decoder
                and self.frame_detail_cls_decoder):
            raise ValueError(
                'frame-evidence and frame-detail classification decoders '
                'are mutually exclusive')
        if self.terminal_factorized_confidence not in {
                'none', 'common', 'detail', 'both'}:
            raise ValueError(
                'terminal_factorized_confidence must be one of '
                "'none', 'common', 'detail', or 'both'")
        if (self.terminal_factorized_confidence != 'none'
                and not self.terminal_factorized_evidence_decoder):
            raise ValueError(
                'terminal_factorized_confidence requires '
                'terminal_factorized_evidence_decoder')
        if (self.terminal_factorized_diagonal_gates
                and not self.terminal_factorized_evidence_decoder):
            raise ValueError(
                'terminal_factorized_diagonal_gates requires '
                'terminal_factorized_evidence_decoder')
        if (self.terminal_factorized_coupled_gate
                and not self.terminal_factorized_diagonal_gates):
            raise ValueError(
                'terminal_factorized_coupled_gate requires '
                'terminal_factorized_diagonal_gates')
        if (self.terminal_factorized_center_motion_only
                and not self.terminal_factorized_evidence_decoder):
            raise ValueError(
                'terminal_factorized_center_motion_only requires '
                'terminal_factorized_evidence_decoder')
        if (self.terminal_factorized_detail_only
                and not self.terminal_factorized_evidence_decoder):
            raise ValueError(
                'terminal_factorized_detail_only requires '
                'terminal_factorized_evidence_decoder')
        if (self.terminal_factorized_detail_only
                and self.terminal_factorized_coupled_gate):
            raise ValueError(
                'terminal_factorized_detail_only is incompatible with '
                'terminal_factorized_coupled_gate')
        if (self.terminal_factorized_detail_only
                and self.terminal_factorized_confidence in {
                    'common', 'both'}):
            raise ValueError(
                'terminal_factorized_detail_only cannot apply common '
                'confidence')
        if self.dual_output_cls_scale < 0:
            raise ValueError('dual_output_cls_scale must be non-negative')
        if self.dual_output_reg_scale < 0:
            raise ValueError('dual_output_reg_scale must be non-negative')
        if sum((
                self.tristate_decoder,
                self.dual_output_adapter,
                self.common_motion_decoder,
                self.antisymmetric_detail_decoder,
                (self.enveloped_detail_decoder
                 or self.regression_enveloped_detail_decoder
                 or self.midpoint_regression_enveloped_detail_decoder
                 or self.classification_enveloped_detail_decoder
                 or self._terminal_enveloped_detail_enabled
                 or self._common_evidence_bypass_enabled
                 or
                 self.terminal_classification_common_evidence_decoder),
        )) > 1:
            raise ValueError(
                'tristate_decoder, dual_output_adapter, and '
                'common_motion_decoder, antisymmetric_detail_decoder, '
                'and the bounded evidence decoder family are mutually '
                'exclusive')
        if self.shared_evidence_decoder and (
                self.tristate_decoder
                or self.dual_output_adapter
                or self.antisymmetric_detail_decoder
                or self.enveloped_detail_decoder
                or self.regression_enveloped_detail_decoder
                or self.midpoint_regression_enveloped_detail_decoder
                or self.classification_enveloped_detail_decoder
                or self._terminal_enveloped_detail_enabled
                or self._common_evidence_bypass_enabled
                or
                self.terminal_classification_common_evidence_decoder):
            raise ValueError(
                'shared_evidence_decoder is incompatible with tristate_decoder '
                'dual_output_adapter, and antisymmetric_detail_decoder')
        if self.competitive_evidence_decoder and (
                self.tristate_decoder
                or self.dual_output_adapter
                or self.shared_evidence_decoder
                or self.antisymmetric_detail_decoder
                or self.enveloped_detail_decoder
                or self.regression_enveloped_detail_decoder
                or self.midpoint_regression_enveloped_detail_decoder
                or self.classification_enveloped_detail_decoder
                or self._terminal_enveloped_detail_enabled
                or self._common_evidence_bypass_enabled
                or
                self.terminal_classification_common_evidence_decoder):
            raise ValueError(
                'competitive_evidence_decoder is incompatible with '
                'tristate_decoder, dual_output_adapter, and '
                'shared_evidence_decoder')
        if self.motion_trust_decoder and any((
                self.tristate_decoder,
                self.dual_output_adapter,
                self.common_motion_decoder,
                self.competitive_evidence_decoder,
                self.antisymmetric_detail_decoder,
                self.enveloped_detail_decoder,
                self.regression_enveloped_detail_decoder,
                self.midpoint_regression_enveloped_detail_decoder,
                self.classification_enveloped_detail_decoder,
                self._terminal_enveloped_detail_enabled,
                self.terminal_classification_common_evidence_decoder,
                self._common_evidence_bypass_enabled,
        )):
            raise ValueError(
                'motion_trust_decoder is incompatible with tristate, '
                'dual-output, common-motion, and competitive-evidence '
                'decoder variants')
        if self.symmetric_pair_decoder and any((
                self.symmetric_position_decoder,
                self.symmetric_feature_decoder,
                self.residual_preserving_fusion_decoder,
                self.tristate_decoder,
                self.dual_output_adapter,
                self.common_motion_decoder,
                self.shared_evidence_decoder,
                self.competitive_evidence_decoder,
                self.motion_trust_decoder,
                self.shared_routing_decoder,
                self.shared_attention_decoder,
                self.antisymmetric_detail_decoder,
                self.enveloped_detail_decoder,
                self.regression_enveloped_detail_decoder,
                self.midpoint_regression_enveloped_detail_decoder,
                self.classification_enveloped_detail_decoder,
                self._terminal_enveloped_detail_enabled,
                self.terminal_classification_common_evidence_decoder,
                self._common_evidence_bypass_enabled,
        )):
            raise ValueError(
                'symmetric_pair_decoder is incompatible with all other '
                'decoder variants')
        if self.symmetric_feature_decoder and any((
                self.symmetric_position_decoder,
                self.residual_preserving_fusion_decoder,
                self.tristate_decoder,
                self.dual_output_adapter,
                self.common_motion_decoder,
                self.shared_evidence_decoder,
                self.competitive_evidence_decoder,
                self.motion_trust_decoder,
                self.shared_routing_decoder,
                self.shared_attention_decoder,
                self.antisymmetric_detail_decoder,
                self.enveloped_detail_decoder,
                self.regression_enveloped_detail_decoder,
                self.midpoint_regression_enveloped_detail_decoder,
                self.classification_enveloped_detail_decoder,
                self._terminal_enveloped_detail_enabled,
                self.terminal_classification_common_evidence_decoder,
                self._common_evidence_bypass_enabled,
        )):
            raise ValueError(
                'symmetric_feature_decoder is incompatible with all other '
                'decoder variants')
        if self.residual_preserving_fusion_decoder and any((
                self.symmetric_pair_decoder,
                self.symmetric_feature_decoder,
                self.tristate_decoder,
                self.dual_output_adapter,
                self.common_motion_decoder,
                self.shared_evidence_decoder,
                self.competitive_evidence_decoder,
                self.motion_trust_decoder,
                self.shared_routing_decoder,
                self.shared_attention_decoder,
                self.antisymmetric_detail_decoder,
                self.enveloped_detail_decoder,
                self.regression_enveloped_detail_decoder,
                self.midpoint_regression_enveloped_detail_decoder,
                self.classification_enveloped_detail_decoder,
                self._terminal_enveloped_detail_enabled,
                self.terminal_classification_common_evidence_decoder,
                self._common_evidence_bypass_enabled,
        )):
            raise ValueError(
                'residual_preserving_fusion_decoder is incompatible with '
                'decoder variants other than symmetric_position_decoder')
        if self.shared_routing_decoder and any((
                self.tristate_decoder,
                self.dual_output_adapter,
                self.common_motion_decoder,
                self.shared_evidence_decoder,
                self.competitive_evidence_decoder,
                self.motion_trust_decoder,
                self.symmetric_pair_decoder,
                self.shared_attention_decoder,
                self.antisymmetric_detail_decoder,
                self.enveloped_detail_decoder,
                self.regression_enveloped_detail_decoder,
                self.midpoint_regression_enveloped_detail_decoder,
                self.classification_enveloped_detail_decoder,
                self._terminal_enveloped_detail_enabled,
                self._common_evidence_bypass_enabled,
                self.terminal_classification_common_evidence_decoder,
        )):
            raise ValueError(
                'shared_routing_decoder is incompatible with all other '
                'decoder variants')
        if self.shared_attention_decoder and any((
                self.tristate_decoder,
                self.dual_output_adapter,
                self.common_motion_decoder,
                self.competitive_evidence_decoder,
                self.symmetric_pair_decoder,
                self.shared_routing_decoder,
                self._common_evidence_bypass_enabled,
        )):
            raise ValueError(
                'shared_attention_decoder is incompatible with tristate, '
                'dual-output, common-motion, competitive-evidence, '
                'symmetric-pair, and shared-routing '
                'decoder variants')
        if self.antisymmetric_detail_decoder and any((
                self.shared_evidence_decoder,
                self.competitive_evidence_decoder,
                self.motion_trust_decoder,
                self.symmetric_pair_decoder,
                self.shared_routing_decoder,
                self.enveloped_detail_decoder,
                self.regression_enveloped_detail_decoder,
                self.midpoint_regression_enveloped_detail_decoder,
                self.classification_enveloped_detail_decoder,
                self._terminal_enveloped_detail_enabled,
                self._common_evidence_bypass_enabled,
                self.terminal_classification_common_evidence_decoder,
        )):
            raise ValueError(
                'antisymmetric_detail_decoder is incompatible with '
                'shared-evidence, competitive-evidence, motion-trust, '
                'symmetric-pair, and shared-routing decoder variants')
        if self.enveloped_detail_decoder and any((
                self.shared_evidence_decoder,
                self.competitive_evidence_decoder,
                self.motion_trust_decoder,
                self.symmetric_pair_decoder,
                self.shared_routing_decoder,
                self.regression_enveloped_detail_decoder,
                self.midpoint_regression_enveloped_detail_decoder,
                self.classification_enveloped_detail_decoder,
                self._terminal_enveloped_detail_enabled,
                self.terminal_classification_common_evidence_decoder,
        )):
            raise ValueError(
                'enveloped_detail_decoder is incompatible with decoder '
                'variants other than shared_attention_decoder')
        if self._common_evidence_bypass_enabled and any((
                self.shared_evidence_decoder,
                self.competitive_evidence_decoder,
                self.motion_trust_decoder,
                self.symmetric_pair_decoder,
                self.shared_routing_decoder,
                self.shared_attention_decoder,
                self.antisymmetric_detail_decoder,
                self.regression_enveloped_detail_decoder,
                self.midpoint_regression_enveloped_detail_decoder,
                self.classification_enveloped_detail_decoder,
                self._terminal_enveloped_detail_enabled,
                self.terminal_classification_common_evidence_decoder,
        )):
            raise ValueError(
                'common evidence bypass decoder is incompatible with other '
                'decoder variants')
        if (self.common_evidence_bypass_decoder
                and self.terminal_common_evidence_bypass_decoder):
            raise ValueError(
                'common_evidence_bypass_decoder and '
                'terminal_common_evidence_bypass_decoder are mutually '
                'exclusive')
        if self.regression_enveloped_detail_decoder and any((
                self.shared_evidence_decoder,
                self.competitive_evidence_decoder,
                self.motion_trust_decoder,
                self.symmetric_pair_decoder,
                self.shared_routing_decoder,
                self.antisymmetric_detail_decoder,
                self.enveloped_detail_decoder,
                self.midpoint_regression_enveloped_detail_decoder,
                self.classification_enveloped_detail_decoder,
                self._terminal_enveloped_detail_enabled,
                self._common_evidence_bypass_enabled,
                self.terminal_classification_common_evidence_decoder,
        )):
            raise ValueError(
                'regression_enveloped_detail_decoder is incompatible with '
                'decoder variants other than shared_attention_decoder')
        if self.midpoint_regression_enveloped_detail_decoder and any((
                self.shared_evidence_decoder,
                self.competitive_evidence_decoder,
                self.motion_trust_decoder,
                self.symmetric_pair_decoder,
                self.shared_routing_decoder,
                self.antisymmetric_detail_decoder,
                self.enveloped_detail_decoder,
                self.regression_enveloped_detail_decoder,
                self.classification_enveloped_detail_decoder,
                self._terminal_enveloped_detail_enabled,
                self._common_evidence_bypass_enabled,
                self.terminal_classification_common_evidence_decoder,
        )):
            raise ValueError(
                'midpoint_regression_enveloped_detail_decoder is '
                'incompatible with decoder variants other than '
                'shared_attention_decoder')
        if self.classification_enveloped_detail_decoder and any((
                self.shared_evidence_decoder,
                self.competitive_evidence_decoder,
                self.motion_trust_decoder,
                self.symmetric_pair_decoder,
                self.shared_routing_decoder,
                self.antisymmetric_detail_decoder,
                self.enveloped_detail_decoder,
                self.regression_enveloped_detail_decoder,
                self.midpoint_regression_enveloped_detail_decoder,
                self._terminal_enveloped_detail_enabled,
                self._common_evidence_bypass_enabled,
                self.terminal_classification_common_evidence_decoder,
        )):
            raise ValueError(
                'classification_enveloped_detail_decoder is incompatible '
                'with decoder variants other than shared_attention_decoder')
        terminal_detail_modes = (
            self.terminal_enveloped_detail_decoder,
            self.terminal_midpoint_enveloped_detail_decoder,
            self.terminal_regression_enveloped_detail_decoder,
            self.terminal_midpoint_regression_enveloped_detail_decoder,
            self.terminal_factorized_evidence_decoder,
            self.terminal_position_tangent_product_decoder,
            self.terminal_position_tangent_transport_decoder,
            self.terminal_position_tangent_plane_decoder,
        )
        if sum(bool(mode) for mode in terminal_detail_modes) > 1:
            raise ValueError(
                'terminal detail decoder modes are mutually exclusive')
        if self._terminal_enveloped_detail_enabled and any((
                self.shared_evidence_decoder,
                self.competitive_evidence_decoder,
                self.motion_trust_decoder,
                self.symmetric_pair_decoder,
                self.shared_routing_decoder,
                self.antisymmetric_detail_decoder,
                self.enveloped_detail_decoder,
                self.regression_enveloped_detail_decoder,
                self.midpoint_regression_enveloped_detail_decoder,
                self.classification_enveloped_detail_decoder,
                self._common_evidence_bypass_enabled,
                self.terminal_classification_common_evidence_decoder,
        )):
            raise ValueError(
                'terminal detail decoder is incompatible with '
                'decoder variants other than shared_attention_decoder')
        super().__init__(*args, **kwargs)
        if self.shared_routing_decoder:
            for layer in self.layers:
                # Adjacent frames should use the same geometric sampling
                # policy, but retaining frame-specific value/output
                # projections preserves the capacity needed to encode
                # direction-dependent appearance changes.
                layer.cross_attn_curr.sampling_offsets = (
                    layer.cross_attn_prev.sampling_offsets)
                layer.cross_attn_curr.attention_weights = (
                    layer.cross_attn_prev.attention_weights)
        if self.shared_attention_decoder:
            for layer in self.layers:
                # Share only how sampled points are aggregated.  Each frame
                # keeps independent sampling offsets and value/output
                # projections, preserving the localization freedom removed
                # by the hard shared-routing decoder.
                layer.cross_attn_curr.attention_weights = (
                    layer.cross_attn_prev.attention_weights)
        if self.symmetric_pair_decoder:
            for layer in self.layers:
                layer.symmetric_pair_decoder = True
                # Adjacent frames share the same modality and encoder.  A
                # single cross-attention module removes ordered-frame
                # parameter bias and makes a frame swap an exact reordering
                # rather than a change of function.
                layer.cross_attn_curr = layer.cross_attn_prev
        if self.symmetric_feature_decoder:
            for layer in self.layers:
                layer.symmetric_feature_decoder = True
        if self.residual_preserving_fusion_decoder:
            for layer in self.layers:
                layer.residual_preserving_fusion_decoder = True

    @property
    def _terminal_enveloped_detail_enabled(self) -> bool:
        return bool(
            self.terminal_enveloped_detail_decoder
            or self.terminal_midpoint_enveloped_detail_decoder
            or self.terminal_regression_enveloped_detail_decoder
            or self.terminal_midpoint_regression_enveloped_detail_decoder
            or self.terminal_factorized_evidence_decoder
            or self.terminal_position_tangent_product_decoder
            or self.terminal_position_tangent_transport_decoder
            or self.terminal_position_tangent_plane_decoder)

    @property
    def _terminal_position_tangent_enabled(self) -> bool:
        return (self.terminal_position_tangent_product_decoder
                or self.terminal_position_tangent_transport_decoder
                or self.terminal_position_tangent_plane_decoder)

    @property
    def _common_evidence_bypass_enabled(self) -> bool:
        return bool(
            self.common_evidence_bypass_decoder
            or self.terminal_common_evidence_bypass_decoder)

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
        if self.competitive_evidence_decoder:
            self.competitive_evidence_adapters = ModuleList([
                nn.Linear(self.embed_dims, self.embed_dims, bias=False)
                for _ in range(self.num_layers)
            ])
        if self.motion_trust_decoder:
            self.motion_trust_adapters = ModuleList([
                nn.Linear(self.embed_dims + 5, 5, bias=False)
                for _ in range(self.num_layers)
            ])
        if self.antisymmetric_detail_decoder:
            self.antisymmetric_detail_adapters = ModuleList([
                nn.Linear(self.embed_dims, self.embed_dims, bias=False)
                for _ in range(self.num_layers)
            ])
        if (self.enveloped_detail_decoder
                or self.regression_enveloped_detail_decoder
                or self.midpoint_regression_enveloped_detail_decoder
                or self.classification_enveloped_detail_decoder):
            self.enveloped_detail_gates = ModuleList([
                nn.Linear(self.embed_dims, self.embed_dims, bias=False)
                for _ in range(self.num_layers)
            ])
        if (self._terminal_enveloped_detail_enabled
                and not self._terminal_position_tangent_enabled):
            # Only the final prediction layer needs a gate.  Allocating gates
            # for earlier layers would create intentionally unused DDP
            # parameters and obscure the structural invariant.
            if (self.terminal_factorized_evidence_decoder
                    and self.terminal_factorized_diagonal_gates):
                if self.terminal_factorized_coupled_gate:
                    self.terminal_enveloped_detail_gates = nn.ParameterList()
                else:
                    self.terminal_enveloped_detail_gates = nn.ParameterList([
                        nn.Parameter(torch.zeros(self.embed_dims))
                    ])
            else:
                self.terminal_enveloped_detail_gates = ModuleList([
                    nn.Linear(self.embed_dims, self.embed_dims, bias=False)
                ])
        if self.common_evidence_bypass_decoder:
            self.common_evidence_bypass_gates = ModuleList([
                nn.Linear(self.embed_dims, self.embed_dims, bias=False)
                for _ in range(self.num_layers)
            ])
        if (self.terminal_common_evidence_bypass_decoder
                or self.terminal_classification_common_evidence_decoder
                or (self.terminal_factorized_evidence_decoder
                    and not self.terminal_factorized_detail_only)):
            # A terminal-only bypass cannot perturb recurrent references.
            # Allocate exactly one gate so DDP sees no intentionally unused
            # parameters on the auxiliary decoder layers.
            if (self.terminal_factorized_evidence_decoder
                    and self.terminal_factorized_diagonal_gates):
                if self.terminal_factorized_coupled_gate:
                    self.terminal_common_evidence_bypass_gates = (
                        nn.ParameterList())
                else:
                    self.terminal_common_evidence_bypass_gates = (
                        nn.ParameterList([
                            nn.Parameter(torch.zeros(self.embed_dims))
                        ]))
            else:
                self.terminal_common_evidence_bypass_gates = ModuleList([
                    nn.Linear(self.embed_dims, self.embed_dims, bias=False)
                ])
        if self.terminal_factorized_coupled_gate:
            # One channel-wise gate jointly regulates common recovery and
            # frame detail. This prevents either route from independently
            # increasing its learned gain while adding only embed_dims
            # parameters and no extra matrix multiplication.
            self.terminal_coupled_evidence_gate = nn.Parameter(
                torch.zeros(self.embed_dims))
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

    def _fuse_pair_position(
        self,
        query_pos_prev: Tensor,
        query_pos_curr: Tensor,
    ) -> Tensor:
        """Fuse dual references into the shared self-attention position.

        The position-only symmetric mode removes ordered-frame bias from the
        shared query position while retaining independent cross-attention and
        the original ordered feature fusion. Repeating the pair mean uses the
        existing linear layer once, so parameter count and forward complexity
        remain unchanged.
        """
        if self.symmetric_pair_decoder:
            return 0.5 * (
                self.pair_pos_fusion(
                    torch.cat([query_pos_prev, query_pos_curr], dim=-1))
                + self.pair_pos_fusion(
                    torch.cat([query_pos_curr, query_pos_prev], dim=-1)))
        if self.symmetric_position_decoder:
            pair_mean = 0.5 * (query_pos_prev + query_pos_curr)
            return self.pair_pos_fusion(
                torch.cat([pair_mean, pair_mean], dim=-1))
        return self.pair_pos_fusion(
            torch.cat([query_pos_prev, query_pos_curr], dim=-1))

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

    def _competitive_evidence_correction(
        self,
        lid: int,
        out_prev: Tensor,
        out_curr: Tensor,
    ) -> Tensor:
        """Select frame detail without introducing an ordered-frame bias.

        The learned gate is odd under a frame swap, while the signed detail is
        also odd.  Their product is therefore swap invariant.  ``tanh`` keeps
        the correction within the two-frame detail envelope, equivalent to a
        per-channel convex competition around the equal-weight common path.
        Inputs are detached so the new branch cannot create a shortcut into
        either cross-attention module.
        """
        evidence = self._normalized_motion_evidence(out_prev, out_curr)
        detail = (0.5 * (out_curr - out_prev)).detach()
        gate = self.competitive_evidence_adapters[lid](evidence).tanh()
        return gate * detail

    def _motion_trust_correction(
        self,
        lid: int,
        out_prev: Tensor,
        out_curr: Tensor,
        reference_prev: Tensor,
        reference_curr: Tensor,
        layer_output: Tensor,
        cls_branch_prev: nn.Module,
        cls_branch_curr: nn.Module,
    ) -> Tensor:
        """Return a bounded, detection-confident antisymmetric box update.

        The previous common-motion decoder could emit an unbounded logit-space
        box residual for every query.  Here the learned direction is bounded
        by ``tanh`` and its elementwise envelope is half of the already
        observed inter-frame reference displacement.  Applying ``-delta`` to
        prev and ``+delta`` to curr therefore preserves the pair midpoint and
        cannot change their separation by more than the current separation.
        A detached geometric mean of the two frame confidences suppresses
        motion hallucination for unilateral or background queries.
        """
        evidence = self._normalized_motion_evidence(out_prev, out_curr)
        reference_motion = self._reference_motion(
            reference_prev, reference_curr)
        score_prev = cls_branch_prev(layer_output).detach().sigmoid().amax(
            dim=-1, keepdim=True)
        score_curr = cls_branch_curr(layer_output).detach().sigmoid().amax(
            dim=-1, keepdim=True)
        bilateral_confidence = (
            score_prev * score_curr).clamp_min(0.0).sqrt()
        direction = self.motion_trust_adapters[lid](
            torch.cat([evidence, reference_motion], dim=-1)).tanh()
        envelope = 0.5 * reference_motion.abs()
        return bilateral_confidence * envelope * direction

    def _antisymmetric_detail_correction(
        self,
        lid: int,
        out_prev: Tensor,
        out_curr: Tensor,
    ) -> Tensor:
        """Return a bounded, swap-odd frame-detail head correction.

        The common decoder query remains the sole recurrent state. Only the
        two frame heads receive ``-detail`` and ``+detail`` corrections, so
        their midpoint is exactly the common representation. The evidence is
        detached to prevent a second gradient path into cross-attention, while
        one shared adapter makes a frame swap an exact sign reversal.
        """
        evidence = self._normalized_motion_evidence(out_prev, out_curr)
        return self.antisymmetric_detail_adapters[lid](evidence).tanh()

    def _enveloped_detail_correction(
        self,
        lid: int,
        out_prev: Tensor,
        out_curr: Tensor,
    ) -> Tensor:
        """Return swap-odd head detail bounded by observed frame evidence.

        This branch can only reveal detail already present between the two
        cross-attention outputs. A swap-invariant learned gate multiplies the
        detached signed detail, making the correction exactly swap odd,
        midpoint preserving, and elementwise bounded by the observed detail.
        """
        evidence = self._normalized_shared_evidence(out_prev, out_curr)
        gate = self.enveloped_detail_gates[lid](evidence).tanh()
        detail = (0.5 * (out_curr - out_prev)).detach()
        return gate * detail

    def _terminal_enveloped_detail_correction(
        self,
        out_prev: Tensor,
        out_curr: Tensor,
    ) -> Tensor:
        """Return bounded frame detail for the terminal prediction only.

        Earlier decoder layers and their iterative references stay on the
        shared baseline path.  The final layer alone receives swap-odd detail,
        so frame specialization cannot recursively perturb later query
        positions or classification features.
        """
        evidence = self._normalized_shared_evidence(out_prev, out_curr)
        if self.terminal_factorized_coupled_gate:
            gate = (evidence * self.terminal_coupled_evidence_gate).tanh()
        elif (self.terminal_factorized_evidence_decoder
                and self.terminal_factorized_diagonal_gates):
            gate = (
                evidence * self.terminal_enveloped_detail_gates[0]).tanh()
        else:
            gate = self.terminal_enveloped_detail_gates[0](evidence).tanh()
        detail = (0.5 * (out_curr - out_prev)).detach()
        return gate * detail

    def _common_evidence_bypass_correction(
        self,
        lid: int,
        layer_output: Tensor,
        out_prev: Tensor,
        out_curr: Tensor,
    ) -> Tensor:
        """Recover common detection evidence bypassed by fusion and FFN.

        The recurrent query is untouched. Heads receive a gated residual
        toward detached raw two-frame common evidence. The gate depends only
        on swap-invariant disagreement, starts at zero, and bounds every
        correction by the existing common-to-head residual.
        """
        evidence = self._normalized_shared_evidence(out_prev, out_curr)
        gate = self.common_evidence_bypass_gates[lid](evidence).tanh()
        common = (0.5 * (out_prev + out_curr)).detach()
        residual = common - layer_output.detach()
        return gate * residual

    def _terminal_common_evidence_bypass_correction(
        self,
        layer_output: Tensor,
        out_prev: Tensor,
        out_curr: Tensor,
    ) -> Tensor:
        """Recover common evidence only for the terminal predictions.

        The correction is identical for both frame heads and is applied
        after the final recurrent decoder update. Therefore all auxiliary
        outputs and every reference consumed by a later layer remain exactly
        on the parent path.
        """
        evidence = self._normalized_shared_evidence(out_prev, out_curr)
        if self.terminal_factorized_coupled_gate:
            gate = (evidence * self.terminal_coupled_evidence_gate).tanh()
        elif (self.terminal_factorized_evidence_decoder
                and self.terminal_factorized_diagonal_gates):
            gate = (
                evidence
                * self.terminal_common_evidence_bypass_gates[0]).tanh()
        else:
            gate = self.terminal_common_evidence_bypass_gates[0](
                evidence).tanh()
        common = (0.5 * (out_prev + out_curr)).detach()
        residual = common - layer_output.detach()
        return gate * residual

    @staticmethod
    def _terminal_bilateral_confidence(
        layer_output: Tensor,
        cls_branch_prev: nn.Module,
        cls_branch_curr: nn.Module,
    ) -> Tensor:
        """Return detached object reliability shared by both frame routes.

        The geometric mean is high only when both parent-path classifiers
        support an object query.  It adds no threshold or learned scale and
        cannot create a shortcut into either classifier or decoder feature.
        """
        score_prev = cls_branch_prev(layer_output).detach().sigmoid().amax(
            dim=-1, keepdim=True)
        score_curr = cls_branch_curr(layer_output).detach().sigmoid().amax(
            dim=-1, keepdim=True)
        return (score_prev * score_curr).clamp_min(0.0).sqrt()

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
        if self.competitive_evidence_decoder:
            for adapter in self.competitive_evidence_adapters:
                nn.init.zeros_(adapter.weight)
        if self.motion_trust_decoder:
            for adapter in self.motion_trust_adapters:
                nn.init.zeros_(adapter.weight)
        if self.antisymmetric_detail_decoder:
            for adapter in self.antisymmetric_detail_adapters:
                nn.init.zeros_(adapter.weight)
        if (self.enveloped_detail_decoder
                or self.regression_enveloped_detail_decoder
                or self.midpoint_regression_enveloped_detail_decoder
                or self.classification_enveloped_detail_decoder):
            for gate in self.enveloped_detail_gates:
                nn.init.zeros_(gate.weight)
        if (self._terminal_enveloped_detail_enabled
                and not self._terminal_position_tangent_enabled):
            for gate in self.terminal_enveloped_detail_gates:
                if isinstance(gate, nn.Parameter):
                    nn.init.zeros_(gate)
                else:
                    nn.init.zeros_(gate.weight)
        if self.common_evidence_bypass_decoder:
            for gate in self.common_evidence_bypass_gates:
                nn.init.zeros_(gate.weight)
        if (self.terminal_common_evidence_bypass_decoder
                or self.terminal_classification_common_evidence_decoder
                or (self.terminal_factorized_evidence_decoder
                    and not self.terminal_factorized_detail_only)):
            for gate in self.terminal_common_evidence_bypass_gates:
                if isinstance(gate, nn.Parameter):
                    nn.init.zeros_(gate)
                else:
                    nn.init.zeros_(gate.weight)
        if self.terminal_factorized_coupled_gate:
            nn.init.zeros_(self.terminal_coupled_evidence_gate)
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
        if (self.tristate_decoder or self.motion_trust_decoder
                or self.terminal_factorized_confidence != 'none'):
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
            query_pos = self._fuse_pair_position(
                query_pos_prev, query_pos_curr)

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
                    or self.shared_evidence_decoder
                    or self.competitive_evidence_decoder
                    or self.motion_trust_decoder
                    or self.antisymmetric_detail_decoder
                    or self.enveloped_detail_decoder
                    or self.regression_enveloped_detail_decoder
                    or self.midpoint_regression_enveloped_detail_decoder
                    or self.classification_enveloped_detail_decoder
                    or self._terminal_enveloped_detail_enabled
                    or self._common_evidence_bypass_enabled
                    or
                    self.terminal_classification_common_evidence_decoder
                    or self.frame_evidence_cls_decoder
                    or self.frame_detail_cls_decoder)
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
                if self.competitive_evidence_decoder:
                    query = query + self._competitive_evidence_correction(
                        lid, frame_evidence_prev, frame_evidence_curr)

                layer_output = self.norm(query)
                if self.frame_evidence_cls_decoder:
                    # The recurrent query and both iterative references stay
                    # on the parent shared path. Classification alone sees
                    # each frame's already-computed cross-attention evidence,
                    # recovering visibility-specific information without an
                    # adapter, extra attention, or an additional parameter.
                    layer_output_prev = frame_evidence_prev
                    layer_output_curr = frame_evidence_curr
                    tmp_prev = reg_branches_prev[lid](layer_output)
                    tmp_curr = reg_branches_curr[lid](layer_output)
                elif self.frame_detail_cls_decoder:
                    # Preserve the shared classification state as the exact
                    # pair midpoint. Only the swap-odd component of the two
                    # already-computed frame observations distinguishes the
                    # classification inputs. Regression and recurrent state
                    # remain exactly on the parent path.
                    frame_detail = 0.5 * (
                        frame_evidence_prev - frame_evidence_curr)
                    layer_output_prev = layer_output + frame_detail
                    layer_output_curr = layer_output - frame_detail
                    tmp_prev = reg_branches_prev[lid](layer_output)
                    tmp_curr = reg_branches_curr[lid](layer_output)
                elif self.dual_output_adapter:
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
                elif self.antisymmetric_detail_decoder:
                    frame_detail = self._antisymmetric_detail_correction(
                        lid, frame_evidence_prev, frame_evidence_curr)
                    layer_output_prev = layer_output - frame_detail
                    layer_output_curr = layer_output + frame_detail
                    tmp_prev = reg_branches_prev[lid](layer_output_prev)
                    tmp_curr = reg_branches_curr[lid](layer_output_curr)
                elif self.enveloped_detail_decoder:
                    if self.common_evidence_bypass_decoder:
                        layer_output = layer_output + (
                            self._common_evidence_bypass_correction(
                                lid,
                                layer_output,
                                frame_evidence_prev,
                                frame_evidence_curr))
                    frame_detail = self._enveloped_detail_correction(
                        lid, frame_evidence_prev, frame_evidence_curr)
                    layer_output_prev = layer_output - frame_detail
                    layer_output_curr = layer_output + frame_detail
                    tmp_prev = reg_branches_prev[lid](layer_output_prev)
                    tmp_curr = reg_branches_curr[lid](layer_output_curr)
                elif self.regression_enveloped_detail_decoder:
                    frame_detail = self._enveloped_detail_correction(
                        lid, frame_evidence_prev, frame_evidence_curr)
                    # Keep classification features on the shared decoder
                    # path.  Only iterative box refinement receives the
                    # bounded, swap-odd frame correction.
                    reg_output_prev = layer_output - frame_detail
                    reg_output_curr = layer_output + frame_detail
                    tmp_prev = reg_branches_prev[lid](reg_output_prev)
                    tmp_curr = reg_branches_curr[lid](reg_output_curr)
                elif self.midpoint_regression_enveloped_detail_decoder:
                    frame_detail = self._enveloped_detail_correction(
                        lid, frame_evidence_prev, frame_evidence_curr)
                    # Convert the feature-domain frame detail into one
                    # antisymmetric box-residual correction. This keeps the
                    # added detail's pair midpoint exactly zero in 5D logit
                    # space even though the two regression heads are
                    # independent nonlinear functions.
                    base_prev = reg_branches_prev[lid](layer_output)
                    base_curr = reg_branches_curr[lid](layer_output)
                    detailed_prev = reg_branches_prev[lid](
                        layer_output - frame_detail)
                    detailed_curr = reg_branches_curr[lid](
                        layer_output + frame_detail)
                    box_detail = 0.5 * (
                        (detailed_curr - base_curr)
                        - (detailed_prev - base_prev))
                    tmp_prev = base_prev - box_detail
                    tmp_curr = base_curr + box_detail
                elif self.classification_enveloped_detail_decoder:
                    frame_detail = self._enveloped_detail_correction(
                        lid, frame_evidence_prev, frame_evidence_curr)
                    # Keep iterative box refinement exactly on the shared
                    # decoder path.  The bounded, swap-odd frame correction
                    # is exposed only to the frame-specific classifiers.
                    layer_output_prev = layer_output - frame_detail
                    layer_output_curr = layer_output + frame_detail
                    tmp_prev = reg_branches_prev[lid](layer_output)
                    tmp_curr = reg_branches_curr[lid](layer_output)
                elif self._terminal_position_tangent_enabled:
                    # The recurrent query and all auxiliary outputs remain on
                    # the parent shared path. At the terminal layer, retain
                    # only frame evidence detail aligned with the detached
                    # positional displacement already encoded by the two
                    # references. The orthogonal projection cannot increase
                    # detail energy and is swap equivariant. Regression starts
                    # from the parent heads here and is projected by the
                    # selected terminal geometry tangent block below.
                    layer_output_prev = layer_output
                    layer_output_curr = layer_output
                    tmp_prev = reg_branches_prev[lid](layer_output)
                    tmp_curr = reg_branches_curr[lid](layer_output)
                    if lid == self.num_layers - 1:
                        num_dn = max(tmp_prev.shape[1] - self.num_queries, 0)
                        frame_detail = (
                            self._pair_position_tangent_feature_detail(
                                frame_evidence_prev,
                                frame_evidence_curr,
                                query_pos_prev,
                                query_pos_curr,
                                num_dn))
                        layer_output_prev = layer_output - frame_detail
                        layer_output_curr = layer_output + frame_detail
                elif self.terminal_enveloped_detail_decoder:
                    if lid == self.num_layers - 1:
                        frame_detail = (
                            self._terminal_enveloped_detail_correction(
                                frame_evidence_prev,
                                frame_evidence_curr))
                        layer_output_prev = layer_output - frame_detail
                        layer_output_curr = layer_output + frame_detail
                    else:
                        # Auxiliary layers remain exact shared-path outputs;
                        # only the terminal predictions specialize by frame.
                        layer_output_prev = layer_output
                        layer_output_curr = layer_output
                    tmp_prev = reg_branches_prev[lid](layer_output_prev)
                    tmp_curr = reg_branches_curr[lid](layer_output_curr)
                elif self.terminal_midpoint_enveloped_detail_decoder:
                    if lid == self.num_layers - 1:
                        frame_detail = (
                            self._terminal_enveloped_detail_correction(
                                frame_evidence_prev,
                                frame_evidence_curr))
                        # Classification keeps the final-only frame detail.
                        layer_output_prev = layer_output - frame_detail
                        layer_output_curr = layer_output + frame_detail

                        # Regression receives the same final-only evidence,
                        # but its added 5D logit residual is antisymmetric.
                        # This prevents two independent nonlinear heads from
                        # shifting the pair midpoint.
                        base_prev = reg_branches_prev[lid](layer_output)
                        base_curr = reg_branches_curr[lid](layer_output)
                        detailed_prev = reg_branches_prev[lid](
                            layer_output_prev)
                        detailed_curr = reg_branches_curr[lid](
                            layer_output_curr)
                        box_detail = 0.5 * (
                            (detailed_curr - base_curr)
                            - (detailed_prev - base_prev))
                        tmp_prev = base_prev - box_detail
                        tmp_curr = base_curr + box_detail
                    else:
                        # Recurrent references and all auxiliary outputs stay
                        # exactly on the shared-attention parent path.
                        layer_output_prev = layer_output
                        layer_output_curr = layer_output
                        tmp_prev = reg_branches_prev[lid](layer_output)
                        tmp_curr = reg_branches_curr[lid](layer_output)
                elif self.terminal_regression_enveloped_detail_decoder:
                    # Keep all classification features on the shared path.
                    # Only the final box prediction receives frame detail,
                    # and no corrected reference is fed into another layer.
                    layer_output_prev = layer_output
                    layer_output_curr = layer_output
                    if lid == self.num_layers - 1:
                        frame_detail = (
                            self._terminal_enveloped_detail_correction(
                                frame_evidence_prev,
                                frame_evidence_curr))
                        tmp_prev = reg_branches_prev[lid](
                            layer_output - frame_detail)
                        tmp_curr = reg_branches_curr[lid](
                            layer_output + frame_detail)
                    else:
                        tmp_prev = reg_branches_prev[lid](layer_output)
                        tmp_curr = reg_branches_curr[lid](layer_output)
                elif (
                        self.
                        terminal_midpoint_regression_enveloped_detail_decoder):
                    # Classification and all recurrent references remain
                    # exactly shared.  At the terminal layer, convert the
                    # frame evidence into a strictly antisymmetric 5D
                    # box-logit residual so the added pair midpoint is zero.
                    layer_output_prev = layer_output
                    layer_output_curr = layer_output
                    if lid == self.num_layers - 1:
                        frame_detail = (
                            self._terminal_enveloped_detail_correction(
                                frame_evidence_prev,
                                frame_evidence_curr))
                        base_prev = reg_branches_prev[lid](layer_output)
                        base_curr = reg_branches_curr[lid](layer_output)
                        detailed_prev = reg_branches_prev[lid](
                            layer_output - frame_detail)
                        detailed_curr = reg_branches_curr[lid](
                            layer_output + frame_detail)
                        box_detail = 0.5 * (
                            (detailed_curr - base_curr)
                            - (detailed_prev - base_prev))
                        tmp_prev = base_prev - box_detail
                        tmp_curr = base_curr + box_detail
                    else:
                        tmp_prev = reg_branches_prev[lid](layer_output)
                        tmp_curr = reg_branches_curr[lid](layer_output)
                elif (
                        self.
                        terminal_classification_common_evidence_decoder):
                    # Common evidence changes final classification features
                    # only. Box regression, auxiliary outputs, and recurrent
                    # references are bitwise-identical to the parent path.
                    layer_output_prev = layer_output
                    layer_output_curr = layer_output
                    if lid == self.num_layers - 1:
                        common_output = layer_output + (
                            self.
                            _terminal_common_evidence_bypass_correction(
                                layer_output,
                                frame_evidence_prev,
                                frame_evidence_curr))
                        layer_output_prev = common_output
                        layer_output_curr = common_output
                    tmp_prev = reg_branches_prev[lid](layer_output)
                    tmp_curr = reg_branches_curr[lid](layer_output)
                elif self.terminal_factorized_evidence_decoder:
                    # Factor the terminal evidence tensors into a symmetric
                    # common component and an antisymmetric frame component.
                    # This algebra remains exact whether the two cross-
                    # attention branches are independent or share their
                    # attention-weight projection.
                    # Classification sees only the common correction. Box
                    # specialization receives an antisymmetric 5D residual,
                    # so its added pair midpoint is exactly zero. Auxiliary
                    # outputs and all recurrent references remain unchanged.
                    layer_output_prev = layer_output
                    layer_output_curr = layer_output
                    if lid == self.num_layers - 1:
                        confidence = None
                        if self.terminal_factorized_confidence != 'none':
                            confidence = self._terminal_bilateral_confidence(
                                layer_output,
                                cls_branches_prev[lid],
                                cls_branches_curr[lid])
                        if not self.terminal_factorized_detail_only:
                            common_correction = (
                                self.
                                _terminal_common_evidence_bypass_correction(
                                    layer_output,
                                    frame_evidence_prev,
                                    frame_evidence_curr))
                            if self.terminal_factorized_confidence in {
                                    'common', 'both'}:
                                common_correction = (
                                    confidence * common_correction)
                            common_output = layer_output + common_correction
                            layer_output_prev = common_output
                            layer_output_curr = common_output
                        frame_detail = (
                            self._terminal_enveloped_detail_correction(
                                frame_evidence_prev,
                                frame_evidence_curr))
                        if self.terminal_factorized_confidence in {
                                'detail', 'both'}:
                            frame_detail = confidence * frame_detail
                        # Keep the box midpoint on the unmodified parent
                        # representation.  Common evidence is classification
                        # only; otherwise it can improve association while
                        # silently degrading DetA through both box heads.
                        base_prev = reg_branches_prev[lid](layer_output)
                        base_curr = reg_branches_curr[lid](layer_output)
                        detailed_prev = reg_branches_prev[lid](
                            layer_output - frame_detail)
                        detailed_curr = reg_branches_curr[lid](
                            layer_output + frame_detail)
                        box_detail = 0.5 * (
                            (detailed_curr - base_curr)
                            - (detailed_prev - base_prev))
                        if self.terminal_factorized_center_motion_only:
                            # Adjacent-frame detail represents motion. Keep
                            # its antisymmetric correction on center x/y and
                            # leave width, height, and angle on the parent
                            # geometry. This preserves the exact pair
                            # midpoint while preventing temporal appearance
                            # detail from perturbing shared object shape.
                            box_detail = torch.cat((
                                box_detail[..., :2],
                                torch.zeros_like(box_detail[..., 2:])),
                                dim=-1)
                        tmp_prev = base_prev - box_detail
                        tmp_curr = base_curr + box_detail
                    else:
                        tmp_prev = reg_branches_prev[lid](layer_output)
                        tmp_curr = reg_branches_curr[lid](layer_output)
                elif self.common_evidence_bypass_decoder:
                    layer_output = layer_output + (
                        self._common_evidence_bypass_correction(
                            lid,
                            layer_output,
                            frame_evidence_prev,
                            frame_evidence_curr))
                    tmp_prev = reg_branches_prev[lid](layer_output)
                    tmp_curr = reg_branches_curr[lid](layer_output)
                elif self.terminal_common_evidence_bypass_decoder:
                    if lid == self.num_layers - 1:
                        layer_output = layer_output + (
                            self.
                            _terminal_common_evidence_bypass_correction(
                                layer_output,
                                frame_evidence_prev,
                                frame_evidence_curr))
                    tmp_prev = reg_branches_prev[lid](layer_output)
                    tmp_curr = reg_branches_curr[lid](layer_output)
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
                elif self.motion_trust_decoder:
                    tmp_prev = reg_branches_prev[lid](layer_output)
                    tmp_curr = reg_branches_curr[lid](layer_output)
                    trusted_motion = self._motion_trust_correction(
                        lid,
                        frame_evidence_prev,
                        frame_evidence_curr,
                        reference_prev,
                        reference_curr,
                        layer_output,
                        cls_branches_prev[lid],
                        cls_branches_curr[lid])
                    tmp_prev = tmp_prev - trusted_motion
                    tmp_curr = tmp_curr + trusted_motion
                else:
                    tmp_prev = reg_branches_prev[lid](layer_output)
                    tmp_curr = reg_branches_curr[lid](layer_output)

            if self.pair_shared_shape_refinement_decoder:
                num_dn = max(tmp_prev.shape[1] - self.num_queries, 0)
                tmp_prev, tmp_curr = self._pair_shared_shape_residual(
                    tmp_prev, tmp_curr, num_dn)
            elif self.pair_shared_angle_refinement_decoder:
                num_dn = max(tmp_prev.shape[1] - self.num_queries, 0)
                tmp_prev, tmp_curr = self._pair_shared_angle_residual(
                    tmp_prev, tmp_curr, num_dn)
            elif self.pair_shared_periodic_angle_refinement_decoder:
                num_dn = max(tmp_prev.shape[1] - self.num_queries, 0)
                tmp_prev, tmp_curr = (
                    self._pair_shared_periodic_angle_residual(
                        tmp_prev, tmp_curr, reference_prev, reference_curr,
                        num_dn))
            elif self.pair_shared_log_size_periodic_angle_refinement_decoder:
                num_dn = max(tmp_prev.shape[1] - self.num_queries, 0)
                tmp_prev, tmp_curr = self._pair_shared_log_size_residual(
                    tmp_prev, tmp_curr, reference_prev, reference_curr,
                    num_dn)
                tmp_prev, tmp_curr = (
                    self._pair_shared_periodic_angle_residual(
                        tmp_prev, tmp_curr, reference_prev, reference_curr,
                        num_dn))
            elif self.pair_shared_log_area_periodic_angle_refinement_decoder:
                num_dn = max(tmp_prev.shape[1] - self.num_queries, 0)
                tmp_prev, tmp_curr = self._pair_shared_log_area_residual(
                    tmp_prev, tmp_curr, reference_prev, reference_curr,
                    num_dn)
                tmp_prev, tmp_curr = (
                    self._pair_shared_periodic_angle_residual(
                        tmp_prev, tmp_curr, reference_prev, reference_curr,
                        num_dn))
            elif (self.
                  pair_shared_late_log_size_periodic_angle_refinement_decoder
                  and lid >= max(self.num_layers - 2, 0)):
                num_dn = max(tmp_prev.shape[1] - self.num_queries, 0)
                tmp_prev, tmp_curr = self._pair_shared_log_size_residual(
                    tmp_prev, tmp_curr, reference_prev, reference_curr,
                    num_dn)
                tmp_prev, tmp_curr = (
                    self._pair_shared_periodic_angle_residual(
                        tmp_prev, tmp_curr, reference_prev, reference_curr,
                        num_dn))
            elif (self.
                  pair_shared_terminal_log_size_periodic_angle_refinement_decoder
                  and lid == self.num_layers - 1):
                num_dn = max(tmp_prev.shape[1] - self.num_queries, 0)
                tmp_prev, tmp_curr = self._pair_shared_log_size_residual(
                    tmp_prev, tmp_curr, reference_prev, reference_curr,
                    num_dn)
                tmp_prev, tmp_curr = (
                    self._pair_shared_periodic_angle_residual(
                        tmp_prev, tmp_curr, reference_prev, reference_curr,
                        num_dn))
            elif (self.
                  pair_shared_terminal_log_area_periodic_angle_refinement_decoder
                  and lid == self.num_layers - 1):
                num_dn = max(tmp_prev.shape[1] - self.num_queries, 0)
                tmp_prev, tmp_curr = self._pair_shared_log_area_residual(
                    tmp_prev, tmp_curr, reference_prev, reference_curr,
                    num_dn)
                tmp_prev, tmp_curr = (
                    self._pair_shared_periodic_angle_residual(
                        tmp_prev, tmp_curr, reference_prev, reference_curr,
                        num_dn))
            elif (self.pair_shared_terminal_periodic_angle_refinement_decoder
                  and lid == self.num_layers - 1):
                num_dn = max(tmp_prev.shape[1] - self.num_queries, 0)
                tmp_prev, tmp_curr = (
                    self._pair_shared_periodic_angle_residual(
                        tmp_prev, tmp_curr, reference_prev, reference_curr,
                        num_dn))
            elif (self.pair_shared_terminal_log_size_refinement_decoder
                  and lid == self.num_layers - 1):
                num_dn = max(tmp_prev.shape[1] - self.num_queries, 0)
                tmp_prev, tmp_curr = self._pair_shared_log_size_residual(
                    tmp_prev, tmp_curr, reference_prev, reference_curr,
                    num_dn)
            elif (
                    self.
                    pair_shared_terminal_normalized_center_refinement_decoder
                    and lid == self.num_layers - 1):
                num_dn = max(tmp_prev.shape[1] - self.num_queries, 0)
                tmp_prev, tmp_curr = (
                    self._pair_shared_normalized_center_residual(
                        tmp_prev, tmp_curr, reference_prev, reference_curr,
                        num_dn))
            elif (self.pair_shared_terminal_full_tangent_refinement_decoder
                  and lid == self.num_layers - 1):
                num_dn = max(tmp_prev.shape[1] - self.num_queries, 0)
                tmp_prev, tmp_curr = (
                    self._pair_shared_normalized_center_residual(
                        tmp_prev, tmp_curr, reference_prev, reference_curr,
                        num_dn))
                tmp_prev, tmp_curr = self._pair_shared_log_size_residual(
                    tmp_prev, tmp_curr, reference_prev, reference_curr,
                    num_dn)
                tmp_prev, tmp_curr = (
                    self._pair_shared_periodic_angle_residual(
                        tmp_prev, tmp_curr, reference_prev, reference_curr,
                        num_dn))
            elif (
                    self.
                    pair_shared_terminal_transport_center_tangent_refinement_decoder
                    and lid == self.num_layers - 1):
                num_dn = max(tmp_prev.shape[1] - self.num_queries, 0)
                tmp_prev, tmp_curr = (
                    self._pair_transport_center_tangent_residual(
                        tmp_prev, tmp_curr, reference_prev, reference_curr,
                        num_dn))
            elif (
                    self.
                    pair_shared_terminal_transport_shape_tangent_refinement_decoder
                    and lid == self.num_layers - 1):
                num_dn = max(tmp_prev.shape[1] - self.num_queries, 0)
                tmp_prev, tmp_curr = (
                    self._pair_transport_shape_tangent_residual(
                        tmp_prev, tmp_curr, reference_prev, reference_curr,
                        num_dn))
            elif ((
                    self.
                    pair_shared_terminal_transport_shared_metric_product_tangent_refinement_decoder)
                  and lid == self.num_layers - 1):
                num_dn = max(tmp_prev.shape[1] - self.num_queries, 0)
                # Express both frame updates in one geometric-mean box metric
                # before the established rank-one transport. Shape transport
                # is unchanged, isolating cross-frame metric consistency.
                tmp_prev, tmp_curr = (
                    self._pair_transport_shared_metric_center_tangent_residual(
                        tmp_prev, tmp_curr, reference_prev, reference_curr,
                        num_dn))
                tmp_prev, tmp_curr = (
                    self._pair_transport_shape_tangent_residual(
                        tmp_prev, tmp_curr, reference_prev, reference_curr,
                        num_dn))
            elif ((
                    self.
                    pair_shared_terminal_transport_axis_frenet_product_tangent_refinement_decoder)
                  and lid == self.num_layers - 1):
                num_dn = max(tmp_prev.shape[1] - self.num_queries, 0)
                # Preserve the strong axis-normalized product tangent while
                # replacing its single chord projector with constant-turn
                # endpoint directions. Shape transport is unchanged.
                tmp_prev, tmp_curr = (
                    self._pair_transport_axis_frenet_center_tangent_residual(
                        tmp_prev, tmp_curr, reference_prev, reference_curr,
                        num_dn))
                tmp_prev, tmp_curr = (
                    self._pair_transport_shape_tangent_residual(
                        tmp_prev, tmp_curr, reference_prev, reference_curr,
                        num_dn))
            elif ((
                    self.
                    pair_shared_terminal_transport_frenet_product_tangent_refinement_decoder)
                  and lid == self.num_layers - 1):
                num_dn = max(tmp_prev.shape[1] - self.num_queries, 0)
                # Preserve the established shape tangent, while replacing
                # the shared chord projector for center detail with the two
                # endpoint tangents of a constant-turn Frenet arc.
                tmp_prev, tmp_curr = (
                    self._pair_transport_frenet_center_tangent_residual(
                        tmp_prev, tmp_curr, reference_prev, reference_curr,
                        num_dn))
                tmp_prev, tmp_curr = (
                    self._pair_transport_shape_tangent_residual(
                        tmp_prev, tmp_curr, reference_prev, reference_curr,
                        num_dn))
            elif ((
                    self.
                    pair_shared_terminal_transport_se2_product_tangent_refinement_decoder)
                  and lid == self.num_layers - 1):
                num_dn = max(tmp_prev.shape[1] - self.num_queries, 0)
                # First establish the transported terminal orientation, then
                # express center corrections as finite SE(2) midpoint twists.
                # This couples translation only to the already selected
                # angle increment while leaving shape transport unchanged.
                tmp_prev, tmp_curr = (
                    self._pair_transport_shape_tangent_residual(
                        tmp_prev, tmp_curr, reference_prev, reference_curr,
                        num_dn))
                tmp_prev, tmp_curr = (
                    self._pair_transport_se2_center_tangent_residual(
                        tmp_prev, tmp_curr, reference_prev, reference_curr,
                        num_dn))
            elif ((
                    self.
                    pair_shared_terminal_transport_body_frame_product_tangent_refinement_decoder)
                  and lid == self.num_layers - 1):
                num_dn = max(tmp_prev.shape[1] - self.num_queries, 0)
                # Express translation in the pair's oriented body frame
                # before applying the same factorized product-tangent
                # transport used by the axis-aligned predecessor. Shape
                # transport is deliberately unchanged, isolating coordinate
                # alignment as the only structural factor.
                tmp_prev, tmp_curr = (
                    self._pair_transport_body_frame_center_tangent_residual(
                        tmp_prev, tmp_curr, reference_prev, reference_curr,
                        num_dn))
                tmp_prev, tmp_curr = (
                    self._pair_transport_shape_tangent_residual(
                        tmp_prev, tmp_curr, reference_prev, reference_curr,
                        num_dn))
            elif ((
                    self.
                    pair_shared_terminal_transport_product_tangent_refinement_decoder
                    or self.terminal_position_tangent_product_decoder)
                  and lid == self.num_layers - 1):
                num_dn = max(tmp_prev.shape[1] - self.num_queries, 0)
                # Factor the product geometry into independent translation
                # and shape tangent bundles. This retains the full terminal
                # motion prior without allowing center energy to rotate into
                # scale/angle detail (or vice versa) through one 5D dot
                # product. Both projections are parameter-free and preserve
                # the DN prefix exactly.
                tmp_prev, tmp_curr = (
                    self._pair_transport_center_tangent_residual(
                        tmp_prev, tmp_curr, reference_prev, reference_curr,
                        num_dn))
                tmp_prev, tmp_curr = (
                    self._pair_transport_shape_tangent_residual(
                        tmp_prev, tmp_curr, reference_prev, reference_curr,
                        num_dn))
            elif ((
                    self.
                    pair_shared_terminal_transport_plane_refinement_decoder
                    or self.terminal_position_tangent_plane_decoder)
                  and lid == self.num_layers - 1):
                num_dn = max(tmp_prev.shape[1] - self.num_queries, 0)
                tmp_prev, tmp_curr = (
                    self._pair_transport_osculating_plane_residual(
                        tmp_prev, tmp_curr, reference_prev, reference_curr,
                        num_dn))
            elif ((
                    self.
                    pair_shared_terminal_transport_tangent_refinement_decoder
                    or self.terminal_position_tangent_transport_decoder)
                  and lid == self.num_layers - 1):
                num_dn = max(tmp_prev.shape[1] - self.num_queries, 0)
                tmp_prev, tmp_curr = (
                    self._pair_transport_full_tangent_residual(
                        tmp_prev, tmp_curr, reference_prev, reference_curr,
                        num_dn))
            elif (self.
                  pair_shared_progressive_log_shape_periodic_angle_refinement_decoder
                  and lid >= max(self.num_layers - 2, 0)):
                num_dn = max(tmp_prev.shape[1] - self.num_queries, 0)
                if lid == self.num_layers - 1:
                    tmp_prev, tmp_curr = self._pair_shared_log_size_residual(
                        tmp_prev, tmp_curr, reference_prev, reference_curr,
                        num_dn)
                else:
                    tmp_prev, tmp_curr = self._pair_shared_log_area_residual(
                        tmp_prev, tmp_curr, reference_prev, reference_curr,
                        num_dn)
                tmp_prev, tmp_curr = (
                    self._pair_shared_periodic_angle_residual(
                        tmp_prev, tmp_curr, reference_prev, reference_curr,
                        num_dn))
            elif self.pair_shared_normalized_center_refinement_decoder:
                num_dn = max(tmp_prev.shape[1] - self.num_queries, 0)
                tmp_prev, tmp_curr = (
                    self._pair_shared_normalized_center_residual(
                        tmp_prev, tmp_curr, reference_prev, reference_curr,
                        num_dn))

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
                if (self.frame_evidence_cls_decoder
                        or self.frame_detail_cls_decoder
                        or self.dual_output_adapter
                        or self.antisymmetric_detail_decoder
                        or self.enveloped_detail_decoder
                        or self.classification_enveloped_detail_decoder
                        or
                        self.terminal_classification_common_evidence_decoder
                        or self._terminal_enveloped_detail_enabled):
                    hidden_states_prev.append(layer_output_prev)
                    hidden_states_curr.append(layer_output_curr)
            references_prev.append(new_reference_prev)
            references_curr.append(new_reference_curr)

        if (self.tristate_decoder
                or self.frame_evidence_cls_decoder
                or self.frame_detail_cls_decoder
                or self.dual_output_adapter
                or self.antisymmetric_detail_decoder
                or self.enveloped_detail_decoder
                or self.classification_enveloped_detail_decoder
                or self.terminal_classification_common_evidence_decoder
                or self._terminal_enveloped_detail_enabled):
            return (hidden_states, references_prev, references_curr,
                    hidden_states_prev, hidden_states_curr)
        return hidden_states, references_prev, references_curr

    @staticmethod
    def _pair_shared_shape_residual(
            residual_prev: Tensor, residual_curr: Tensor,
            num_dn: int) -> Tuple[Tensor, Tensor]:
        """Share normal-query shape refinement while preserving motion.

        Center x/y residuals stay frame-specific. Width, height, and angle
        residuals are replaced by their pair mean, which is swap-equivariant
        and class-agnostic. The DN prefix is untouched because denoising
        queries do not share the normal proposal alignment contract.
        """
        if residual_prev.shape != residual_curr.shape:
            raise ValueError(
                'pair-shared shape refinement requires aligned residuals')
        if residual_prev.shape[-1] != 5:
            raise ValueError(
                'pair-shared shape refinement requires 5D box residuals')
        if num_dn < 0 or num_dn > residual_prev.shape[1]:
            raise ValueError(
                f'invalid DN prefix length {num_dn} for pair residuals')

        normal_prev = residual_prev[:, num_dn:]
        normal_curr = residual_curr[:, num_dn:]
        shared_shape = 0.5 * (
            normal_prev[..., 2:] + normal_curr[..., 2:])
        normal_prev = torch.cat((normal_prev[..., :2], shared_shape), dim=-1)
        normal_curr = torch.cat((normal_curr[..., :2], shared_shape), dim=-1)
        if not num_dn:
            return normal_prev, normal_curr
        return (
            torch.cat((residual_prev[:, :num_dn], normal_prev), dim=1),
            torch.cat((residual_curr[:, :num_dn], normal_curr), dim=1),
        )

    @staticmethod
    def _pair_transport_osculating_plane_residual(
            residual_prev: Tensor, residual_curr: Tensor,
            reference_prev: Tensor, reference_curr: Tensor,
            num_dn: int) -> Tuple[Tensor, Tensor]:
        """Transport terminal detail in a local box-motion tangent plane.

        The one-dimensional full-tangent projection can discard legitimate
        late refinement whenever established frame motion and the decoder's
        current shared correction are not collinear.  This variant projects
        pair detail onto the at-most two-dimensional plane spanned by those
        two detached directions.  Gram-Schmidt makes the two retained
        components orthogonal, so the projection cannot increase detail
        energy.  It remains parameter-free, swap equivariant, class agnostic,
        terminal-only at the caller, and preserves the DN prefix exactly.
        """
        if residual_prev.shape != residual_curr.shape:
            raise ValueError(
                'pair-transport tangent-plane refinement requires aligned '
                'residuals')
        if reference_prev.shape != reference_curr.shape:
            raise ValueError(
                'pair-transport tangent-plane refinement requires aligned '
                'references')
        if residual_prev.shape != reference_prev.shape:
            raise ValueError(
                'pair-transport tangent-plane refinement requires residual '
                'and reference shapes to match')
        if residual_prev.shape[-1] != 5:
            raise ValueError(
                'pair-transport tangent-plane refinement requires 5D boxes')
        if num_dn < 0 or num_dn > residual_prev.shape[1]:
            raise ValueError(
                f'invalid DN prefix length {num_dn} for pair residuals')

        normal_prev = residual_prev[:, num_dn:]
        normal_curr = residual_curr[:, num_dn:]
        ref_prev = reference_prev[:, num_dn:]
        ref_curr = reference_curr[:, num_dn:]
        ref_logit_prev = inverse_sigmoid(ref_prev, eps=1e-3)
        ref_logit_curr = inverse_sigmoid(ref_curr, eps=1e-3)

        proposed_prev = (normal_prev + ref_logit_prev).sigmoid()
        proposed_curr = (normal_curr + ref_logit_curr).sigmoid()
        size_prev = ref_prev[..., 2:4].clamp_min(1e-6)
        size_curr = ref_curr[..., 2:4].clamp_min(1e-6)

        def wrap_period(value: Tensor) -> Tensor:
            return torch.remainder(value + 0.5, 1.0) - 0.5

        tangent_prev = torch.cat((
            (proposed_prev[..., :2] - ref_prev[..., :2]) / size_prev,
            torch.log(proposed_prev[..., 2:4].clamp_min(1e-6) / size_prev),
            wrap_period(proposed_prev[..., 4:] - ref_prev[..., 4:])),
            dim=-1)
        tangent_curr = torch.cat((
            (proposed_curr[..., :2] - ref_curr[..., :2]) / size_curr,
            torch.log(proposed_curr[..., 2:4].clamp_min(1e-6) / size_curr),
            wrap_period(proposed_curr[..., 4:] - ref_curr[..., 4:])),
            dim=-1)

        common_tangent = 0.5 * (tangent_prev + tangent_curr)
        detail_tangent = 0.5 * (tangent_curr - tangent_prev)
        pair_size = torch.sqrt(size_prev * size_curr).clamp_min(1e-6)
        motion_basis = torch.cat((
            (ref_curr[..., :2] - ref_prev[..., :2]) / pair_size,
            torch.log(size_curr / size_prev),
            wrap_period(ref_curr[..., 4:] - ref_prev[..., 4:])),
            dim=-1).detach()
        common_basis = common_tangent.detach()

        def safe_unit(direction: Tensor) -> Tensor:
            norm = direction.square().sum(
                dim=-1, keepdim=True).sqrt()
            return torch.where(
                norm > 1e-6,
                direction / norm.clamp_min(1e-6),
                torch.zeros_like(direction))

        motion_unit = safe_unit(motion_basis)
        motion_component = motion_unit * (
            (detail_tangent * motion_unit).sum(dim=-1, keepdim=True))
        common_orthogonal = common_basis - motion_unit * (
            (common_basis * motion_unit).sum(dim=-1, keepdim=True))
        common_unit = safe_unit(common_orthogonal)
        common_component = common_unit * (
            (detail_tangent * common_unit).sum(dim=-1, keepdim=True))
        transported_detail = motion_component + common_component
        tangent_prev = common_tangent - transported_detail
        tangent_curr = common_tangent + transported_detail

        def encode_tangent(tangent: Tensor, reference: Tensor,
                           reference_logit: Tensor, reference_size: Tensor
                           ) -> Tensor:
            target_center = (
                reference[..., :2]
                + tangent[..., :2] * reference_size).clamp(
                    1e-3, 1 - 1e-3)
            target_log_size = (
                torch.log(reference_size) + tangent[..., 2:4]).clamp(
                    min=-13.815510557964274,
                    max=-1.0000005000003334e-6)
            target_size = torch.exp(target_log_size)
            target_angle = torch.remainder(
                reference[..., 4:] + tangent[..., 4:], 1.0)
            target = torch.cat(
                (target_center, target_size, target_angle), dim=-1)
            return inverse_sigmoid(target, eps=1e-3) - reference_logit

        normal_prev = encode_tangent(
            tangent_prev, ref_prev, ref_logit_prev, size_prev)
        normal_curr = encode_tangent(
            tangent_curr, ref_curr, ref_logit_curr, size_curr)
        if not num_dn:
            return normal_prev, normal_curr
        return (
            torch.cat((residual_prev[:, :num_dn], normal_prev), dim=1),
            torch.cat((residual_curr[:, :num_dn], normal_curr), dim=1),
        )

    @staticmethod
    def _pair_shared_angle_residual(
            residual_prev: Tensor, residual_curr: Tensor,
            num_dn: int) -> Tuple[Tensor, Tensor]:
        """Share only normal-query angle refinement across the pair.

        Center, width, and height residuals remain frame-specific. Only the
        angle residual is replaced by the pair mean, which suppresses noisy
        orientation drift without constraining translation or scale. The DN
        prefix stays untouched because it has no aligned-pair contract.
        """
        if residual_prev.shape != residual_curr.shape:
            raise ValueError(
                'pair-shared angle refinement requires aligned residuals')
        if residual_prev.shape[-1] != 5:
            raise ValueError(
                'pair-shared angle refinement requires 5D box residuals')
        if num_dn < 0 or num_dn > residual_prev.shape[1]:
            raise ValueError(
                f'invalid DN prefix length {num_dn} for pair residuals')

        normal_prev = residual_prev[:, num_dn:]
        normal_curr = residual_curr[:, num_dn:]
        shared_angle = 0.5 * (
            normal_prev[..., 4:] + normal_curr[..., 4:])
        normal_prev = torch.cat(
            (normal_prev[..., :4], shared_angle), dim=-1)
        normal_curr = torch.cat(
            (normal_curr[..., :4], shared_angle), dim=-1)
        if not num_dn:
            return normal_prev, normal_curr
        return (
            torch.cat((residual_prev[:, :num_dn], normal_prev), dim=1),
            torch.cat((residual_curr[:, :num_dn], normal_curr), dim=1),
        )

    @staticmethod
    def _pair_shared_periodic_angle_residual(
            residual_prev: Tensor, residual_curr: Tensor,
            reference_prev: Tensor, reference_curr: Tensor,
            num_dn: int) -> Tuple[Tensor, Tensor]:
        """Share normal-query angle increments in the periodic tangent space.

        Rotated boxes use a pi-periodic orientation, represented here by a
        normalized unit interval. Averaging raw regression logits distorts an
        equal physical rotation near the sigmoid or angle-wrap boundaries.
        This projection first decodes each frame's proposed angle, measures
        its shortest circular increment from that frame's own reference, and
        replaces the two increments by their circular midpoint. Each shared
        increment is then re-encoded relative to its original frame reference.

        Normal-query center, width, and height residuals stay independent.
        The DN prefix is unchanged because it has no aligned-pair contract.
        The operation is parameter-free, swap-equivariant away from the
        unavoidable antipodal midpoint ambiguity, and class agnostic.
        """
        if residual_prev.shape != residual_curr.shape:
            raise ValueError(
                'pair-shared periodic angle refinement requires aligned '
                'residuals')
        if reference_prev.shape != reference_curr.shape:
            raise ValueError(
                'pair-shared periodic angle refinement requires aligned '
                'references')
        if residual_prev.shape != reference_prev.shape:
            raise ValueError(
                'pair-shared periodic angle refinement requires residual '
                'and reference shapes to match')
        if residual_prev.shape[-1] != 5:
            raise ValueError(
                'pair-shared periodic angle refinement requires 5D boxes')
        if num_dn < 0 or num_dn > residual_prev.shape[1]:
            raise ValueError(
                f'invalid DN prefix length {num_dn} for pair residuals')

        normal_prev = residual_prev[:, num_dn:]
        normal_curr = residual_curr[:, num_dn:]
        reference_normal_prev = reference_prev[:, num_dn:]
        reference_normal_curr = reference_curr[:, num_dn:]
        reference_logit_prev = inverse_sigmoid(
            reference_normal_prev, eps=1e-3)
        reference_logit_curr = inverse_sigmoid(
            reference_normal_curr, eps=1e-3)

        proposed_angle_prev = (
            normal_prev[..., 4:] + reference_logit_prev[..., 4:]).sigmoid()
        proposed_angle_curr = (
            normal_curr[..., 4:] + reference_logit_curr[..., 4:]).sigmoid()

        def wrap_period(value: Tensor) -> Tensor:
            return torch.remainder(value + 0.5, 1.0) - 0.5

        delta_prev = wrap_period(
            proposed_angle_prev - reference_normal_prev[..., 4:])
        delta_curr = wrap_period(
            proposed_angle_curr - reference_normal_curr[..., 4:])
        shared_delta = wrap_period(
            delta_prev + 0.5 * wrap_period(delta_curr - delta_prev))

        target_angle_prev = torch.remainder(
            reference_normal_prev[..., 4:] + shared_delta, 1.0)
        target_angle_curr = torch.remainder(
            reference_normal_curr[..., 4:] + shared_delta, 1.0)
        shared_residual_prev = inverse_sigmoid(
            target_angle_prev, eps=1e-3) - reference_logit_prev[..., 4:]
        shared_residual_curr = inverse_sigmoid(
            target_angle_curr, eps=1e-3) - reference_logit_curr[..., 4:]

        normal_prev = torch.cat(
            (normal_prev[..., :4], shared_residual_prev), dim=-1)
        normal_curr = torch.cat(
            (normal_curr[..., :4], shared_residual_curr), dim=-1)
        if not num_dn:
            return normal_prev, normal_curr
        return (
            torch.cat((residual_prev[:, :num_dn], normal_prev), dim=1),
            torch.cat((residual_curr[:, :num_dn], normal_curr), dim=1),
        )

    @staticmethod
    def _pair_shared_log_size_residual(
            residual_prev: Tensor, residual_curr: Tensor,
            reference_prev: Tensor, reference_curr: Tensor,
            num_dn: int) -> Tuple[Tensor, Tensor]:
        """Share multiplicative width/height increments across the pair.

        Width and height are positive physical quantities. Averaging raw
        sigmoid logits does not represent an equal relative size correction
        when the two frame references differ. Decode each proposed size,
        average its log ratio to its own reference, then re-encode the common
        multiplicative correction relative to each original reference.
        Centers, angle, and the DN prefix remain unchanged.
        """
        if residual_prev.shape != residual_curr.shape:
            raise ValueError(
                'pair-shared log-size refinement requires aligned residuals')
        if reference_prev.shape != reference_curr.shape:
            raise ValueError(
                'pair-shared log-size refinement requires aligned references')
        if residual_prev.shape != reference_prev.shape:
            raise ValueError(
                'pair-shared log-size refinement requires residual and '
                'reference shapes to match')
        if residual_prev.shape[-1] != 5:
            raise ValueError(
                'pair-shared log-size refinement requires 5D boxes')
        if num_dn < 0 or num_dn > residual_prev.shape[1]:
            raise ValueError(
                f'invalid DN prefix length {num_dn} for pair residuals')

        normal_prev = residual_prev[:, num_dn:]
        normal_curr = residual_curr[:, num_dn:]
        reference_normal_prev = reference_prev[:, num_dn:]
        reference_normal_curr = reference_curr[:, num_dn:]
        reference_logit_prev = inverse_sigmoid(
            reference_normal_prev, eps=1e-3)
        reference_logit_curr = inverse_sigmoid(
            reference_normal_curr, eps=1e-3)

        proposed_size_prev = (
            normal_prev[..., 2:4]
            + reference_logit_prev[..., 2:4]).sigmoid()
        proposed_size_curr = (
            normal_curr[..., 2:4]
            + reference_logit_curr[..., 2:4]).sigmoid()
        reference_size_prev = reference_normal_prev[..., 2:4].clamp_min(1e-6)
        reference_size_curr = reference_normal_curr[..., 2:4].clamp_min(1e-6)
        log_delta_prev = torch.log(
            proposed_size_prev.clamp_min(1e-6) / reference_size_prev)
        log_delta_curr = torch.log(
            proposed_size_curr.clamp_min(1e-6) / reference_size_curr)
        shared_log_delta = 0.5 * (log_delta_prev + log_delta_curr)

        target_size_prev = (
            reference_size_prev * torch.exp(shared_log_delta)).clamp(
                min=1e-6, max=1 - 1e-6)
        target_size_curr = (
            reference_size_curr * torch.exp(shared_log_delta)).clamp(
                min=1e-6, max=1 - 1e-6)
        shared_residual_prev = inverse_sigmoid(
            target_size_prev, eps=1e-6) - reference_logit_prev[..., 2:4]
        shared_residual_curr = inverse_sigmoid(
            target_size_curr, eps=1e-6) - reference_logit_curr[..., 2:4]

        normal_prev = torch.cat((
            normal_prev[..., :2], shared_residual_prev,
            normal_prev[..., 4:]), dim=-1)
        normal_curr = torch.cat((
            normal_curr[..., :2], shared_residual_curr,
            normal_curr[..., 4:]), dim=-1)
        if not num_dn:
            return normal_prev, normal_curr
        return (
            torch.cat((residual_prev[:, :num_dn], normal_prev), dim=1),
            torch.cat((residual_curr[:, :num_dn], normal_curr), dim=1),
        )

    @staticmethod
    def _pair_shared_log_area_residual(
            residual_prev: Tensor, residual_curr: Tensor,
            reference_prev: Tensor, reference_curr: Tensor,
            num_dn: int) -> Tuple[Tensor, Tensor]:
        """Share isotropic log-area increments, preserving aspect changes."""
        if residual_prev.shape != residual_curr.shape:
            raise ValueError(
                'pair-shared log-area refinement requires aligned residuals')
        if reference_prev.shape != reference_curr.shape:
            raise ValueError(
                'pair-shared log-area refinement requires aligned references')
        if residual_prev.shape != reference_prev.shape:
            raise ValueError(
                'pair-shared log-area refinement requires residual and '
                'reference shapes to match')
        if residual_prev.shape[-1] != 5:
            raise ValueError(
                'pair-shared log-area refinement requires 5D boxes')
        if num_dn < 0 or num_dn > residual_prev.shape[1]:
            raise ValueError(
                f'invalid DN prefix length {num_dn} for pair residuals')

        normal_prev = residual_prev[:, num_dn:]
        normal_curr = residual_curr[:, num_dn:]
        reference_normal_prev = reference_prev[:, num_dn:]
        reference_normal_curr = reference_curr[:, num_dn:]
        reference_logit_prev = inverse_sigmoid(
            reference_normal_prev, eps=1e-3)
        reference_logit_curr = inverse_sigmoid(
            reference_normal_curr, eps=1e-3)
        proposed_size_prev = (
            normal_prev[..., 2:4]
            + reference_logit_prev[..., 2:4]).sigmoid()
        proposed_size_curr = (
            normal_curr[..., 2:4]
            + reference_logit_curr[..., 2:4]).sigmoid()
        reference_size_prev = reference_normal_prev[..., 2:4].clamp_min(1e-6)
        reference_size_curr = reference_normal_curr[..., 2:4].clamp_min(1e-6)
        log_delta_prev = torch.log(
            proposed_size_prev.clamp_min(1e-6) / reference_size_prev)
        log_delta_curr = torch.log(
            proposed_size_curr.clamp_min(1e-6) / reference_size_curr)

        area_prev = 0.5 * log_delta_prev.sum(dim=-1, keepdim=True)
        area_curr = 0.5 * log_delta_curr.sum(dim=-1, keepdim=True)
        shared_area = 0.5 * (area_prev + area_curr)
        aspect_prev = 0.5 * (
            log_delta_prev[..., :1] - log_delta_prev[..., 1:2])
        aspect_curr = 0.5 * (
            log_delta_curr[..., :1] - log_delta_curr[..., 1:2])
        target_delta_prev = torch.cat(
            (shared_area + aspect_prev, shared_area - aspect_prev), dim=-1)
        target_delta_curr = torch.cat(
            (shared_area + aspect_curr, shared_area - aspect_curr), dim=-1)

        target_size_prev = (
            reference_size_prev * torch.exp(target_delta_prev)).clamp(
                min=1e-6, max=1 - 1e-6)
        target_size_curr = (
            reference_size_curr * torch.exp(target_delta_curr)).clamp(
                min=1e-6, max=1 - 1e-6)
        shared_residual_prev = inverse_sigmoid(
            target_size_prev, eps=1e-6) - reference_logit_prev[..., 2:4]
        shared_residual_curr = inverse_sigmoid(
            target_size_curr, eps=1e-6) - reference_logit_curr[..., 2:4]
        normal_prev = torch.cat((
            normal_prev[..., :2], shared_residual_prev,
            normal_prev[..., 4:]), dim=-1)
        normal_curr = torch.cat((
            normal_curr[..., :2], shared_residual_curr,
            normal_curr[..., 4:]), dim=-1)
        if not num_dn:
            return normal_prev, normal_curr
        return (
            torch.cat((residual_prev[:, :num_dn], normal_prev), dim=1),
            torch.cat((residual_curr[:, :num_dn], normal_curr), dim=1),
        )

    @staticmethod
    def _pair_shared_normalized_center_residual(
            residual_prev: Tensor, residual_curr: Tensor,
            reference_prev: Tensor, reference_curr: Tensor,
            num_dn: int) -> Tuple[Tensor, Tensor]:
        """Share center correction in each reference box's local frame.

        Equal raw logits do not represent equal spatial corrections when the
        two references have different centers or sizes. This projection first
        decodes each proposed center, measures its displacement from the
        frame-specific reference in units of that reference's width/height,
        and averages only those normalized displacements. The shared local
        correction is then mapped back through each frame's own reference.

        Consequently, the correction remains anchored to each frame's own
        reference motion and scale instead of collapsing the two centers.
        Width, height, angle, and the DN prefix remain untouched. The
        operation is parameter-free,
        swap-equivariant, and class agnostic.
        """
        if residual_prev.shape != residual_curr.shape:
            raise ValueError(
                'pair-shared normalized-center refinement requires aligned '
                'residuals')
        if reference_prev.shape != reference_curr.shape:
            raise ValueError(
                'pair-shared normalized-center refinement requires aligned '
                'references')
        if residual_prev.shape != reference_prev.shape:
            raise ValueError(
                'pair-shared normalized-center refinement requires residual '
                'and reference shapes to match')
        if residual_prev.shape[-1] != 5:
            raise ValueError(
                'pair-shared normalized-center refinement requires 5D boxes')
        if num_dn < 0 or num_dn > residual_prev.shape[1]:
            raise ValueError(
                f'invalid DN prefix length {num_dn} for pair residuals')

        normal_prev = residual_prev[:, num_dn:]
        normal_curr = residual_curr[:, num_dn:]
        reference_normal_prev = reference_prev[:, num_dn:]
        reference_normal_curr = reference_curr[:, num_dn:]
        reference_logit_prev = inverse_sigmoid(
            reference_normal_prev, eps=1e-3)
        reference_logit_curr = inverse_sigmoid(
            reference_normal_curr, eps=1e-3)

        proposed_center_prev = (
            normal_prev[..., :2] + reference_logit_prev[..., :2]).sigmoid()
        proposed_center_curr = (
            normal_curr[..., :2] + reference_logit_curr[..., :2]).sigmoid()
        reference_size_prev = reference_normal_prev[..., 2:4].clamp_min(1e-3)
        reference_size_curr = reference_normal_curr[..., 2:4].clamp_min(1e-3)
        local_delta_prev = (
            proposed_center_prev - reference_normal_prev[..., :2]
        ) / reference_size_prev
        local_delta_curr = (
            proposed_center_curr - reference_normal_curr[..., :2]
        ) / reference_size_curr
        shared_local_delta = 0.5 * (local_delta_prev + local_delta_curr)

        target_center_prev = (
            reference_normal_prev[..., :2]
            + shared_local_delta * reference_size_prev).clamp(1e-3, 1 - 1e-3)
        target_center_curr = (
            reference_normal_curr[..., :2]
            + shared_local_delta * reference_size_curr).clamp(1e-3, 1 - 1e-3)
        shared_residual_prev = inverse_sigmoid(
            target_center_prev, eps=1e-3) - reference_logit_prev[..., :2]
        shared_residual_curr = inverse_sigmoid(
            target_center_curr, eps=1e-3) - reference_logit_curr[..., :2]

        normal_prev = torch.cat(
            (shared_residual_prev, normal_prev[..., 2:]), dim=-1)
        normal_curr = torch.cat(
            (shared_residual_curr, normal_curr[..., 2:]), dim=-1)
        if not num_dn:
            return normal_prev, normal_curr
        return (
            torch.cat((residual_prev[:, :num_dn], normal_prev), dim=1),
            torch.cat((residual_curr[:, :num_dn], normal_curr), dim=1),
        )

    @staticmethod
    def _pair_position_tangent_feature_detail(
            evidence_prev: Tensor, evidence_curr: Tensor,
            query_pos_prev: Tensor, query_pos_curr: Tensor,
            num_dn: int) -> Tensor:
        """Project terminal frame evidence onto established position motion.

        The existing reference-point MLP already embeds each frame's oriented
        box geometry in the decoder feature space. Its detached pair
        difference defines a local positional tangent. Only the component of
        the swap-odd cross-attention evidence lying on that tangent is kept;
        transverse appearance noise is removed without averaging away
        motion-aligned evidence. The projection is parameter-free, cannot
        increase detail energy, and preserves the unaligned DN prefix.
        """
        if evidence_prev.shape != evidence_curr.shape:
            raise ValueError(
                'position-tangent feature transport requires aligned '
                'frame evidence')
        if query_pos_prev.shape != query_pos_curr.shape:
            raise ValueError(
                'position-tangent feature transport requires aligned '
                'position features')
        if evidence_prev.shape != query_pos_prev.shape:
            raise ValueError(
                'position-tangent feature transport requires evidence and '
                'position features to have equal shapes')
        if num_dn < 0 or num_dn > evidence_prev.shape[1]:
            raise ValueError(
                f'invalid DN prefix length {num_dn} for pair features')

        detail = 0.5 * (
            evidence_curr[:, num_dn:] - evidence_prev[:, num_dn:])
        transport = (
            query_pos_curr[:, num_dn:] - query_pos_prev[:, num_dn:]).detach()
        transport_energy = transport.square().sum(
            dim=-1, keepdim=True).clamp_min(1e-6)
        transported_detail = transport * (
            (detail * transport).sum(dim=-1, keepdim=True)
            / transport_energy)
        if not num_dn:
            return transported_detail
        return torch.cat((
            torch.zeros_like(evidence_prev[:, :num_dn]),
            transported_detail), dim=1)

    @staticmethod
    def _pair_transport_center_tangent_residual(
            residual_prev: Tensor, residual_curr: Tensor,
            reference_prev: Tensor, reference_curr: Tensor,
            num_dn: int) -> Tuple[Tensor, Tensor]:
        """Transport terminal center detail along established translation.

        Final normal-query center updates are expressed as displacement in
        each reference box's local width/height coordinates. Their common
        component is retained, while their pair detail is projected onto the
        detached relative translation already accumulated in the references.
        Width, height, angle, and the DN prefix remain exactly frame-specific.

        This is a zero-state, swap-equivariant, class-agnostic projection.
        """
        if residual_prev.shape != residual_curr.shape:
            raise ValueError(
                'pair-transport center-tangent refinement requires aligned '
                'residuals')
        if reference_prev.shape != reference_curr.shape:
            raise ValueError(
                'pair-transport center-tangent refinement requires aligned '
                'references')
        if residual_prev.shape != reference_prev.shape:
            raise ValueError(
                'pair-transport center-tangent refinement requires residual '
                'and reference shapes to match')
        if residual_prev.shape[-1] != 5:
            raise ValueError(
                'pair-transport center-tangent refinement requires 5D boxes')
        if num_dn < 0 or num_dn > residual_prev.shape[1]:
            raise ValueError(
                f'invalid DN prefix length {num_dn} for pair residuals')

        normal_prev = residual_prev[:, num_dn:]
        normal_curr = residual_curr[:, num_dn:]
        ref_prev = reference_prev[:, num_dn:]
        ref_curr = reference_curr[:, num_dn:]
        ref_logit_prev = inverse_sigmoid(ref_prev, eps=1e-3)
        ref_logit_curr = inverse_sigmoid(ref_curr, eps=1e-3)
        proposed_prev = (normal_prev + ref_logit_prev).sigmoid()
        proposed_curr = (normal_curr + ref_logit_curr).sigmoid()
        size_prev = ref_prev[..., 2:4].clamp_min(1e-6)
        size_curr = ref_curr[..., 2:4].clamp_min(1e-6)

        tangent_prev = (
            proposed_prev[..., :2] - ref_prev[..., :2]) / size_prev
        tangent_curr = (
            proposed_curr[..., :2] - ref_curr[..., :2]) / size_curr
        common_tangent = 0.5 * (tangent_prev + tangent_curr)
        detail_tangent = 0.5 * (tangent_curr - tangent_prev)
        pair_size = torch.sqrt(size_prev * size_curr).clamp_min(1e-6)
        transport = (
            (ref_curr[..., :2] - ref_prev[..., :2]) / pair_size).detach()
        transport_energy = transport.square().sum(
            dim=-1, keepdim=True).clamp_min(1e-6)
        transported_detail = transport * (
            (detail_tangent * transport).sum(dim=-1, keepdim=True)
            / transport_energy)
        tangent_prev = common_tangent - transported_detail
        tangent_curr = common_tangent + transported_detail

        def encode_center(tangent: Tensor, reference: Tensor,
                          reference_logit: Tensor,
                          reference_size: Tensor,
                          original_residual: Tensor) -> Tensor:
            target_center = (
                reference[..., :2] + tangent * reference_size).clamp(
                    1e-3, 1 - 1e-3)
            center_residual = (
                inverse_sigmoid(target_center, eps=1e-3)
                - reference_logit[..., :2])
            return torch.cat(
                (center_residual, original_residual[..., 2:]), dim=-1)

        normal_prev = encode_center(
            tangent_prev, ref_prev, ref_logit_prev, size_prev, normal_prev)
        normal_curr = encode_center(
            tangent_curr, ref_curr, ref_logit_curr, size_curr, normal_curr)
        if not num_dn:
            return normal_prev, normal_curr
        return (
            torch.cat((residual_prev[:, :num_dn], normal_prev), dim=1),
            torch.cat((residual_curr[:, :num_dn], normal_curr), dim=1),
        )

    @staticmethod
    def _pair_transport_shared_metric_center_tangent_residual(
            residual_prev: Tensor, residual_curr: Tensor,
            reference_prev: Tensor, reference_curr: Tensor,
            num_dn: int) -> Tuple[Tensor, Tensor]:
        """Transport center detail in one shared pair-size metric.

        The original product tangent normalizes the two proposed center
        updates by different frame sizes, but compares their detail against a
        chord normalized by the geometric pair size. This variant uses that
        same geometric-mean width/height metric for both updates, projection,
        and reconstruction. It reduces to the original operation when the two
        reference sizes agree, remains swap-equivariant, and adds no state.
        """
        if residual_prev.shape != residual_curr.shape:
            raise ValueError(
                'pair shared-metric center-tangent refinement requires '
                'aligned residuals')
        if reference_prev.shape != reference_curr.shape:
            raise ValueError(
                'pair shared-metric center-tangent refinement requires '
                'aligned references')
        if residual_prev.shape != reference_prev.shape:
            raise ValueError(
                'pair shared-metric center-tangent refinement requires '
                'residual and reference shapes to match')
        if residual_prev.shape[-1] != 5:
            raise ValueError(
                'pair shared-metric center-tangent refinement requires 5D '
                'boxes')
        if num_dn < 0 or num_dn > residual_prev.shape[1]:
            raise ValueError(
                f'invalid DN prefix length {num_dn} for pair residuals')

        normal_prev = residual_prev[:, num_dn:]
        normal_curr = residual_curr[:, num_dn:]
        ref_prev = reference_prev[:, num_dn:]
        ref_curr = reference_curr[:, num_dn:]
        ref_logit_prev = inverse_sigmoid(ref_prev, eps=1e-3)
        ref_logit_curr = inverse_sigmoid(ref_curr, eps=1e-3)
        proposed_prev = (normal_prev + ref_logit_prev).sigmoid()
        proposed_curr = (normal_curr + ref_logit_curr).sigmoid()
        size_prev = ref_prev[..., 2:4].clamp_min(1e-6)
        size_curr = ref_curr[..., 2:4].clamp_min(1e-6)
        pair_size = torch.sqrt(size_prev * size_curr).clamp_min(1e-6)

        tangent_prev = (
            proposed_prev[..., :2] - ref_prev[..., :2]) / pair_size
        tangent_curr = (
            proposed_curr[..., :2] - ref_curr[..., :2]) / pair_size
        common_tangent = 0.5 * (tangent_prev + tangent_curr)
        detail_tangent = 0.5 * (tangent_curr - tangent_prev)
        transport = (
            (ref_curr[..., :2] - ref_prev[..., :2]) / pair_size).detach()
        transport_energy = transport.square().sum(
            dim=-1, keepdim=True).clamp_min(1e-6)
        transported_detail = transport * (
            (detail_tangent * transport).sum(dim=-1, keepdim=True)
            / transport_energy)
        tangent_prev = common_tangent - transported_detail
        tangent_curr = common_tangent + transported_detail

        def encode_center(tangent: Tensor, reference: Tensor,
                          reference_logit: Tensor,
                          original_residual: Tensor) -> Tensor:
            target_center = (
                reference[..., :2] + tangent * pair_size).clamp(
                    1e-3, 1 - 1e-3)
            center_residual = (
                inverse_sigmoid(target_center, eps=1e-3)
                - reference_logit[..., :2])
            return torch.cat(
                (center_residual, original_residual[..., 2:]), dim=-1)

        normal_prev = encode_center(
            tangent_prev, ref_prev, ref_logit_prev, normal_prev)
        normal_curr = encode_center(
            tangent_curr, ref_curr, ref_logit_curr, normal_curr)
        if not num_dn:
            return normal_prev, normal_curr
        return (
            torch.cat((residual_prev[:, :num_dn], normal_prev), dim=1),
            torch.cat((residual_curr[:, :num_dn], normal_curr), dim=1),
        )

    @staticmethod
    def _pair_transport_axis_frenet_center_tangent_residual(
            residual_prev: Tensor, residual_curr: Tensor,
            reference_prev: Tensor, reference_curr: Tensor,
            num_dn: int) -> Tuple[Tensor, Tensor]:
        """Project axis-normalized center detail onto endpoint tangents.

        The factorized product tangent is strongest in the original
        width/height-normalized image coordinates, while one shared chord
        direction is only exact for straight motion. This variant preserves
        that established metric and rotates only the detached reference chord
        by minus/plus half the pi-periodic orientation turn. Previous/current
        detail is then projected onto its own constant-turn endpoint tangent.

        Zero turn reduces exactly to the axis-normalized product tangent.
        Frame reversal swaps endpoint projectors and detail signs, making the
        operation swap-equivariant. It is parameter-free, class agnostic, and
        preserves the DN prefix and all shape residuals.
        """
        if residual_prev.shape != residual_curr.shape:
            raise ValueError(
                'pair axis-Frenet center-tangent refinement requires aligned '
                'residuals')
        if reference_prev.shape != reference_curr.shape:
            raise ValueError(
                'pair axis-Frenet center-tangent refinement requires aligned '
                'references')
        if residual_prev.shape != reference_prev.shape:
            raise ValueError(
                'pair axis-Frenet center-tangent refinement requires residual '
                'and reference shapes to match')
        if residual_prev.shape[-1] != 5:
            raise ValueError(
                'pair axis-Frenet center-tangent refinement requires 5D boxes')
        if num_dn < 0 or num_dn > residual_prev.shape[1]:
            raise ValueError(
                f'invalid DN prefix length {num_dn} for pair residuals')

        normal_prev = residual_prev[:, num_dn:]
        normal_curr = residual_curr[:, num_dn:]
        ref_prev = reference_prev[:, num_dn:]
        ref_curr = reference_curr[:, num_dn:]
        ref_logit_prev = inverse_sigmoid(ref_prev, eps=1e-3)
        ref_logit_curr = inverse_sigmoid(ref_curr, eps=1e-3)
        proposed_prev = (normal_prev + ref_logit_prev).sigmoid()
        proposed_curr = (normal_curr + ref_logit_curr).sigmoid()
        size_prev = ref_prev[..., 2:4].clamp_min(1e-6)
        size_curr = ref_curr[..., 2:4].clamp_min(1e-6)

        tangent_prev = (
            proposed_prev[..., :2] - ref_prev[..., :2]) / size_prev
        tangent_curr = (
            proposed_curr[..., :2] - ref_curr[..., :2]) / size_curr
        common_tangent = 0.5 * (tangent_prev + tangent_curr)
        detail_tangent = 0.5 * (tangent_curr - tangent_prev)
        pair_size = torch.sqrt(size_prev * size_curr).clamp_min(1e-6)
        chord = (
            (ref_curr[..., :2] - ref_prev[..., :2]) / pair_size).detach()

        def wrap_period(value: Tensor) -> Tensor:
            return torch.remainder(value + 0.5, 1.0) - 0.5

        half_turn = (
            0.5 * wrap_period(ref_curr[..., 4:] - ref_prev[..., 4:])
            * torch.pi).detach()

        def rotate(vector: Tensor, angle: Tensor) -> Tensor:
            cosine = torch.cos(angle)
            sine = torch.sin(angle)
            return torch.cat((
                cosine * vector[..., :1] - sine * vector[..., 1:2],
                sine * vector[..., :1] + cosine * vector[..., 1:2]),
                dim=-1)

        direction_prev = rotate(chord, -half_turn)
        direction_curr = rotate(chord, half_turn)

        def project(detail: Tensor, direction: Tensor) -> Tensor:
            energy = direction.square().sum(
                dim=-1, keepdim=True).clamp_min(1e-6)
            return direction * (
                (detail * direction).sum(dim=-1, keepdim=True) / energy)

        tangent_prev = common_tangent - project(
            detail_tangent, direction_prev)
        tangent_curr = common_tangent + project(
            detail_tangent, direction_curr)

        def encode_center(tangent: Tensor, reference: Tensor,
                          reference_logit: Tensor,
                          reference_size: Tensor,
                          original_residual: Tensor) -> Tensor:
            target_center = (
                reference[..., :2] + tangent * reference_size).clamp(
                    1e-3, 1 - 1e-3)
            center_residual = (
                inverse_sigmoid(target_center, eps=1e-3)
                - reference_logit[..., :2])
            return torch.cat(
                (center_residual, original_residual[..., 2:]), dim=-1)

        normal_prev = encode_center(
            tangent_prev, ref_prev, ref_logit_prev, size_prev, normal_prev)
        normal_curr = encode_center(
            tangent_curr, ref_curr, ref_logit_curr, size_curr, normal_curr)
        if not num_dn:
            return normal_prev, normal_curr
        return (
            torch.cat((residual_prev[:, :num_dn], normal_prev), dim=1),
            torch.cat((residual_curr[:, :num_dn], normal_curr), dim=1),
        )

    @staticmethod
    def _pair_transport_body_frame_center_tangent_residual(
            residual_prev: Tensor, residual_curr: Tensor,
            reference_prev: Tensor, reference_curr: Tensor,
            num_dn: int) -> Tuple[Tensor, Tensor]:
        """Transport center detail in an oriented pair-local body frame.

        Axis-aligned width/height normalization treats the image axes as the
        object's translation axes. For rotated boxes this can suppress a
        valid along-object correction as transverse jitter. This variant
        builds a swap-invariant body frame from the shortest pi-periodic
        midpoint of the two reference orientations and uses the geometric
        mean pair size as its metric. Both terminal center corrections and
        the established inter-frame translation are expressed in that common
        frame before the usual one-dimensional tangent projection, then
        mapped back to image coordinates.

        Shape residuals and the DN prefix remain exactly frame-specific. The
        operation is parameter-free, swap-equivariant, class agnostic, and
        adds only elementwise trigonometry to the terminal layer.
        """
        if residual_prev.shape != residual_curr.shape:
            raise ValueError(
                'pair body-frame center-tangent refinement requires aligned '
                'residuals')
        if reference_prev.shape != reference_curr.shape:
            raise ValueError(
                'pair body-frame center-tangent refinement requires aligned '
                'references')
        if residual_prev.shape != reference_prev.shape:
            raise ValueError(
                'pair body-frame center-tangent refinement requires residual '
                'and reference shapes to match')
        if residual_prev.shape[-1] != 5:
            raise ValueError(
                'pair body-frame center-tangent refinement requires 5D boxes')
        if num_dn < 0 or num_dn > residual_prev.shape[1]:
            raise ValueError(
                f'invalid DN prefix length {num_dn} for pair residuals')

        normal_prev = residual_prev[:, num_dn:]
        normal_curr = residual_curr[:, num_dn:]
        ref_prev = reference_prev[:, num_dn:]
        ref_curr = reference_curr[:, num_dn:]
        ref_logit_prev = inverse_sigmoid(ref_prev, eps=1e-3)
        ref_logit_curr = inverse_sigmoid(ref_curr, eps=1e-3)
        proposed_prev = (normal_prev + ref_logit_prev).sigmoid()
        proposed_curr = (normal_curr + ref_logit_curr).sigmoid()

        def wrap_period(value: Tensor) -> Tensor:
            return torch.remainder(value + 0.5, 1.0) - 0.5

        pair_angle = torch.remainder(
            ref_prev[..., 4:]
            + 0.5 * wrap_period(ref_curr[..., 4:] - ref_prev[..., 4:]),
            1.0).detach()
        theta = pair_angle * torch.pi
        cos_theta = torch.cos(theta)
        sin_theta = torch.sin(theta)

        def to_body(vector: Tensor) -> Tensor:
            return torch.cat((
                cos_theta * vector[..., :1]
                + sin_theta * vector[..., 1:2],
                -sin_theta * vector[..., :1]
                + cos_theta * vector[..., 1:2]), dim=-1)

        def from_body(vector: Tensor) -> Tensor:
            return torch.cat((
                cos_theta * vector[..., :1]
                - sin_theta * vector[..., 1:2],
                sin_theta * vector[..., :1]
                + cos_theta * vector[..., 1:2]), dim=-1)

        pair_size = torch.sqrt(
            ref_prev[..., 2:4].clamp_min(1e-6)
            * ref_curr[..., 2:4].clamp_min(1e-6)).detach().clamp_min(1e-6)
        tangent_prev = to_body(
            proposed_prev[..., :2] - ref_prev[..., :2]) / pair_size
        tangent_curr = to_body(
            proposed_curr[..., :2] - ref_curr[..., :2]) / pair_size
        common_tangent = 0.5 * (tangent_prev + tangent_curr)
        detail_tangent = 0.5 * (tangent_curr - tangent_prev)
        transport = (
            to_body(ref_curr[..., :2] - ref_prev[..., :2])
            / pair_size).detach()
        transport_energy = transport.square().sum(
            dim=-1, keepdim=True).clamp_min(1e-6)
        transported_detail = transport * (
            (detail_tangent * transport).sum(dim=-1, keepdim=True)
            / transport_energy)
        tangent_prev = common_tangent - transported_detail
        tangent_curr = common_tangent + transported_detail

        def encode_center(tangent: Tensor, reference: Tensor,
                          reference_logit: Tensor,
                          original_residual: Tensor) -> Tensor:
            target_center = (
                reference[..., :2]
                + from_body(tangent * pair_size)).clamp(1e-3, 1 - 1e-3)
            center_residual = (
                inverse_sigmoid(target_center, eps=1e-3)
                - reference_logit[..., :2])
            return torch.cat(
                (center_residual, original_residual[..., 2:]), dim=-1)

        normal_prev = encode_center(
            tangent_prev, ref_prev, ref_logit_prev, normal_prev)
        normal_curr = encode_center(
            tangent_curr, ref_curr, ref_logit_curr, normal_curr)
        if not num_dn:
            return normal_prev, normal_curr
        return (
            torch.cat((residual_prev[:, :num_dn], normal_prev), dim=1),
            torch.cat((residual_curr[:, :num_dn], normal_curr), dim=1),
        )

    @staticmethod
    def _pair_transport_se2_center_tangent_residual(
            residual_prev: Tensor, residual_curr: Tensor,
            reference_prev: Tensor, reference_curr: Tensor,
            num_dn: int) -> Tuple[Tensor, Tensor]:
        """Transport center detail as finite SE(2) midpoint twists.

        A planar rigid motion with rotation ``omega`` maps its translational
        Lie-algebra coordinate through the SE(2) left Jacobian. In midpoint
        coordinates that Jacobian is the scalar
        ``sinc(omega / 2)`` followed by the half-angle rotation. The common
        pair body frame absorbs that rotation, so dividing by the even sinc
        factor yields a swap-equivariant finite-motion tangent. We project
        only the antisymmetric center detail onto the established reference
        motion, then apply the matching sinc retraction for each frame.

        The transported shape projection runs before this method, making its
        terminal angle the sole new coupling signal. DN entries and shape
        residuals are preserved exactly. The operation is parameter-free,
        class agnostic, and reduces exactly to body-frame center transport
        when all angular increments are zero.
        """
        if residual_prev.shape != residual_curr.shape:
            raise ValueError(
                'pair SE(2) center-tangent refinement requires aligned '
                'residuals')
        if reference_prev.shape != reference_curr.shape:
            raise ValueError(
                'pair SE(2) center-tangent refinement requires aligned '
                'references')
        if residual_prev.shape != reference_prev.shape:
            raise ValueError(
                'pair SE(2) center-tangent refinement requires residual and '
                'reference shapes to match')
        if residual_prev.shape[-1] != 5:
            raise ValueError(
                'pair SE(2) center-tangent refinement requires 5D boxes')
        if num_dn < 0 or num_dn > residual_prev.shape[1]:
            raise ValueError(
                f'invalid DN prefix length {num_dn} for pair residuals')

        normal_prev = residual_prev[:, num_dn:]
        normal_curr = residual_curr[:, num_dn:]
        ref_prev = reference_prev[:, num_dn:]
        ref_curr = reference_curr[:, num_dn:]
        ref_logit_prev = inverse_sigmoid(ref_prev, eps=1e-3)
        ref_logit_curr = inverse_sigmoid(ref_curr, eps=1e-3)
        proposed_prev = (normal_prev + ref_logit_prev).sigmoid()
        proposed_curr = (normal_curr + ref_logit_curr).sigmoid()

        def wrap_period(value: Tensor) -> Tensor:
            return torch.remainder(value + 0.5, 1.0) - 0.5

        pair_angle = torch.remainder(
            ref_prev[..., 4:]
            + 0.5 * wrap_period(ref_curr[..., 4:] - ref_prev[..., 4:]),
            1.0).detach()
        theta = pair_angle * torch.pi
        cos_theta = torch.cos(theta)
        sin_theta = torch.sin(theta)

        def to_body(vector: Tensor) -> Tensor:
            return torch.cat((
                cos_theta * vector[..., :1]
                + sin_theta * vector[..., 1:2],
                -sin_theta * vector[..., :1]
                + cos_theta * vector[..., 1:2]), dim=-1)

        def from_body(vector: Tensor) -> Tensor:
            return torch.cat((
                cos_theta * vector[..., :1]
                - sin_theta * vector[..., 1:2],
                sin_theta * vector[..., :1]
                + cos_theta * vector[..., 1:2]), dim=-1)

        def midpoint_jacobian(angle_delta: Tensor) -> Tensor:
            omega = wrap_period(angle_delta) * torch.pi
            # torch.sinc(x) is sin(pi*x)/(pi*x), including its stable x=0
            # limit. Our normalized argument therefore evaluates
            # sin(omega/2)/(omega/2) without a fragile conditional branch.
            return torch.sinc(omega / (2.0 * torch.pi)).clamp_min(1e-4)

        pair_size = torch.sqrt(
            ref_prev[..., 2:4].clamp_min(1e-6)
            * ref_curr[..., 2:4].clamp_min(1e-6)).detach().clamp_min(1e-6)
        jacobian_prev = midpoint_jacobian(
            proposed_prev[..., 4:] - ref_prev[..., 4:])
        jacobian_curr = midpoint_jacobian(
            proposed_curr[..., 4:] - ref_curr[..., 4:])
        tangent_prev = (
            to_body(proposed_prev[..., :2] - ref_prev[..., :2])
            / pair_size / jacobian_prev)
        tangent_curr = (
            to_body(proposed_curr[..., :2] - ref_curr[..., :2])
            / pair_size / jacobian_curr)
        common_tangent = 0.5 * (tangent_prev + tangent_curr)
        detail_tangent = 0.5 * (tangent_curr - tangent_prev)

        transport_jacobian = midpoint_jacobian(
            ref_curr[..., 4:] - ref_prev[..., 4:]).detach()
        transport = (
            to_body(ref_curr[..., :2] - ref_prev[..., :2])
            / pair_size / transport_jacobian).detach()
        transport_energy = transport.square().sum(
            dim=-1, keepdim=True).clamp_min(1e-6)
        transported_detail = transport * (
            (detail_tangent * transport).sum(dim=-1, keepdim=True)
            / transport_energy)
        tangent_prev = common_tangent - transported_detail
        tangent_curr = common_tangent + transported_detail

        def encode_center(tangent: Tensor, reference: Tensor,
                          reference_logit: Tensor, jacobian: Tensor,
                          original_residual: Tensor) -> Tensor:
            target_center = (
                reference[..., :2]
                + from_body(tangent * pair_size * jacobian)).clamp(
                    1e-3, 1 - 1e-3)
            center_residual = (
                inverse_sigmoid(target_center, eps=1e-3)
                - reference_logit[..., :2])
            return torch.cat(
                (center_residual, original_residual[..., 2:]), dim=-1)

        normal_prev = encode_center(
            tangent_prev, ref_prev, ref_logit_prev, jacobian_prev,
            normal_prev)
        normal_curr = encode_center(
            tangent_curr, ref_curr, ref_logit_curr, jacobian_curr,
            normal_curr)
        if not num_dn:
            return normal_prev, normal_curr
        return (
            torch.cat((residual_prev[:, :num_dn], normal_prev), dim=1),
            torch.cat((residual_curr[:, :num_dn], normal_curr), dim=1),
        )

    @staticmethod
    def _pair_transport_frenet_center_tangent_residual(
            residual_prev: Tensor, residual_curr: Tensor,
            reference_prev: Tensor, reference_curr: Tensor,
            num_dn: int) -> Tuple[Tensor, Tensor]:
        """Project center detail onto constant-turn endpoint tangents.

        A single chord direction is exact only for straight translation. When
        the reference orientation turns, the physically consistent endpoint
        velocities rotate by minus/plus half the turn relative to the chord.
        This method therefore projects the previous and current detail onto
        their own Frenet endpoint tangents in the established pair body frame.
        It retains curved-motion localization without opening an unconstrained
        two-dimensional center plane.

        Reversing frame order swaps the endpoint projectors and negates the
        detail, so the operation is exactly swap-equivariant. At zero turn it
        reduces to body-frame center transport. It is parameter-free, class
        agnostic, and preserves DN and shape residuals exactly.
        """
        if residual_prev.shape != residual_curr.shape:
            raise ValueError(
                'pair Frenet center-tangent refinement requires aligned '
                'residuals')
        if reference_prev.shape != reference_curr.shape:
            raise ValueError(
                'pair Frenet center-tangent refinement requires aligned '
                'references')
        if residual_prev.shape != reference_prev.shape:
            raise ValueError(
                'pair Frenet center-tangent refinement requires residual '
                'and reference shapes to match')
        if residual_prev.shape[-1] != 5:
            raise ValueError(
                'pair Frenet center-tangent refinement requires 5D boxes')
        if num_dn < 0 or num_dn > residual_prev.shape[1]:
            raise ValueError(
                f'invalid DN prefix length {num_dn} for pair residuals')

        normal_prev = residual_prev[:, num_dn:]
        normal_curr = residual_curr[:, num_dn:]
        ref_prev = reference_prev[:, num_dn:]
        ref_curr = reference_curr[:, num_dn:]
        ref_logit_prev = inverse_sigmoid(ref_prev, eps=1e-3)
        ref_logit_curr = inverse_sigmoid(ref_curr, eps=1e-3)
        proposed_prev = (normal_prev + ref_logit_prev).sigmoid()
        proposed_curr = (normal_curr + ref_logit_curr).sigmoid()

        def wrap_period(value: Tensor) -> Tensor:
            return torch.remainder(value + 0.5, 1.0) - 0.5

        pair_angle = torch.remainder(
            ref_prev[..., 4:]
            + 0.5 * wrap_period(ref_curr[..., 4:] - ref_prev[..., 4:]),
            1.0).detach()
        theta = pair_angle * torch.pi
        cos_theta = torch.cos(theta)
        sin_theta = torch.sin(theta)

        def to_body(vector: Tensor) -> Tensor:
            return torch.cat((
                cos_theta * vector[..., :1]
                + sin_theta * vector[..., 1:2],
                -sin_theta * vector[..., :1]
                + cos_theta * vector[..., 1:2]), dim=-1)

        def from_body(vector: Tensor) -> Tensor:
            return torch.cat((
                cos_theta * vector[..., :1]
                - sin_theta * vector[..., 1:2],
                sin_theta * vector[..., :1]
                + cos_theta * vector[..., 1:2]), dim=-1)

        def rotate(vector: Tensor, angle: Tensor) -> Tensor:
            cosine = torch.cos(angle)
            sine = torch.sin(angle)
            return torch.cat((
                cosine * vector[..., :1] - sine * vector[..., 1:2],
                sine * vector[..., :1] + cosine * vector[..., 1:2]),
                dim=-1)

        pair_size = torch.sqrt(
            ref_prev[..., 2:4].clamp_min(1e-6)
            * ref_curr[..., 2:4].clamp_min(1e-6)).detach().clamp_min(1e-6)
        tangent_prev = (
            to_body(proposed_prev[..., :2] - ref_prev[..., :2])
            / pair_size)
        tangent_curr = (
            to_body(proposed_curr[..., :2] - ref_curr[..., :2])
            / pair_size)
        common_tangent = 0.5 * (tangent_prev + tangent_curr)
        detail_tangent = 0.5 * (tangent_curr - tangent_prev)

        chord = (
            to_body(ref_curr[..., :2] - ref_prev[..., :2])
            / pair_size).detach()
        half_turn = (
            0.5 * wrap_period(ref_curr[..., 4:] - ref_prev[..., 4:])
            * torch.pi).detach()
        tangent_direction_prev = rotate(chord, -half_turn)
        tangent_direction_curr = rotate(chord, half_turn)

        def project(detail: Tensor, direction: Tensor) -> Tensor:
            energy = direction.square().sum(
                dim=-1, keepdim=True).clamp_min(1e-6)
            return direction * (
                (detail * direction).sum(dim=-1, keepdim=True) / energy)

        tangent_prev = common_tangent - project(
            detail_tangent, tangent_direction_prev)
        tangent_curr = common_tangent + project(
            detail_tangent, tangent_direction_curr)

        def encode_center(tangent: Tensor, reference: Tensor,
                          reference_logit: Tensor,
                          original_residual: Tensor) -> Tensor:
            target_center = (
                reference[..., :2]
                + from_body(tangent * pair_size)).clamp(1e-3, 1 - 1e-3)
            center_residual = (
                inverse_sigmoid(target_center, eps=1e-3)
                - reference_logit[..., :2])
            return torch.cat(
                (center_residual, original_residual[..., 2:]), dim=-1)

        normal_prev = encode_center(
            tangent_prev, ref_prev, ref_logit_prev, normal_prev)
        normal_curr = encode_center(
            tangent_curr, ref_curr, ref_logit_curr, normal_curr)
        if not num_dn:
            return normal_prev, normal_curr
        return (
            torch.cat((residual_prev[:, :num_dn], normal_prev), dim=1),
            torch.cat((residual_curr[:, :num_dn], normal_curr), dim=1),
        )

    @staticmethod
    def _pair_transport_shape_tangent_residual(
            residual_prev: Tensor, residual_curr: Tensor,
            reference_prev: Tensor, reference_curr: Tensor,
            num_dn: int) -> Tuple[Tensor, Tensor]:
        """Transport terminal size/angle detail along established shape motion.

        Center residuals remain exactly frame-specific. Size and angle updates
        are represented by log-width, log-height, and shortest pi-periodic
        angle tangents. Their common component is retained, while their detail
        is projected onto the detached relative size/angle transform already
        accumulated in the input references. This isolates the conservative
        terminal geometry of the log-size/periodic-angle decoder from center
        transport and suppresses only transverse shape jitter.

        The projection is parameter-free, swap-equivariant, class agnostic,
        and leaves the DN prefix unchanged.
        """
        if residual_prev.shape != residual_curr.shape:
            raise ValueError(
                'pair-transport shape-tangent refinement requires aligned '
                'residuals')
        if reference_prev.shape != reference_curr.shape:
            raise ValueError(
                'pair-transport shape-tangent refinement requires aligned '
                'references')
        if residual_prev.shape != reference_prev.shape:
            raise ValueError(
                'pair-transport shape-tangent refinement requires residual '
                'and reference shapes to match')
        if residual_prev.shape[-1] != 5:
            raise ValueError(
                'pair-transport shape-tangent refinement requires 5D boxes')
        if num_dn < 0 or num_dn > residual_prev.shape[1]:
            raise ValueError(
                f'invalid DN prefix length {num_dn} for pair residuals')

        normal_prev = residual_prev[:, num_dn:]
        normal_curr = residual_curr[:, num_dn:]
        ref_prev = reference_prev[:, num_dn:]
        ref_curr = reference_curr[:, num_dn:]
        ref_logit_prev = inverse_sigmoid(ref_prev, eps=1e-3)
        ref_logit_curr = inverse_sigmoid(ref_curr, eps=1e-3)
        proposed_prev = (normal_prev + ref_logit_prev).sigmoid()
        proposed_curr = (normal_curr + ref_logit_curr).sigmoid()
        size_prev = ref_prev[..., 2:4].clamp_min(1e-6)
        size_curr = ref_curr[..., 2:4].clamp_min(1e-6)

        def wrap_period(value: Tensor) -> Tensor:
            return torch.remainder(value + 0.5, 1.0) - 0.5

        tangent_prev = torch.cat((
            torch.log(proposed_prev[..., 2:4].clamp_min(1e-6) / size_prev),
            wrap_period(proposed_prev[..., 4:] - ref_prev[..., 4:])),
            dim=-1)
        tangent_curr = torch.cat((
            torch.log(proposed_curr[..., 2:4].clamp_min(1e-6) / size_curr),
            wrap_period(proposed_curr[..., 4:] - ref_curr[..., 4:])),
            dim=-1)
        common_tangent = 0.5 * (tangent_prev + tangent_curr)
        detail_tangent = 0.5 * (tangent_curr - tangent_prev)
        transport = torch.cat((
            torch.log(size_curr / size_prev),
            wrap_period(ref_curr[..., 4:] - ref_prev[..., 4:])),
            dim=-1).detach()
        transport_energy = transport.square().sum(
            dim=-1, keepdim=True).clamp_min(1e-6)
        transported_detail = transport * (
            (detail_tangent * transport).sum(dim=-1, keepdim=True)
            / transport_energy)
        tangent_prev = common_tangent - transported_detail
        tangent_curr = common_tangent + transported_detail

        def encode_shape(tangent: Tensor, reference: Tensor,
                         reference_logit: Tensor,
                         reference_size: Tensor,
                         original_residual: Tensor) -> Tensor:
            target_log_size = (
                torch.log(reference_size) + tangent[..., :2]).clamp(
                    min=-13.815510557964274,
                    max=-1.0000005000003334e-6)
            target_size = torch.exp(target_log_size)
            target_angle = torch.remainder(
                reference[..., 4:] + tangent[..., 2:], 1.0)
            target_shape = torch.cat((target_size, target_angle), dim=-1)
            shape_residual = (
                inverse_sigmoid(target_shape, eps=1e-3)
                - reference_logit[..., 2:])
            return torch.cat(
                (original_residual[..., :2], shape_residual), dim=-1)

        normal_prev = encode_shape(
            tangent_prev, ref_prev, ref_logit_prev, size_prev, normal_prev)
        normal_curr = encode_shape(
            tangent_curr, ref_curr, ref_logit_curr, size_curr, normal_curr)
        if not num_dn:
            return normal_prev, normal_curr
        return (
            torch.cat((residual_prev[:, :num_dn], normal_prev), dim=1),
            torch.cat((residual_curr[:, :num_dn], normal_curr), dim=1),
        )

    @staticmethod
    def _pair_transport_full_tangent_residual(
            residual_prev: Tensor, residual_curr: Tensor,
            reference_prev: Tensor, reference_curr: Tensor,
            num_dn: int) -> Tuple[Tensor, Tensor]:
        """Transport terminal pair detail along established box motion.

        Final box increments are represented in a reference-local product
        tangent: normalized center displacement, log-size change, and the
        shortest pi-periodic angle change. Their pair-common component is
        retained, while the frame-specific component is projected onto the
        detached relative transform already accumulated in the input
        references. The terminal layer can therefore continue established
        translation, scale, or rotation, but cannot introduce a new
        transverse pair discrepancy.

        This is a parameter-free, swap-equivariant geometric projection. DN
        queries remain unchanged because they have no aligned pair contract.
        """
        if residual_prev.shape != residual_curr.shape:
            raise ValueError(
                'pair-transport tangent refinement requires aligned '
                'residuals')
        if reference_prev.shape != reference_curr.shape:
            raise ValueError(
                'pair-transport tangent refinement requires aligned '
                'references')
        if residual_prev.shape != reference_prev.shape:
            raise ValueError(
                'pair-transport tangent refinement requires residual and '
                'reference shapes to match')
        if residual_prev.shape[-1] != 5:
            raise ValueError(
                'pair-transport tangent refinement requires 5D boxes')
        if num_dn < 0 or num_dn > residual_prev.shape[1]:
            raise ValueError(
                f'invalid DN prefix length {num_dn} for pair residuals')

        normal_prev = residual_prev[:, num_dn:]
        normal_curr = residual_curr[:, num_dn:]
        ref_prev = reference_prev[:, num_dn:]
        ref_curr = reference_curr[:, num_dn:]
        ref_logit_prev = inverse_sigmoid(ref_prev, eps=1e-3)
        ref_logit_curr = inverse_sigmoid(ref_curr, eps=1e-3)

        proposed_prev = (normal_prev + ref_logit_prev).sigmoid()
        proposed_curr = (normal_curr + ref_logit_curr).sigmoid()
        size_prev = ref_prev[..., 2:4].clamp_min(1e-6)
        size_curr = ref_curr[..., 2:4].clamp_min(1e-6)

        def wrap_period(value: Tensor) -> Tensor:
            return torch.remainder(value + 0.5, 1.0) - 0.5

        tangent_prev = torch.cat((
            (proposed_prev[..., :2] - ref_prev[..., :2]) / size_prev,
            torch.log(proposed_prev[..., 2:4].clamp_min(1e-6) / size_prev),
            wrap_period(proposed_prev[..., 4:] - ref_prev[..., 4:])),
            dim=-1)
        tangent_curr = torch.cat((
            (proposed_curr[..., :2] - ref_curr[..., :2]) / size_curr,
            torch.log(proposed_curr[..., 2:4].clamp_min(1e-6) / size_curr),
            wrap_period(proposed_curr[..., 4:] - ref_curr[..., 4:])),
            dim=-1)

        common_tangent = 0.5 * (tangent_prev + tangent_curr)
        detail_tangent = 0.5 * (tangent_curr - tangent_prev)
        pair_size = torch.sqrt(size_prev * size_curr).clamp_min(1e-6)
        transport = torch.cat((
            (ref_curr[..., :2] - ref_prev[..., :2]) / pair_size,
            torch.log(size_curr / size_prev),
            wrap_period(ref_curr[..., 4:] - ref_prev[..., 4:])),
            dim=-1).detach()
        transport_energy = transport.square().sum(
            dim=-1, keepdim=True).clamp_min(1e-6)
        transported_detail = transport * (
            (detail_tangent * transport).sum(dim=-1, keepdim=True)
            / transport_energy)
        tangent_prev = common_tangent - transported_detail
        tangent_curr = common_tangent + transported_detail

        def encode_tangent(tangent: Tensor, reference: Tensor,
                           reference_logit: Tensor, reference_size: Tensor
                           ) -> Tensor:
            target_center = (
                reference[..., :2]
                + tangent[..., :2] * reference_size).clamp(
                    1e-3, 1 - 1e-3)
            # Clamp in log space before exponentiation. This is equivalent to
            # clamping the decoded size, but avoids exp overflow and the
            # resulting inf*0 NaN gradient when a tiny reference's large
            # center tangent projects onto a scale dimension.
            target_log_size = (
                torch.log(reference_size) + tangent[..., 2:4]).clamp(
                    min=-13.815510557964274,
                    max=-1.0000005000003334e-6)
            target_size = torch.exp(target_log_size)
            target_angle = torch.remainder(
                reference[..., 4:] + tangent[..., 4:], 1.0)
            target = torch.cat(
                (target_center, target_size, target_angle), dim=-1)
            return inverse_sigmoid(target, eps=1e-3) - reference_logit

        normal_prev = encode_tangent(
            tangent_prev, ref_prev, ref_logit_prev, size_prev)
        normal_curr = encode_tangent(
            tangent_curr, ref_curr, ref_logit_curr, size_curr)
        if not num_dn:
            return normal_prev, normal_curr
        return (
            torch.cat((residual_prev[:, :num_dn], normal_prev), dim=1),
            torch.cat((residual_curr[:, :num_dn], normal_curr), dim=1),
        )
