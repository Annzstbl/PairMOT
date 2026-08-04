# Copyright (c) AI4RS. All rights reserved.
"""Unit tests for PairRotatedRTDETRTransformerDecoder (M3j / M3-2)."""

import copy
import os.path as osp
import sys
import unittest
from unittest import mock

import torch
from mmengine.config import Config

_AI4RS_ROOT = osp.abspath(osp.join(osp.dirname(__file__), '../..'))
if _AI4RS_ROOT not in sys.path:
    sys.path.insert(0, _AI4RS_ROOT)

from mmrotate.utils import register_all_modules
from projects.multispec_pair_rotated_rtdetr.multispec_pair_rotated_rtdetr.multispec_pair_rotated_rtdetr import (  # noqa: E501
    MultispecPairRotatedRTDETR,
)
from projects.multispec_pair_rotated_rtdetr.multispec_pair_rotated_rtdetr.pair_rotated_rtdetr_layers import (  # noqa: E501
    PairRotatedRTDETRTransformerDecoder,
    PairRotatedRTDETRTransformerDecoderLayer,
)
from projects.rotated_rtdetr.rotated_rtdetr import RotatedRTDETR


def _spatial_meta(device: torch.device):
    """Small 3-level spatial shapes matching typical RT-DETR neck."""
    spatial_shapes = torch.tensor(
        [[20, 25], [10, 13], [5, 7]], device=device, dtype=torch.long)
    level_start_index = torch.cat([
        spatial_shapes.new_zeros((1, )),
        spatial_shapes.prod(1).cumsum(0)[:-1],
    ])
    num_value = int(spatial_shapes.prod(1).sum())
    return spatial_shapes, level_start_index, num_value


def _build_reg_branches(num_layers: int,
                        embed_dims: int,
                        device: torch.device,
                        seed: int = 0):
    torch.manual_seed(seed)
    branches = torch.nn.ModuleList([
        torch.nn.Linear(embed_dims, 5) for _ in range(num_layers)
    ]).to(device)
    for branch in branches:
        torch.nn.init.normal_(branch.weight, std=0.01)
        torch.nn.init.zeros_(branch.bias)
    return branches


def _build_cls_branches(num_layers: int,
                        embed_dims: int,
                        num_classes: int,
                        device: torch.device,
                        seed: int = 0):
    torch.manual_seed(seed)
    branches = torch.nn.ModuleList([
        torch.nn.Linear(embed_dims, num_classes)
        for _ in range(num_layers)
    ]).to(device)
    for branch in branches:
        torch.nn.init.normal_(branch.weight, std=0.01)
        torch.nn.init.zeros_(branch.bias)
    return branches


def _build_decoder(num_layers: int = 2,
                   num_queries: int = 8,
                   embed_dims: int = 64,
                   device: torch.device = torch.device('cpu'),
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
                   pair_shared_periodic_angle_refinement_decoder:
                   bool = False,
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
                    pair_shared_terminal_transport_tangent_refinement_decoder:
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
                   terminal_factorized_detail_only: bool = False):
    layer_cfg = dict(
        self_attn_cfg=dict(
            embed_dims=embed_dims, num_heads=4, dropout=0.0, batch_first=True),
        cross_attn_cfg=dict(
            embed_dims=embed_dims,
            num_heads=4,
            num_levels=3,
            num_points=4,
            dropout=0.0,
            batch_first=True),
        ffn_cfg=dict(
            embed_dims=embed_dims,
            feedforward_channels=128,
            ffn_drop=0.0,
            act_cfg=dict(type='GELU')),
    )
    decoder = PairRotatedRTDETRTransformerDecoder(
        num_layers=num_layers,
        num_queries=num_queries,
        return_intermediate=True,
        layer_cfg=layer_cfg,
        post_norm_cfg=None,
        angle_factor=3.141592653589793,
        tristate_decoder=tristate_decoder,
        tristate_separate_ffn=tristate_separate_ffn,
        tristate_zero_init_coupling=tristate_zero_init_coupling,
        dual_output_adapter=dual_output_adapter,
        dual_output_cls_scale=dual_output_cls_scale,
        dual_output_reg_scale=dual_output_reg_scale,
        dual_output_detach_adapter_input=dual_output_detach_adapter_input,
        common_motion_decoder=common_motion_decoder,
        shared_evidence_decoder=shared_evidence_decoder,
        competitive_evidence_decoder=competitive_evidence_decoder,
        motion_trust_decoder=motion_trust_decoder,
        symmetric_pair_decoder=symmetric_pair_decoder,
        symmetric_position_decoder=symmetric_position_decoder,
        symmetric_feature_decoder=symmetric_feature_decoder,
        residual_preserving_fusion_decoder=(
            residual_preserving_fusion_decoder),
        pair_shared_shape_refinement_decoder=(
            pair_shared_shape_refinement_decoder),
        pair_shared_angle_refinement_decoder=(
            pair_shared_angle_refinement_decoder),
        pair_shared_periodic_angle_refinement_decoder=(
            pair_shared_periodic_angle_refinement_decoder),
        pair_shared_log_size_periodic_angle_refinement_decoder=(
            pair_shared_log_size_periodic_angle_refinement_decoder),
        pair_shared_log_area_periodic_angle_refinement_decoder=(
            pair_shared_log_area_periodic_angle_refinement_decoder),
        pair_shared_late_log_size_periodic_angle_refinement_decoder=(
            pair_shared_late_log_size_periodic_angle_refinement_decoder),
        pair_shared_terminal_log_size_periodic_angle_refinement_decoder=(
            pair_shared_terminal_log_size_periodic_angle_refinement_decoder),
        pair_shared_terminal_log_area_periodic_angle_refinement_decoder=(
            pair_shared_terminal_log_area_periodic_angle_refinement_decoder),
        pair_shared_terminal_periodic_angle_refinement_decoder=(
            pair_shared_terminal_periodic_angle_refinement_decoder),
        pair_shared_terminal_normalized_center_refinement_decoder=(
            pair_shared_terminal_normalized_center_refinement_decoder),
        pair_shared_terminal_full_tangent_refinement_decoder=(
            pair_shared_terminal_full_tangent_refinement_decoder),
        pair_shared_terminal_transport_center_tangent_refinement_decoder=(
            pair_shared_terminal_transport_center_tangent_refinement_decoder),
        pair_shared_terminal_transport_shape_tangent_refinement_decoder=(
            pair_shared_terminal_transport_shape_tangent_refinement_decoder),
        pair_shared_terminal_transport_product_tangent_refinement_decoder=(
            pair_shared_terminal_transport_product_tangent_refinement_decoder),
        pair_shared_terminal_transport_tangent_refinement_decoder=(
            pair_shared_terminal_transport_tangent_refinement_decoder),
        terminal_position_tangent_product_decoder=(
            terminal_position_tangent_product_decoder),
        terminal_position_tangent_transport_decoder=(
            terminal_position_tangent_transport_decoder),
        terminal_position_tangent_plane_decoder=(
            terminal_position_tangent_plane_decoder),
        pair_shared_progressive_log_shape_periodic_angle_refinement_decoder=(
            pair_shared_progressive_log_shape_periodic_angle_refinement_decoder),
        pair_shared_normalized_center_refinement_decoder=(
            pair_shared_normalized_center_refinement_decoder),
        frame_evidence_cls_decoder=frame_evidence_cls_decoder,
        frame_detail_cls_decoder=frame_detail_cls_decoder,
        shared_routing_decoder=shared_routing_decoder,
        shared_attention_decoder=shared_attention_decoder,
        antisymmetric_detail_decoder=antisymmetric_detail_decoder,
        enveloped_detail_decoder=enveloped_detail_decoder,
        regression_enveloped_detail_decoder=(
            regression_enveloped_detail_decoder),
        midpoint_regression_enveloped_detail_decoder=(
            midpoint_regression_enveloped_detail_decoder),
        classification_enveloped_detail_decoder=(
            classification_enveloped_detail_decoder),
        terminal_enveloped_detail_decoder=(
            terminal_enveloped_detail_decoder),
        terminal_midpoint_enveloped_detail_decoder=(
            terminal_midpoint_enveloped_detail_decoder),
        terminal_regression_enveloped_detail_decoder=(
            terminal_regression_enveloped_detail_decoder),
        terminal_midpoint_regression_enveloped_detail_decoder=(
            terminal_midpoint_regression_enveloped_detail_decoder),
        common_evidence_bypass_decoder=common_evidence_bypass_decoder,
        terminal_common_evidence_bypass_decoder=(
            terminal_common_evidence_bypass_decoder),
        terminal_classification_common_evidence_decoder=(
            terminal_classification_common_evidence_decoder),
        terminal_factorized_evidence_decoder=(
            terminal_factorized_evidence_decoder),
        terminal_factorized_confidence=terminal_factorized_confidence,
        terminal_factorized_diagonal_gates=(
            terminal_factorized_diagonal_gates),
        terminal_factorized_coupled_gate=(
            terminal_factorized_coupled_gate),
        terminal_factorized_center_motion_only=(
            terminal_factorized_center_motion_only),
        terminal_factorized_detail_only=terminal_factorized_detail_only,
    ).to(device)
    reg_branches_prev = _build_reg_branches(
        num_layers, embed_dims, device, seed=0)
    reg_branches_curr = _build_reg_branches(
        num_layers, embed_dims, device, seed=1)
    return decoder, reg_branches_prev, reg_branches_curr


def _random_memories(batch_size: int, num_value: int, embed_dims: int,
                     device: torch.device):
    torch.manual_seed(0)
    memory_prev = torch.randn(
        batch_size, num_value, embed_dims, device=device, requires_grad=True)
    memory_curr = torch.randn(
        batch_size, num_value, embed_dims, device=device, requires_grad=True)
    return memory_prev, memory_curr


class TestPairRotatedRTDETRDecoder(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        register_all_modules()
        cls.device = torch.device('cpu')

    def _forward(self, batch_size: int, decoder=None,
                 reg_branches_prev=None, reg_branches_curr=None, **kwargs):
        if decoder is None:
            decoder, reg_branches_prev, reg_branches_curr = _build_decoder(
                device=self.device)
        spatial_shapes, level_start_index, num_value = _spatial_meta(
            self.device)
        memory_prev, memory_curr = _random_memories(
            batch_size, num_value, decoder.embed_dims, self.device)
        hidden, refs_prev, refs_curr = decoder(
            memory_prev=memory_prev,
            memory_curr=memory_curr,
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index,
            reg_branches_prev=reg_branches_prev,
            reg_branches_curr=reg_branches_curr,
            **kwargs,
        )
        return hidden, refs_prev, refs_curr, memory_prev, memory_curr

    def test_pair_shared_shape_residual_preserves_dn_and_centers(self):
        prev = torch.tensor([[[1., 2., 3., 4., 5.],
                              [6., 7., 8., 9., 10.],
                              [11., 12., 13., 14., 15.]]])
        curr = torch.tensor([[[-1., -2., -3., -4., -5.],
                              [-6., -7., -8., -9., -10.],
                              [-11., -12., -13., -14., -15.]]])
        projected_prev, projected_curr = (
            PairRotatedRTDETRTransformerDecoder.
            _pair_shared_shape_residual(prev, curr, num_dn=1))

        self.assertTrue(torch.equal(projected_prev[:, :1], prev[:, :1]))
        self.assertTrue(torch.equal(projected_curr[:, :1], curr[:, :1]))
        self.assertTrue(torch.equal(
            projected_prev[:, 1:, :2], prev[:, 1:, :2]))
        self.assertTrue(torch.equal(
            projected_curr[:, 1:, :2], curr[:, 1:, :2]))
        expected_shape = 0.5 * (prev[:, 1:, 2:] + curr[:, 1:, 2:])
        self.assertTrue(torch.equal(
            projected_prev[:, 1:, 2:], expected_shape))
        self.assertTrue(torch.equal(
            projected_curr[:, 1:, 2:], expected_shape))

    def test_pair_shared_shape_residual_is_swap_equivariant_and_coupled(self):
        prev = torch.randn(2, 5, 5, requires_grad=True)
        curr = torch.randn(2, 5, 5, requires_grad=True)
        projected_prev, projected_curr = (
            PairRotatedRTDETRTransformerDecoder.
            _pair_shared_shape_residual(prev, curr, num_dn=0))
        swapped_curr, swapped_prev = (
            PairRotatedRTDETRTransformerDecoder.
            _pair_shared_shape_residual(curr, prev, num_dn=0))
        self.assertTrue(torch.equal(projected_prev, swapped_prev))
        self.assertTrue(torch.equal(projected_curr, swapped_curr))

        projected_prev[..., 2:].sum().backward()
        self.assertGreater(curr.grad[..., 2:].abs().sum().item(), 0.0)
        self.assertEqual(curr.grad[..., :2].abs().sum().item(), 0.0)

    def test_pair_shared_shape_refinement_adds_no_parameters_or_state(self):
        parent, _, _ = _build_decoder(num_layers=3, device=self.device)
        projected, reg_prev, reg_curr = _build_decoder(
            num_layers=3,
            device=self.device,
            pair_shared_shape_refinement_decoder=True)
        self.assertEqual(
            sum(parameter.numel() for parameter in parent.parameters()),
            sum(parameter.numel() for parameter in projected.parameters()))
        self.assertEqual(
            {key: tuple(value.shape)
             for key, value in parent.state_dict().items()},
            {key: tuple(value.shape)
             for key, value in projected.state_dict().items()})

        _, refs_prev, refs_curr, _, _ = self._forward(
            1,
            decoder=projected,
            reg_branches_prev=reg_prev,
            reg_branches_curr=reg_curr)
        self.assertEqual(len(refs_prev), 3)
        self.assertEqual(len(refs_curr), 3)
        for prev_ref, curr_ref in zip(refs_prev, refs_curr):
            prev_logit = torch.logit(prev_ref.clamp(1e-6, 1 - 1e-6))
            curr_logit = torch.logit(curr_ref.clamp(1e-6, 1 - 1e-6))
            self.assertTrue(torch.isfinite(prev_logit).all())
            self.assertTrue(torch.isfinite(curr_logit).all())

    def test_pair_shared_angle_residual_preserves_dn_and_non_angle_box(self):
        prev = torch.tensor([[[1., 2., 3., 4., 5.],
                              [6., 7., 8., 9., 10.],
                              [11., 12., 13., 14., 15.]]])
        curr = torch.tensor([[[-1., -2., -3., -4., -5.],
                              [-6., -7., -8., -9., -10.],
                              [-11., -12., -13., -14., -15.]]])
        projected_prev, projected_curr = (
            PairRotatedRTDETRTransformerDecoder.
            _pair_shared_angle_residual(prev, curr, num_dn=1))

        self.assertTrue(torch.equal(projected_prev[:, :1], prev[:, :1]))
        self.assertTrue(torch.equal(projected_curr[:, :1], curr[:, :1]))
        self.assertTrue(torch.equal(
            projected_prev[:, 1:, :4], prev[:, 1:, :4]))
        self.assertTrue(torch.equal(
            projected_curr[:, 1:, :4], curr[:, 1:, :4]))
        expected_angle = 0.5 * (prev[:, 1:, 4:] + curr[:, 1:, 4:])
        self.assertTrue(torch.equal(
            projected_prev[:, 1:, 4:], expected_angle))
        self.assertTrue(torch.equal(
            projected_curr[:, 1:, 4:], expected_angle))

    def test_pair_shared_angle_residual_is_swap_equivariant_and_local(self):
        prev = torch.randn(2, 5, 5, requires_grad=True)
        curr = torch.randn(2, 5, 5, requires_grad=True)
        projected_prev, projected_curr = (
            PairRotatedRTDETRTransformerDecoder.
            _pair_shared_angle_residual(prev, curr, num_dn=0))
        swapped_curr, swapped_prev = (
            PairRotatedRTDETRTransformerDecoder.
            _pair_shared_angle_residual(curr, prev, num_dn=0))
        self.assertTrue(torch.equal(projected_prev, swapped_prev))
        self.assertTrue(torch.equal(projected_curr, swapped_curr))

        projected_prev[..., 4:].sum().backward()
        self.assertGreater(curr.grad[..., 4:].abs().sum().item(), 0.0)
        self.assertEqual(curr.grad[..., :4].abs().sum().item(), 0.0)

    def test_pair_shared_angle_refinement_is_parameter_free_and_exclusive(self):
        parent, _, _ = _build_decoder(num_layers=3, device=self.device)
        projected, reg_prev, reg_curr = _build_decoder(
            num_layers=3,
            device=self.device,
            pair_shared_angle_refinement_decoder=True)
        self.assertEqual(
            sum(parameter.numel() for parameter in parent.parameters()),
            sum(parameter.numel() for parameter in projected.parameters()))
        self.assertEqual(
            {key: tuple(value.shape)
             for key, value in parent.state_dict().items()},
            {key: tuple(value.shape)
             for key, value in projected.state_dict().items()})

        _, refs_prev, refs_curr, _, _ = self._forward(
            1,
            decoder=projected,
            reg_branches_prev=reg_prev,
            reg_branches_curr=reg_curr)
        for reference in refs_prev + refs_curr:
            self.assertTrue(torch.isfinite(reference).all())

        with self.assertRaisesRegex(ValueError, 'mutually exclusive'):
            _build_decoder(
                device=self.device,
                pair_shared_shape_refinement_decoder=True,
                pair_shared_angle_refinement_decoder=True)

    def test_periodic_angle_residual_uses_wrapped_increment_midpoint(self):
        reference_prev = torch.full((1, 3, 5), 0.4)
        reference_curr = torch.full((1, 3, 5), 0.6)
        reference_prev[:, 1:, 4] = torch.tensor([0.90, 0.95])
        reference_curr[:, 1:, 4] = torch.tensor([0.10, 0.05])
        proposed_prev = torch.tensor([0.05, 0.05])
        proposed_curr = torch.tensor([0.30, 0.95])

        residual_prev = torch.randn(1, 3, 5)
        residual_curr = torch.randn(1, 3, 5)
        residual_prev[:, 1:, 4] = (
            torch.logit(proposed_prev)
            - torch.logit(reference_prev[:, 1:, 4]))
        residual_curr[:, 1:, 4] = (
            torch.logit(proposed_curr)
            - torch.logit(reference_curr[:, 1:, 4]))
        original_prev = residual_prev.clone()
        original_curr = residual_curr.clone()

        projected_prev, projected_curr = (
            PairRotatedRTDETRTransformerDecoder.
            _pair_shared_periodic_angle_residual(
                residual_prev, residual_curr,
                reference_prev, reference_curr, num_dn=1))
        decoded_prev = (
            projected_prev + torch.logit(reference_prev)).sigmoid()
        decoded_curr = (
            projected_curr + torch.logit(reference_curr)).sigmoid()

        self.assertTrue(torch.equal(
            projected_prev[:, :1], original_prev[:, :1]))
        self.assertTrue(torch.equal(
            projected_curr[:, :1], original_curr[:, :1]))
        self.assertTrue(torch.equal(
            projected_prev[:, 1:, :4], original_prev[:, 1:, :4]))
        self.assertTrue(torch.equal(
            projected_curr[:, 1:, :4], original_curr[:, 1:, :4]))
        self.assertTrue(torch.allclose(
            decoded_prev[:, 1:, 4], torch.tensor([[0.075, 0.95]]),
            atol=1e-5, rtol=0.0))
        self.assertTrue(torch.allclose(
            decoded_curr[:, 1:, 4], torch.tensor([[0.275, 0.05]]),
            atol=1e-5, rtol=0.0))

    def test_periodic_angle_residual_is_swap_equivariant_and_local(self):
        reference_prev = torch.rand(2, 5, 5) * 0.8 + 0.1
        reference_curr = torch.rand(2, 5, 5) * 0.8 + 0.1
        prev = torch.randn(2, 5, 5, requires_grad=True)
        curr = torch.randn(2, 5, 5, requires_grad=True)
        projected_prev, projected_curr = (
            PairRotatedRTDETRTransformerDecoder.
            _pair_shared_periodic_angle_residual(
                prev, curr, reference_prev, reference_curr, num_dn=0))
        swapped_curr, swapped_prev = (
            PairRotatedRTDETRTransformerDecoder.
            _pair_shared_periodic_angle_residual(
                curr, prev, reference_curr, reference_prev, num_dn=0))
        self.assertTrue(torch.allclose(projected_prev, swapped_prev))
        self.assertTrue(torch.allclose(projected_curr, swapped_curr))

        projected_prev[..., 4:].sum().backward()
        self.assertGreater(curr.grad[..., 4:].abs().sum().item(), 0.0)
        self.assertEqual(curr.grad[..., :4].abs().sum().item(), 0.0)

    def test_periodic_angle_refinement_is_parameter_free_and_exclusive(self):
        parent, _, _ = _build_decoder(num_layers=3, device=self.device)
        projected, reg_prev, reg_curr = _build_decoder(
            num_layers=3,
            device=self.device,
            pair_shared_periodic_angle_refinement_decoder=True)
        self.assertEqual(
            sum(parameter.numel() for parameter in parent.parameters()),
            sum(parameter.numel() for parameter in projected.parameters()))
        self.assertEqual(
            {key: tuple(value.shape)
             for key, value in parent.state_dict().items()},
            {key: tuple(value.shape)
             for key, value in projected.state_dict().items()})

        _, refs_prev, refs_curr, _, _ = self._forward(
            1,
            decoder=projected,
            reg_branches_prev=reg_prev,
            reg_branches_curr=reg_curr)
        for reference in refs_prev + refs_curr:
            self.assertTrue(torch.isfinite(reference).all())

        with self.assertRaisesRegex(ValueError, 'mutually exclusive'):
            _build_decoder(
                device=self.device,
                pair_shared_angle_refinement_decoder=True,
                pair_shared_periodic_angle_refinement_decoder=True)

    def test_log_size_residual_shares_multiplicative_increment(self):
        reference_prev = torch.rand(2, 4, 5) * 0.6 + 0.2
        reference_curr = torch.rand(2, 4, 5) * 0.6 + 0.2
        # Keep normal queries away from the unavoidable sigmoid-size ceiling;
        # the geometric invariant is exact before boundary clipping.
        prev = 0.2 * torch.randn(2, 4, 5)
        curr = 0.2 * torch.randn(2, 4, 5)
        original_prev = prev.clone()
        original_curr = curr.clone()

        projected_prev, projected_curr = (
            PairRotatedRTDETRTransformerDecoder.
            _pair_shared_log_size_residual(
                prev, curr, reference_prev, reference_curr, num_dn=1))
        decoded_prev = (
            projected_prev + torch.logit(reference_prev)).sigmoid()
        decoded_curr = (
            projected_curr + torch.logit(reference_curr)).sigmoid()
        log_ratio_prev = torch.log(
            decoded_prev[:, 1:, 2:4] / reference_prev[:, 1:, 2:4])
        log_ratio_curr = torch.log(
            decoded_curr[:, 1:, 2:4] / reference_curr[:, 1:, 2:4])

        self.assertTrue(torch.equal(
            projected_prev[:, :1], original_prev[:, :1]))
        self.assertTrue(torch.equal(
            projected_curr[:, :1], original_curr[:, :1]))
        self.assertTrue(torch.equal(
            projected_prev[:, 1:, :2], original_prev[:, 1:, :2]))
        self.assertTrue(torch.equal(
            projected_curr[:, 1:, :2], original_curr[:, 1:, :2]))
        self.assertTrue(torch.equal(
            projected_prev[:, 1:, 4:], original_prev[:, 1:, 4:]))
        self.assertTrue(torch.equal(
            projected_curr[:, 1:, 4:], original_curr[:, 1:, 4:]))
        torch.testing.assert_close(
            log_ratio_prev, log_ratio_curr, atol=2e-5, rtol=2e-5)

    def test_log_size_periodic_angle_is_parameter_free_and_exclusive(self):
        parent, _, _ = _build_decoder(num_layers=3, device=self.device)
        projected, reg_prev, reg_curr = _build_decoder(
            num_layers=3,
            device=self.device,
            pair_shared_log_size_periodic_angle_refinement_decoder=True)
        self.assertEqual(
            sum(parameter.numel() for parameter in parent.parameters()),
            sum(parameter.numel() for parameter in projected.parameters()))
        self.assertEqual(
            {key: tuple(value.shape)
             for key, value in parent.state_dict().items()},
            {key: tuple(value.shape)
             for key, value in projected.state_dict().items()})

        _, refs_prev, refs_curr, _, _ = self._forward(
            1,
            decoder=projected,
            reg_branches_prev=reg_prev,
            reg_branches_curr=reg_curr)
        for reference in refs_prev + refs_curr:
            self.assertTrue(torch.isfinite(reference).all())

        with self.assertRaisesRegex(ValueError, 'mutually exclusive'):
            _build_decoder(
                pair_shared_periodic_angle_refinement_decoder=True,
                pair_shared_log_size_periodic_angle_refinement_decoder=True,
                device=self.device)

    def test_log_area_residual_preserves_aspect_increment(self):
        reference_prev = torch.rand(2, 4, 5) * 0.5 + 0.2
        reference_curr = torch.rand(2, 4, 5) * 0.5 + 0.2
        prev = 0.15 * torch.randn(2, 4, 5)
        curr = 0.15 * torch.randn(2, 4, 5)
        original_prev = prev.clone()
        original_curr = curr.clone()
        proposed_prev = (prev + torch.logit(reference_prev)).sigmoid()
        proposed_curr = (curr + torch.logit(reference_curr)).sigmoid()
        original_delta_prev = torch.log(
            proposed_prev[:, 1:, 2:4] / reference_prev[:, 1:, 2:4])
        original_delta_curr = torch.log(
            proposed_curr[:, 1:, 2:4] / reference_curr[:, 1:, 2:4])

        projected_prev, projected_curr = (
            PairRotatedRTDETRTransformerDecoder.
            _pair_shared_log_area_residual(
                prev, curr, reference_prev, reference_curr, num_dn=1))
        decoded_prev = (
            projected_prev + torch.logit(reference_prev)).sigmoid()
        decoded_curr = (
            projected_curr + torch.logit(reference_curr)).sigmoid()
        delta_prev = torch.log(
            decoded_prev[:, 1:, 2:4] / reference_prev[:, 1:, 2:4])
        delta_curr = torch.log(
            decoded_curr[:, 1:, 2:4] / reference_curr[:, 1:, 2:4])
        area_prev = 0.5 * delta_prev.sum(dim=-1)
        area_curr = 0.5 * delta_curr.sum(dim=-1)
        aspect_prev = 0.5 * (delta_prev[..., 0] - delta_prev[..., 1])
        aspect_curr = 0.5 * (delta_curr[..., 0] - delta_curr[..., 1])
        original_aspect_prev = 0.5 * (
            original_delta_prev[..., 0] - original_delta_prev[..., 1])
        original_aspect_curr = 0.5 * (
            original_delta_curr[..., 0] - original_delta_curr[..., 1])

        self.assertTrue(torch.equal(
            projected_prev[:, :1], original_prev[:, :1]))
        self.assertTrue(torch.equal(
            projected_curr[:, :1], original_curr[:, :1]))
        torch.testing.assert_close(area_prev, area_curr, atol=2e-5, rtol=2e-5)
        torch.testing.assert_close(
            aspect_prev, original_aspect_prev, atol=2e-5, rtol=2e-5)
        torch.testing.assert_close(
            aspect_curr, original_aspect_curr, atol=2e-5, rtol=2e-5)

    def test_log_area_periodic_angle_is_parameter_free_and_exclusive(self):
        parent, _, _ = _build_decoder(num_layers=3, device=self.device)
        projected, reg_prev, reg_curr = _build_decoder(
            num_layers=3,
            device=self.device,
            pair_shared_log_area_periodic_angle_refinement_decoder=True)
        self.assertEqual(
            sum(parameter.numel() for parameter in parent.parameters()),
            sum(parameter.numel() for parameter in projected.parameters()))
        self.assertEqual(
            {key: tuple(value.shape)
             for key, value in parent.state_dict().items()},
            {key: tuple(value.shape)
             for key, value in projected.state_dict().items()})
        _, refs_prev, refs_curr, _, _ = self._forward(
            1, decoder=projected,
            reg_branches_prev=reg_prev, reg_branches_curr=reg_curr)
        for reference in refs_prev + refs_curr:
            self.assertTrue(torch.isfinite(reference).all())
        with self.assertRaisesRegex(ValueError, 'mutually exclusive'):
            _build_decoder(
                pair_shared_log_size_periodic_angle_refinement_decoder=True,
                pair_shared_log_area_periodic_angle_refinement_decoder=True,
                device=self.device)

    def test_late_log_size_periodic_angle_only_projects_last_two_layers(self):
        parent, _, _ = _build_decoder(num_layers=4, device=self.device)
        projected, reg_prev, reg_curr = _build_decoder(
            num_layers=4,
            device=self.device,
            pair_shared_late_log_size_periodic_angle_refinement_decoder=True)
        self.assertEqual(
            sum(parameter.numel() for parameter in parent.parameters()),
            sum(parameter.numel() for parameter in projected.parameters()))
        self.assertEqual(
            {key: tuple(value.shape)
             for key, value in parent.state_dict().items()},
            {key: tuple(value.shape)
             for key, value in projected.state_dict().items()})

        log_size = projected._pair_shared_log_size_residual
        periodic_angle = projected._pair_shared_periodic_angle_residual
        with mock.patch.object(
                projected, '_pair_shared_log_size_residual',
                side_effect=log_size) as log_size_mock, mock.patch.object(
                    projected, '_pair_shared_periodic_angle_residual',
                    side_effect=periodic_angle) as angle_mock:
            _, refs_prev, refs_curr, _, _ = self._forward(
                1, decoder=projected,
                reg_branches_prev=reg_prev, reg_branches_curr=reg_curr)
        self.assertEqual(log_size_mock.call_count, 2)
        self.assertEqual(angle_mock.call_count, 2)
        for reference in refs_prev + refs_curr:
            self.assertTrue(torch.isfinite(reference).all())
        with self.assertRaisesRegex(ValueError, 'mutually exclusive'):
            _build_decoder(
                pair_shared_log_size_periodic_angle_refinement_decoder=True,
                pair_shared_late_log_size_periodic_angle_refinement_decoder=(
                    True),
                device=self.device)

    def test_progressive_log_shape_uses_area_then_full_size(self):
        parent, _, _ = _build_decoder(num_layers=4, device=self.device)
        projected, reg_prev, reg_curr = _build_decoder(
            num_layers=4,
            device=self.device,
            pair_shared_progressive_log_shape_periodic_angle_refinement_decoder=(
                True))
        self.assertEqual(
            sum(parameter.numel() for parameter in parent.parameters()),
            sum(parameter.numel() for parameter in projected.parameters()))
        self.assertEqual(
            {key: tuple(value.shape)
             for key, value in parent.state_dict().items()},
            {key: tuple(value.shape)
             for key, value in projected.state_dict().items()})

        log_area = projected._pair_shared_log_area_residual
        log_size = projected._pair_shared_log_size_residual
        periodic_angle = projected._pair_shared_periodic_angle_residual
        with mock.patch.object(
                projected, '_pair_shared_log_area_residual',
                side_effect=log_area) as area_mock, mock.patch.object(
                    projected, '_pair_shared_log_size_residual',
                    side_effect=log_size) as size_mock, mock.patch.object(
                        projected, '_pair_shared_periodic_angle_residual',
                        side_effect=periodic_angle) as angle_mock:
            _, refs_prev, refs_curr, _, _ = self._forward(
                1, decoder=projected,
                reg_branches_prev=reg_prev, reg_branches_curr=reg_curr)
        self.assertEqual(area_mock.call_count, 1)
        self.assertEqual(size_mock.call_count, 1)
        self.assertEqual(angle_mock.call_count, 2)
        for reference in refs_prev + refs_curr:
            self.assertTrue(torch.isfinite(reference).all())
        with self.assertRaisesRegex(ValueError, 'mutually exclusive'):
            _build_decoder(
                pair_shared_late_log_size_periodic_angle_refinement_decoder=(
                    True),
                pair_shared_progressive_log_shape_periodic_angle_refinement_decoder=(
                    True),
                device=self.device)

    def test_terminal_log_size_periodic_angle_only_projects_final_layer(self):
        parent, _, _ = _build_decoder(num_layers=4, device=self.device)
        projected, reg_prev, reg_curr = _build_decoder(
            num_layers=4,
            device=self.device,
            pair_shared_terminal_log_size_periodic_angle_refinement_decoder=(
                True))
        self.assertEqual(
            sum(parameter.numel() for parameter in parent.parameters()),
            sum(parameter.numel() for parameter in projected.parameters()))
        self.assertEqual(
            {key: tuple(value.shape)
             for key, value in parent.state_dict().items()},
            {key: tuple(value.shape)
             for key, value in projected.state_dict().items()})

        log_size = projected._pair_shared_log_size_residual
        periodic_angle = projected._pair_shared_periodic_angle_residual
        with mock.patch.object(
                projected, '_pair_shared_log_size_residual',
                side_effect=log_size) as log_size_mock, mock.patch.object(
                    projected, '_pair_shared_periodic_angle_residual',
                    side_effect=periodic_angle) as angle_mock:
            _, refs_prev, refs_curr, _, _ = self._forward(
                1, decoder=projected,
                reg_branches_prev=reg_prev, reg_branches_curr=reg_curr)
        self.assertEqual(log_size_mock.call_count, 1)
        self.assertEqual(angle_mock.call_count, 1)
        for reference in refs_prev + refs_curr:
            self.assertTrue(torch.isfinite(reference).all())
        with self.assertRaisesRegex(ValueError, 'mutually exclusive'):
            _build_decoder(
                pair_shared_late_log_size_periodic_angle_refinement_decoder=(
                    True),
                pair_shared_terminal_log_size_periodic_angle_refinement_decoder=(
                    True),
                device=self.device)

    def test_terminal_log_area_periodic_angle_only_projects_final_layer(self):
        parent, _, _ = _build_decoder(num_layers=4, device=self.device)
        projected, reg_prev, reg_curr = _build_decoder(
            num_layers=4,
            device=self.device,
            pair_shared_terminal_log_area_periodic_angle_refinement_decoder=(
                True))
        self.assertEqual(
            sum(parameter.numel() for parameter in parent.parameters()),
            sum(parameter.numel() for parameter in projected.parameters()))
        self.assertEqual(
            {key: tuple(value.shape)
             for key, value in parent.state_dict().items()},
            {key: tuple(value.shape)
             for key, value in projected.state_dict().items()})

        log_area = projected._pair_shared_log_area_residual
        periodic_angle = projected._pair_shared_periodic_angle_residual
        with mock.patch.object(
                projected, '_pair_shared_log_area_residual',
                side_effect=log_area) as area_mock, mock.patch.object(
                    projected, '_pair_shared_periodic_angle_residual',
                    side_effect=periodic_angle) as angle_mock:
            _, refs_prev, refs_curr, _, _ = self._forward(
                1, decoder=projected,
                reg_branches_prev=reg_prev, reg_branches_curr=reg_curr)
        self.assertEqual(area_mock.call_count, 1)
        self.assertEqual(angle_mock.call_count, 1)
        for reference in refs_prev + refs_curr:
            self.assertTrue(torch.isfinite(reference).all())
        with self.assertRaisesRegex(ValueError, 'mutually exclusive'):
            _build_decoder(
                pair_shared_terminal_log_size_periodic_angle_refinement_decoder=(
                    True),
                pair_shared_terminal_log_area_periodic_angle_refinement_decoder=(
                    True),
                device=self.device)

    def test_terminal_periodic_angle_only_projects_final_layer(self):
        parent, _, _ = _build_decoder(num_layers=4, device=self.device)
        projected, reg_prev, reg_curr = _build_decoder(
            num_layers=4,
            device=self.device,
            pair_shared_terminal_periodic_angle_refinement_decoder=True)
        self.assertEqual(
            sum(parameter.numel() for parameter in parent.parameters()),
            sum(parameter.numel() for parameter in projected.parameters()))
        self.assertEqual(
            {key: tuple(value.shape)
             for key, value in parent.state_dict().items()},
            {key: tuple(value.shape)
             for key, value in projected.state_dict().items()})

        periodic_angle = projected._pair_shared_periodic_angle_residual
        with mock.patch.object(
                projected, '_pair_shared_periodic_angle_residual',
                side_effect=periodic_angle) as angle_mock:
            _, refs_prev, refs_curr, _, _ = self._forward(
                1, decoder=projected,
                reg_branches_prev=reg_prev, reg_branches_curr=reg_curr)
        self.assertEqual(angle_mock.call_count, 1)
        for reference in refs_prev + refs_curr:
            self.assertTrue(torch.isfinite(reference).all())
        with self.assertRaisesRegex(ValueError, 'mutually exclusive'):
            _build_decoder(
                pair_shared_terminal_log_area_periodic_angle_refinement_decoder=(
                    True),
                pair_shared_terminal_periodic_angle_refinement_decoder=True,
                device=self.device)

    def test_terminal_normalized_center_only_projects_final_layer(self):
        parent, _, _ = _build_decoder(num_layers=4, device=self.device)
        projected, reg_prev, reg_curr = _build_decoder(
            num_layers=4,
            device=self.device,
            pair_shared_terminal_normalized_center_refinement_decoder=True)
        self.assertEqual(
            sum(parameter.numel() for parameter in parent.parameters()),
            sum(parameter.numel() for parameter in projected.parameters()))
        self.assertEqual(
            {key: tuple(value.shape)
             for key, value in parent.state_dict().items()},
            {key: tuple(value.shape)
             for key, value in projected.state_dict().items()})

        normalized_center = (
            projected._pair_shared_normalized_center_residual)
        with mock.patch.object(
                projected, '_pair_shared_normalized_center_residual',
                side_effect=normalized_center) as center_mock:
            _, refs_prev, refs_curr, _, _ = self._forward(
                1, decoder=projected,
                reg_branches_prev=reg_prev, reg_branches_curr=reg_curr)
        self.assertEqual(center_mock.call_count, 1)
        for reference in refs_prev + refs_curr:
            self.assertTrue(torch.isfinite(reference).all())
        with self.assertRaisesRegex(ValueError, 'mutually exclusive'):
            _build_decoder(
                pair_shared_terminal_periodic_angle_refinement_decoder=True,
                pair_shared_terminal_normalized_center_refinement_decoder=(
                    True),
                device=self.device)

    def test_terminal_full_tangent_only_projects_final_layer(self):
        parent, _, _ = _build_decoder(num_layers=4, device=self.device)
        projected, reg_prev, reg_curr = _build_decoder(
            num_layers=4,
            device=self.device,
            pair_shared_terminal_full_tangent_refinement_decoder=True)
        self.assertEqual(
            sum(parameter.numel() for parameter in parent.parameters()),
            sum(parameter.numel() for parameter in projected.parameters()))
        self.assertEqual(
            {key: tuple(value.shape)
             for key, value in parent.state_dict().items()},
            {key: tuple(value.shape)
             for key, value in projected.state_dict().items()})

        normalized_center = (
            projected._pair_shared_normalized_center_residual)
        log_size = projected._pair_shared_log_size_residual
        periodic_angle = projected._pair_shared_periodic_angle_residual
        with mock.patch.object(
                projected, '_pair_shared_normalized_center_residual',
                side_effect=normalized_center) as center_mock, mock.patch.object(
                    projected, '_pair_shared_log_size_residual',
                    side_effect=log_size) as size_mock, mock.patch.object(
                        projected, '_pair_shared_periodic_angle_residual',
                        side_effect=periodic_angle) as angle_mock:
            _, refs_prev, refs_curr, _, _ = self._forward(
                1, decoder=projected,
                reg_branches_prev=reg_prev, reg_branches_curr=reg_curr)
        self.assertEqual(center_mock.call_count, 1)
        self.assertEqual(size_mock.call_count, 1)
        self.assertEqual(angle_mock.call_count, 1)
        for reference in refs_prev + refs_curr:
            self.assertTrue(torch.isfinite(reference).all())
        with self.assertRaisesRegex(ValueError, 'mutually exclusive'):
            _build_decoder(
                pair_shared_terminal_log_size_periodic_angle_refinement_decoder=(
                    True),
                pair_shared_terminal_full_tangent_refinement_decoder=True,
                device=self.device)

    def test_transport_tangent_preserves_only_established_detail(self):
        reference_prev = torch.tensor([[[
            0.40, 0.40, 0.20, 0.30, 0.45]]], device=self.device)
        reference_curr = torch.tensor([[[
            0.44, 0.40, 0.20, 0.30, 0.45]]], device=self.device)

        # Construct terminal updates in natural tangent coordinates. The
        # pair's established transform is pure horizontal translation, so
        # only horizontal frame detail may survive the projection.
        tangent_prev = torch.tensor([[[
            0.10, -0.20, 0.05, -0.04, 0.03]]], device=self.device)
        tangent_curr = torch.tensor([[[
            0.30, 0.40, -0.07, 0.08, -0.01]]], device=self.device)

        def encode(tangent, reference):
            reference_logit = torch.logit(
                reference.clamp(1e-3, 1 - 1e-3))
            target = torch.cat((
                reference[..., :2]
                + tangent[..., :2] * reference[..., 2:4],
                reference[..., 2:4] * torch.exp(tangent[..., 2:4]),
                torch.remainder(
                    reference[..., 4:] + tangent[..., 4:], 1.0)), dim=-1)
            return torch.logit(
                target.clamp(1e-3, 1 - 1e-3)) - reference_logit

        residual_prev = encode(tangent_prev, reference_prev)
        residual_curr = encode(tangent_curr, reference_curr)
        projected_prev, projected_curr = (
            PairRotatedRTDETRTransformerDecoder.
            _pair_transport_full_tangent_residual(
                residual_prev, residual_curr,
                reference_prev, reference_curr, 0))

        def decode(residual, reference):
            proposed = (
                residual
                + torch.logit(reference.clamp(1e-3, 1 - 1e-3))).sigmoid()
            return torch.cat((
                (proposed[..., :2] - reference[..., :2])
                / reference[..., 2:4],
                torch.log(proposed[..., 2:4] / reference[..., 2:4]),
                torch.remainder(
                    proposed[..., 4:] - reference[..., 4:] + 0.5,
                    1.0) - 0.5), dim=-1)

        output_prev = decode(projected_prev, reference_prev)
        output_curr = decode(projected_curr, reference_curr)
        output_common = 0.5 * (output_prev + output_curr)
        output_detail = 0.5 * (output_curr - output_prev)
        expected_common = 0.5 * (tangent_prev + tangent_curr)
        expected_detail = torch.zeros_like(output_detail)
        expected_detail[..., :1] = 0.5 * (
            tangent_curr[..., :1] - tangent_prev[..., :1])
        self.assertTrue(torch.allclose(
            output_common, expected_common, atol=1e-5, rtol=1e-5))
        self.assertTrue(torch.allclose(
            output_detail, expected_detail, atol=1e-5, rtol=1e-5))

    def test_transport_center_tangent_only_projects_final_layer(self):
        parent, _, _ = _build_decoder(num_layers=4, device=self.device)
        projected, reg_prev, reg_curr = _build_decoder(
            num_layers=4,
            device=self.device,
            pair_shared_terminal_transport_center_tangent_refinement_decoder=(
                True))
        self.assertEqual(
            sum(parameter.numel() for parameter in parent.parameters()),
            sum(parameter.numel() for parameter in projected.parameters()))
        self.assertEqual(
            {key: tuple(value.shape)
             for key, value in parent.state_dict().items()},
            {key: tuple(value.shape)
             for key, value in projected.state_dict().items()})

        projection = projected._pair_transport_center_tangent_residual
        with mock.patch.object(
                projected, '_pair_transport_center_tangent_residual',
                side_effect=projection) as projection_mock:
            _, refs_prev, refs_curr, _, _ = self._forward(
                1, decoder=projected,
                reg_branches_prev=reg_prev, reg_branches_curr=reg_curr)
        self.assertEqual(projection_mock.call_count, 1)
        for reference in refs_prev + refs_curr:
            self.assertTrue(torch.isfinite(reference).all())
        with self.assertRaisesRegex(ValueError, 'mutually exclusive'):
            _build_decoder(
                pair_shared_terminal_log_size_periodic_angle_refinement_decoder=(
                    True),
                pair_shared_terminal_transport_center_tangent_refinement_decoder=(
                    True),
                device=self.device)

    def test_transport_center_tangent_preserves_shape_and_motion_detail(self):
        reference_prev = torch.tensor([[[
            0.40, 0.40, 0.20, 0.30, 0.45]]], device=self.device)
        reference_curr = torch.tensor([[[
            0.44, 0.40, 0.20, 0.30, 0.45]]], device=self.device)
        tangent_prev = torch.tensor([[[0.10, -0.20]]], device=self.device)
        tangent_curr = torch.tensor([[[0.30, 0.40]]], device=self.device)
        shape_prev = torch.tensor([[[0.2, -0.3, 0.1]]], device=self.device)
        shape_curr = torch.tensor([[[-0.4, 0.5, -0.2]]], device=self.device)

        def encode(tangent, shape, reference):
            reference_logit = torch.logit(
                reference.clamp(1e-3, 1 - 1e-3))
            target_center = (
                reference[..., :2]
                + tangent * reference[..., 2:4]).clamp(1e-3, 1 - 1e-3)
            center_residual = (
                torch.logit(target_center)
                - reference_logit[..., :2])
            return torch.cat((center_residual, shape), dim=-1)

        residual_prev = encode(tangent_prev, shape_prev, reference_prev)
        residual_curr = encode(tangent_curr, shape_curr, reference_curr)
        projected_prev, projected_curr = (
            PairRotatedRTDETRTransformerDecoder.
            _pair_transport_center_tangent_residual(
                residual_prev, residual_curr,
                reference_prev, reference_curr, 0))
        self.assertTrue(torch.equal(projected_prev[..., 2:], shape_prev))
        self.assertTrue(torch.equal(projected_curr[..., 2:], shape_curr))

        def decode(residual, reference):
            proposed_center = (
                residual[..., :2]
                + torch.logit(
                    reference[..., :2].clamp(1e-3, 1 - 1e-3))).sigmoid()
            return (
                proposed_center - reference[..., :2]
            ) / reference[..., 2:4]

        output_prev = decode(projected_prev, reference_prev)
        output_curr = decode(projected_curr, reference_curr)
        output_common = 0.5 * (output_prev + output_curr)
        output_detail = 0.5 * (output_curr - output_prev)
        expected_common = 0.5 * (tangent_prev + tangent_curr)
        expected_detail = torch.zeros_like(output_detail)
        expected_detail[..., :1] = 0.5 * (
            tangent_curr[..., :1] - tangent_prev[..., :1])
        self.assertTrue(torch.allclose(
            output_common, expected_common, atol=1e-5, rtol=1e-5))
        self.assertTrue(torch.allclose(
            output_detail, expected_detail, atol=1e-5, rtol=1e-5))

    def test_transport_center_tangent_is_swap_equivariant_and_preserves_dn(
            self):
        torch.manual_seed(25)
        reference_prev = torch.rand(2, 5, 5, device=self.device) * 0.6 + 0.2
        reference_curr = torch.rand(2, 5, 5, device=self.device) * 0.6 + 0.2
        residual_prev = (
            torch.randn(2, 5, 5, device=self.device) * 0.2).requires_grad_()
        residual_curr = (
            torch.randn(2, 5, 5, device=self.device) * 0.2).requires_grad_()
        projected = (
            PairRotatedRTDETRTransformerDecoder.
            _pair_transport_center_tangent_residual(
                residual_prev, residual_curr,
                reference_prev, reference_curr, 2))
        swapped = (
            PairRotatedRTDETRTransformerDecoder.
            _pair_transport_center_tangent_residual(
                residual_curr, residual_prev,
                reference_curr, reference_prev, 2))
        self.assertTrue(torch.equal(projected[0][:, :2], residual_prev[:, :2]))
        self.assertTrue(torch.equal(projected[1][:, :2], residual_curr[:, :2]))
        self.assertTrue(torch.equal(
            projected[0][:, 2:, 2:], residual_prev[:, 2:, 2:]))
        self.assertTrue(torch.equal(
            projected[1][:, 2:, 2:], residual_curr[:, 2:, 2:]))
        self.assertTrue(torch.allclose(
            projected[0], swapped[1], atol=1e-6, rtol=1e-5))
        self.assertTrue(torch.allclose(
            projected[1], swapped[0], atol=1e-6, rtol=1e-5))
        (projected[0].sum() + projected[1].sum()).backward()
        self.assertTrue(torch.isfinite(projected[0]).all())
        self.assertTrue(torch.isfinite(projected[1]).all())
        self.assertTrue(torch.isfinite(residual_prev.grad).all())
        self.assertTrue(torch.isfinite(residual_curr.grad).all())

    def test_transport_shape_tangent_only_projects_final_layer(self):
        parent, _, _ = _build_decoder(num_layers=4, device=self.device)
        projected, reg_prev, reg_curr = _build_decoder(
            num_layers=4,
            device=self.device,
            pair_shared_terminal_transport_shape_tangent_refinement_decoder=(
                True))
        self.assertEqual(
            sum(parameter.numel() for parameter in parent.parameters()),
            sum(parameter.numel() for parameter in projected.parameters()))
        self.assertEqual(
            {key: tuple(value.shape)
             for key, value in parent.state_dict().items()},
            {key: tuple(value.shape)
             for key, value in projected.state_dict().items()})

        projection = projected._pair_transport_shape_tangent_residual
        with mock.patch.object(
                projected, '_pair_transport_shape_tangent_residual',
                side_effect=projection) as projection_mock:
            _, refs_prev, refs_curr, _, _ = self._forward(
                1, decoder=projected,
                reg_branches_prev=reg_prev, reg_branches_curr=reg_curr)
        self.assertEqual(projection_mock.call_count, 1)
        for reference in refs_prev + refs_curr:
            self.assertTrue(torch.isfinite(reference).all())
        with self.assertRaisesRegex(ValueError, 'mutually exclusive'):
            _build_decoder(
                pair_shared_terminal_log_size_periodic_angle_refinement_decoder=(
                    True),
                pair_shared_terminal_transport_shape_tangent_refinement_decoder=(
                    True),
                device=self.device)

    def test_transport_shape_tangent_preserves_centers_and_established_detail(
            self):
        reference_prev = torch.tensor([[[
            0.40, 0.40, 0.20, 0.30, 0.45]]], device=self.device)
        reference_curr = torch.tensor([[[
            0.44, 0.40, 0.40, 0.30, 0.45]]], device=self.device)
        tangent_prev = torch.tensor([[[
            0.10, -0.20, 0.03]]], device=self.device)
        tangent_curr = torch.tensor([[[
            0.30, 0.40, -0.01]]], device=self.device)
        center_prev = torch.tensor([[[0.7, -0.4]]], device=self.device)
        center_curr = torch.tensor([[[-0.2, 0.5]]], device=self.device)

        def encode(tangent, center, reference):
            reference_logit = torch.logit(
                reference.clamp(1e-3, 1 - 1e-3))
            target_shape = torch.cat((
                reference[..., 2:4] * torch.exp(tangent[..., :2]),
                torch.remainder(
                    reference[..., 4:] + tangent[..., 2:], 1.0)), dim=-1)
            shape_residual = (
                torch.logit(target_shape.clamp(1e-3, 1 - 1e-3))
                - reference_logit[..., 2:])
            return torch.cat((center, shape_residual), dim=-1)

        residual_prev = encode(tangent_prev, center_prev, reference_prev)
        residual_curr = encode(tangent_curr, center_curr, reference_curr)
        projected_prev, projected_curr = (
            PairRotatedRTDETRTransformerDecoder.
            _pair_transport_shape_tangent_residual(
                residual_prev, residual_curr,
                reference_prev, reference_curr, 0))
        self.assertTrue(torch.equal(projected_prev[..., :2], center_prev))
        self.assertTrue(torch.equal(projected_curr[..., :2], center_curr))

        def decode(residual, reference):
            proposed = (
                residual[..., 2:]
                + torch.logit(
                    reference[..., 2:].clamp(1e-3, 1 - 1e-3))).sigmoid()
            return torch.cat((
                torch.log(proposed[..., :2] / reference[..., 2:4]),
                torch.remainder(
                    proposed[..., 2:] - reference[..., 4:] + 0.5,
                    1.0) - 0.5), dim=-1)

        output_prev = decode(projected_prev, reference_prev)
        output_curr = decode(projected_curr, reference_curr)
        output_common = 0.5 * (output_prev + output_curr)
        output_detail = 0.5 * (output_curr - output_prev)
        expected_common = 0.5 * (tangent_prev + tangent_curr)
        expected_detail = torch.zeros_like(output_detail)
        expected_detail[..., :1] = 0.5 * (
            tangent_curr[..., :1] - tangent_prev[..., :1])
        self.assertTrue(torch.allclose(
            output_common, expected_common, atol=1e-5, rtol=1e-5))
        self.assertTrue(torch.allclose(
            output_detail, expected_detail, atol=1e-5, rtol=1e-5))

    def test_transport_shape_tangent_is_swap_equivariant_and_preserves_dn(
            self):
        torch.manual_seed(24)
        reference_prev = torch.rand(2, 5, 5, device=self.device) * 0.6 + 0.2
        reference_curr = torch.rand(2, 5, 5, device=self.device) * 0.6 + 0.2
        residual_prev = torch.randn(2, 5, 5, device=self.device) * 0.2
        residual_curr = torch.randn(2, 5, 5, device=self.device) * 0.2
        projected = (
            PairRotatedRTDETRTransformerDecoder.
            _pair_transport_shape_tangent_residual(
                residual_prev, residual_curr,
                reference_prev, reference_curr, 2))
        swapped = (
            PairRotatedRTDETRTransformerDecoder.
            _pair_transport_shape_tangent_residual(
                residual_curr, residual_prev,
                reference_curr, reference_prev, 2))
        self.assertTrue(torch.equal(projected[0][:, :2], residual_prev[:, :2]))
        self.assertTrue(torch.equal(projected[1][:, :2], residual_curr[:, :2]))
        self.assertTrue(torch.equal(
            projected[0][:, 2:, :2], residual_prev[:, 2:, :2]))
        self.assertTrue(torch.equal(
            projected[1][:, 2:, :2], residual_curr[:, 2:, :2]))
        self.assertTrue(torch.allclose(
            projected[0], swapped[1], atol=1e-6, rtol=1e-5))
        self.assertTrue(torch.allclose(
            projected[1], swapped[0], atol=1e-6, rtol=1e-5))

    def test_transport_shape_tangent_tiny_references_have_finite_gradients(
            self):
        reference_prev = torch.tensor([[[
            0.5000, 0.5000, 1.0e-4, 1.0e-4, 0.5000]]],
            device=self.device)
        reference_curr = torch.tensor([[[
            0.5001, 0.5000, 1.1e-4, 1.0e-4, 0.5000]]],
            device=self.device)
        target_prev = torch.tensor([[[
            0.1000, 0.5000, 0.2000, 0.2000, 0.5000]]],
            device=self.device)
        target_curr = torch.tensor([[[
            0.9000, 0.5000, 0.2000, 0.2000, 0.5000]]],
            device=self.device)
        residual_prev = (
            torch.logit(target_prev.clamp(1e-6, 1 - 1e-6))
            - torch.logit(reference_prev.clamp(1e-6, 1 - 1e-6))
        ).requires_grad_()
        residual_curr = (
            torch.logit(target_curr.clamp(1e-6, 1 - 1e-6))
            - torch.logit(reference_curr.clamp(1e-6, 1 - 1e-6))
        ).requires_grad_()
        projected_prev, projected_curr = (
            PairRotatedRTDETRTransformerDecoder.
            _pair_transport_shape_tangent_residual(
                residual_prev, residual_curr,
                reference_prev, reference_curr, 0))
        (projected_prev.sum() + projected_curr.sum()).backward()
        self.assertTrue(torch.isfinite(projected_prev).all())
        self.assertTrue(torch.isfinite(projected_curr).all())
        self.assertTrue(torch.isfinite(residual_prev.grad).all())
        self.assertTrue(torch.isfinite(residual_curr.grad).all())

    def test_transport_product_tangent_is_terminal_factorized_composition(
            self):
        parent, _, _ = _build_decoder(num_layers=4, device=self.device)
        projected, reg_prev, reg_curr = _build_decoder(
            num_layers=4,
            device=self.device,
            pair_shared_terminal_transport_product_tangent_refinement_decoder=(
                True))
        self.assertEqual(
            sum(parameter.numel() for parameter in parent.parameters()),
            sum(parameter.numel() for parameter in projected.parameters()))
        self.assertEqual(
            {key: tuple(value.shape)
             for key, value in parent.state_dict().items()},
            {key: tuple(value.shape)
             for key, value in projected.state_dict().items()})

        center_projection = projected._pair_transport_center_tangent_residual
        shape_projection = projected._pair_transport_shape_tangent_residual
        with mock.patch.object(
                projected, '_pair_transport_center_tangent_residual',
                side_effect=center_projection) as center_mock, mock.patch.object(
                    projected, '_pair_transport_shape_tangent_residual',
                    side_effect=shape_projection) as shape_mock:
            _, refs_prev, refs_curr, _, _ = self._forward(
                1, decoder=projected,
                reg_branches_prev=reg_prev, reg_branches_curr=reg_curr)
        self.assertEqual(center_mock.call_count, 1)
        self.assertEqual(shape_mock.call_count, 1)
        for reference in refs_prev + refs_curr:
            self.assertTrue(torch.isfinite(reference).all())

        torch.manual_seed(26)
        reference_prev = (
            torch.rand(2, 5, 5, device=self.device) * 0.6 + 0.2)
        reference_curr = (
            torch.rand(2, 5, 5, device=self.device) * 0.6 + 0.2)
        residual_prev = (
            torch.randn(2, 5, 5, device=self.device) * 0.2).requires_grad_()
        residual_curr = (
            torch.randn(2, 5, 5, device=self.device) * 0.2).requires_grad_()
        expected = center_projection(
            residual_prev, residual_curr,
            reference_prev, reference_curr, 2)
        expected = shape_projection(
            expected[0], expected[1],
            reference_prev, reference_curr, 2)
        swapped = center_projection(
            residual_curr, residual_prev,
            reference_curr, reference_prev, 2)
        swapped = shape_projection(
            swapped[0], swapped[1],
            reference_curr, reference_prev, 2)
        self.assertTrue(torch.equal(expected[0][:, :2], residual_prev[:, :2]))
        self.assertTrue(torch.equal(expected[1][:, :2], residual_curr[:, :2]))
        self.assertTrue(torch.allclose(
            expected[0], swapped[1], atol=1e-6, rtol=1e-5))
        self.assertTrue(torch.allclose(
            expected[1], swapped[0], atol=1e-6, rtol=1e-5))
        (expected[0].sum() + expected[1].sum()).backward()
        self.assertTrue(torch.isfinite(expected[0]).all())
        self.assertTrue(torch.isfinite(expected[1]).all())
        self.assertTrue(torch.isfinite(residual_prev.grad).all())
        self.assertTrue(torch.isfinite(residual_curr.grad).all())

        with self.assertRaisesRegex(ValueError, 'mutually exclusive'):
            _build_decoder(
                pair_shared_terminal_transport_shape_tangent_refinement_decoder=(
                    True),
                pair_shared_terminal_transport_product_tangent_refinement_decoder=(
                    True),
                device=self.device)

    def test_position_tangent_feature_projection_is_swap_equivariant(self):
        torch.manual_seed(27)
        evidence_prev = torch.randn(
            2, 7, 32, device=self.device, requires_grad=True)
        evidence_curr = torch.randn(
            2, 7, 32, device=self.device, requires_grad=True)
        query_pos_prev = torch.randn(
            2, 7, 32, device=self.device, requires_grad=True)
        query_pos_curr = torch.randn(
            2, 7, 32, device=self.device, requires_grad=True)
        projection = (
            PairRotatedRTDETRTransformerDecoder.
            _pair_position_tangent_feature_detail(
                evidence_prev, evidence_curr,
                query_pos_prev, query_pos_curr, 2))
        swapped = (
            PairRotatedRTDETRTransformerDecoder.
            _pair_position_tangent_feature_detail(
                evidence_curr, evidence_prev,
                query_pos_curr, query_pos_prev, 2))
        self.assertTrue(torch.equal(
            projection[:, :2], torch.zeros_like(projection[:, :2])))
        self.assertTrue(torch.allclose(
            projection, -swapped, atol=1e-6, rtol=1e-5))

        detail = 0.5 * (
            evidence_curr[:, 2:] - evidence_prev[:, 2:])
        transport = query_pos_curr[:, 2:] - query_pos_prev[:, 2:]
        remainder = detail - projection[:, 2:]
        self.assertTrue(torch.allclose(
            (remainder * transport).sum(dim=-1),
            torch.zeros_like(remainder[..., 0]),
            atol=2e-5, rtol=1e-5))
        self.assertTrue(torch.all(
            projection[:, 2:].square().sum(dim=-1)
            <= detail.square().sum(dim=-1) + 1e-6))
        projection.sum().backward()
        self.assertTrue(torch.isfinite(evidence_prev.grad).all())
        self.assertTrue(torch.isfinite(evidence_curr.grad).all())
        self.assertIsNone(query_pos_prev.grad)
        self.assertIsNone(query_pos_curr.grad)

    def test_position_tangent_product_is_terminal_and_parameter_free(self):
        parent, _, _ = _build_decoder(num_layers=3, device=self.device)
        decoder, reg_prev, reg_curr = _build_decoder(
            num_layers=3,
            device=self.device,
            terminal_position_tangent_product_decoder=True)
        self.assertEqual(
            sum(parameter.numel() for parameter in parent.parameters()),
            sum(parameter.numel() for parameter in decoder.parameters()))
        self.assertEqual(
            {key: tuple(value.shape)
             for key, value in parent.state_dict().items()},
            {key: tuple(value.shape)
             for key, value in decoder.state_dict().items()})

        spatial_shapes, level_start_index, num_value = _spatial_meta(
            self.device)
        memory_prev, memory_curr = _random_memories(
            1, num_value, decoder.embed_dims, self.device)
        feature_projection = decoder._pair_position_tangent_feature_detail
        center_projection = decoder._pair_transport_center_tangent_residual
        shape_projection = decoder._pair_transport_shape_tangent_residual
        with mock.patch.object(
                decoder, '_pair_position_tangent_feature_detail',
                side_effect=feature_projection) as feature_mock, \
                mock.patch.object(
                    decoder, '_pair_transport_center_tangent_residual',
                    side_effect=center_projection) as center_mock, \
                mock.patch.object(
                    decoder, '_pair_transport_shape_tangent_residual',
                    side_effect=shape_projection) as shape_mock:
            output = decoder(
                memory_prev=memory_prev,
                memory_curr=memory_curr,
                spatial_shapes=spatial_shapes,
                level_start_index=level_start_index,
                reg_branches_prev=reg_prev,
                reg_branches_curr=reg_curr)
        self.assertEqual(feature_mock.call_count, 1)
        self.assertEqual(center_mock.call_count, 1)
        self.assertEqual(shape_mock.call_count, 1)
        self.assertEqual(len(output), 5)
        hidden, refs_prev, refs_curr, hidden_prev, hidden_curr = output
        for lid in range(decoder.num_layers - 1):
            self.assertTrue(torch.equal(hidden_prev[lid], hidden[lid]))
            self.assertTrue(torch.equal(hidden_curr[lid], hidden[lid]))
        self.assertTrue(torch.allclose(
            0.5 * (hidden_prev[-1] + hidden_curr[-1]),
            hidden[-1], atol=1e-6, rtol=1e-5))
        for reference in refs_prev + refs_curr:
            self.assertTrue(torch.isfinite(reference).all())

        with self.assertRaisesRegex(ValueError, 'mutually exclusive'):
            _build_decoder(
                terminal_position_tangent_product_decoder=True,
                pair_shared_terminal_transport_product_tangent_refinement_decoder=(
                    True),
                device=self.device)

    def test_position_tangent_transport_is_terminal_and_parameter_free(self):
        parent, _, _ = _build_decoder(num_layers=3, device=self.device)
        decoder, reg_prev, reg_curr = _build_decoder(
            num_layers=3,
            device=self.device,
            terminal_position_tangent_transport_decoder=True)
        self.assertEqual(
            sum(parameter.numel() for parameter in parent.parameters()),
            sum(parameter.numel() for parameter in decoder.parameters()))
        self.assertEqual(
            {key: tuple(value.shape)
             for key, value in parent.state_dict().items()},
            {key: tuple(value.shape)
             for key, value in decoder.state_dict().items()})

        spatial_shapes, level_start_index, num_value = _spatial_meta(
            self.device)
        memory_prev, memory_curr = _random_memories(
            1, num_value, decoder.embed_dims, self.device)
        feature_projection = decoder._pair_position_tangent_feature_detail
        full_projection = decoder._pair_transport_full_tangent_residual
        center_projection = decoder._pair_transport_center_tangent_residual
        shape_projection = decoder._pair_transport_shape_tangent_residual
        with mock.patch.object(
                decoder, '_pair_position_tangent_feature_detail',
                side_effect=feature_projection) as feature_mock, \
                mock.patch.object(
                    decoder, '_pair_transport_full_tangent_residual',
                    side_effect=full_projection) as full_mock, \
                mock.patch.object(
                    decoder, '_pair_transport_center_tangent_residual',
                    side_effect=center_projection) as center_mock, \
                mock.patch.object(
                    decoder, '_pair_transport_shape_tangent_residual',
                    side_effect=shape_projection) as shape_mock:
            output = decoder(
                memory_prev=memory_prev,
                memory_curr=memory_curr,
                spatial_shapes=spatial_shapes,
                level_start_index=level_start_index,
                reg_branches_prev=reg_prev,
                reg_branches_curr=reg_curr)
        self.assertEqual(feature_mock.call_count, 1)
        self.assertEqual(full_mock.call_count, 1)
        self.assertEqual(center_mock.call_count, 0)
        self.assertEqual(shape_mock.call_count, 0)
        self.assertEqual(len(output), 5)
        hidden, refs_prev, refs_curr, hidden_prev, hidden_curr = output
        for lid in range(decoder.num_layers - 1):
            self.assertTrue(torch.equal(hidden_prev[lid], hidden[lid]))
            self.assertTrue(torch.equal(hidden_curr[lid], hidden[lid]))
        self.assertTrue(torch.allclose(
            0.5 * (hidden_prev[-1] + hidden_curr[-1]),
            hidden[-1], atol=1e-6, rtol=1e-5))
        for reference in refs_prev + refs_curr:
            self.assertTrue(torch.isfinite(reference).all())

        with self.assertRaisesRegex(ValueError, 'mutually exclusive'):
            _build_decoder(
                terminal_position_tangent_transport_decoder=True,
                pair_shared_terminal_transport_tangent_refinement_decoder=True,
                device=self.device)

    def test_position_tangent_plane_is_terminal_and_parameter_free(self):
        parent, _, _ = _build_decoder(num_layers=3, device=self.device)
        decoder, reg_prev, reg_curr = _build_decoder(
            num_layers=3,
            device=self.device,
            terminal_position_tangent_plane_decoder=True)
        self.assertEqual(
            sum(parameter.numel() for parameter in parent.parameters()),
            sum(parameter.numel() for parameter in decoder.parameters()))
        self.assertEqual(
            {key: tuple(value.shape)
             for key, value in parent.state_dict().items()},
            {key: tuple(value.shape)
             for key, value in decoder.state_dict().items()})

        spatial_shapes, level_start_index, num_value = _spatial_meta(
            self.device)
        memory_prev, memory_curr = _random_memories(
            1, num_value, decoder.embed_dims, self.device)
        feature_projection = decoder._pair_position_tangent_feature_detail
        plane_projection = decoder._pair_transport_osculating_plane_residual
        full_projection = decoder._pair_transport_full_tangent_residual
        with mock.patch.object(
                decoder, '_pair_position_tangent_feature_detail',
                side_effect=feature_projection) as feature_mock, \
                mock.patch.object(
                    decoder, '_pair_transport_osculating_plane_residual',
                    side_effect=plane_projection) as plane_mock, \
                mock.patch.object(
                    decoder, '_pair_transport_full_tangent_residual',
                    side_effect=full_projection) as full_mock:
            output = decoder(
                memory_prev=memory_prev,
                memory_curr=memory_curr,
                spatial_shapes=spatial_shapes,
                level_start_index=level_start_index,
                reg_branches_prev=reg_prev,
                reg_branches_curr=reg_curr)
        self.assertEqual(feature_mock.call_count, 1)
        self.assertEqual(plane_mock.call_count, 1)
        self.assertEqual(full_mock.call_count, 0)
        hidden, refs_prev, refs_curr, hidden_prev, hidden_curr = output
        for lid in range(decoder.num_layers - 1):
            self.assertTrue(torch.equal(hidden_prev[lid], hidden[lid]))
            self.assertTrue(torch.equal(hidden_curr[lid], hidden[lid]))
        self.assertTrue(torch.allclose(
            0.5 * (hidden_prev[-1] + hidden_curr[-1]),
            hidden[-1], atol=1e-6, rtol=1e-5))
        for reference in refs_prev + refs_curr:
            self.assertTrue(torch.isfinite(reference).all())

        with self.assertRaisesRegex(ValueError, 'mutually exclusive'):
            _build_decoder(
                terminal_position_tangent_plane_decoder=True,
                terminal_position_tangent_transport_decoder=True,
                device=self.device)

    def test_transport_plane_is_swap_equivariant_and_preserves_dn(self):
        torch.manual_seed(29)
        reference_prev = torch.rand(2, 6, 5, device=self.device) * 0.4 + 0.3
        reference_curr = torch.rand(2, 6, 5, device=self.device) * 0.4 + 0.3
        residual_prev = (
            torch.randn(2, 6, 5, device=self.device) * 0.08
        ).requires_grad_()
        residual_curr = (
            torch.randn(2, 6, 5, device=self.device) * 0.08
        ).requires_grad_()
        projected = (
            PairRotatedRTDETRTransformerDecoder.
            _pair_transport_osculating_plane_residual(
                residual_prev, residual_curr,
                reference_prev, reference_curr, 2))
        swapped = (
            PairRotatedRTDETRTransformerDecoder.
            _pair_transport_osculating_plane_residual(
                residual_curr, residual_prev,
                reference_curr, reference_prev, 2))
        self.assertTrue(torch.equal(projected[0][:, :2], residual_prev[:, :2]))
        self.assertTrue(torch.equal(projected[1][:, :2], residual_curr[:, :2]))
        self.assertTrue(torch.allclose(
            projected[0], swapped[1], atol=2e-5, rtol=1e-5))
        self.assertTrue(torch.allclose(
            projected[1], swapped[0], atol=2e-5, rtol=1e-5))

        def tangent(residual, reference):
            reference_logit = torch.logit(
                reference.clamp(1e-3, 1 - 1e-3))
            proposed = (residual + reference_logit).sigmoid()
            size = reference[..., 2:4].clamp_min(1e-6)
            angle = torch.remainder(
                proposed[..., 4:] - reference[..., 4:] + 0.5,
                1.0) - 0.5
            return torch.cat((
                (proposed[..., :2] - reference[..., :2]) / size,
                torch.log(proposed[..., 2:4].clamp_min(1e-6) / size),
                angle), dim=-1)

        input_detail = 0.5 * (
            tangent(residual_curr[:, 2:], reference_curr[:, 2:])
            - tangent(residual_prev[:, 2:], reference_prev[:, 2:]))
        output_detail = 0.5 * (
            tangent(projected[1][:, 2:], reference_curr[:, 2:])
            - tangent(projected[0][:, 2:], reference_prev[:, 2:]))
        self.assertTrue(torch.all(
            output_detail.square().sum(dim=-1)
            <= input_detail.square().sum(dim=-1) + 2e-5))
        sum(value.square().mean() for value in projected).backward()
        self.assertTrue(torch.isfinite(residual_prev.grad).all())
        self.assertTrue(torch.isfinite(residual_curr.grad).all())

    def test_transport_tangent_is_swap_equivariant_and_preserves_dn(self):
        torch.manual_seed(23)
        reference_prev = torch.rand(2, 5, 5, device=self.device) * 0.6 + 0.2
        reference_curr = torch.rand(2, 5, 5, device=self.device) * 0.6 + 0.2
        residual_prev = torch.randn(2, 5, 5, device=self.device) * 0.2
        residual_curr = torch.randn(2, 5, 5, device=self.device) * 0.2
        projected = (
            PairRotatedRTDETRTransformerDecoder.
            _pair_transport_full_tangent_residual(
                residual_prev, residual_curr,
                reference_prev, reference_curr, 2))
        swapped = (
            PairRotatedRTDETRTransformerDecoder.
            _pair_transport_full_tangent_residual(
                residual_curr, residual_prev,
                reference_curr, reference_prev, 2))
        self.assertTrue(torch.equal(projected[0][:, :2], residual_prev[:, :2]))
        self.assertTrue(torch.equal(projected[1][:, :2], residual_curr[:, :2]))
        self.assertTrue(torch.allclose(
            projected[0], swapped[1], atol=1e-6, rtol=1e-5))
        self.assertTrue(torch.allclose(
            projected[1], swapped[0], atol=1e-6, rtol=1e-5))

    def test_transport_tangent_tiny_references_have_finite_gradients(self):
        reference_prev = torch.tensor([[[
            0.5000, 0.5000, 1.0e-4, 1.0e-4, 0.5000]]],
            device=self.device)
        reference_curr = torch.tensor([[[
            0.5001, 0.5000, 1.1e-4, 1.0e-4, 0.5000]]],
            device=self.device)
        target_prev = torch.tensor([[[
            0.1000, 0.5000, 0.2000, 0.2000, 0.5000]]],
            device=self.device)
        target_curr = torch.tensor([[[
            0.9000, 0.5000, 0.2000, 0.2000, 0.5000]]],
            device=self.device)

        residual_prev = (
            torch.logit(target_prev.clamp(1e-6, 1 - 1e-6))
            - torch.logit(reference_prev.clamp(1e-6, 1 - 1e-6))
        ).requires_grad_()
        residual_curr = (
            torch.logit(target_curr.clamp(1e-6, 1 - 1e-6))
            - torch.logit(reference_curr.clamp(1e-6, 1 - 1e-6))
        ).requires_grad_()
        projected_prev, projected_curr = (
            PairRotatedRTDETRTransformerDecoder.
            _pair_transport_full_tangent_residual(
                residual_prev, residual_curr,
                reference_prev, reference_curr, 0))
        (projected_prev.sum() + projected_curr.sum()).backward()

        self.assertTrue(torch.isfinite(projected_prev).all())
        self.assertTrue(torch.isfinite(projected_curr).all())
        self.assertTrue(torch.isfinite(residual_prev.grad).all())
        self.assertTrue(torch.isfinite(residual_curr.grad).all())

    def test_normalized_center_residual_uses_reference_local_coordinates(self):
        reference_prev = torch.tensor([[[0.4, 0.4, 0.2, 0.4, 0.5],
                                        [0.3, 0.4, 0.2, 0.4, 0.5]]])
        reference_curr = torch.tensor([[[0.6, 0.5, 0.4, 0.2, 0.5],
                                        [0.7, 0.5, 0.4, 0.2, 0.5]]])
        proposed_prev = reference_prev.clone()
        proposed_curr = reference_curr.clone()
        proposed_prev[:, 1:, :2] += torch.tensor([[[0.02, 0.08]]])
        proposed_curr[:, 1:, :2] += torch.tensor([[[0.12, -0.02]]])
        residual_prev = (
            torch.logit(proposed_prev) - torch.logit(reference_prev))
        residual_curr = (
            torch.logit(proposed_curr) - torch.logit(reference_curr))
        original_prev = residual_prev.clone()
        original_curr = residual_curr.clone()

        projected_prev, projected_curr = (
            PairRotatedRTDETRTransformerDecoder.
            _pair_shared_normalized_center_residual(
                residual_prev, residual_curr,
                reference_prev, reference_curr, num_dn=1))
        decoded_prev = (
            projected_prev + torch.logit(reference_prev)).sigmoid()
        decoded_curr = (
            projected_curr + torch.logit(reference_curr)).sigmoid()

        # Local deltas are [0.1, 0.2] and [0.3, -0.1], so their common
        # local correction is [0.2, 0.05].
        self.assertTrue(torch.equal(
            projected_prev[:, :1], original_prev[:, :1]))
        self.assertTrue(torch.equal(
            projected_curr[:, :1], original_curr[:, :1]))
        self.assertTrue(torch.equal(
            projected_prev[:, 1:, 2:], original_prev[:, 1:, 2:]))
        self.assertTrue(torch.equal(
            projected_curr[:, 1:, 2:], original_curr[:, 1:, 2:]))
        self.assertTrue(torch.allclose(
            decoded_prev[:, 1:, :2], torch.tensor([[[0.34, 0.42]]]),
            atol=1e-5, rtol=0.0))
        self.assertTrue(torch.allclose(
            decoded_curr[:, 1:, :2], torch.tensor([[[0.78, 0.51]]]),
            atol=1e-5, rtol=0.0))

    def test_normalized_center_residual_is_swap_equivariant_and_local(self):
        reference_prev = torch.rand(2, 5, 5) * 0.8 + 0.1
        reference_curr = torch.rand(2, 5, 5) * 0.8 + 0.1
        prev = torch.randn(2, 5, 5, requires_grad=True)
        curr = torch.randn(2, 5, 5, requires_grad=True)
        projected_prev, projected_curr = (
            PairRotatedRTDETRTransformerDecoder.
            _pair_shared_normalized_center_residual(
                prev, curr, reference_prev, reference_curr, num_dn=0))
        swapped_curr, swapped_prev = (
            PairRotatedRTDETRTransformerDecoder.
            _pair_shared_normalized_center_residual(
                curr, prev, reference_curr, reference_prev, num_dn=0))
        self.assertTrue(torch.allclose(projected_prev, swapped_prev))
        self.assertTrue(torch.allclose(projected_curr, swapped_curr))

        projected_prev[..., :2].sum().backward()
        self.assertGreater(curr.grad[..., :2].abs().sum().item(), 0.0)
        self.assertEqual(curr.grad[..., 2:].abs().sum().item(), 0.0)

    def test_normalized_center_refinement_is_parameter_free_and_exclusive(self):
        parent, _, _ = _build_decoder(num_layers=3, device=self.device)
        projected, reg_prev, reg_curr = _build_decoder(
            num_layers=3,
            device=self.device,
            pair_shared_normalized_center_refinement_decoder=True)
        self.assertEqual(
            sum(parameter.numel() for parameter in parent.parameters()),
            sum(parameter.numel() for parameter in projected.parameters()))
        self.assertEqual(
            {key: tuple(value.shape)
             for key, value in parent.state_dict().items()},
            {key: tuple(value.shape)
             for key, value in projected.state_dict().items()})

        _, refs_prev, refs_curr, _, _ = self._forward(
            1,
            decoder=projected,
            reg_branches_prev=reg_prev,
            reg_branches_curr=reg_curr)
        for reference in refs_prev + refs_curr:
            self.assertTrue(torch.isfinite(reference).all())

        with self.assertRaisesRegex(ValueError, 'mutually exclusive'):
            _build_decoder(
                device=self.device,
                pair_shared_shape_refinement_decoder=True,
                pair_shared_normalized_center_refinement_decoder=True)

    def test_frame_evidence_cls_preserves_recurrent_decoder_and_references(self):
        parent, parent_reg_prev, parent_reg_curr = _build_decoder(
            num_layers=3, device=self.device)
        routed, routed_reg_prev, routed_reg_curr = _build_decoder(
            num_layers=3,
            device=self.device,
            frame_evidence_cls_decoder=True)
        routed.load_state_dict(parent.state_dict())
        self.assertEqual(
            sum(parameter.numel() for parameter in parent.parameters()),
            sum(parameter.numel() for parameter in routed.parameters()))
        self.assertEqual(
            {key: tuple(value.shape)
             for key, value in parent.state_dict().items()},
            {key: tuple(value.shape)
             for key, value in routed.state_dict().items()})

        spatial_shapes, level_start_index, num_value = _spatial_meta(
            self.device)
        memory_prev, memory_curr = _random_memories(
            2, num_value, parent.embed_dims, self.device)
        parent_out = parent(
            memory_prev=memory_prev,
            memory_curr=memory_curr,
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index,
            reg_branches_prev=parent_reg_prev,
            reg_branches_curr=parent_reg_curr)
        routed_out = routed(
            memory_prev=memory_prev,
            memory_curr=memory_curr,
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index,
            reg_branches_prev=routed_reg_prev,
            reg_branches_curr=routed_reg_curr)

        self.assertEqual(len(parent_out), 3)
        self.assertEqual(len(routed_out), 5)
        hidden, refs_prev, refs_curr = parent_out
        (routed_hidden, routed_refs_prev, routed_refs_curr,
         evidence_prev, evidence_curr) = routed_out
        for expected, actual in zip(hidden, routed_hidden):
            self.assertTrue(torch.equal(expected, actual))
        for expected, actual in zip(refs_prev, routed_refs_prev):
            self.assertTrue(torch.equal(expected, actual))
        for expected, actual in zip(refs_curr, routed_refs_curr):
            self.assertTrue(torch.equal(expected, actual))
        self.assertEqual(len(evidence_prev), 3)
        self.assertEqual(len(evidence_curr), 3)
        self.assertFalse(torch.equal(evidence_prev[0], evidence_curr[0]))
        self.assertFalse(torch.equal(evidence_prev[0], routed_hidden[0]))

    def test_frame_evidence_cls_first_layer_keeps_frame_local_gradient(self):
        decoder, reg_prev, reg_curr = _build_decoder(
            frame_evidence_cls_decoder=True, device=self.device)
        spatial_shapes, level_start_index, num_value = _spatial_meta(
            self.device)
        memory_prev, memory_curr = _random_memories(
            1, num_value, decoder.embed_dims, self.device)
        _, _, _, evidence_prev, _ = decoder(
            memory_prev=memory_prev,
            memory_curr=memory_curr,
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index,
            reg_branches_prev=reg_prev,
            reg_branches_curr=reg_curr)
        evidence_prev[0].sum().backward()
        self.assertGreater(memory_prev.grad.abs().sum().item(), 0.0)
        self.assertTrue(
            memory_curr.grad is None
            or memory_curr.grad.abs().sum().item() == 0.0)

    def test_frame_detail_cls_preserves_shared_midpoint_and_references(self):
        parent, parent_reg_prev, parent_reg_curr = _build_decoder(
            num_layers=3, device=self.device)
        detailed, detailed_reg_prev, detailed_reg_curr = _build_decoder(
            num_layers=3,
            frame_detail_cls_decoder=True,
            device=self.device)
        detailed.load_state_dict(parent.state_dict())
        self.assertEqual(
            sum(parameter.numel() for parameter in parent.parameters()),
            sum(parameter.numel() for parameter in detailed.parameters()))
        self.assertEqual(
            {key: tuple(value.shape)
             for key, value in parent.state_dict().items()},
            {key: tuple(value.shape)
             for key, value in detailed.state_dict().items()})

        spatial_shapes, level_start_index, num_value = _spatial_meta(
            self.device)
        memory_prev, memory_curr = _random_memories(
            2, num_value, parent.embed_dims, self.device)
        parent_out = parent(
            memory_prev=memory_prev,
            memory_curr=memory_curr,
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index,
            reg_branches_prev=parent_reg_prev,
            reg_branches_curr=parent_reg_curr)
        detailed_out = detailed(
            memory_prev=memory_prev,
            memory_curr=memory_curr,
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index,
            reg_branches_prev=detailed_reg_prev,
            reg_branches_curr=detailed_reg_curr)

        hidden, refs_prev, refs_curr = parent_out
        (detailed_hidden, detailed_refs_prev, detailed_refs_curr,
         cls_prev, cls_curr) = detailed_out
        for expected, actual in zip(hidden, detailed_hidden):
            self.assertTrue(torch.equal(expected, actual))
        for expected, actual in zip(refs_prev, detailed_refs_prev):
            self.assertTrue(torch.equal(expected, actual))
        for expected, actual in zip(refs_curr, detailed_refs_curr):
            self.assertTrue(torch.equal(expected, actual))
        for shared, prev, curr in zip(
                detailed_hidden, cls_prev, cls_curr):
            torch.testing.assert_close(0.5 * (prev + curr), shared)
            self.assertFalse(torch.equal(prev, curr))

        with self.assertRaisesRegex(ValueError, 'mutually exclusive'):
            _build_decoder(
                frame_evidence_cls_decoder=True,
                frame_detail_cls_decoder=True,
                device=self.device)

    def test_frame_evidence_cls_is_orthogonal_to_periodic_angle(self):
        periodic, periodic_reg_prev, periodic_reg_curr = _build_decoder(
            num_layers=3,
            pair_shared_periodic_angle_refinement_decoder=True,
            device=self.device)
        combined, combined_reg_prev, combined_reg_curr = _build_decoder(
            num_layers=3,
            pair_shared_periodic_angle_refinement_decoder=True,
            frame_evidence_cls_decoder=True,
            device=self.device)
        combined.load_state_dict(periodic.state_dict())
        self.assertEqual(
            sum(parameter.numel() for parameter in periodic.parameters()),
            sum(parameter.numel() for parameter in combined.parameters()))

        spatial_shapes, level_start_index, num_value = _spatial_meta(
            self.device)
        memory_prev, memory_curr = _random_memories(
            2, num_value, periodic.embed_dims, self.device)
        periodic_out = periodic(
            memory_prev=memory_prev,
            memory_curr=memory_curr,
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index,
            reg_branches_prev=periodic_reg_prev,
            reg_branches_curr=periodic_reg_curr)
        combined_out = combined(
            memory_prev=memory_prev,
            memory_curr=memory_curr,
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index,
            reg_branches_prev=combined_reg_prev,
            reg_branches_curr=combined_reg_curr)

        hidden, refs_prev, refs_curr = periodic_out
        (combined_hidden, combined_refs_prev, combined_refs_curr,
         evidence_prev, evidence_curr) = combined_out
        for expected, actual in zip(hidden, combined_hidden):
            self.assertTrue(torch.equal(expected, actual))
        for expected, actual in zip(refs_prev, combined_refs_prev):
            self.assertTrue(torch.equal(expected, actual))
        for expected, actual in zip(refs_curr, combined_refs_curr):
            self.assertTrue(torch.equal(expected, actual))
        self.assertFalse(torch.equal(evidence_prev[0], evidence_curr[0]))

    def test_output_shapes_batch1(self):
        decoder, reg_prev, reg_curr = _build_decoder(device=self.device)
        hidden, refs_prev, refs_curr, _, _ = self._forward(
            1, decoder, reg_prev, reg_curr)
        self.assertEqual(len(hidden), decoder.num_layers)
        self.assertEqual(hidden[0].shape, (1, decoder.num_queries,
                                           decoder.embed_dims))
        self.assertEqual(refs_prev[0].shape, (1, decoder.num_queries, 5))
        self.assertEqual(refs_curr[0].shape, (1, decoder.num_queries, 5))

    def test_stacked_reference_shape(self):
        decoder, reg_prev, reg_curr = _build_decoder(device=self.device)
        _, refs_prev, refs_curr, _, _ = self._forward(
            2, decoder, reg_prev, reg_curr)
        stacked_prev = torch.stack(refs_prev)
        stacked_curr = torch.stack(refs_curr)
        self.assertEqual(stacked_prev.shape,
                         (decoder.num_layers, 2, decoder.num_queries, 5))
        self.assertEqual(stacked_curr.shape,
                         (decoder.num_layers, 2, decoder.num_queries, 5))

    def test_output_shapes_batch2(self):
        decoder, reg_prev, reg_curr = _build_decoder(device=self.device)
        hidden, refs_prev, refs_curr, _, _ = self._forward(
            2, decoder, reg_prev, reg_curr)
        self.assertEqual(hidden[-1].shape[0], 2)
        self.assertEqual(refs_prev[-1].shape[0], 2)
        self.assertEqual(refs_curr[-1].shape[0], 2)

    def test_tristate_decoder_outputs_frame_hidden_states(self):
        decoder, reg_prev, reg_curr = _build_decoder(
            device=self.device, tristate_decoder=True)
        cls_prev = _build_cls_branches(
            decoder.num_layers, decoder.embed_dims, 4, self.device, seed=2)
        cls_curr = _build_cls_branches(
            decoder.num_layers, decoder.embed_dims, 4, self.device, seed=3)
        spatial_shapes, level_start_index, num_value = _spatial_meta(
            self.device)
        memory_prev, memory_curr = _random_memories(
            2, num_value, decoder.embed_dims, self.device)

        out = decoder(
            memory_prev=memory_prev,
            memory_curr=memory_curr,
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index,
            reg_branches_prev=reg_prev,
            reg_branches_curr=reg_curr,
            cls_branches_prev=cls_prev,
            cls_branches_curr=cls_curr,
        )

        self.assertEqual(len(out), 5)
        hidden, refs_prev, refs_curr, hidden_prev, hidden_curr = out
        self.assertEqual(len(hidden), decoder.num_layers)
        self.assertEqual(len(hidden_prev), decoder.num_layers)
        self.assertEqual(len(hidden_curr), decoder.num_layers)
        self.assertEqual(hidden[-1].shape, (2, decoder.num_queries,
                                            decoder.embed_dims))
        self.assertEqual(hidden_prev[-1].shape, hidden[-1].shape)
        self.assertEqual(hidden_curr[-1].shape, hidden[-1].shape)
        self.assertEqual(refs_prev[-1].shape, (2, decoder.num_queries, 5))
        self.assertEqual(refs_curr[-1].shape, (2, decoder.num_queries, 5))

        loss = (hidden[-1].sum() + hidden_prev[-1].sum() +
                hidden_curr[-1].sum() + refs_prev[-1].sum() +
                refs_curr[-1].sum())
        loss.backward()
        self.assertIsNotNone(memory_prev.grad)
        self.assertIsNotNone(memory_curr.grad)
        self.assertGreater(memory_prev.grad.abs().sum().item(), 0.0)
        self.assertGreater(memory_curr.grad.abs().sum().item(), 0.0)

    def test_tristate_separate_ffn_forward(self):
        decoder, reg_prev, reg_curr = _build_decoder(
            device=self.device,
            tristate_decoder=True,
            tristate_separate_ffn=True)
        self.assertTrue(hasattr(decoder.layers[0], 'ffn_prev'))
        self.assertTrue(hasattr(decoder.layers[0], 'ffn_curr'))
        cls_prev = _build_cls_branches(
            decoder.num_layers, decoder.embed_dims, 4, self.device, seed=4)
        cls_curr = _build_cls_branches(
            decoder.num_layers, decoder.embed_dims, 4, self.device, seed=5)
        spatial_shapes, level_start_index, num_value = _spatial_meta(
            self.device)
        memory_prev, memory_curr = _random_memories(
            1, num_value, decoder.embed_dims, self.device)
        out = decoder(
            memory_prev=memory_prev,
            memory_curr=memory_curr,
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index,
            reg_branches_prev=reg_prev,
            reg_branches_curr=reg_curr,
            cls_branches_prev=cls_prev,
            cls_branches_curr=cls_curr,
        )
        _, refs_prev, refs_curr, hidden_prev, hidden_curr = out
        self.assertEqual(hidden_prev[-1].shape, hidden_curr[-1].shape)

    def test_tristate_zero_init_coupling(self):
        decoder, _, _ = _build_decoder(
            device=self.device,
            tristate_decoder=True,
            tristate_zero_init_coupling=True)
        decoder.init_weights()
        for layer in decoder.layers:
            for module in (layer.pointer_to_prev, layer.pointer_to_curr,
                           layer.pointer_update):
                self.assertEqual(module.weight.abs().sum().item(), 0.0)
                self.assertEqual(module.bias.abs().sum().item(), 0.0)

    def test_detector_init_preserves_tristate_structural_weights(self):
        """Detector-level Xavier must not overwrite Pair decoder invariants."""
        decoder, _, _ = _build_decoder(
            device=self.device,
            tristate_decoder=True,
            tristate_zero_init_coupling=True)
        model = MultispecPairRotatedRTDETR.__new__(
            MultispecPairRotatedRTDETR)
        torch.nn.Module.__init__(model)
        model.decoder = decoder

        def detector_level_xavier(model_self):
            for param in model_self.decoder.parameters():
                if param.dim() > 1:
                    torch.nn.init.xavier_uniform_(param)

        with mock.patch.object(
                RotatedRTDETR, 'init_weights', detector_level_xavier):
            model.init_weights()

        eye = torch.eye(decoder.embed_dims, device=self.device)
        for module in (
                decoder.query_to_prev,
                decoder.query_to_curr,
                decoder.query_to_pointer):
            self.assertTrue(torch.equal(module.weight, eye))
            self.assertEqual(module.bias.abs().sum().item(), 0.0)
        self.assertTrue(torch.equal(
            decoder.pointer_init_fusion.weight[:, :decoder.embed_dims], eye))
        self.assertEqual(
            decoder.pointer_init_fusion.weight[:, decoder.embed_dims:]
            .abs().sum().item(), 0.0)
        self.assertEqual(
            decoder.pointer_init_fusion.bias.abs().sum().item(), 0.0)
        average_fusion = torch.zeros(
            decoder.embed_dims,
            decoder.embed_dims * 2,
            device=self.device)
        average_fusion[:, :decoder.embed_dims] = 0.5 * eye
        average_fusion[:, decoder.embed_dims:] = 0.5 * eye
        self.assertTrue(torch.equal(
            decoder.pair_pos_fusion.weight, average_fusion))
        self.assertEqual(
            decoder.pair_pos_fusion.bias.abs().sum().item(), 0.0)
        for layer in decoder.layers:
            self.assertTrue(torch.equal(
                layer.cross_fusion.weight, average_fusion))
            self.assertEqual(layer.cross_fusion.bias.abs().sum().item(), 0.0)
            for module in (layer.pointer_to_prev, layer.pointer_to_curr,
                           layer.pointer_update):
                self.assertEqual(module.weight.abs().sum().item(), 0.0)
                self.assertEqual(module.bias.abs().sum().item(), 0.0)

    def test_dual_output_adapter_is_baseline_preserving_at_init(self):
        decoder, reg_prev, reg_curr = _build_decoder(
            device=self.device, dual_output_adapter=True)
        decoder.init_weights()
        spatial_shapes, level_start_index, num_value = _spatial_meta(
            self.device)
        memory_prev, memory_curr = _random_memories(
            1, num_value, decoder.embed_dims, self.device)
        output = decoder(
            memory_prev=memory_prev,
            memory_curr=memory_curr,
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index,
            reg_branches_prev=reg_prev,
            reg_branches_curr=reg_curr,
        )
        hidden, _, _, hidden_prev, hidden_curr = output
        for shared, prev, curr in zip(hidden, hidden_prev, hidden_curr):
            self.assertTrue(torch.equal(prev, shared))
            self.assertTrue(torch.equal(curr, shared))
        for adapter in (
                list(decoder.dual_output_prev_adapters)
                + list(decoder.dual_output_curr_adapters)):
            self.assertEqual(adapter.weight.abs().sum().item(), 0.0)
            self.assertEqual(adapter.bias.abs().sum().item(), 0.0)

    def test_box_only_adapter_keeps_classification_features_shared(self):
        decoder, reg_prev, reg_curr = _build_decoder(
            device=self.device,
            dual_output_adapter=True,
            dual_output_cls_scale=0.0,
            dual_output_reg_scale=0.5,
            dual_output_detach_adapter_input=True)
        decoder.init_weights()
        with torch.no_grad():
            for adapter in decoder.dual_output_prev_adapters:
                adapter.weight.copy_(torch.eye(decoder.embed_dims))
            for adapter in decoder.dual_output_curr_adapters:
                adapter.weight.copy_(-torch.eye(decoder.embed_dims))
        spatial_shapes, level_start_index, num_value = _spatial_meta(
            self.device)
        memory_prev, memory_curr = _random_memories(
            1, num_value, decoder.embed_dims, self.device)
        hidden, refs_prev, refs_curr, hidden_prev, hidden_curr = decoder(
            memory_prev=memory_prev,
            memory_curr=memory_curr,
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index,
            reg_branches_prev=reg_prev,
            reg_branches_curr=reg_curr)
        for shared, prev, curr in zip(hidden, hidden_prev, hidden_curr):
            self.assertTrue(torch.equal(prev, shared))
            self.assertTrue(torch.equal(curr, shared))
        self.assertGreater(
            (refs_prev[-1] - refs_curr[-1]).abs().max().item(), 1e-6)

    def test_dual_output_adapter_rejects_negative_reg_scale(self):
        with self.assertRaisesRegex(ValueError, 'reg_scale'):
            _build_decoder(
                device=self.device,
                dual_output_adapter=True,
                dual_output_reg_scale=-0.1)

    def test_detector_init_preserves_dual_output_zero_start(self):
        decoder, _, _ = _build_decoder(
            device=self.device, dual_output_adapter=True)
        model = MultispecPairRotatedRTDETR.__new__(
            MultispecPairRotatedRTDETR)
        torch.nn.Module.__init__(model)
        model.decoder = decoder

        def detector_level_xavier(model_self):
            for param in model_self.decoder.parameters():
                if param.dim() > 1:
                    torch.nn.init.xavier_uniform_(param)

        with mock.patch.object(
                RotatedRTDETR, 'init_weights', detector_level_xavier):
            model.init_weights()
        for adapter in (
                list(decoder.dual_output_prev_adapters)
                + list(decoder.dual_output_curr_adapters)):
            self.assertEqual(adapter.weight.abs().sum().item(), 0.0)
            self.assertEqual(adapter.bias.abs().sum().item(), 0.0)

    def test_dual_output_adapters_receive_first_step_gradients(self):
        decoder, reg_prev, reg_curr = _build_decoder(
            device=self.device, dual_output_adapter=True)
        decoder.init_weights()
        spatial_shapes, level_start_index, num_value = _spatial_meta(
            self.device)
        memory_prev, memory_curr = _random_memories(
            1, num_value, decoder.embed_dims, self.device)
        output = decoder(
            memory_prev=memory_prev,
            memory_curr=memory_curr,
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index,
            reg_branches_prev=reg_prev,
            reg_branches_curr=reg_curr,
        )
        _, _, _, hidden_prev, hidden_curr = output
        torch.manual_seed(17)
        loss = sum(
            (prev * torch.randn_like(prev)).mean()
            + (curr * torch.randn_like(curr)).mean()
            for prev, curr in zip(hidden_prev, hidden_curr))
        loss.backward()
        for adapter in (
                list(decoder.dual_output_prev_adapters)
                + list(decoder.dual_output_curr_adapters)):
            self.assertIsNotNone(adapter.weight.grad)
            self.assertGreater(adapter.weight.grad.abs().max().item(), 0.0)

    def test_dual_output_adapter_rejects_tristate_combination(self):
        with self.assertRaisesRegex(ValueError, 'mutually exclusive'):
            _build_decoder(
                device=self.device,
                tristate_decoder=True,
                dual_output_adapter=True)

    def test_common_motion_is_exact_baseline_at_zero_start(self):
        decoder, reg_prev, reg_curr = _build_decoder(
            device=self.device, common_motion_decoder=True)
        decoder.init_weights()
        spatial_shapes, level_start_index, num_value = _spatial_meta(
            self.device)
        memory_prev, memory_curr = _random_memories(
            1, num_value, decoder.embed_dims, self.device)
        common_output = decoder(
            memory_prev=memory_prev,
            memory_curr=memory_curr,
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index,
            reg_branches_prev=reg_prev,
            reg_branches_curr=reg_curr)
        decoder.common_motion_decoder = False
        baseline_output = decoder(
            memory_prev=memory_prev,
            memory_curr=memory_curr,
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index,
            reg_branches_prev=reg_prev,
            reg_branches_curr=reg_curr)
        for common_group, baseline_group in zip(
                common_output, baseline_output):
            for common_tensor, baseline_tensor in zip(
                    common_group, baseline_group):
                self.assertTrue(torch.equal(
                    common_tensor, baseline_tensor))
        for adapter in decoder.common_motion_adapters:
            self.assertEqual(adapter.weight.abs().sum().item(), 0.0)

    def test_common_motion_correction_is_swap_antisymmetric(self):
        decoder, _, _ = _build_decoder(
            device=self.device, common_motion_decoder=True)
        decoder.init_weights()
        torch.manual_seed(23)
        with torch.no_grad():
            torch.nn.init.normal_(
                decoder.common_motion_adapters[0].weight, std=0.02)
        out_prev = torch.randn(
            2, 7, decoder.embed_dims, device=self.device)
        out_curr = torch.randn_like(out_prev)
        reference_prev = torch.rand(2, 7, 5, device=self.device)
        reference_curr = torch.rand(2, 7, 5, device=self.device)
        correction = decoder._common_motion_correction(
            0, out_prev, out_curr, reference_prev, reference_curr)
        swapped = decoder._common_motion_correction(
            0, out_curr, out_prev, reference_curr, reference_prev)
        self.assertTrue(torch.allclose(
            correction, -swapped, atol=1e-6, rtol=1e-5))

    def test_common_motion_adapters_receive_first_step_gradients(self):
        decoder, reg_prev, reg_curr = _build_decoder(
            num_layers=3,
            device=self.device,
            common_motion_decoder=True)
        decoder.init_weights()
        spatial_shapes, level_start_index, num_value = _spatial_meta(
            self.device)
        memory_prev, memory_curr = _random_memories(
            1, num_value, decoder.embed_dims, self.device)
        _, references_prev, references_curr = decoder(
            memory_prev=memory_prev,
            memory_curr=memory_curr,
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index,
            reg_branches_prev=reg_prev,
            reg_branches_curr=reg_curr)
        torch.manual_seed(29)
        loss = sum(
            (prev * torch.randn_like(prev)).mean()
            + (curr * torch.randn_like(curr)).mean()
            for prev, curr in zip(references_prev, references_curr))
        loss.backward()
        for adapter in decoder.common_motion_adapters:
            self.assertIsNotNone(adapter.weight.grad)
            self.assertGreater(adapter.weight.grad.abs().max().item(), 0.0)

    def test_detector_init_preserves_common_motion_zero_start(self):
        decoder, _, _ = _build_decoder(
            device=self.device, common_motion_decoder=True)
        model = MultispecPairRotatedRTDETR.__new__(
            MultispecPairRotatedRTDETR)
        torch.nn.Module.__init__(model)
        model.decoder = decoder

        def detector_level_xavier(model_self):
            for param in model_self.decoder.parameters():
                if param.dim() > 1:
                    torch.nn.init.xavier_uniform_(param)

        with mock.patch.object(
                RotatedRTDETR, 'init_weights', detector_level_xavier):
            model.init_weights()
        for adapter in decoder.common_motion_adapters:
            self.assertEqual(adapter.weight.abs().sum().item(), 0.0)

    def test_common_motion_rejects_other_decoder_variants(self):
        for other in (
                dict(tristate_decoder=True),
                dict(dual_output_adapter=True),
        ):
            with self.assertRaisesRegex(ValueError, 'mutually exclusive'):
                _build_decoder(
                    device=self.device,
                    common_motion_decoder=True,
                    **other)

    def test_shared_evidence_is_exact_baseline_at_zero_start(self):
        decoder, reg_prev, reg_curr = _build_decoder(
            device=self.device, shared_evidence_decoder=True)
        decoder.init_weights()
        spatial_shapes, level_start_index, num_value = _spatial_meta(
            self.device)
        memory_prev, memory_curr = _random_memories(
            1, num_value, decoder.embed_dims, self.device)
        evidence_output = decoder(
            memory_prev=memory_prev,
            memory_curr=memory_curr,
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index,
            reg_branches_prev=reg_prev,
            reg_branches_curr=reg_curr)
        decoder.shared_evidence_decoder = False
        baseline_output = decoder(
            memory_prev=memory_prev,
            memory_curr=memory_curr,
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index,
            reg_branches_prev=reg_prev,
            reg_branches_curr=reg_curr)
        for evidence_group, baseline_group in zip(
                evidence_output, baseline_output):
            for evidence_tensor, baseline_tensor in zip(
                    evidence_group, baseline_group):
                self.assertTrue(torch.equal(
                    evidence_tensor, baseline_tensor))
        for adapter in decoder.shared_evidence_adapters:
            self.assertEqual(adapter.weight.abs().sum().item(), 0.0)

    def test_shared_evidence_is_swap_invariant(self):
        decoder, _, _ = _build_decoder(
            device=self.device, shared_evidence_decoder=True)
        decoder.init_weights()
        torch.manual_seed(31)
        with torch.no_grad():
            torch.nn.init.normal_(
                decoder.shared_evidence_adapters[0].weight, std=0.02)
        out_prev = torch.randn(
            2, 7, decoder.embed_dims, device=self.device)
        out_curr = torch.randn_like(out_prev)
        correction = decoder._shared_evidence_correction(
            0, out_prev, out_curr)
        swapped = decoder._shared_evidence_correction(
            0, out_curr, out_prev)
        self.assertTrue(torch.equal(correction, swapped))

    def test_shared_evidence_adapters_receive_first_step_gradients(self):
        decoder, reg_prev, reg_curr = _build_decoder(
            num_layers=3,
            device=self.device,
            shared_evidence_decoder=True)
        decoder.init_weights()
        spatial_shapes, level_start_index, num_value = _spatial_meta(
            self.device)
        memory_prev, memory_curr = _random_memories(
            1, num_value, decoder.embed_dims, self.device)
        hidden_states, references_prev, references_curr = decoder(
            memory_prev=memory_prev,
            memory_curr=memory_curr,
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index,
            reg_branches_prev=reg_prev,
            reg_branches_curr=reg_curr)
        torch.manual_seed(37)
        loss = sum(
            (hidden * torch.randn_like(hidden)).mean()
            for hidden in hidden_states)
        loss = loss + sum(
            (prev * torch.randn_like(prev)).mean()
            + (curr * torch.randn_like(curr)).mean()
            for prev, curr in zip(references_prev, references_curr))
        loss.backward()
        for adapter in decoder.shared_evidence_adapters:
            self.assertIsNotNone(adapter.weight.grad)
            self.assertGreater(adapter.weight.grad.abs().max().item(), 0.0)

    def test_detector_init_preserves_shared_evidence_zero_start(self):
        decoder, _, _ = _build_decoder(
            device=self.device, shared_evidence_decoder=True)
        model = MultispecPairRotatedRTDETR.__new__(
            MultispecPairRotatedRTDETR)
        torch.nn.Module.__init__(model)
        model.decoder = decoder

        def detector_level_xavier(model_self):
            for param in model_self.decoder.parameters():
                if param.dim() > 1:
                    torch.nn.init.xavier_uniform_(param)

        with mock.patch.object(
                RotatedRTDETR, 'init_weights', detector_level_xavier):
            model.init_weights()
        for adapter in decoder.shared_evidence_adapters:
            self.assertEqual(adapter.weight.abs().sum().item(), 0.0)

    def test_common_motion_and_shared_evidence_compose(self):
        """The two orthogonal decoder paths remain zero-start and trainable."""
        decoder, reg_prev, reg_curr = _build_decoder(
            num_layers=3,
            device=self.device,
            common_motion_decoder=True,
            shared_evidence_decoder=True)
        decoder.init_weights()
        spatial_shapes, level_start_index, num_value = _spatial_meta(
            self.device)
        memory_prev, memory_curr = _random_memories(
            1, num_value, decoder.embed_dims, self.device)

        combined_output = decoder(
            memory_prev=memory_prev,
            memory_curr=memory_curr,
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index,
            reg_branches_prev=reg_prev,
            reg_branches_curr=reg_curr)
        decoder.common_motion_decoder = False
        decoder.shared_evidence_decoder = False
        baseline_output = decoder(
            memory_prev=memory_prev,
            memory_curr=memory_curr,
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index,
            reg_branches_prev=reg_prev,
            reg_branches_curr=reg_curr)
        for combined_group, baseline_group in zip(
                combined_output, baseline_output):
            for combined_tensor, baseline_tensor in zip(
                    combined_group, baseline_group):
                self.assertTrue(torch.equal(
                    combined_tensor, baseline_tensor))

        decoder.common_motion_decoder = True
        decoder.shared_evidence_decoder = True
        torch.manual_seed(41)
        hidden_states, references_prev, references_curr = decoder(
            memory_prev=memory_prev,
            memory_curr=memory_curr,
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index,
            reg_branches_prev=reg_prev,
            reg_branches_curr=reg_curr)
        loss = sum(
            (hidden * torch.randn_like(hidden)).mean()
            for hidden in hidden_states)
        loss = loss + sum(
            (prev * torch.randn_like(prev)).mean()
            + (curr * torch.randn_like(curr)).mean()
            for prev, curr in zip(references_prev, references_curr))
        loss.backward()
        for adapter in (
                list(decoder.common_motion_adapters)
                + list(decoder.shared_evidence_adapters)):
            self.assertIsNotNone(adapter.weight.grad)
            self.assertGreater(adapter.weight.grad.abs().max().item(), 0.0)

    def test_shared_evidence_rejects_incompatible_decoders(self):
        for other in (
                dict(tristate_decoder=True),
                dict(dual_output_adapter=True),
        ):
            with self.assertRaisesRegex(ValueError, 'incompatible'):
                _build_decoder(
                    device=self.device,
                    shared_evidence_decoder=True,
                    **other)

    def test_competitive_evidence_is_exact_baseline_at_zero_start(self):
        decoder, reg_prev, reg_curr = _build_decoder(
            device=self.device, competitive_evidence_decoder=True)
        decoder.init_weights()
        spatial_shapes, level_start_index, num_value = _spatial_meta(
            self.device)
        memory_prev, memory_curr = _random_memories(
            1, num_value, decoder.embed_dims, self.device)
        competitive_output = decoder(
            memory_prev=memory_prev,
            memory_curr=memory_curr,
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index,
            reg_branches_prev=reg_prev,
            reg_branches_curr=reg_curr)
        decoder.competitive_evidence_decoder = False
        baseline_output = decoder(
            memory_prev=memory_prev,
            memory_curr=memory_curr,
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index,
            reg_branches_prev=reg_prev,
            reg_branches_curr=reg_curr)
        for competitive_group, baseline_group in zip(
                competitive_output, baseline_output):
            for competitive_tensor, baseline_tensor in zip(
                    competitive_group, baseline_group):
                self.assertTrue(torch.equal(
                    competitive_tensor, baseline_tensor))
        for adapter in decoder.competitive_evidence_adapters:
            self.assertEqual(adapter.weight.abs().sum().item(), 0.0)

    def test_competitive_evidence_correction_is_swap_invariant(self):
        decoder, _, _ = _build_decoder(
            device=self.device, competitive_evidence_decoder=True)
        decoder.init_weights()
        torch.manual_seed(43)
        with torch.no_grad():
            torch.nn.init.normal_(
                decoder.competitive_evidence_adapters[0].weight, std=0.02)
        out_prev = torch.randn(
            2, 7, decoder.embed_dims, device=self.device)
        out_curr = torch.randn_like(out_prev)
        correction = decoder._competitive_evidence_correction(
            0, out_prev, out_curr)
        swapped = decoder._competitive_evidence_correction(
            0, out_curr, out_prev)
        self.assertTrue(torch.allclose(
            correction, swapped, atol=1e-6, rtol=1e-5))
        detail = 0.5 * (out_curr - out_prev)
        self.assertTrue(torch.all(correction.abs() <= detail.abs() + 1e-7))

    def test_competitive_evidence_adapters_receive_first_step_gradients(self):
        decoder, reg_prev, reg_curr = _build_decoder(
            num_layers=3,
            device=self.device,
            competitive_evidence_decoder=True)
        decoder.init_weights()
        spatial_shapes, level_start_index, num_value = _spatial_meta(
            self.device)
        memory_prev, memory_curr = _random_memories(
            1, num_value, decoder.embed_dims, self.device)
        hidden_states, references_prev, references_curr = decoder(
            memory_prev=memory_prev,
            memory_curr=memory_curr,
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index,
            reg_branches_prev=reg_prev,
            reg_branches_curr=reg_curr)
        torch.manual_seed(47)
        loss = sum(
            (hidden * torch.randn_like(hidden)).mean()
            for hidden in hidden_states)
        loss = loss + sum(
            (prev * torch.randn_like(prev)).mean()
            + (curr * torch.randn_like(curr)).mean()
            for prev, curr in zip(references_prev, references_curr))
        loss.backward()
        for adapter in decoder.competitive_evidence_adapters:
            self.assertIsNotNone(adapter.weight.grad)
            self.assertGreater(adapter.weight.grad.abs().max().item(), 0.0)

    def test_detector_init_preserves_competitive_evidence_zero_start(self):
        decoder, _, _ = _build_decoder(
            device=self.device, competitive_evidence_decoder=True)
        model = MultispecPairRotatedRTDETR.__new__(
            MultispecPairRotatedRTDETR)
        torch.nn.Module.__init__(model)
        model.decoder = decoder

        def detector_level_xavier(model_self):
            for param in model_self.decoder.parameters():
                if param.dim() > 1:
                    torch.nn.init.xavier_uniform_(param)

        with mock.patch.object(
                RotatedRTDETR, 'init_weights', detector_level_xavier):
            model.init_weights()
        for adapter in decoder.competitive_evidence_adapters:
            self.assertEqual(adapter.weight.abs().sum().item(), 0.0)

    def test_competitive_evidence_rejects_incompatible_decoders(self):
        for other in (
                dict(tristate_decoder=True),
                dict(dual_output_adapter=True),
                dict(shared_evidence_decoder=True),
        ):
            with self.assertRaisesRegex(ValueError, 'incompatible'):
                _build_decoder(
                    device=self.device,
                    competitive_evidence_decoder=True,
                    **other)

    def test_motion_trust_is_exact_baseline_at_zero_start(self):
        decoder, reg_prev, reg_curr = _build_decoder(
            device=self.device, motion_trust_decoder=True)
        decoder.init_weights()
        cls_prev = _build_cls_branches(
            decoder.num_layers, decoder.embed_dims, 8, self.device, seed=2)
        cls_curr = _build_cls_branches(
            decoder.num_layers, decoder.embed_dims, 8, self.device, seed=3)
        spatial_shapes, level_start_index, num_value = _spatial_meta(
            self.device)
        memory_prev, memory_curr = _random_memories(
            1, num_value, decoder.embed_dims, self.device)
        trusted_output = decoder(
            memory_prev=memory_prev,
            memory_curr=memory_curr,
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index,
            reg_branches_prev=reg_prev,
            reg_branches_curr=reg_curr,
            cls_branches_prev=cls_prev,
            cls_branches_curr=cls_curr)
        decoder.motion_trust_decoder = False
        baseline_output = decoder(
            memory_prev=memory_prev,
            memory_curr=memory_curr,
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index,
            reg_branches_prev=reg_prev,
            reg_branches_curr=reg_curr)
        for trusted_group, baseline_group in zip(
                trusted_output, baseline_output):
            for trusted_tensor, baseline_tensor in zip(
                    trusted_group, baseline_group):
                self.assertTrue(torch.equal(trusted_tensor, baseline_tensor))
        for adapter in decoder.motion_trust_adapters:
            self.assertEqual(adapter.weight.abs().sum().item(), 0.0)

    def test_motion_trust_is_swap_antisymmetric_and_bounded(self):
        decoder, _, _ = _build_decoder(
            device=self.device, motion_trust_decoder=True)
        decoder.init_weights()
        torch.manual_seed(53)
        with torch.no_grad():
            torch.nn.init.normal_(
                decoder.motion_trust_adapters[0].weight, std=0.2)
        out_prev = torch.randn(
            2, 7, decoder.embed_dims, device=self.device)
        out_curr = torch.randn_like(out_prev)
        reference_prev = torch.rand(
            2, 7, 5, device=self.device).mul(0.3).add(0.1)
        reference_curr = torch.rand(
            2, 7, 5, device=self.device).mul(0.3).add(0.6)
        layer_output = torch.randn_like(out_prev)
        cls_prev = _build_cls_branches(
            1, decoder.embed_dims, 8, self.device, seed=4)[0]
        cls_curr = _build_cls_branches(
            1, decoder.embed_dims, 8, self.device, seed=5)[0]
        correction = decoder._motion_trust_correction(
            0, out_prev, out_curr, reference_prev, reference_curr,
            layer_output, cls_prev, cls_curr)
        swapped = decoder._motion_trust_correction(
            0, out_curr, out_prev, reference_curr, reference_prev,
            layer_output, cls_curr, cls_prev)
        self.assertTrue(torch.allclose(
            correction, -swapped, atol=1e-6, rtol=1e-5))
        envelope = 0.5 * decoder._reference_motion(
            reference_prev, reference_curr).abs()
        self.assertTrue(torch.all(correction.abs() <= envelope + 1e-7))

    def test_motion_trust_suppresses_unilateral_confidence(self):
        decoder, _, _ = _build_decoder(
            device=self.device, motion_trust_decoder=True)
        decoder.init_weights()
        with torch.no_grad():
            decoder.motion_trust_adapters[0].weight.fill_(0.05)
        out_prev = torch.randn(
            1, 5, decoder.embed_dims, device=self.device)
        out_curr = torch.randn_like(out_prev)
        reference_prev = out_prev.new_full((1, 5, 5), 0.25)
        reference_curr = out_prev.new_full((1, 5, 5), 0.75)
        layer_output = torch.zeros_like(out_prev)
        high_prev = torch.nn.Linear(decoder.embed_dims, 8).to(self.device)
        high_curr = torch.nn.Linear(decoder.embed_dims, 8).to(self.device)
        low_curr = torch.nn.Linear(decoder.embed_dims, 8).to(self.device)
        with torch.no_grad():
            for branch in (high_prev, high_curr, low_curr):
                branch.weight.zero_()
            high_prev.bias.fill_(8.0)
            high_curr.bias.fill_(8.0)
            low_curr.bias.fill_(-8.0)
        high = decoder._motion_trust_correction(
            0, out_prev, out_curr, reference_prev, reference_curr,
            layer_output, high_prev, high_curr)
        unilateral = decoder._motion_trust_correction(
            0, out_prev, out_curr, reference_prev, reference_curr,
            layer_output, high_prev, low_curr)
        self.assertLess(
            unilateral.abs().max().item(), 0.03 * high.abs().max().item())

    def test_motion_trust_adapters_receive_first_step_gradients(self):
        decoder, reg_prev, reg_curr = _build_decoder(
            num_layers=3,
            device=self.device,
            motion_trust_decoder=True)
        decoder.init_weights()
        cls_prev = _build_cls_branches(
            decoder.num_layers, decoder.embed_dims, 8, self.device, seed=6)
        cls_curr = _build_cls_branches(
            decoder.num_layers, decoder.embed_dims, 8, self.device, seed=7)
        spatial_shapes, level_start_index, num_value = _spatial_meta(
            self.device)
        memory_prev, memory_curr = _random_memories(
            1, num_value, decoder.embed_dims, self.device)
        reference_prev = torch.rand(
            1, decoder.num_queries, 5, device=self.device).mul(0.2).add(0.2)
        reference_curr = torch.rand(
            1, decoder.num_queries, 5, device=self.device).mul(0.2).add(0.6)
        _, references_prev, references_curr = decoder(
            memory_prev=memory_prev,
            memory_curr=memory_curr,
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index,
            reg_branches_prev=reg_prev,
            reg_branches_curr=reg_curr,
            cls_branches_prev=cls_prev,
            cls_branches_curr=cls_curr,
            reference_prev=reference_prev,
            reference_curr=reference_curr)
        torch.manual_seed(59)
        loss = sum(
            (prev * torch.randn_like(prev)).mean()
            + (curr * torch.randn_like(curr)).mean()
            for prev, curr in zip(references_prev, references_curr))
        loss.backward()
        for adapter in decoder.motion_trust_adapters:
            self.assertIsNotNone(adapter.weight.grad)
            self.assertGreater(adapter.weight.grad.abs().max().item(), 0.0)

    def test_detector_init_preserves_motion_trust_zero_start(self):
        decoder, _, _ = _build_decoder(
            device=self.device, motion_trust_decoder=True)
        model = MultispecPairRotatedRTDETR.__new__(
            MultispecPairRotatedRTDETR)
        torch.nn.Module.__init__(model)
        model.decoder = decoder

        def detector_level_xavier(model_self):
            for param in model_self.decoder.parameters():
                if param.dim() > 1:
                    torch.nn.init.xavier_uniform_(param)

        with mock.patch.object(
                RotatedRTDETR, 'init_weights', detector_level_xavier):
            model.init_weights()
        for adapter in decoder.motion_trust_adapters:
            self.assertEqual(adapter.weight.abs().sum().item(), 0.0)

    def test_motion_trust_rejects_incompatible_decoders(self):
        for other in (
                dict(tristate_decoder=True),
                dict(dual_output_adapter=True),
                dict(common_motion_decoder=True),
                dict(competitive_evidence_decoder=True),
                dict(shared_routing_decoder=True),
        ):
            with self.assertRaisesRegex(ValueError, 'incompatible'):
                _build_decoder(
                    device=self.device,
                    motion_trust_decoder=True,
                    **other)

    def test_motion_trust_and_shared_evidence_compose(self):
        """Detection-protected motion and shared evidence stay orthogonal."""
        decoder, reg_prev, reg_curr = _build_decoder(
            num_layers=3,
            device=self.device,
            shared_evidence_decoder=True,
            motion_trust_decoder=True)
        decoder.init_weights()
        cls_prev = _build_cls_branches(
            decoder.num_layers, decoder.embed_dims, 8, self.device, seed=8)
        cls_curr = _build_cls_branches(
            decoder.num_layers, decoder.embed_dims, 8, self.device, seed=9)
        spatial_shapes, level_start_index, num_value = _spatial_meta(
            self.device)
        memory_prev, memory_curr = _random_memories(
            1, num_value, decoder.embed_dims, self.device)
        reference_prev = torch.rand(
            1, decoder.num_queries, 5, device=self.device).mul(0.2).add(0.2)
        reference_curr = torch.rand(
            1, decoder.num_queries, 5, device=self.device).mul(0.2).add(0.6)

        combined_output = decoder(
            memory_prev=memory_prev,
            memory_curr=memory_curr,
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index,
            reg_branches_prev=reg_prev,
            reg_branches_curr=reg_curr,
            cls_branches_prev=cls_prev,
            cls_branches_curr=cls_curr,
            reference_prev=reference_prev,
            reference_curr=reference_curr)
        decoder.shared_evidence_decoder = False
        decoder.motion_trust_decoder = False
        baseline_output = decoder(
            memory_prev=memory_prev,
            memory_curr=memory_curr,
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index,
            reg_branches_prev=reg_prev,
            reg_branches_curr=reg_curr,
            reference_prev=reference_prev,
            reference_curr=reference_curr)
        for combined_group, baseline_group in zip(
                combined_output, baseline_output):
            for combined_tensor, baseline_tensor in zip(
                    combined_group, baseline_group):
                self.assertTrue(torch.equal(
                    combined_tensor, baseline_tensor))

        decoder.shared_evidence_decoder = True
        decoder.motion_trust_decoder = True
        torch.manual_seed(60)
        hidden_states, references_prev, references_curr = decoder(
            memory_prev=memory_prev,
            memory_curr=memory_curr,
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index,
            reg_branches_prev=reg_prev,
            reg_branches_curr=reg_curr,
            cls_branches_prev=cls_prev,
            cls_branches_curr=cls_curr,
            reference_prev=reference_prev,
            reference_curr=reference_curr)
        loss = sum(
            (hidden * torch.randn_like(hidden)).mean()
            for hidden in hidden_states)
        loss = loss + sum(
            (prev * torch.randn_like(prev)).mean()
            + (curr * torch.randn_like(curr)).mean()
            for prev, curr in zip(references_prev, references_curr))
        loss.backward()
        for adapter in (
                list(decoder.shared_evidence_adapters)
                + list(decoder.motion_trust_adapters)):
            self.assertIsNotNone(adapter.weight.grad)
            self.assertGreater(adapter.weight.grad.abs().max().item(), 0.0)

    def test_symmetric_pair_decoder_shares_cross_attention(self):
        decoder, _, _ = _build_decoder(
            device=self.device, symmetric_pair_decoder=True)
        for layer in decoder.layers:
            self.assertIs(layer.cross_attn_prev, layer.cross_attn_curr)
            self.assertTrue(layer.symmetric_pair_decoder)

    def test_symmetric_pair_decoder_matches_equal_weight_baseline(self):
        torch.manual_seed(61)
        baseline, reg_prev, reg_curr = _build_decoder(device=self.device)
        baseline.init_weights()
        for layer in baseline.layers:
            layer.cross_attn_curr.load_state_dict(
                layer.cross_attn_prev.state_dict())
        symmetric = copy.deepcopy(baseline)
        symmetric.symmetric_pair_decoder = True
        for layer in symmetric.layers:
            layer.symmetric_pair_decoder = True
            layer.cross_attn_curr = layer.cross_attn_prev

        spatial_shapes, level_start_index, num_value = _spatial_meta(
            self.device)
        memory_prev, memory_curr = _random_memories(
            1, num_value, baseline.embed_dims, self.device)
        reference_prev = torch.rand(
            1, baseline.num_queries, 5, device=self.device).clamp(
                1e-3, 1 - 1e-3)
        reference_curr = torch.rand_like(reference_prev).clamp(
            1e-3, 1 - 1e-3)
        with torch.no_grad():
            baseline_output = baseline(
                memory_prev=memory_prev,
                memory_curr=memory_curr,
                spatial_shapes=spatial_shapes,
                level_start_index=level_start_index,
                reg_branches_prev=reg_prev,
                reg_branches_curr=reg_curr,
                reference_prev=reference_prev,
                reference_curr=reference_curr)
            symmetric_output = symmetric(
                memory_prev=memory_prev,
                memory_curr=memory_curr,
                spatial_shapes=spatial_shapes,
                level_start_index=level_start_index,
                reg_branches_prev=reg_prev,
                reg_branches_curr=reg_curr,
                reference_prev=reference_prev,
                reference_curr=reference_curr)
        for baseline_group, symmetric_group in zip(
                baseline_output, symmetric_output):
            for baseline_tensor, symmetric_tensor in zip(
                    baseline_group, symmetric_group):
                self.assertTrue(torch.equal(
                    baseline_tensor, symmetric_tensor))

    def test_symmetric_pair_decoder_is_frame_swap_equivariant(self):
        decoder, reg_prev, reg_curr = _build_decoder(
            device=self.device, symmetric_pair_decoder=True)
        decoder.init_weights()
        spatial_shapes, level_start_index, num_value = _spatial_meta(
            self.device)
        memory_prev, memory_curr = _random_memories(
            1, num_value, decoder.embed_dims, self.device)
        reference_prev = torch.rand(
            1, decoder.num_queries, 5, device=self.device).clamp(
                1e-3, 1 - 1e-3)
        reference_curr = torch.rand_like(reference_prev).clamp(
            1e-3, 1 - 1e-3)
        with torch.no_grad():
            hidden, refs_prev, refs_curr = decoder(
                memory_prev=memory_prev,
                memory_curr=memory_curr,
                spatial_shapes=spatial_shapes,
                level_start_index=level_start_index,
                reg_branches_prev=reg_prev,
                reg_branches_curr=reg_curr,
                reference_prev=reference_prev,
                reference_curr=reference_curr)
            swapped_hidden, swapped_prev, swapped_curr = decoder(
                memory_prev=memory_curr,
                memory_curr=memory_prev,
                spatial_shapes=spatial_shapes,
                level_start_index=level_start_index,
                reg_branches_prev=reg_curr,
                reg_branches_curr=reg_prev,
                reference_prev=reference_curr,
                reference_curr=reference_prev)
        for original, swapped in zip(hidden, swapped_hidden):
            self.assertTrue(torch.allclose(
                original, swapped, atol=1e-6, rtol=1e-5))
        for original, swapped in zip(refs_prev, swapped_curr):
            self.assertTrue(torch.allclose(
                original, swapped, atol=1e-6, rtol=1e-5))
        for original, swapped in zip(refs_curr, swapped_prev):
            self.assertTrue(torch.allclose(
                original, swapped, atol=1e-6, rtol=1e-5))

    def test_detector_init_preserves_symmetric_pair_structure(self):
        decoder, _, _ = _build_decoder(
            device=self.device, symmetric_pair_decoder=True)
        model = MultispecPairRotatedRTDETR.__new__(
            MultispecPairRotatedRTDETR)
        torch.nn.Module.__init__(model)
        model.decoder = decoder

        def detector_level_xavier(model_self):
            for param in model_self.decoder.parameters():
                if param.dim() > 1:
                    torch.nn.init.xavier_uniform_(param)

        with mock.patch.object(
                RotatedRTDETR, 'init_weights', detector_level_xavier):
            model.init_weights()
        pair_weight = decoder.pair_pos_fusion.weight
        self.assertTrue(torch.equal(
            pair_weight[:, :decoder.embed_dims],
            pair_weight[:, decoder.embed_dims:]))
        for layer in decoder.layers:
            self.assertIs(layer.cross_attn_prev, layer.cross_attn_curr)
            fusion_weight = layer.cross_fusion.weight
            self.assertTrue(torch.equal(
                fusion_weight[:, :decoder.embed_dims],
                fusion_weight[:, decoder.embed_dims:]))

    def test_symmetric_pair_decoder_rejects_other_variants(self):
        for other in (
                dict(tristate_decoder=True),
                dict(dual_output_adapter=True),
                dict(common_motion_decoder=True),
                dict(shared_evidence_decoder=True),
                dict(competitive_evidence_decoder=True),
                dict(motion_trust_decoder=True),
                dict(shared_routing_decoder=True),
                dict(shared_attention_decoder=True),
        ):
            with self.assertRaisesRegex(ValueError, 'incompatible'):
                _build_decoder(
                    device=self.device,
                    symmetric_pair_decoder=True,
                    **other)

    def test_symmetric_position_decoder_only_symmetrizes_position(self):
        decoder, _, _ = _build_decoder(
            device=self.device, symmetric_position_decoder=True)
        decoder.init_weights()
        query_pos_prev = torch.randn(
            2, decoder.num_queries, decoder.embed_dims, device=self.device)
        query_pos_curr = torch.randn_like(query_pos_prev)
        fused = decoder._fuse_pair_position(
            query_pos_prev, query_pos_curr)
        swapped = decoder._fuse_pair_position(
            query_pos_curr, query_pos_prev)
        self.assertTrue(torch.equal(fused, swapped))
        for layer in decoder.layers:
            self.assertIsNot(layer.cross_attn_prev, layer.cross_attn_curr)
            self.assertFalse(layer.symmetric_pair_decoder)

    def test_symmetric_position_decoder_preserves_parent_at_init(self):
        torch.manual_seed(63)
        baseline, _, _ = _build_decoder(device=self.device)
        baseline.init_weights()
        symmetric = copy.deepcopy(baseline)
        symmetric.symmetric_position_decoder = True
        query_pos_prev = torch.randn(
            2, baseline.num_queries, baseline.embed_dims, device=self.device)
        query_pos_curr = torch.randn_like(query_pos_prev)
        baseline_fused = baseline._fuse_pair_position(
            query_pos_prev, query_pos_curr)
        symmetric_fused = symmetric._fuse_pair_position(
            query_pos_prev, query_pos_curr)
        self.assertTrue(torch.allclose(
            baseline_fused, symmetric_fused, atol=1e-6, rtol=1e-6))

    def test_symmetric_position_decoder_rejects_full_symmetry(self):
        with self.assertRaisesRegex(ValueError, 'incompatible'):
            _build_decoder(
                device=self.device,
                symmetric_pair_decoder=True,
                symmetric_position_decoder=True)

    def test_symmetric_feature_decoder_only_symmetrizes_feature_fusion(self):
        decoder, _, _ = _build_decoder(
            device=self.device, symmetric_feature_decoder=True)
        decoder.init_weights()
        for layer in decoder.layers:
            out_prev = torch.randn(
                2, decoder.num_queries, decoder.embed_dims,
                device=self.device)
            out_curr = torch.randn_like(out_prev)
            fused = layer._fuse_frame_features(out_prev, out_curr)
            swapped = layer._fuse_frame_features(out_curr, out_prev)
            self.assertTrue(torch.equal(fused, swapped))
            self.assertIsNot(layer.cross_attn_prev, layer.cross_attn_curr)
            self.assertFalse(layer.symmetric_pair_decoder)
            self.assertTrue(layer.symmetric_feature_decoder)

    def test_symmetric_feature_decoder_preserves_parent_at_init(self):
        torch.manual_seed(65)
        baseline, _, _ = _build_decoder(device=self.device)
        baseline.init_weights()
        symmetric = copy.deepcopy(baseline)
        symmetric.symmetric_feature_decoder = True
        for layer in symmetric.layers:
            layer.symmetric_feature_decoder = True
        for baseline_layer, symmetric_layer in zip(
                baseline.layers, symmetric.layers):
            out_prev = torch.randn(
                2, baseline.num_queries, baseline.embed_dims,
                device=self.device)
            out_curr = torch.randn_like(out_prev)
            baseline_fused = baseline_layer._fuse_frame_features(
                out_prev, out_curr)
            symmetric_fused = symmetric_layer._fuse_frame_features(
                out_prev, out_curr)
            self.assertTrue(torch.allclose(
                baseline_fused, symmetric_fused, atol=1e-6, rtol=1e-6))

    def test_symmetric_feature_decoder_rejects_other_variants(self):
        for other in (
                dict(symmetric_pair_decoder=True),
                dict(symmetric_position_decoder=True),
                dict(shared_attention_decoder=True),
                dict(terminal_factorized_evidence_decoder=True),
        ):
            with self.assertRaisesRegex(ValueError, 'incompatible'):
                _build_decoder(
                    device=self.device,
                    symmetric_feature_decoder=True,
                    **other)

    def test_residual_preserving_fusion_matches_parent_at_init(self):
        torch.manual_seed(66)
        baseline, _, _ = _build_decoder(device=self.device)
        baseline.init_weights()
        residual = copy.deepcopy(baseline)
        residual.residual_preserving_fusion_decoder = True
        for layer in residual.layers:
            layer.residual_preserving_fusion_decoder = True
        for baseline_layer, residual_layer in zip(
                baseline.layers, residual.layers):
            shared_query = torch.randn(
                2, baseline.num_queries, baseline.embed_dims,
                device=self.device)
            out_prev = torch.randn_like(shared_query)
            out_curr = torch.randn_like(shared_query)
            baseline_fused = baseline_layer._fuse_frame_features(
                out_prev, out_curr, shared_query=shared_query)
            residual_fused = residual_layer._fuse_frame_features(
                out_prev, out_curr, shared_query=shared_query)
            self.assertTrue(torch.allclose(
                baseline_fused, residual_fused, atol=1e-6, rtol=1e-6))

    def test_residual_preserving_fusion_keeps_explicit_query_identity(self):
        decoder, _, _ = _build_decoder(
            device=self.device,
            symmetric_position_decoder=True,
            residual_preserving_fusion_decoder=True)
        decoder.init_weights()
        for layer in decoder.layers:
            shared_query = torch.randn(
                2, decoder.num_queries, decoder.embed_dims,
                device=self.device)
            out_prev = torch.randn_like(shared_query)
            out_curr = torch.randn_like(shared_query)
            with torch.no_grad():
                torch.nn.init.normal_(layer.cross_fusion.weight, std=0.03)
                torch.nn.init.normal_(layer.cross_fusion.bias, std=0.01)
            fused = layer._fuse_frame_features(
                out_prev, out_curr, shared_query=shared_query)
            expected = shared_query + layer.cross_fusion(torch.cat([
                out_prev - shared_query,
                out_curr - shared_query,
            ], dim=-1))
            self.assertTrue(torch.equal(fused, expected))
            self.assertTrue(layer.residual_preserving_fusion_decoder)
            self.assertIsNot(layer.cross_attn_prev, layer.cross_attn_curr)

    def test_residual_preserving_fusion_requires_shared_query(self):
        decoder, _, _ = _build_decoder(
            device=self.device,
            residual_preserving_fusion_decoder=True)
        layer = decoder.layers[0]
        out_prev = torch.randn(
            2, decoder.num_queries, decoder.embed_dims,
            device=self.device)
        with self.assertRaisesRegex(ValueError, 'shared_query is required'):
            layer._fuse_frame_features(out_prev, torch.randn_like(out_prev))

    def test_residual_preserving_fusion_rejects_other_variants(self):
        for other in (
                dict(symmetric_pair_decoder=True),
                dict(symmetric_feature_decoder=True),
                dict(shared_attention_decoder=True),
                dict(terminal_factorized_evidence_decoder=True),
        ):
            with self.assertRaisesRegex(ValueError, 'incompatible'):
                _build_decoder(
                    device=self.device,
                    residual_preserving_fusion_decoder=True,
                    **other)

    def test_shared_routing_ties_only_sampling_policy(self):
        decoder, _, _ = _build_decoder(
            device=self.device, shared_routing_decoder=True)
        for layer in decoder.layers:
            prev = layer.cross_attn_prev
            curr = layer.cross_attn_curr
            self.assertIs(prev.sampling_offsets, curr.sampling_offsets)
            self.assertIs(prev.attention_weights, curr.attention_weights)
            self.assertIsNot(prev.value_proj, curr.value_proj)
            self.assertIsNot(prev.output_proj, curr.output_proj)
            self.assertTrue(prev.value_proj.weight.requires_grad)
            self.assertTrue(curr.value_proj.weight.requires_grad)
            self.assertTrue(prev.output_proj.weight.requires_grad)
            self.assertTrue(curr.output_proj.weight.requires_grad)

    def test_shared_routing_matches_equal_routing_baseline(self):
        torch.manual_seed(67)
        baseline, reg_prev, reg_curr = _build_decoder(device=self.device)
        baseline.init_weights()
        for layer in baseline.layers:
            layer.cross_attn_curr.sampling_offsets.load_state_dict(
                layer.cross_attn_prev.sampling_offsets.state_dict())
            layer.cross_attn_curr.attention_weights.load_state_dict(
                layer.cross_attn_prev.attention_weights.state_dict())
        shared = copy.deepcopy(baseline)
        shared.shared_routing_decoder = True
        for layer in shared.layers:
            layer.cross_attn_curr.sampling_offsets = (
                layer.cross_attn_prev.sampling_offsets)
            layer.cross_attn_curr.attention_weights = (
                layer.cross_attn_prev.attention_weights)

        spatial_shapes, level_start_index, num_value = _spatial_meta(
            self.device)
        memory_prev, memory_curr = _random_memories(
            1, num_value, baseline.embed_dims, self.device)
        reference_prev = torch.rand(
            1, baseline.num_queries, 5, device=self.device).clamp(
                1e-3, 1 - 1e-3)
        reference_curr = torch.rand_like(reference_prev).clamp(
            1e-3, 1 - 1e-3)
        with torch.no_grad():
            baseline_output = baseline(
                memory_prev=memory_prev,
                memory_curr=memory_curr,
                spatial_shapes=spatial_shapes,
                level_start_index=level_start_index,
                reg_branches_prev=reg_prev,
                reg_branches_curr=reg_curr,
                reference_prev=reference_prev,
                reference_curr=reference_curr)
            shared_output = shared(
                memory_prev=memory_prev,
                memory_curr=memory_curr,
                spatial_shapes=spatial_shapes,
                level_start_index=level_start_index,
                reg_branches_prev=reg_prev,
                reg_branches_curr=reg_curr,
                reference_prev=reference_prev,
                reference_curr=reference_curr)
        for baseline_group, shared_group in zip(
                baseline_output, shared_output):
            for baseline_tensor, shared_tensor in zip(
                    baseline_group, shared_group):
                self.assertTrue(torch.equal(
                    baseline_tensor, shared_tensor))

    def test_detector_init_preserves_shared_routing_structure(self):
        decoder, _, _ = _build_decoder(
            device=self.device, shared_routing_decoder=True)
        model = MultispecPairRotatedRTDETR.__new__(
            MultispecPairRotatedRTDETR)
        torch.nn.Module.__init__(model)
        model.decoder = decoder

        def detector_level_xavier(model_self):
            for param in model_self.decoder.parameters():
                if param.dim() > 1:
                    torch.nn.init.xavier_uniform_(param)

        with mock.patch.object(
                RotatedRTDETR, 'init_weights', detector_level_xavier):
            model.init_weights()
        for layer in decoder.layers:
            prev = layer.cross_attn_prev
            curr = layer.cross_attn_curr
            self.assertIs(prev.sampling_offsets, curr.sampling_offsets)
            self.assertIs(prev.attention_weights, curr.attention_weights)
            self.assertIsNot(prev.value_proj, curr.value_proj)
            self.assertIsNot(prev.output_proj, curr.output_proj)

    def test_shared_routing_rejects_other_variants(self):
        for other in (
                dict(tristate_decoder=True),
                dict(dual_output_adapter=True),
                dict(common_motion_decoder=True),
                dict(shared_evidence_decoder=True),
                dict(competitive_evidence_decoder=True),
                dict(motion_trust_decoder=True),
                dict(symmetric_pair_decoder=True),
                dict(shared_attention_decoder=True),
        ):
            with self.assertRaisesRegex(ValueError, 'incompatible'):
                _build_decoder(
                    device=self.device,
                    shared_routing_decoder=True,
                    **other)

    def test_shared_attention_ties_only_attention_weights(self):
        decoder, _, _ = _build_decoder(
            device=self.device, shared_attention_decoder=True)
        for layer in decoder.layers:
            prev = layer.cross_attn_prev
            curr = layer.cross_attn_curr
            self.assertIsNot(prev.sampling_offsets, curr.sampling_offsets)
            self.assertIs(prev.attention_weights, curr.attention_weights)
            self.assertIsNot(prev.value_proj, curr.value_proj)
            self.assertIsNot(prev.output_proj, curr.output_proj)
            self.assertTrue(prev.sampling_offsets.weight.requires_grad)
            self.assertTrue(curr.sampling_offsets.weight.requires_grad)

    def test_shared_attention_matches_equal_attention_baseline(self):
        torch.manual_seed(68)
        baseline, reg_prev, reg_curr = _build_decoder(device=self.device)
        baseline.init_weights()
        for layer in baseline.layers:
            layer.cross_attn_curr.attention_weights.load_state_dict(
                layer.cross_attn_prev.attention_weights.state_dict())
        shared = copy.deepcopy(baseline)
        shared.shared_attention_decoder = True
        for layer in shared.layers:
            layer.cross_attn_curr.attention_weights = (
                layer.cross_attn_prev.attention_weights)

        spatial_shapes, level_start_index, num_value = _spatial_meta(
            self.device)
        memory_prev, memory_curr = _random_memories(
            1, num_value, baseline.embed_dims, self.device)
        reference_prev = torch.rand(
            1, baseline.num_queries, 5, device=self.device).clamp(
                1e-3, 1 - 1e-3)
        reference_curr = torch.rand_like(reference_prev).clamp(
                1e-3, 1 - 1e-3)
        with torch.no_grad():
            baseline_output = baseline(
                memory_prev=memory_prev,
                memory_curr=memory_curr,
                spatial_shapes=spatial_shapes,
                level_start_index=level_start_index,
                reg_branches_prev=reg_prev,
                reg_branches_curr=reg_curr,
                reference_prev=reference_prev,
                reference_curr=reference_curr)
            shared_output = shared(
                memory_prev=memory_prev,
                memory_curr=memory_curr,
                spatial_shapes=spatial_shapes,
                level_start_index=level_start_index,
                reg_branches_prev=reg_prev,
                reg_branches_curr=reg_curr,
                reference_prev=reference_prev,
                reference_curr=reference_curr)
        for baseline_group, shared_group in zip(
                baseline_output, shared_output):
            for baseline_tensor, shared_tensor in zip(
                    baseline_group, shared_group):
                self.assertTrue(torch.equal(
                    baseline_tensor, shared_tensor))

    def test_detector_init_preserves_shared_attention_structure(self):
        decoder, _, _ = _build_decoder(
            device=self.device, shared_attention_decoder=True)
        model = MultispecPairRotatedRTDETR.__new__(
            MultispecPairRotatedRTDETR)
        torch.nn.Module.__init__(model)
        model.decoder = decoder

        def detector_level_xavier(model_self):
            for param in model_self.decoder.parameters():
                if param.dim() > 1:
                    torch.nn.init.xavier_uniform_(param)

        with mock.patch.object(
                RotatedRTDETR, 'init_weights', detector_level_xavier):
            model.init_weights()
        for layer in decoder.layers:
            prev = layer.cross_attn_prev
            curr = layer.cross_attn_curr
            self.assertIsNot(prev.sampling_offsets, curr.sampling_offsets)
            self.assertIs(prev.attention_weights, curr.attention_weights)
            self.assertIsNot(prev.value_proj, curr.value_proj)
            self.assertIsNot(prev.output_proj, curr.output_proj)

    def test_shared_attention_rejects_other_variants(self):
        for other in (
                dict(tristate_decoder=True),
                dict(dual_output_adapter=True),
                dict(common_motion_decoder=True),
                dict(competitive_evidence_decoder=True),
                dict(symmetric_pair_decoder=True),
                dict(shared_routing_decoder=True),
        ):
            with self.assertRaisesRegex(ValueError, 'incompatible'):
                _build_decoder(
                    device=self.device,
                    shared_attention_decoder=True,
                    **other)

    def test_motion_trust_and_shared_attention_compose(self):
        decoder, _, _ = _build_decoder(
            device=self.device,
            motion_trust_decoder=True,
            shared_attention_decoder=True)
        decoder.init_weights()
        for layer in decoder.layers:
            self.assertIs(
                layer.cross_attn_prev.attention_weights,
                layer.cross_attn_curr.attention_weights)
            self.assertIsNot(
                layer.cross_attn_prev.sampling_offsets,
                layer.cross_attn_curr.sampling_offsets)
        for adapter in decoder.motion_trust_adapters:
            self.assertEqual(adapter.weight.abs().sum().item(), 0.0)

    def test_shared_evidence_and_shared_attention_compose(self):
        decoder, _, _ = _build_decoder(
            device=self.device,
            shared_evidence_decoder=True,
            shared_attention_decoder=True)
        decoder.init_weights()
        for layer in decoder.layers:
            self.assertIs(
                layer.cross_attn_prev.attention_weights,
                layer.cross_attn_curr.attention_weights)
            self.assertIsNot(
                layer.cross_attn_prev.sampling_offsets,
                layer.cross_attn_curr.sampling_offsets)
        for adapter in decoder.shared_evidence_adapters:
            self.assertEqual(adapter.weight.abs().sum().item(), 0.0)

    def test_antisymmetric_detail_is_exact_baseline_at_zero_start(self):
        decoder, reg_prev, reg_curr = _build_decoder(
            device=self.device, antisymmetric_detail_decoder=True)
        decoder.init_weights()
        spatial_shapes, level_start_index, num_value = _spatial_meta(
            self.device)
        memory_prev, memory_curr = _random_memories(
            1, num_value, decoder.embed_dims, self.device)
        detail_output = decoder(
            memory_prev=memory_prev,
            memory_curr=memory_curr,
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index,
            reg_branches_prev=reg_prev,
            reg_branches_curr=reg_curr)
        hidden, _, _, hidden_prev, hidden_curr = detail_output
        for shared, prev, curr in zip(hidden, hidden_prev, hidden_curr):
            self.assertTrue(torch.equal(prev, shared))
            self.assertTrue(torch.equal(curr, shared))

        decoder.antisymmetric_detail_decoder = False
        baseline_output = decoder(
            memory_prev=memory_prev,
            memory_curr=memory_curr,
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index,
            reg_branches_prev=reg_prev,
            reg_branches_curr=reg_curr)
        for detail_group, baseline_group in zip(
                detail_output[:3], baseline_output):
            for detail_tensor, baseline_tensor in zip(
                    detail_group, baseline_group):
                self.assertTrue(torch.equal(detail_tensor, baseline_tensor))
        for adapter in decoder.antisymmetric_detail_adapters:
            self.assertEqual(adapter.weight.abs().sum().item(), 0.0)

    def test_antisymmetric_detail_is_swap_odd_bounded_and_midpoint_preserving(
            self):
        decoder, reg_prev, reg_curr = _build_decoder(
            device=self.device, antisymmetric_detail_decoder=True)
        decoder.init_weights()
        torch.manual_seed(71)
        with torch.no_grad():
            for adapter in decoder.antisymmetric_detail_adapters:
                torch.nn.init.normal_(adapter.weight, std=0.05)
        out_prev = torch.randn(
            2, 7, decoder.embed_dims, device=self.device)
        out_curr = torch.randn_like(out_prev)
        correction = decoder._antisymmetric_detail_correction(
            0, out_prev, out_curr)
        swapped = decoder._antisymmetric_detail_correction(
            0, out_curr, out_prev)
        self.assertTrue(torch.allclose(correction, -swapped, atol=1e-6))
        self.assertLessEqual(correction.abs().max().item(), 1.0)

        spatial_shapes, level_start_index, num_value = _spatial_meta(
            self.device)
        memory_prev, memory_curr = _random_memories(
            1, num_value, decoder.embed_dims, self.device)
        hidden, _, _, hidden_prev, hidden_curr = decoder(
            memory_prev=memory_prev,
            memory_curr=memory_curr,
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index,
            reg_branches_prev=reg_prev,
            reg_branches_curr=reg_curr)
        for shared, prev, curr in zip(hidden, hidden_prev, hidden_curr):
            self.assertTrue(torch.allclose(
                0.5 * (prev + curr), shared, atol=1e-6))

    def test_antisymmetric_detail_adapters_receive_first_step_gradients(self):
        decoder, reg_prev, reg_curr = _build_decoder(
            num_layers=3,
            device=self.device,
            antisymmetric_detail_decoder=True)
        decoder.init_weights()
        spatial_shapes, level_start_index, num_value = _spatial_meta(
            self.device)
        memory_prev, memory_curr = _random_memories(
            1, num_value, decoder.embed_dims, self.device)
        _, references_prev, references_curr, hidden_prev, hidden_curr = decoder(
            memory_prev=memory_prev,
            memory_curr=memory_curr,
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index,
            reg_branches_prev=reg_prev,
            reg_branches_curr=reg_curr)
        torch.manual_seed(73)
        loss = sum(
            (prev * torch.randn_like(prev)).mean()
            + (curr * torch.randn_like(curr)).mean()
            for prev, curr in zip(hidden_prev, hidden_curr))
        loss = loss + sum(
            (prev * torch.randn_like(prev)).mean()
            + (curr * torch.randn_like(curr)).mean()
            for prev, curr in zip(references_prev, references_curr))
        loss.backward()
        for adapter in decoder.antisymmetric_detail_adapters:
            self.assertIsNotNone(adapter.weight.grad)
            self.assertGreater(adapter.weight.grad.abs().max().item(), 0.0)

    def test_detector_init_preserves_antisymmetric_detail_zero_start(self):
        decoder, _, _ = _build_decoder(
            device=self.device, antisymmetric_detail_decoder=True)
        model = MultispecPairRotatedRTDETR.__new__(
            MultispecPairRotatedRTDETR)
        torch.nn.Module.__init__(model)
        model.decoder = decoder

        def detector_level_xavier(model_self):
            for param in model_self.decoder.parameters():
                if param.dim() > 1:
                    torch.nn.init.xavier_uniform_(param)

        with mock.patch.object(
                RotatedRTDETR, 'init_weights', detector_level_xavier):
            model.init_weights()
        for adapter in decoder.antisymmetric_detail_adapters:
            self.assertEqual(adapter.weight.abs().sum().item(), 0.0)

    def test_antisymmetric_detail_rejects_competing_decoder_variants(self):
        for other in (
                dict(tristate_decoder=True),
                dict(dual_output_adapter=True),
                dict(common_motion_decoder=True),
                dict(shared_evidence_decoder=True),
                dict(competitive_evidence_decoder=True),
                dict(motion_trust_decoder=True),
                dict(symmetric_pair_decoder=True),
                dict(shared_routing_decoder=True),
        ):
            with self.assertRaisesRegex(
                    ValueError, 'mutually exclusive|incompatible'):
                _build_decoder(
                    device=self.device,
                    antisymmetric_detail_decoder=True,
                    **other)

    def test_antisymmetric_detail_and_shared_attention_compose(self):
        decoder, _, _ = _build_decoder(
            device=self.device,
            antisymmetric_detail_decoder=True,
            shared_attention_decoder=True)
        decoder.init_weights()
        for layer in decoder.layers:
            self.assertIs(
                layer.cross_attn_prev.attention_weights,
                layer.cross_attn_curr.attention_weights)
            self.assertIsNot(
                layer.cross_attn_prev.sampling_offsets,
                layer.cross_attn_curr.sampling_offsets)
        for adapter in decoder.antisymmetric_detail_adapters:
            self.assertEqual(adapter.weight.abs().sum().item(), 0.0)

    def test_enveloped_detail_is_exact_baseline_at_zero_start(self):
        decoder, reg_prev, reg_curr = _build_decoder(
            device=self.device, enveloped_detail_decoder=True)
        decoder.init_weights()
        spatial_shapes, level_start_index, num_value = _spatial_meta(
            self.device)
        memory_prev, memory_curr = _random_memories(
            1, num_value, decoder.embed_dims, self.device)
        detail_output = decoder(
            memory_prev=memory_prev,
            memory_curr=memory_curr,
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index,
            reg_branches_prev=reg_prev,
            reg_branches_curr=reg_curr)
        hidden, _, _, hidden_prev, hidden_curr = detail_output
        for shared, prev, curr in zip(hidden, hidden_prev, hidden_curr):
            self.assertTrue(torch.equal(prev, shared))
            self.assertTrue(torch.equal(curr, shared))

        decoder.enveloped_detail_decoder = False
        baseline_output = decoder(
            memory_prev=memory_prev,
            memory_curr=memory_curr,
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index,
            reg_branches_prev=reg_prev,
            reg_branches_curr=reg_curr)
        for detail_group, baseline_group in zip(
                detail_output[:3], baseline_output):
            for detail_tensor, baseline_tensor in zip(
                    detail_group, baseline_group):
                self.assertTrue(torch.equal(detail_tensor, baseline_tensor))

    def test_enveloped_detail_and_shared_attention_compose(self):
        decoder, reg_prev, reg_curr = _build_decoder(
            device=self.device,
            shared_attention_decoder=True,
            enveloped_detail_decoder=True)
        decoder.init_weights()
        for layer in decoder.layers:
            self.assertIs(
                layer.cross_attn_prev.attention_weights,
                layer.cross_attn_curr.attention_weights)
            self.assertIsNot(
                layer.cross_attn_prev.sampling_offsets,
                layer.cross_attn_curr.sampling_offsets)
        for gate in decoder.enveloped_detail_gates:
            self.assertEqual(gate.weight.abs().sum().item(), 0.0)

        spatial_shapes, level_start_index, num_value = _spatial_meta(
            self.device)
        memory_prev, memory_curr = _random_memories(
            1, num_value, decoder.embed_dims, self.device)
        composed_output = decoder(
            memory_prev=memory_prev,
            memory_curr=memory_curr,
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index,
            reg_branches_prev=reg_prev,
            reg_branches_curr=reg_curr)
        decoder.enveloped_detail_decoder = False
        shared_attention_output = decoder(
            memory_prev=memory_prev,
            memory_curr=memory_curr,
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index,
            reg_branches_prev=reg_prev,
            reg_branches_curr=reg_curr)
        for composed_group, baseline_group in zip(
                composed_output[:3], shared_attention_output):
            for composed_tensor, baseline_tensor in zip(
                    composed_group, baseline_group):
                self.assertTrue(torch.equal(
                    composed_tensor, baseline_tensor))

    def test_enveloped_detail_is_swap_odd_and_evidence_bounded(self):
        decoder, _, _ = _build_decoder(
            device=self.device, enveloped_detail_decoder=True)
        decoder.init_weights()
        torch.manual_seed(79)
        with torch.no_grad():
            for gate in decoder.enveloped_detail_gates:
                torch.nn.init.normal_(gate.weight, std=0.05)
        out_prev = torch.randn(
            2, 7, decoder.embed_dims, device=self.device)
        out_curr = torch.randn_like(out_prev)
        correction = decoder._enveloped_detail_correction(
            0, out_prev, out_curr)
        swapped = decoder._enveloped_detail_correction(
            0, out_curr, out_prev)
        envelope = 0.5 * (out_curr - out_prev).abs()
        self.assertTrue(torch.allclose(correction, -swapped, atol=1e-6))
        self.assertTrue(torch.all(correction.abs() <= envelope + 1e-7))

    def test_regression_enveloped_detail_preserves_shared_cls_states(self):
        decoder, reg_prev, reg_curr = _build_decoder(
            device=self.device,
            shared_attention_decoder=True,
            regression_enveloped_detail_decoder=True)
        decoder.init_weights()
        for layer in decoder.layers:
            self.assertIs(
                layer.cross_attn_prev.attention_weights,
                layer.cross_attn_curr.attention_weights)
            self.assertIsNot(
                layer.cross_attn_prev.sampling_offsets,
                layer.cross_attn_curr.sampling_offsets)
        for gate in decoder.enveloped_detail_gates:
            self.assertEqual(gate.weight.abs().sum().item(), 0.0)

        spatial_shapes, level_start_index, num_value = _spatial_meta(
            self.device)
        memory_prev, memory_curr = _random_memories(
            1, num_value, decoder.embed_dims, self.device)
        regression_detail_output = decoder(
            memory_prev=memory_prev,
            memory_curr=memory_curr,
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index,
            reg_branches_prev=reg_prev,
            reg_branches_curr=reg_curr)
        self.assertEqual(len(regression_detail_output), 3)
        decoder.regression_enveloped_detail_decoder = False
        shared_attention_output = decoder(
            memory_prev=memory_prev,
            memory_curr=memory_curr,
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index,
            reg_branches_prev=reg_prev,
            reg_branches_curr=reg_curr)
        for detail_group, baseline_group in zip(
                regression_detail_output, shared_attention_output):
            for detail_tensor, baseline_tensor in zip(
                    detail_group, baseline_group):
                self.assertTrue(torch.equal(
                    detail_tensor, baseline_tensor))

        decoder.regression_enveloped_detail_decoder = True
        _, references_prev, references_curr = decoder(
            memory_prev=memory_prev,
            memory_curr=memory_curr,
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index,
            reg_branches_prev=reg_prev,
            reg_branches_curr=reg_curr)
        torch.manual_seed(89)
        loss = sum(
            (prev * torch.randn_like(prev)).mean()
            + (curr * torch.randn_like(curr)).mean()
            for prev, curr in zip(references_prev, references_curr))
        loss.backward()
        for gate in decoder.enveloped_detail_gates:
            self.assertIsNotNone(gate.weight.grad)
            self.assertGreater(gate.weight.grad.abs().max().item(), 0.0)

    def test_midpoint_regression_detail_is_zero_start_and_trainable(self):
        decoder, reg_prev, reg_curr = _build_decoder(
            device=self.device,
            shared_attention_decoder=True,
            midpoint_regression_enveloped_detail_decoder=True)
        decoder.init_weights()
        for gate in decoder.enveloped_detail_gates:
            self.assertEqual(gate.weight.abs().sum().item(), 0.0)

        spatial_shapes, level_start_index, num_value = _spatial_meta(
            self.device)
        memory_prev, memory_curr = _random_memories(
            1, num_value, decoder.embed_dims, self.device)
        midpoint_output = decoder(
            memory_prev=memory_prev,
            memory_curr=memory_curr,
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index,
            reg_branches_prev=reg_prev,
            reg_branches_curr=reg_curr)
        self.assertEqual(len(midpoint_output), 3)
        decoder.midpoint_regression_enveloped_detail_decoder = False
        baseline_output = decoder(
            memory_prev=memory_prev,
            memory_curr=memory_curr,
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index,
            reg_branches_prev=reg_prev,
            reg_branches_curr=reg_curr)
        for detail_group, baseline_group in zip(
                midpoint_output, baseline_output):
            for detail_tensor, baseline_tensor in zip(
                    detail_group, baseline_group):
                self.assertTrue(torch.equal(
                    detail_tensor, baseline_tensor))

        decoder.midpoint_regression_enveloped_detail_decoder = True
        _, references_prev, references_curr = decoder(
            memory_prev=memory_prev,
            memory_curr=memory_curr,
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index,
            reg_branches_prev=reg_prev,
            reg_branches_curr=reg_curr)
        torch.manual_seed(113)
        loss = sum(
            (prev * torch.randn_like(prev)).mean()
            + (curr * torch.randn_like(curr)).mean()
            for prev, curr in zip(references_prev, references_curr))
        loss.backward()
        for gate in decoder.enveloped_detail_gates:
            self.assertIsNotNone(gate.weight.grad)
            self.assertGreater(gate.weight.grad.abs().max().item(), 0.0)

    def test_midpoint_regression_detail_preserves_box_logit_midpoint(self):
        decoder, reg_prev, reg_curr = _build_decoder(
            num_layers=1,
            device=self.device,
            shared_attention_decoder=True,
            midpoint_regression_enveloped_detail_decoder=True)
        decoder.init_weights()
        for gate in decoder.enveloped_detail_gates:
            torch.nn.init.normal_(gate.weight, std=0.05)

        spatial_shapes, level_start_index, num_value = _spatial_meta(
            self.device)
        memory_prev, memory_curr = _random_memories(
            1, num_value, decoder.embed_dims, self.device)
        torch.manual_seed(127)
        query = torch.randn(
            1, decoder.num_queries, decoder.embed_dims, device=self.device)
        reference_prev = torch.rand(
            1, decoder.num_queries, 5, device=self.device).mul_(0.6).add_(0.2)
        reference_curr = torch.rand(
            1, decoder.num_queries, 5, device=self.device).mul_(0.6).add_(0.2)

        detail_output = decoder(
            memory_prev=memory_prev,
            memory_curr=memory_curr,
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index,
            reg_branches_prev=reg_prev,
            reg_branches_curr=reg_curr,
            query=query,
            reference_prev=reference_prev,
            reference_curr=reference_curr)
        decoder.midpoint_regression_enveloped_detail_decoder = False
        baseline_output = decoder(
            memory_prev=memory_prev,
            memory_curr=memory_curr,
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index,
            reg_branches_prev=reg_prev,
            reg_branches_curr=reg_curr,
            query=query,
            reference_prev=reference_prev,
            reference_curr=reference_curr)

        detail_prev = torch.logit(
            detail_output[1][0].clamp(1e-6, 1 - 1e-6))
        detail_curr = torch.logit(
            detail_output[2][0].clamp(1e-6, 1 - 1e-6))
        baseline_prev = torch.logit(
            baseline_output[1][0].clamp(1e-6, 1 - 1e-6))
        baseline_curr = torch.logit(
            baseline_output[2][0].clamp(1e-6, 1 - 1e-6))
        detail_delta_sum = (
            detail_prev - baseline_prev + detail_curr - baseline_curr)
        self.assertTrue(torch.allclose(
            detail_delta_sum,
            torch.zeros_like(detail_delta_sum),
            atol=2e-5,
            rtol=2e-5))
        for detail_hidden, baseline_hidden in zip(
                detail_output[0], baseline_output[0]):
            self.assertTrue(torch.equal(detail_hidden, baseline_hidden))

    def test_classification_enveloped_detail_preserves_box_path(self):
        decoder, reg_prev, reg_curr = _build_decoder(
            device=self.device,
            classification_enveloped_detail_decoder=True)
        decoder.init_weights()
        for gate in decoder.enveloped_detail_gates:
            self.assertEqual(gate.weight.abs().sum().item(), 0.0)

        spatial_shapes, level_start_index, num_value = _spatial_meta(
            self.device)
        memory_prev, memory_curr = _random_memories(
            1, num_value, decoder.embed_dims, self.device)
        zero_start_output = decoder(
            memory_prev=memory_prev,
            memory_curr=memory_curr,
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index,
            reg_branches_prev=reg_prev,
            reg_branches_curr=reg_curr)
        self.assertEqual(len(zero_start_output), 5)
        torch.manual_seed(103)
        cls_path_loss = sum(
            (prev * torch.randn_like(prev)).mean()
            + (curr * torch.randn_like(curr)).mean()
            for prev, curr in zip(
                zero_start_output[3], zero_start_output[4]))
        cls_path_loss.backward()
        for gate in decoder.enveloped_detail_gates:
            self.assertIsNotNone(gate.weight.grad)
            self.assertGreater(gate.weight.grad.abs().max().item(), 0.0)

        decoder.classification_enveloped_detail_decoder = False
        baseline_output = decoder(
            memory_prev=memory_prev,
            memory_curr=memory_curr,
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index,
            reg_branches_prev=reg_prev,
            reg_branches_curr=reg_curr)
        for detail_group, baseline_group in zip(
                zero_start_output[:3], baseline_output):
            for detail_tensor, baseline_tensor in zip(
                    detail_group, baseline_group):
                self.assertTrue(torch.equal(
                    detail_tensor, baseline_tensor))
        for detail_group in zero_start_output[3:]:
            for detail_tensor, shared_tensor in zip(
                    detail_group, baseline_output[0]):
                self.assertTrue(torch.equal(detail_tensor, shared_tensor))

        decoder.classification_enveloped_detail_decoder = True
        with torch.no_grad():
            for gate in decoder.enveloped_detail_gates:
                torch.nn.init.normal_(gate.weight, std=0.05)
        detail_output = decoder(
            memory_prev=memory_prev,
            memory_curr=memory_curr,
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index,
            reg_branches_prev=reg_prev,
            reg_branches_curr=reg_curr)
        decoder.classification_enveloped_detail_decoder = False
        nonzero_baseline = decoder(
            memory_prev=memory_prev,
            memory_curr=memory_curr,
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index,
            reg_branches_prev=reg_prev,
            reg_branches_curr=reg_curr)
        for detail_refs, baseline_refs in zip(
                detail_output[1:3], nonzero_baseline[1:3]):
            for detail_tensor, baseline_tensor in zip(
                    detail_refs, baseline_refs):
                self.assertTrue(torch.equal(
                    detail_tensor, baseline_tensor))
        self.assertTrue(any(
            not torch.equal(detail_tensor, shared_tensor)
            for detail_group in detail_output[3:]
            for detail_tensor, shared_tensor in zip(
                detail_group, nonzero_baseline[0])))

    def test_classification_enveloped_detail_composes_with_shared_attention(
            self):
        decoder, _, _ = _build_decoder(
            device=self.device,
            shared_attention_decoder=True,
            classification_enveloped_detail_decoder=True)
        decoder.init_weights()
        for layer in decoder.layers:
            self.assertIs(
                layer.cross_attn_prev.attention_weights,
                layer.cross_attn_curr.attention_weights)
            self.assertIsNot(
                layer.cross_attn_prev.sampling_offsets,
                layer.cross_attn_curr.sampling_offsets)

    def test_terminal_enveloped_detail_is_exact_until_final_layer(self):
        decoder, reg_prev, reg_curr = _build_decoder(
            num_layers=3,
            device=self.device,
            shared_attention_decoder=True,
            terminal_enveloped_detail_decoder=True)
        decoder.init_weights()
        self.assertEqual(len(decoder.terminal_enveloped_detail_gates), 1)
        terminal_gate = decoder.terminal_enveloped_detail_gates[0]
        self.assertEqual(terminal_gate.weight.abs().sum().item(), 0.0)

        spatial_shapes, level_start_index, num_value = _spatial_meta(
            self.device)
        memory_prev, memory_curr = _random_memories(
            1, num_value, decoder.embed_dims, self.device)
        zero_start_output = decoder(
            memory_prev=memory_prev,
            memory_curr=memory_curr,
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index,
            reg_branches_prev=reg_prev,
            reg_branches_curr=reg_curr)
        self.assertEqual(len(zero_start_output), 5)

        decoder.terminal_enveloped_detail_decoder = False
        baseline_output = decoder(
            memory_prev=memory_prev,
            memory_curr=memory_curr,
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index,
            reg_branches_prev=reg_prev,
            reg_branches_curr=reg_curr)
        for terminal_group, baseline_group in zip(
                zero_start_output[:3], baseline_output):
            for terminal_tensor, baseline_tensor in zip(
                    terminal_group, baseline_group):
                self.assertTrue(torch.equal(
                    terminal_tensor, baseline_tensor))
        for terminal_group in zero_start_output[3:]:
            for terminal_tensor, baseline_tensor in zip(
                    terminal_group, baseline_output[0]):
                self.assertTrue(torch.equal(
                    terminal_tensor, baseline_tensor))

        decoder.terminal_enveloped_detail_decoder = True
        with torch.no_grad():
            torch.nn.init.normal_(terminal_gate.weight, std=0.05)
        terminal_output = decoder(
            memory_prev=memory_prev,
            memory_curr=memory_curr,
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index,
            reg_branches_prev=reg_prev,
            reg_branches_curr=reg_curr)
        decoder.terminal_enveloped_detail_decoder = False
        nonzero_baseline = decoder(
            memory_prev=memory_prev,
            memory_curr=memory_curr,
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index,
            reg_branches_prev=reg_prev,
            reg_branches_curr=reg_curr)

        for lid in range(decoder.num_layers - 1):
            self.assertTrue(torch.equal(
                terminal_output[1][lid], nonzero_baseline[1][lid]))
            self.assertTrue(torch.equal(
                terminal_output[2][lid], nonzero_baseline[2][lid]))
            self.assertTrue(torch.equal(
                terminal_output[3][lid], nonzero_baseline[0][lid]))
            self.assertTrue(torch.equal(
                terminal_output[4][lid], nonzero_baseline[0][lid]))
        final_lid = decoder.num_layers - 1
        self.assertTrue(any((
            not torch.equal(
                terminal_output[1][final_lid],
                nonzero_baseline[1][final_lid]),
            not torch.equal(
                terminal_output[2][final_lid],
                nonzero_baseline[2][final_lid]),
        )))
        self.assertTrue(any((
            not torch.equal(
                terminal_output[3][final_lid],
                nonzero_baseline[0][final_lid]),
            not torch.equal(
                terminal_output[4][final_lid],
                nonzero_baseline[0][final_lid]),
        )))

    def test_terminal_enveloped_detail_gate_receives_final_loss_gradient(self):
        decoder, reg_prev, reg_curr = _build_decoder(
            num_layers=3,
            device=self.device,
            shared_attention_decoder=True,
            terminal_enveloped_detail_decoder=True)
        decoder.init_weights()
        spatial_shapes, level_start_index, num_value = _spatial_meta(
            self.device)
        memory_prev, memory_curr = _random_memories(
            1, num_value, decoder.embed_dims, self.device)
        _, references_prev, references_curr, hidden_prev, hidden_curr = decoder(
            memory_prev=memory_prev,
            memory_curr=memory_curr,
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index,
            reg_branches_prev=reg_prev,
            reg_branches_curr=reg_curr)
        torch.manual_seed(137)
        loss = (
            (hidden_prev[-1] * torch.randn_like(hidden_prev[-1])).mean()
            + (hidden_curr[-1] * torch.randn_like(hidden_curr[-1])).mean()
            + (references_prev[-1]
               * torch.randn_like(references_prev[-1])).mean()
            + (references_curr[-1]
               * torch.randn_like(references_curr[-1])).mean())
        loss.backward()
        terminal_gate = decoder.terminal_enveloped_detail_gates[0]
        self.assertIsNotNone(terminal_gate.weight.grad)
        self.assertGreater(terminal_gate.weight.grad.abs().max().item(), 0.0)

    def test_terminal_midpoint_detail_is_final_only_and_preserves_midpoint(
            self):
        decoder, reg_prev, reg_curr = _build_decoder(
            num_layers=3,
            device=self.device,
            shared_attention_decoder=True,
            terminal_midpoint_enveloped_detail_decoder=True)
        decoder.init_weights()
        self.assertEqual(len(decoder.terminal_enveloped_detail_gates), 1)
        terminal_gate = decoder.terminal_enveloped_detail_gates[0]
        with torch.no_grad():
            torch.nn.init.normal_(terminal_gate.weight, std=0.05)

        spatial_shapes, level_start_index, num_value = _spatial_meta(
            self.device)
        memory_prev, memory_curr = _random_memories(
            1, num_value, decoder.embed_dims, self.device)
        detail_output = decoder(
            memory_prev=memory_prev,
            memory_curr=memory_curr,
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index,
            reg_branches_prev=reg_prev,
            reg_branches_curr=reg_curr)
        decoder.terminal_midpoint_enveloped_detail_decoder = False
        baseline_output = decoder(
            memory_prev=memory_prev,
            memory_curr=memory_curr,
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index,
            reg_branches_prev=reg_prev,
            reg_branches_curr=reg_curr)

        for lid in range(decoder.num_layers - 1):
            self.assertTrue(torch.equal(
                detail_output[1][lid], baseline_output[1][lid]))
            self.assertTrue(torch.equal(
                detail_output[2][lid], baseline_output[2][lid]))
            self.assertTrue(torch.equal(
                detail_output[3][lid], baseline_output[0][lid]))
            self.assertTrue(torch.equal(
                detail_output[4][lid], baseline_output[0][lid]))

        final_lid = decoder.num_layers - 1
        detail_prev = torch.logit(
            detail_output[1][final_lid].clamp(1e-6, 1 - 1e-6))
        detail_curr = torch.logit(
            detail_output[2][final_lid].clamp(1e-6, 1 - 1e-6))
        baseline_prev = torch.logit(
            baseline_output[1][final_lid].clamp(1e-6, 1 - 1e-6))
        baseline_curr = torch.logit(
            baseline_output[2][final_lid].clamp(1e-6, 1 - 1e-6))
        detail_delta_sum = (
            detail_prev - baseline_prev + detail_curr - baseline_curr)
        self.assertTrue(torch.allclose(
            detail_delta_sum,
            torch.zeros_like(detail_delta_sum),
            atol=2e-5,
            rtol=2e-5))
        self.assertTrue(any((
            not torch.equal(
                detail_output[3][final_lid],
                baseline_output[0][final_lid]),
            not torch.equal(
                detail_output[4][final_lid],
                baseline_output[0][final_lid]),
        )))

    def test_terminal_regression_detail_keeps_classification_shared(self):
        for flag in (
                'terminal_regression_enveloped_detail_decoder',
                'terminal_midpoint_regression_enveloped_detail_decoder'):
            with self.subTest(flag=flag):
                decoder, reg_prev, reg_curr = _build_decoder(
                    num_layers=3,
                    device=self.device,
                    shared_attention_decoder=True,
                    **{flag: True})
                decoder.init_weights()
                terminal_gate = decoder.terminal_enveloped_detail_gates[0]
                with torch.no_grad():
                    torch.manual_seed(149)
                    torch.nn.init.normal_(terminal_gate.weight, std=0.05)

                spatial_shapes, level_start_index, num_value = _spatial_meta(
                    self.device)
                memory_prev, memory_curr = _random_memories(
                    1, num_value, decoder.embed_dims, self.device)
                detail_output = decoder(
                    memory_prev=memory_prev,
                    memory_curr=memory_curr,
                    spatial_shapes=spatial_shapes,
                    level_start_index=level_start_index,
                    reg_branches_prev=reg_prev,
                    reg_branches_curr=reg_curr)
                setattr(decoder, flag, False)
                baseline_output = decoder(
                    memory_prev=memory_prev,
                    memory_curr=memory_curr,
                    spatial_shapes=spatial_shapes,
                    level_start_index=level_start_index,
                    reg_branches_prev=reg_prev,
                    reg_branches_curr=reg_curr)

                for hidden_group in detail_output[3:]:
                    for hidden_tensor, baseline_tensor in zip(
                            hidden_group, baseline_output[0]):
                        self.assertTrue(torch.equal(
                            hidden_tensor, baseline_tensor))
                for lid in range(decoder.num_layers - 1):
                    self.assertTrue(torch.equal(
                        detail_output[1][lid], baseline_output[1][lid]))
                    self.assertTrue(torch.equal(
                        detail_output[2][lid], baseline_output[2][lid]))
                self.assertTrue(any((
                    not torch.equal(
                        detail_output[1][-1], baseline_output[1][-1]),
                    not torch.equal(
                        detail_output[2][-1], baseline_output[2][-1]),
                )))

    def test_terminal_midpoint_regression_preserves_added_box_midpoint(
            self):
        decoder, reg_prev, reg_curr = _build_decoder(
            num_layers=3,
            device=self.device,
            shared_attention_decoder=True,
            terminal_midpoint_regression_enveloped_detail_decoder=True)
        decoder.init_weights()
        with torch.no_grad():
            torch.manual_seed(151)
            torch.nn.init.normal_(
                decoder.terminal_enveloped_detail_gates[0].weight, std=0.05)
        spatial_shapes, level_start_index, num_value = _spatial_meta(
            self.device)
        memory_prev, memory_curr = _random_memories(
            1, num_value, decoder.embed_dims, self.device)
        detail_output = decoder(
            memory_prev=memory_prev,
            memory_curr=memory_curr,
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index,
            reg_branches_prev=reg_prev,
            reg_branches_curr=reg_curr)
        decoder.terminal_midpoint_regression_enveloped_detail_decoder = False
        baseline_output = decoder(
            memory_prev=memory_prev,
            memory_curr=memory_curr,
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index,
            reg_branches_prev=reg_prev,
            reg_branches_curr=reg_curr)
        detail_prev = torch.logit(
            detail_output[1][-1].clamp(1e-6, 1 - 1e-6))
        detail_curr = torch.logit(
            detail_output[2][-1].clamp(1e-6, 1 - 1e-6))
        baseline_prev = torch.logit(
            baseline_output[1][-1].clamp(1e-6, 1 - 1e-6))
        baseline_curr = torch.logit(
            baseline_output[2][-1].clamp(1e-6, 1 - 1e-6))
        midpoint_error = (
            detail_prev - baseline_prev + detail_curr - baseline_curr)
        self.assertTrue(torch.allclose(
            midpoint_error,
            torch.zeros_like(midpoint_error),
            atol=5e-5,
            rtol=5e-5),
            msg=float(midpoint_error.abs().max()))

    def test_enveloped_detail_gates_receive_first_step_gradients(self):
        decoder, reg_prev, reg_curr = _build_decoder(
            num_layers=3,
            device=self.device,
            enveloped_detail_decoder=True)
        decoder.init_weights()
        spatial_shapes, level_start_index, num_value = _spatial_meta(
            self.device)
        memory_prev, memory_curr = _random_memories(
            1, num_value, decoder.embed_dims, self.device)
        _, references_prev, references_curr, hidden_prev, hidden_curr = decoder(
            memory_prev=memory_prev,
            memory_curr=memory_curr,
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index,
            reg_branches_prev=reg_prev,
            reg_branches_curr=reg_curr)
        torch.manual_seed(83)
        loss = sum(
            (prev * torch.randn_like(prev)).mean()
            + (curr * torch.randn_like(curr)).mean()
            for prev, curr in zip(hidden_prev, hidden_curr))
        loss = loss + sum(
            (prev * torch.randn_like(prev)).mean()
            + (curr * torch.randn_like(curr)).mean()
            for prev, curr in zip(references_prev, references_curr))
        loss.backward()
        for gate in decoder.enveloped_detail_gates:
            self.assertIsNotNone(gate.weight.grad)
            self.assertGreater(gate.weight.grad.abs().max().item(), 0.0)

    def test_common_evidence_bypass_is_exact_baseline_at_zero_start(self):
        decoder, reg_prev, reg_curr = _build_decoder(
            device=self.device, common_evidence_bypass_decoder=True)
        decoder.init_weights()
        spatial_shapes, level_start_index, num_value = _spatial_meta(
            self.device)
        memory_prev, memory_curr = _random_memories(
            1, num_value, decoder.embed_dims, self.device)
        bypass_output = decoder(
            memory_prev=memory_prev,
            memory_curr=memory_curr,
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index,
            reg_branches_prev=reg_prev,
            reg_branches_curr=reg_curr)
        decoder.common_evidence_bypass_decoder = False
        baseline_output = decoder(
            memory_prev=memory_prev,
            memory_curr=memory_curr,
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index,
            reg_branches_prev=reg_prev,
            reg_branches_curr=reg_curr)
        for bypass_group, baseline_group in zip(
                bypass_output, baseline_output):
            for bypass_tensor, baseline_tensor in zip(
                    bypass_group, baseline_group):
                self.assertTrue(torch.equal(bypass_tensor, baseline_tensor))

    def test_common_evidence_bypass_is_swap_invariant_and_bounded(self):
        decoder, _, _ = _build_decoder(
            device=self.device, common_evidence_bypass_decoder=True)
        decoder.init_weights()
        torch.manual_seed(89)
        with torch.no_grad():
            for gate in decoder.common_evidence_bypass_gates:
                torch.nn.init.normal_(gate.weight, std=0.05)
        layer_output = torch.randn(
            2, 7, decoder.embed_dims, device=self.device)
        out_prev = torch.randn_like(layer_output)
        out_curr = torch.randn_like(layer_output)
        correction = decoder._common_evidence_bypass_correction(
            0, layer_output, out_prev, out_curr)
        swapped = decoder._common_evidence_bypass_correction(
            0, layer_output, out_curr, out_prev)
        residual = 0.5 * (out_prev + out_curr) - layer_output
        self.assertTrue(torch.allclose(correction, swapped, atol=1e-6))
        self.assertTrue(torch.all(correction.abs() <= residual.abs() + 1e-7))

    def test_common_evidence_bypass_gates_receive_first_step_gradients(self):
        decoder, reg_prev, reg_curr = _build_decoder(
            num_layers=3,
            device=self.device,
            common_evidence_bypass_decoder=True)
        decoder.init_weights()
        spatial_shapes, level_start_index, num_value = _spatial_meta(
            self.device)
        memory_prev, memory_curr = _random_memories(
            1, num_value, decoder.embed_dims, self.device)
        hidden, references_prev, references_curr = decoder(
            memory_prev=memory_prev,
            memory_curr=memory_curr,
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index,
            reg_branches_prev=reg_prev,
            reg_branches_curr=reg_curr)
        torch.manual_seed(97)
        loss = sum(
            (value * torch.randn_like(value)).mean()
            for value in hidden + references_prev + references_curr)
        loss.backward()
        for gate in decoder.common_evidence_bypass_gates:
            self.assertIsNotNone(gate.weight.grad)
            self.assertGreater(gate.weight.grad.abs().max().item(), 0.0)

    def test_terminal_common_evidence_bypass_is_exact_zero_start(self):
        decoder, reg_prev, reg_curr = _build_decoder(
            num_layers=3,
            device=self.device,
            terminal_common_evidence_bypass_decoder=True)
        decoder.init_weights()
        spatial_shapes, level_start_index, num_value = _spatial_meta(
            self.device)
        memory_prev, memory_curr = _random_memories(
            1, num_value, decoder.embed_dims, self.device)
        terminal_output = decoder(
            memory_prev=memory_prev,
            memory_curr=memory_curr,
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index,
            reg_branches_prev=reg_prev,
            reg_branches_curr=reg_curr)
        decoder.terminal_common_evidence_bypass_decoder = False
        baseline_output = decoder(
            memory_prev=memory_prev,
            memory_curr=memory_curr,
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index,
            reg_branches_prev=reg_prev,
            reg_branches_curr=reg_curr)
        for terminal_group, baseline_group in zip(
                terminal_output, baseline_output):
            for terminal_tensor, baseline_tensor in zip(
                    terminal_group, baseline_group):
                self.assertTrue(torch.equal(
                    terminal_tensor, baseline_tensor))

    def test_terminal_common_evidence_bypass_changes_only_final_output(self):
        decoder, reg_prev, reg_curr = _build_decoder(
            num_layers=3,
            device=self.device,
            terminal_common_evidence_bypass_decoder=True)
        decoder.init_weights()
        torch.manual_seed(103)
        with torch.no_grad():
            torch.nn.init.normal_(
                decoder.terminal_common_evidence_bypass_gates[0].weight,
                std=0.05)
        spatial_shapes, level_start_index, num_value = _spatial_meta(
            self.device)
        memory_prev, memory_curr = _random_memories(
            1, num_value, decoder.embed_dims, self.device)
        terminal_output = decoder(
            memory_prev=memory_prev,
            memory_curr=memory_curr,
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index,
            reg_branches_prev=reg_prev,
            reg_branches_curr=reg_curr)
        decoder.terminal_common_evidence_bypass_decoder = False
        baseline_output = decoder(
            memory_prev=memory_prev,
            memory_curr=memory_curr,
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index,
            reg_branches_prev=reg_prev,
            reg_branches_curr=reg_curr)
        for terminal_group, baseline_group in zip(
                terminal_output, baseline_output):
            for terminal_tensor, baseline_tensor in zip(
                    terminal_group[:-1], baseline_group[:-1]):
                self.assertTrue(torch.equal(
                    terminal_tensor, baseline_tensor))
        self.assertFalse(torch.equal(
            terminal_output[0][-1], baseline_output[0][-1]))
        self.assertFalse(torch.equal(
            terminal_output[1][-1], baseline_output[1][-1]))
        self.assertFalse(torch.equal(
            terminal_output[2][-1], baseline_output[2][-1]))

    def test_terminal_common_evidence_bypass_is_invariant_and_bounded(self):
        decoder, _, _ = _build_decoder(
            device=self.device,
            terminal_common_evidence_bypass_decoder=True)
        decoder.init_weights()
        torch.manual_seed(107)
        with torch.no_grad():
            torch.nn.init.normal_(
                decoder.terminal_common_evidence_bypass_gates[0].weight,
                std=0.05)
        layer_output = torch.randn(
            2, 7, decoder.embed_dims, device=self.device)
        out_prev = torch.randn_like(layer_output)
        out_curr = torch.randn_like(layer_output)
        correction = decoder._terminal_common_evidence_bypass_correction(
            layer_output, out_prev, out_curr)
        swapped = decoder._terminal_common_evidence_bypass_correction(
            layer_output, out_curr, out_prev)
        residual = 0.5 * (out_prev + out_curr) - layer_output
        self.assertTrue(torch.allclose(correction, swapped, atol=1e-6))
        self.assertTrue(torch.all(correction.abs() <= residual.abs() + 1e-7))

    def test_terminal_common_evidence_bypass_gate_receives_gradient(self):
        decoder, reg_prev, reg_curr = _build_decoder(
            num_layers=3,
            device=self.device,
            terminal_common_evidence_bypass_decoder=True)
        decoder.init_weights()
        spatial_shapes, level_start_index, num_value = _spatial_meta(
            self.device)
        memory_prev, memory_curr = _random_memories(
            1, num_value, decoder.embed_dims, self.device)
        hidden, references_prev, references_curr = decoder(
            memory_prev=memory_prev,
            memory_curr=memory_curr,
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index,
            reg_branches_prev=reg_prev,
            reg_branches_curr=reg_curr)
        torch.manual_seed(109)
        loss = sum(
            (value * torch.randn_like(value)).mean()
            for value in hidden + references_prev + references_curr)
        loss.backward()
        gate = decoder.terminal_common_evidence_bypass_gates[0]
        self.assertIsNotNone(gate.weight.grad)
        self.assertGreater(gate.weight.grad.abs().max().item(), 0.0)

    def test_terminal_classification_common_is_final_only_and_box_exact(self):
        for shared_attention in (False, True):
            decoder, reg_prev, reg_curr = _build_decoder(
                num_layers=3,
                device=self.device,
                shared_attention_decoder=shared_attention,
                terminal_classification_common_evidence_decoder=True)
            decoder.init_weights()
            gate = decoder.terminal_common_evidence_bypass_gates[0]
            torch.manual_seed(149 + int(shared_attention))
            with torch.no_grad():
                torch.nn.init.normal_(gate.weight, std=0.05)
            spatial_shapes, level_start_index, num_value = _spatial_meta(
                self.device)
            memory_prev, memory_curr = _random_memories(
                1, num_value, decoder.embed_dims, self.device)
            classified = decoder(
                memory_prev=memory_prev,
                memory_curr=memory_curr,
                spatial_shapes=spatial_shapes,
                level_start_index=level_start_index,
                reg_branches_prev=reg_prev,
                reg_branches_curr=reg_curr)
            decoder.terminal_classification_common_evidence_decoder = False
            parent = decoder(
                memory_prev=memory_prev,
                memory_curr=memory_curr,
                spatial_shapes=spatial_shapes,
                level_start_index=level_start_index,
                reg_branches_prev=reg_prev,
                reg_branches_curr=reg_curr)

            for group in (1, 2):
                for classified_tensor, parent_tensor in zip(
                        classified[group], parent[group]):
                    self.assertTrue(torch.equal(
                        classified_tensor, parent_tensor))
            for lid in range(decoder.num_layers - 1):
                for group in (3, 4):
                    self.assertTrue(torch.equal(
                        classified[group][lid], parent[0][lid]))
            self.assertTrue(torch.equal(
                classified[3][-1], classified[4][-1]))
            self.assertFalse(torch.equal(
                classified[3][-1], parent[0][-1]))

    def test_terminal_classification_common_gate_receives_gradient(self):
        decoder, reg_prev, reg_curr = _build_decoder(
            num_layers=3,
            device=self.device,
            terminal_classification_common_evidence_decoder=True)
        decoder.init_weights()
        spatial_shapes, level_start_index, num_value = _spatial_meta(
            self.device)
        memory_prev, memory_curr = _random_memories(
            1, num_value, decoder.embed_dims, self.device)
        output = decoder(
            memory_prev=memory_prev,
            memory_curr=memory_curr,
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index,
            reg_branches_prev=reg_prev,
            reg_branches_curr=reg_curr)
        torch.manual_seed(151)
        loss = sum(
            (value * torch.randn_like(value)).mean()
            for group in output[3:]
            for value in group)
        loss.backward()
        gate = decoder.terminal_common_evidence_bypass_gates[0]
        self.assertIsNotNone(gate.weight.grad)
        self.assertGreater(gate.weight.grad.abs().max().item(), 0.0)

    def test_terminal_factorized_evidence_supports_independent_attention(self):
        decoder, reg_prev, reg_curr = _build_decoder(
            num_layers=3,
            device=self.device,
            shared_attention_decoder=False,
            terminal_factorized_evidence_decoder=True)
        decoder.init_weights()
        for layer in decoder.layers:
            self.assertIsNot(
                layer.cross_attn_prev.attention_weights,
                layer.cross_attn_curr.attention_weights)

        common_gate = decoder.terminal_common_evidence_bypass_gates[0]
        detail_gate = decoder.terminal_enveloped_detail_gates[0]
        torch.manual_seed(167)
        with torch.no_grad():
            torch.nn.init.normal_(common_gate.weight, std=0.05)
            torch.nn.init.normal_(detail_gate.weight, std=0.05)

        spatial_shapes, level_start_index, num_value = _spatial_meta(
            self.device)
        memory_prev, memory_curr = _random_memories(
            1, num_value, decoder.embed_dims, self.device)
        factorized = decoder(
            memory_prev=memory_prev,
            memory_curr=memory_curr,
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index,
            reg_branches_prev=reg_prev,
            reg_branches_curr=reg_curr)

        detail_weight = detail_gate.weight.detach().clone()
        with torch.no_grad():
            detail_gate.weight.zero_()
        common_only = decoder(
            memory_prev=memory_prev,
            memory_curr=memory_curr,
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index,
            reg_branches_prev=reg_prev,
            reg_branches_curr=reg_curr)

        common_weight = common_gate.weight.detach().clone()
        with torch.no_grad():
            common_gate.weight.zero_()
        parent = decoder(
            memory_prev=memory_prev,
            memory_curr=memory_curr,
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index,
            reg_branches_prev=reg_prev,
            reg_branches_curr=reg_curr)
        with torch.no_grad():
            common_gate.weight.copy_(common_weight)
            detail_gate.weight.copy_(detail_weight)

        for group in (1, 2):
            for common_tensor, parent_tensor in zip(
                    common_only[group], parent[group]):
                self.assertTrue(torch.equal(common_tensor, parent_tensor))
        factorized_prev = torch.logit(
            factorized[1][-1].clamp(1e-6, 1 - 1e-6))
        factorized_curr = torch.logit(
            factorized[2][-1].clamp(1e-6, 1 - 1e-6))
        common_prev = torch.logit(
            common_only[1][-1].clamp(1e-6, 1 - 1e-6))
        common_curr = torch.logit(
            common_only[2][-1].clamp(1e-6, 1 - 1e-6))
        midpoint_error = (
            factorized_prev - common_prev
            + factorized_curr - common_curr)
        self.assertTrue(torch.allclose(
            midpoint_error,
            torch.zeros_like(midpoint_error),
            atol=5e-5,
            rtol=5e-5),
            msg=float(midpoint_error.abs().max()))

    def test_terminal_factorized_evidence_is_exact_zero_start(self):
        decoder, reg_prev, reg_curr = _build_decoder(
            num_layers=3,
            device=self.device,
            shared_attention_decoder=True,
            terminal_factorized_evidence_decoder=True)
        decoder.init_weights()
        self.assertEqual(len(decoder.terminal_common_evidence_bypass_gates), 1)
        self.assertEqual(len(decoder.terminal_enveloped_detail_gates), 1)
        self.assertEqual(
            decoder.terminal_common_evidence_bypass_gates[
                0].weight.abs().sum().item(), 0.0)
        self.assertEqual(
            decoder.terminal_enveloped_detail_gates[
                0].weight.abs().sum().item(), 0.0)

        spatial_shapes, level_start_index, num_value = _spatial_meta(
            self.device)
        memory_prev, memory_curr = _random_memories(
            1, num_value, decoder.embed_dims, self.device)
        factorized_output = decoder(
            memory_prev=memory_prev,
            memory_curr=memory_curr,
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index,
            reg_branches_prev=reg_prev,
            reg_branches_curr=reg_curr)
        decoder.terminal_factorized_evidence_decoder = False
        baseline_output = decoder(
            memory_prev=memory_prev,
            memory_curr=memory_curr,
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index,
            reg_branches_prev=reg_prev,
            reg_branches_curr=reg_curr)
        for factorized_group, baseline_group in zip(
                factorized_output[:3], baseline_output):
            for factorized_tensor, baseline_tensor in zip(
                    factorized_group, baseline_group):
                self.assertTrue(torch.equal(
                    factorized_tensor, baseline_tensor))
        for factorized_group in factorized_output[3:]:
            for factorized_tensor, baseline_tensor in zip(
                    factorized_group, baseline_output[0]):
                self.assertTrue(torch.equal(
                    factorized_tensor, baseline_tensor))

    def test_terminal_factorized_evidence_is_final_only_and_orthogonal(self):
        decoder, reg_prev, reg_curr = _build_decoder(
            num_layers=3,
            device=self.device,
            shared_attention_decoder=True,
            terminal_factorized_evidence_decoder=True)
        decoder.init_weights()
        common_gate = decoder.terminal_common_evidence_bypass_gates[0]
        detail_gate = decoder.terminal_enveloped_detail_gates[0]
        torch.manual_seed(157)
        with torch.no_grad():
            torch.nn.init.normal_(common_gate.weight, std=0.05)
            torch.nn.init.normal_(detail_gate.weight, std=0.05)

        spatial_shapes, level_start_index, num_value = _spatial_meta(
            self.device)
        memory_prev, memory_curr = _random_memories(
            1, num_value, decoder.embed_dims, self.device)
        factorized_output = decoder(
            memory_prev=memory_prev,
            memory_curr=memory_curr,
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index,
            reg_branches_prev=reg_prev,
            reg_branches_curr=reg_curr)
        detail_weight = detail_gate.weight.detach().clone()
        with torch.no_grad():
            detail_gate.weight.zero_()
        common_only_output = decoder(
            memory_prev=memory_prev,
            memory_curr=memory_curr,
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index,
            reg_branches_prev=reg_prev,
            reg_branches_curr=reg_curr)
        common_weight = common_gate.weight.detach().clone()
        with torch.no_grad():
            common_gate.weight.zero_()
        parent_output = decoder(
            memory_prev=memory_prev,
            memory_curr=memory_curr,
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index,
            reg_branches_prev=reg_prev,
            reg_branches_curr=reg_curr)
        with torch.no_grad():
            common_gate.weight.copy_(common_weight)
            detail_gate.weight.copy_(detail_weight)

        for group in (1, 2):
            for common_tensor, parent_tensor in zip(
                    common_only_output[group], parent_output[group]):
                self.assertTrue(torch.equal(common_tensor, parent_tensor))
        self.assertTrue(any(
            not torch.equal(
                common_only_output[group][-1], parent_output[group][-1])
            for group in (3, 4)))
        for lid in range(decoder.num_layers - 1):
            for group in range(1, 5):
                self.assertTrue(torch.equal(
                    factorized_output[group][lid],
                    common_only_output[group][lid]))
        self.assertTrue(torch.equal(
            factorized_output[3][-1], factorized_output[4][-1]))
        self.assertTrue(torch.equal(
            factorized_output[3][-1], common_only_output[3][-1]))
        self.assertTrue(torch.equal(
            factorized_output[4][-1], common_only_output[4][-1]))

        factorized_prev = torch.logit(
            factorized_output[1][-1].clamp(1e-6, 1 - 1e-6))
        factorized_curr = torch.logit(
            factorized_output[2][-1].clamp(1e-6, 1 - 1e-6))
        common_prev = torch.logit(
            common_only_output[1][-1].clamp(1e-6, 1 - 1e-6))
        common_curr = torch.logit(
            common_only_output[2][-1].clamp(1e-6, 1 - 1e-6))
        midpoint_error = (
            factorized_prev - common_prev
            + factorized_curr - common_curr)
        self.assertTrue(torch.allclose(
            midpoint_error,
            torch.zeros_like(midpoint_error),
            atol=5e-5,
            rtol=5e-5),
            msg=float(midpoint_error.abs().max()))
        self.assertTrue(any((
            not torch.equal(
                factorized_output[1][-1], common_only_output[1][-1]),
            not torch.equal(
                factorized_output[2][-1], common_only_output[2][-1]),
        )))

    def test_terminal_factorized_evidence_gates_receive_gradients(self):
        decoder, reg_prev, reg_curr = _build_decoder(
            num_layers=3,
            device=self.device,
            shared_attention_decoder=True,
            terminal_factorized_evidence_decoder=True)
        decoder.init_weights()
        spatial_shapes, level_start_index, num_value = _spatial_meta(
            self.device)
        memory_prev, memory_curr = _random_memories(
            1, num_value, decoder.embed_dims, self.device)
        _, references_prev, references_curr, hidden_prev, hidden_curr = decoder(
            memory_prev=memory_prev,
            memory_curr=memory_curr,
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index,
            reg_branches_prev=reg_prev,
            reg_branches_curr=reg_curr)
        torch.manual_seed(163)
        loss = sum(
            (value * torch.randn_like(value)).mean()
            for value in (
                hidden_prev[-1],
                hidden_curr[-1],
                references_prev[-1],
                references_curr[-1]))
        loss.backward()
        for gate in (
                decoder.terminal_common_evidence_bypass_gates[0],
                decoder.terminal_enveloped_detail_gates[0]):
            self.assertIsNotNone(gate.weight.grad)
            self.assertGreater(gate.weight.grad.abs().max().item(), 0.0)

    def test_terminal_factorized_confidence_validation(self):
        with self.assertRaisesRegex(ValueError, 'must be one of'):
            _build_decoder(
                device=self.device,
                terminal_factorized_evidence_decoder=True,
                terminal_factorized_confidence='invalid')
        with self.assertRaisesRegex(
                ValueError, 'requires terminal_factorized_evidence_decoder'):
            _build_decoder(
                device=self.device,
                terminal_factorized_confidence='common')

    def test_terminal_factorized_diagonal_gates_are_lightweight_and_exact(self):
        decoder, reg_prev, reg_curr = _build_decoder(
            num_layers=3,
            device=self.device,
            terminal_factorized_evidence_decoder=True,
            terminal_factorized_diagonal_gates=True)
        decoder.init_weights()
        common_gate = decoder.terminal_common_evidence_bypass_gates[0]
        detail_gate = decoder.terminal_enveloped_detail_gates[0]
        self.assertEqual(tuple(common_gate.shape), (decoder.embed_dims,))
        self.assertEqual(tuple(detail_gate.shape), (decoder.embed_dims,))
        self.assertEqual(common_gate.numel() + detail_gate.numel(),
                         2 * decoder.embed_dims)
        self.assertEqual(common_gate.abs().sum().item(), 0.0)
        self.assertEqual(detail_gate.abs().sum().item(), 0.0)

        spatial_shapes, level_start_index, num_value = _spatial_meta(
            self.device)
        memory_prev, memory_curr = _random_memories(
            1, num_value, decoder.embed_dims, self.device)
        zero_output = decoder(
            memory_prev=memory_prev,
            memory_curr=memory_curr,
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index,
            reg_branches_prev=reg_prev,
            reg_branches_curr=reg_curr)
        decoder.terminal_factorized_evidence_decoder = False
        parent_output = decoder(
            memory_prev=memory_prev,
            memory_curr=memory_curr,
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index,
            reg_branches_prev=reg_prev,
            reg_branches_curr=reg_curr)
        decoder.terminal_factorized_evidence_decoder = True
        for zero_group, parent_group in zip(zero_output, parent_output):
            for zero_tensor, parent_tensor in zip(zero_group, parent_group):
                self.assertTrue(torch.equal(zero_tensor, parent_tensor))

        with torch.no_grad():
            common_gate.fill_(0.05)
            detail_gate.fill_(0.05)
        active_output = decoder(
            memory_prev=memory_prev,
            memory_curr=memory_curr,
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index,
            reg_branches_prev=reg_prev,
            reg_branches_curr=reg_curr)
        active_prev = torch.logit(
            active_output[1][-1].clamp(1e-6, 1 - 1e-6))
        active_curr = torch.logit(
            active_output[2][-1].clamp(1e-6, 1 - 1e-6))
        parent_prev = torch.logit(
            parent_output[1][-1].clamp(1e-6, 1 - 1e-6))
        parent_curr = torch.logit(
            parent_output[2][-1].clamp(1e-6, 1 - 1e-6))
        midpoint_error = (
            active_prev - parent_prev + active_curr - parent_curr)
        self.assertTrue(torch.allclose(
            midpoint_error,
            torch.zeros_like(midpoint_error),
            atol=5e-5,
            rtol=5e-5),
            msg=float(midpoint_error.abs().max()))

        loss = sum(
            value.square().mean()
            for value in (
                active_output[3][-1], active_output[4][-1],
                active_output[1][-1], active_output[2][-1]))
        loss.backward()
        self.assertIsNotNone(common_gate.grad)
        self.assertIsNotNone(detail_gate.grad)
        self.assertGreater(common_gate.grad.abs().max().item(), 0.0)
        self.assertGreater(detail_gate.grad.abs().max().item(), 0.0)

        with self.assertRaisesRegex(ValueError, 'requires'):
            _build_decoder(
                device=self.device,
                terminal_factorized_diagonal_gates=True)

    def test_terminal_factorized_coupled_gate_is_single_and_exact(self):
        decoder, reg_prev, reg_curr = _build_decoder(
            num_layers=3,
            device=self.device,
            terminal_factorized_evidence_decoder=True,
            terminal_factorized_diagonal_gates=True,
            terminal_factorized_coupled_gate=True)
        decoder.init_weights()
        gate = decoder.terminal_coupled_evidence_gate
        self.assertEqual(tuple(gate.shape), (decoder.embed_dims,))
        self.assertEqual(gate.numel(), decoder.embed_dims)
        self.assertEqual(len(decoder.terminal_common_evidence_bypass_gates), 0)
        self.assertEqual(len(decoder.terminal_enveloped_detail_gates), 0)
        gate_parameters = [
            (name, parameter) for name, parameter in decoder.named_parameters()
            if 'terminal_' in name and 'gate' in name
        ]
        self.assertEqual(
            [(name, parameter.numel()) for name, parameter in gate_parameters],
            [('terminal_coupled_evidence_gate', decoder.embed_dims)])
        self.assertEqual(gate.abs().sum().item(), 0.0)

        spatial_shapes, level_start_index, num_value = _spatial_meta(
            self.device)
        memory_prev, memory_curr = _random_memories(
            1, num_value, decoder.embed_dims, self.device)
        zero_output = decoder(
            memory_prev=memory_prev,
            memory_curr=memory_curr,
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index,
            reg_branches_prev=reg_prev,
            reg_branches_curr=reg_curr)
        decoder.terminal_factorized_evidence_decoder = False
        parent_output = decoder(
            memory_prev=memory_prev,
            memory_curr=memory_curr,
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index,
            reg_branches_prev=reg_prev,
            reg_branches_curr=reg_curr)
        decoder.terminal_factorized_evidence_decoder = True
        for zero_group, parent_group in zip(zero_output, parent_output):
            for zero_tensor, parent_tensor in zip(zero_group, parent_group):
                self.assertTrue(torch.equal(zero_tensor, parent_tensor))

        with torch.no_grad():
            gate.fill_(0.05)
        active_output = decoder(
            memory_prev=memory_prev,
            memory_curr=memory_curr,
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index,
            reg_branches_prev=reg_prev,
            reg_branches_curr=reg_curr)
        self.assertFalse(torch.equal(active_output[3][-1], parent_output[0][-1]))
        self.assertFalse(torch.equal(active_output[1][-1], parent_output[1][-1]))
        active_prev = torch.logit(
            active_output[1][-1].clamp(1e-6, 1 - 1e-6))
        active_curr = torch.logit(
            active_output[2][-1].clamp(1e-6, 1 - 1e-6))
        parent_prev = torch.logit(
            parent_output[1][-1].clamp(1e-6, 1 - 1e-6))
        parent_curr = torch.logit(
            parent_output[2][-1].clamp(1e-6, 1 - 1e-6))
        midpoint_error = (
            active_prev - parent_prev + active_curr - parent_curr)
        self.assertTrue(torch.allclose(
            midpoint_error,
            torch.zeros_like(midpoint_error),
            atol=5e-5,
            rtol=5e-5),
            msg=float(midpoint_error.abs().max()))

        loss = sum(
            value.square().mean()
            for value in (
                active_output[3][-1], active_output[4][-1],
                active_output[1][-1], active_output[2][-1]))
        loss.backward()
        self.assertIsNotNone(gate.grad)
        self.assertGreater(gate.grad.abs().max().item(), 0.0)

        with self.assertRaisesRegex(ValueError, 'requires'):
            _build_decoder(
                device=self.device,
                terminal_factorized_evidence_decoder=True,
                terminal_factorized_coupled_gate=True)

    def test_terminal_factorized_center_motion_preserves_shape_geometry(self):
        decoder, reg_prev, reg_curr = _build_decoder(
            num_layers=3,
            device=self.device,
            terminal_factorized_evidence_decoder=True,
            terminal_factorized_center_motion_only=True)
        decoder.init_weights()
        common_gate = decoder.terminal_common_evidence_bypass_gates[0]
        detail_gate = decoder.terminal_enveloped_detail_gates[0]
        torch.manual_seed(173)
        with torch.no_grad():
            torch.nn.init.normal_(common_gate.weight, std=0.05)
            torch.nn.init.normal_(detail_gate.weight, std=0.05)

        spatial_shapes, level_start_index, num_value = _spatial_meta(
            self.device)
        memory_prev, memory_curr = _random_memories(
            1, num_value, decoder.embed_dims, self.device)
        centered = decoder(
            memory_prev=memory_prev,
            memory_curr=memory_curr,
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index,
            reg_branches_prev=reg_prev,
            reg_branches_curr=reg_curr)

        detail_weight = detail_gate.weight.detach().clone()
        with torch.no_grad():
            detail_gate.weight.zero_()
        common_only = decoder(
            memory_prev=memory_prev,
            memory_curr=memory_curr,
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index,
            reg_branches_prev=reg_prev,
            reg_branches_curr=reg_curr)
        with torch.no_grad():
            detail_gate.weight.copy_(detail_weight)

        centered_prev = torch.logit(
            centered[1][-1].clamp(1e-6, 1 - 1e-6))
        centered_curr = torch.logit(
            centered[2][-1].clamp(1e-6, 1 - 1e-6))
        common_prev = torch.logit(
            common_only[1][-1].clamp(1e-6, 1 - 1e-6))
        common_curr = torch.logit(
            common_only[2][-1].clamp(1e-6, 1 - 1e-6))
        delta_prev = centered_prev - common_prev
        delta_curr = centered_curr - common_curr
        self.assertGreater(delta_prev[..., :2].abs().max().item(), 0.0)
        self.assertGreater(delta_curr[..., :2].abs().max().item(), 0.0)
        self.assertTrue(torch.allclose(
            delta_prev[..., 2:], torch.zeros_like(delta_prev[..., 2:]),
            atol=5e-5, rtol=5e-5))
        self.assertTrue(torch.allclose(
            delta_curr[..., 2:], torch.zeros_like(delta_curr[..., 2:]),
            atol=5e-5, rtol=5e-5))
        self.assertTrue(torch.allclose(
            delta_prev + delta_curr,
            torch.zeros_like(delta_prev), atol=5e-5, rtol=5e-5))
        self.assertTrue(torch.equal(centered[3][-1], common_only[3][-1]))
        self.assertTrue(torch.equal(centered[4][-1], common_only[4][-1]))

        loss = centered[1][-1][..., :2].square().mean()
        loss = loss + centered[2][-1][..., :2].square().mean()
        loss.backward()
        self.assertIsNotNone(detail_gate.weight.grad)
        self.assertGreater(detail_gate.weight.grad.abs().max().item(), 0.0)

        with self.assertRaisesRegex(ValueError, 'requires'):
            _build_decoder(
                device=self.device,
                terminal_factorized_center_motion_only=True)

    def test_terminal_detail_only_keeps_classification_and_shape_parent(self):
        decoder, reg_prev, reg_curr = _build_decoder(
            num_layers=3,
            device=self.device,
            terminal_factorized_evidence_decoder=True,
            terminal_factorized_center_motion_only=True,
            terminal_factorized_detail_only=True)
        decoder.init_weights()
        self.assertFalse(hasattr(
            decoder, 'terminal_common_evidence_bypass_gates'))
        self.assertEqual(len(decoder.terminal_enveloped_detail_gates), 1)
        detail_gate = decoder.terminal_enveloped_detail_gates[0]
        torch.manual_seed(181)
        with torch.no_grad():
            torch.nn.init.normal_(detail_gate.weight, std=0.05)

        spatial_shapes, level_start_index, num_value = _spatial_meta(
            self.device)
        memory_prev, memory_curr = _random_memories(
            1, num_value, decoder.embed_dims, self.device)
        detailed = decoder(
            memory_prev=memory_prev,
            memory_curr=memory_curr,
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index,
            reg_branches_prev=reg_prev,
            reg_branches_curr=reg_curr)

        detail_weight = detail_gate.weight.detach().clone()
        with torch.no_grad():
            detail_gate.weight.zero_()
        parent = decoder(
            memory_prev=memory_prev,
            memory_curr=memory_curr,
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index,
            reg_branches_prev=reg_prev,
            reg_branches_curr=reg_curr)
        with torch.no_grad():
            detail_gate.weight.copy_(detail_weight)

        detailed_prev = torch.logit(
            detailed[1][-1].clamp(1e-6, 1 - 1e-6))
        detailed_curr = torch.logit(
            detailed[2][-1].clamp(1e-6, 1 - 1e-6))
        parent_prev = torch.logit(
            parent[1][-1].clamp(1e-6, 1 - 1e-6))
        parent_curr = torch.logit(
            parent[2][-1].clamp(1e-6, 1 - 1e-6))
        delta_prev = detailed_prev - parent_prev
        delta_curr = detailed_curr - parent_curr
        self.assertGreater(delta_prev[..., :2].abs().max().item(), 0.0)
        self.assertTrue(torch.allclose(
            delta_prev[..., 2:], torch.zeros_like(delta_prev[..., 2:]),
            atol=5e-5, rtol=5e-5))
        self.assertTrue(torch.allclose(
            delta_curr[..., 2:], torch.zeros_like(delta_curr[..., 2:]),
            atol=5e-5, rtol=5e-5))
        self.assertTrue(torch.allclose(
            delta_prev + delta_curr, torch.zeros_like(delta_prev),
            atol=5e-5, rtol=5e-5))
        for detailed_group, parent_group in zip(detailed[3:], parent[3:]):
            for detailed_tensor, parent_tensor in zip(
                    detailed_group, parent_group):
                self.assertTrue(torch.equal(detailed_tensor, parent_tensor))

        loss = detailed[1][-1][..., :2].square().mean()
        loss = loss + detailed[2][-1][..., :2].square().mean()
        loss.backward()
        self.assertIsNotNone(detail_gate.weight.grad)
        self.assertGreater(detail_gate.weight.grad.abs().max().item(), 0.0)

        with self.assertRaisesRegex(ValueError, 'requires'):
            _build_decoder(
                device=self.device,
                terminal_factorized_detail_only=True)
        with self.assertRaisesRegex(ValueError, 'incompatible'):
            _build_decoder(
                device=self.device,
                terminal_factorized_evidence_decoder=True,
                terminal_factorized_diagonal_gates=True,
                terminal_factorized_coupled_gate=True,
                terminal_factorized_detail_only=True)

    def test_terminal_diagonal_detail_only_is_minimal_and_trainable(self):
        decoder, _, _ = _build_decoder(
            num_layers=3,
            device=self.device,
            terminal_factorized_evidence_decoder=True,
            terminal_factorized_diagonal_gates=True,
            terminal_factorized_center_motion_only=True,
            terminal_factorized_detail_only=True)
        decoder.init_weights()
        self.assertFalse(hasattr(
            decoder, 'terminal_common_evidence_bypass_gates'))
        self.assertEqual(len(decoder.terminal_enveloped_detail_gates), 1)
        detail_gate = decoder.terminal_enveloped_detail_gates[0]
        self.assertIsInstance(detail_gate, torch.nn.Parameter)
        self.assertEqual(detail_gate.numel(), decoder.embed_dims)
        self.assertEqual(detail_gate.abs().max().item(), 0.0)

        evidence = torch.randn(
            2, 5, decoder.embed_dims, device=self.device)
        detail = torch.randn_like(evidence)
        correction = (evidence * detail_gate).tanh() * detail
        correction.sum().backward()
        self.assertIsNotNone(detail_gate.grad)
        self.assertGreater(detail_gate.grad.abs().max().item(), 0.0)

    def test_terminal_bilateral_confidence_is_exact_and_detached(self):
        decoder, _, _ = _build_decoder(
            device=self.device,
            terminal_factorized_evidence_decoder=True,
            terminal_factorized_confidence='both')
        layer_output = torch.randn(
            2, 7, decoder.embed_dims, device=self.device,
            requires_grad=True)
        cls_prev = torch.nn.Linear(decoder.embed_dims, 3).to(self.device)
        cls_curr = torch.nn.Linear(decoder.embed_dims, 3).to(self.device)
        with torch.no_grad():
            cls_prev.weight.zero_()
            cls_curr.weight.zero_()
            cls_prev.bias.fill_(torch.logit(torch.tensor(0.25)))
            cls_curr.bias.fill_(torch.logit(torch.tensor(0.81)))
        confidence = decoder._terminal_bilateral_confidence(
            layer_output, cls_prev, cls_curr)
        self.assertTrue(torch.allclose(
            confidence,
            torch.full_like(confidence, 0.45),
            atol=1e-6,
            rtol=1e-6))
        self.assertFalse(confidence.requires_grad)

    def test_orthogonal_evidence_decomposition_is_exact_zero_start(self):
        decoder, reg_prev, reg_curr = _build_decoder(
            device=self.device,
            enveloped_detail_decoder=True,
            common_evidence_bypass_decoder=True)
        decoder.init_weights()
        spatial_shapes, level_start_index, num_value = _spatial_meta(
            self.device)
        memory_prev, memory_curr = _random_memories(
            1, num_value, decoder.embed_dims, self.device)
        decomposed_output = decoder(
            memory_prev=memory_prev,
            memory_curr=memory_curr,
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index,
            reg_branches_prev=reg_prev,
            reg_branches_curr=reg_curr)
        hidden, _, _, hidden_prev, hidden_curr = decomposed_output
        for shared, prev, curr in zip(hidden, hidden_prev, hidden_curr):
            self.assertTrue(torch.equal(prev, shared))
            self.assertTrue(torch.equal(curr, shared))

        decoder.enveloped_detail_decoder = False
        decoder.common_evidence_bypass_decoder = False
        baseline_output = decoder(
            memory_prev=memory_prev,
            memory_curr=memory_curr,
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index,
            reg_branches_prev=reg_prev,
            reg_branches_curr=reg_curr)
        for decomposed_group, baseline_group in zip(
                decomposed_output[:3], baseline_output):
            for decomposed_tensor, baseline_tensor in zip(
                    decomposed_group, baseline_group):
                self.assertTrue(torch.equal(
                    decomposed_tensor, baseline_tensor))

    def test_orthogonal_evidence_decomposition_gates_receive_gradients(self):
        decoder, reg_prev, reg_curr = _build_decoder(
            num_layers=3,
            device=self.device,
            enveloped_detail_decoder=True,
            common_evidence_bypass_decoder=True)
        decoder.init_weights()
        spatial_shapes, level_start_index, num_value = _spatial_meta(
            self.device)
        memory_prev, memory_curr = _random_memories(
            1, num_value, decoder.embed_dims, self.device)
        hidden, references_prev, references_curr, hidden_prev, hidden_curr = (
            decoder(
                memory_prev=memory_prev,
                memory_curr=memory_curr,
                spatial_shapes=spatial_shapes,
                level_start_index=level_start_index,
                reg_branches_prev=reg_prev,
                reg_branches_curr=reg_curr))
        torch.manual_seed(101)
        loss = sum(
            (shared * torch.randn_like(shared)).mean()
            + (prev * torch.randn_like(prev)).mean()
            + (curr * torch.randn_like(curr)).mean()
            for shared, prev, curr in zip(
                hidden, hidden_prev, hidden_curr))
        loss = loss + sum(
            (prev * torch.randn_like(prev)).mean()
            + (curr * torch.randn_like(curr)).mean()
            for prev, curr in zip(references_prev, references_curr))
        loss.backward()
        for gates in (
                decoder.enveloped_detail_gates,
                decoder.common_evidence_bypass_gates):
            for gate in gates:
                self.assertIsNotNone(gate.weight.grad)
                self.assertGreater(gate.weight.grad.abs().max().item(), 0.0)

    def test_new_detection_preserving_gates_survive_detector_init(self):
        for flag, attribute in (
                ('enveloped_detail_decoder', 'enveloped_detail_gates'),
                ('regression_enveloped_detail_decoder',
                 'enveloped_detail_gates'),
                ('midpoint_regression_enveloped_detail_decoder',
                 'enveloped_detail_gates'),
                ('classification_enveloped_detail_decoder',
                 'enveloped_detail_gates'),
                ('terminal_enveloped_detail_decoder',
                 'terminal_enveloped_detail_gates'),
                ('terminal_midpoint_enveloped_detail_decoder',
                 'terminal_enveloped_detail_gates'),
                ('terminal_regression_enveloped_detail_decoder',
                 'terminal_enveloped_detail_gates'),
                ('terminal_midpoint_regression_enveloped_detail_decoder',
                 'terminal_enveloped_detail_gates'),
                ('common_evidence_bypass_decoder',
                 'common_evidence_bypass_gates'),
                ('terminal_common_evidence_bypass_decoder',
                 'terminal_common_evidence_bypass_gates'),
                ('terminal_classification_common_evidence_decoder',
                 'terminal_common_evidence_bypass_gates')):
            decoder, _, _ = _build_decoder(
                device=self.device, **{flag: True})
            model = MultispecPairRotatedRTDETR.__new__(
                MultispecPairRotatedRTDETR)
            torch.nn.Module.__init__(model)
            model.decoder = decoder

            def detector_level_xavier(model_self):
                for param in model_self.decoder.parameters():
                    if param.dim() > 1:
                        torch.nn.init.xavier_uniform_(param)

            with mock.patch.object(
                    RotatedRTDETR, 'init_weights', detector_level_xavier):
                model.init_weights()
            for gate in getattr(decoder, attribute):
                self.assertEqual(gate.weight.abs().sum().item(), 0.0)

    def test_terminal_factorized_gates_survive_detector_init(self):
        decoder, _, _ = _build_decoder(
            device=self.device,
            shared_attention_decoder=True,
            terminal_factorized_evidence_decoder=True)
        model = MultispecPairRotatedRTDETR.__new__(
            MultispecPairRotatedRTDETR)
        torch.nn.Module.__init__(model)
        model.decoder = decoder

        def detector_level_xavier(model_self):
            for param in model_self.decoder.parameters():
                if param.dim() > 1:
                    torch.nn.init.xavier_uniform_(param)

        with mock.patch.object(
                RotatedRTDETR, 'init_weights', detector_level_xavier):
            model.init_weights()
        for gate in (
                decoder.terminal_common_evidence_bypass_gates[0],
                decoder.terminal_enveloped_detail_gates[0]):
            self.assertEqual(gate.weight.abs().sum().item(), 0.0)

    def test_tristate_disables_structurally_unused_parameters(self):
        decoder, _, _ = _build_decoder(
            device=self.device,
            tristate_decoder=True,
            tristate_zero_init_coupling=True)
        for layer in decoder.layers:
            self.assertFalse(layer.cross_fusion.weight.requires_grad)
            self.assertFalse(layer.cross_fusion.bias.requires_grad)
        self.assertTrue(
            decoder.layers[0].pointer_update.weight.requires_grad)
        self.assertTrue(decoder.layers[0].norms[5].weight.requires_grad)
        self.assertFalse(
            decoder.layers[-1].pointer_update.weight.requires_grad)
        self.assertFalse(decoder.layers[-1].norms[5].weight.requires_grad)

    def test_references_change_across_layers(self):
        decoder, reg_prev, reg_curr = _build_decoder(device=self.device)
        _, refs_prev, refs_curr, _, _ = self._forward(
            1, decoder, reg_prev, reg_curr)
        self.assertGreater(
            (refs_prev[0] - refs_prev[-1]).abs().max().item(), 0.0)
        self.assertGreater(
            (refs_curr[0] - refs_curr[-1]).abs().max().item(), 0.0)

    def test_same_init_refs_diverge_with_different_memories(self):
        decoder, reg_prev, reg_curr = _build_decoder(device=self.device)
        spatial_shapes, level_start_index, num_value = _spatial_meta(
            self.device)
        torch.manual_seed(42)
        shared_ref = torch.rand(
            1, decoder.num_queries, 5, device=self.device).clamp(1e-3, 1 - 1e-3)
        memory_prev = torch.randn(
            1, num_value, decoder.embed_dims, device=self.device)
        memory_curr = torch.randn(
            1, num_value, decoder.embed_dims, device=self.device)
        with torch.no_grad():
            _, refs_prev, refs_curr = decoder(
                memory_prev=memory_prev,
                memory_curr=memory_curr,
                spatial_shapes=spatial_shapes,
                level_start_index=level_start_index,
                reg_branches_prev=reg_prev,
                reg_branches_curr=reg_curr,
                reference_prev=shared_ref,
                reference_curr=shared_ref.clone(),
            )
        self.assertGreater(
            (refs_prev[-1] - refs_curr[-1]).abs().max().item(), 1e-6)

    def test_memory_swap_changes_outputs(self):
        decoder, reg_prev, reg_curr = _build_decoder(device=self.device)
        spatial_shapes, level_start_index, num_value = _spatial_meta(
            self.device)
        torch.manual_seed(1)
        memory_a = torch.randn(1, num_value, decoder.embed_dims,
                               device=self.device)
        memory_b = torch.randn(1, num_value, decoder.embed_dims,
                               device=self.device)
        with torch.no_grad():
            out_ab = decoder(
                memory_prev=memory_a,
                memory_curr=memory_b,
                spatial_shapes=spatial_shapes,
                level_start_index=level_start_index,
                reg_branches_prev=reg_prev,
                reg_branches_curr=reg_curr,
            )
            out_ba = decoder(
                memory_prev=memory_b,
                memory_curr=memory_a,
                spatial_shapes=spatial_shapes,
                level_start_index=level_start_index,
                reg_branches_prev=reg_prev,
                reg_branches_curr=reg_curr,
            )
        hidden_ab, refs_prev_ab, refs_curr_ab = out_ab
        hidden_ba, refs_prev_ba, refs_curr_ba = out_ba
        self.assertGreater(
            (hidden_ab[0] - hidden_ba[0]).abs().max().item(), 1e-6)
        self.assertGreater(
            (refs_prev_ab[-1] - refs_prev_ba[-1]).abs().max().item(), 1e-6)
        self.assertGreater(
            (refs_curr_ab[-1] - refs_curr_ba[-1]).abs().max().item(), 1e-6)

    def test_gradients_reach_both_memories(self):
        decoder, reg_prev, reg_curr = _build_decoder(device=self.device)
        spatial_shapes, level_start_index, num_value = _spatial_meta(
            self.device)
        memory_prev, memory_curr = _random_memories(
            1, num_value, decoder.embed_dims, self.device)
        hidden, refs_prev, refs_curr = decoder(
            memory_prev=memory_prev,
            memory_curr=memory_curr,
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index,
            reg_branches_prev=reg_prev,
            reg_branches_curr=reg_curr,
        )
        loss = hidden[-1].sum() + refs_prev[-1].sum() + refs_curr[-1].sum()
        loss.backward()
        self.assertIsNotNone(memory_prev.grad)
        self.assertIsNotNone(memory_curr.grad)
        self.assertGreater(memory_prev.grad.abs().sum().item(), 0.0)
        self.assertGreater(memory_curr.grad.abs().sum().item(), 0.0)

    def test_dual_reg_branches_receive_gradients(self):
        decoder, reg_prev, reg_curr = _build_decoder(device=self.device)
        spatial_shapes, level_start_index, num_value = _spatial_meta(
            self.device)
        memory_prev, memory_curr = _random_memories(
            1, num_value, decoder.embed_dims, self.device)
        _, refs_prev, refs_curr = decoder(
            memory_prev=memory_prev,
            memory_curr=memory_curr,
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index,
            reg_branches_prev=reg_prev,
            reg_branches_curr=reg_curr,
        )
        loss = sum(r.sum() for r in refs_prev) + sum(r.sum() for r in refs_curr)
        loss.backward()
        for branch in reg_prev:
            self.assertIsNotNone(branch.weight.grad)
            self.assertGreater(branch.weight.grad.abs().sum().item(), 0.0)
        for branch in reg_curr:
            self.assertIsNotNone(branch.weight.grad)
            self.assertGreater(branch.weight.grad.abs().sum().item(), 0.0)

    def test_no_nan_or_inf(self):
        decoder, reg_prev, reg_curr = _build_decoder(device=self.device)
        hidden, refs_prev, refs_curr, _, _ = self._forward(
            2, decoder, reg_prev, reg_curr)
        for tensor in hidden + refs_prev + refs_curr:
            self.assertFalse(torch.isnan(tensor).any().item())
            self.assertFalse(torch.isinf(tensor).any().item())

    def test_padding_query_cannot_change_valid_outputs(self):
        decoder, reg_prev, reg_curr = _build_decoder(
            num_layers=2, num_queries=5, device=self.device)
        decoder.eval()
        spatial_shapes, level_start_index, num_value = _spatial_meta(
            self.device)
        torch.manual_seed(17)
        memory_prev = torch.randn(1, num_value, decoder.embed_dims)
        memory_curr = torch.randn(1, num_value, decoder.embed_dims)
        query = torch.randn(1, 5, decoder.embed_dims)
        ref_prev = torch.rand(1, 5, 5).clamp(1e-3, 1 - 1e-3)
        ref_curr = torch.rand(1, 5, 5).clamp(1e-3, 1 - 1e-3)
        query_padding_mask = torch.tensor(
            [[False, False, False, True, True]])

        def run(current_query, current_prev, current_curr):
            with torch.no_grad():
                return decoder(
                    memory_prev=memory_prev,
                    memory_curr=memory_curr,
                    spatial_shapes=spatial_shapes,
                    level_start_index=level_start_index,
                    reg_branches_prev=reg_prev,
                    reg_branches_curr=reg_curr,
                    query=current_query,
                    reference_prev=current_prev,
                    reference_curr=current_curr,
                    query_key_padding_mask=query_padding_mask)

        baseline = run(query, ref_prev, ref_curr)
        perturbed_query = query.clone()
        perturbed_prev = ref_prev.clone()
        perturbed_curr = ref_curr.clone()
        perturbed_query[:, 3:] = torch.randn_like(
            perturbed_query[:, 3:]) * 1000
        perturbed_prev[:, 3:] = torch.rand_like(perturbed_prev[:, 3:])
        perturbed_curr[:, 3:] = torch.rand_like(perturbed_curr[:, 3:])
        perturbed = run(perturbed_query, perturbed_prev, perturbed_curr)

        for base_tensors, changed_tensors in zip(baseline, perturbed):
            for base, changed in zip(base_tensors, changed_tensors):
                self.assertTrue(torch.allclose(
                    base[:, :3], changed[:, :3], atol=1e-6, rtol=1e-6))

    def test_static_import_from_package(self):
        from projects.multispec_pair_rotated_rtdetr.multispec_pair_rotated_rtdetr import (  # noqa: E501
            PairRotatedRTDETRTransformerDecoder as ImportedDecoder,
            PairRotatedRTDETRTransformerDecoderLayer as ImportedLayer,
        )
        self.assertIs(ImportedDecoder, PairRotatedRTDETRTransformerDecoder)
        self.assertIs(ImportedLayer,
                      PairRotatedRTDETRTransformerDecoderLayer)

    def test_learned_embedding_names(self):
        decoder, _, _ = _build_decoder(device=self.device)
        self.assertTrue(hasattr(decoder, 'query_embedding'))
        self.assertTrue(hasattr(decoder, 'ref_prev_embedding'))
        self.assertTrue(hasattr(decoder, 'ref_curr_embedding'))
        self.assertTrue(hasattr(decoder, 'pair_pos_fusion'))
        self.assertEqual(decoder.query_embedding.weight.shape,
                         (decoder.num_queries, decoder.embed_dims))
        self.assertEqual(decoder.ref_prev_embedding.weight.shape,
                         (decoder.num_queries, 5))
        self.assertIsNot(decoder.ref_prev_embedding.weight,
                         decoder.ref_curr_embedding.weight)

    def test_learnable_query_gradients(self):
        decoder, reg_prev, reg_curr = _build_decoder(device=self.device)
        spatial_shapes, level_start_index, num_value = _spatial_meta(
            self.device)
        memory_prev, memory_curr = _random_memories(
            1, num_value, decoder.embed_dims, self.device)
        hidden, refs_prev, refs_curr = decoder(
            memory_prev=memory_prev,
            memory_curr=memory_curr,
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index,
            reg_branches_prev=reg_prev,
            reg_branches_curr=reg_curr,
        )
        loss = hidden[-1].sum() + refs_prev[-1].sum() + refs_curr[-1].sum()
        loss.backward()
        self.assertIsNotNone(decoder.query_embedding.weight.grad)
        self.assertIsNotNone(decoder.ref_prev_embedding.weight.grad)
        self.assertIsNotNone(decoder.ref_curr_embedding.weight.grad)

    def test_config_build_minimal_forward(self):
        """One minimal forward using decoder cfg from O2-RTDETR debug config."""
        cfg_path = osp.join(
            _AI4RS_ROOT,
            'projects/multispec_rotated_rtdetr/configs/'
            'o2_rtdetr_r18vd_1xb1_1e_hsmot_debug.py')
        cfg = Config.fromfile(cfg_path)
        dec_cfg = copy.deepcopy(cfg.model.decoder)
        dec_cfg.pop('type', None)
        dec_cfg['num_queries'] = 10
        dec_cfg['num_layers'] = 2
        embed_dims = dec_cfg['layer_cfg']['self_attn_cfg']['embed_dims']
        decoder = PairRotatedRTDETRTransformerDecoder(**dec_cfg)
        reg_prev = _build_reg_branches(decoder.num_layers, embed_dims,
                                       self.device, seed=0)
        reg_curr = _build_reg_branches(decoder.num_layers, embed_dims,
                                       self.device, seed=1)
        spatial_shapes, level_start_index, num_value = _spatial_meta(
            self.device)
        memory_prev = torch.randn(1, num_value, embed_dims)
        memory_curr = torch.randn(1, num_value, embed_dims)
        with torch.no_grad():
            hidden, refs_prev, refs_curr = decoder(
                memory_prev=memory_prev,
                memory_curr=memory_curr,
                spatial_shapes=spatial_shapes,
                level_start_index=level_start_index,
                reg_branches_prev=reg_prev,
                reg_branches_curr=reg_curr,
            )
        self.assertEqual(len(hidden), 2)
        self.assertEqual(refs_prev[0].shape[-1], 5)


if __name__ == '__main__':
    unittest.main()
