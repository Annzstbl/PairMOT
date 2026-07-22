# Copyright (c) AI4RS. All rights reserved.
import copy
import tempfile
import unittest

import numpy as np
import torch
import torch.nn.functional as F

from projects.multispec_rotated_rtdetr.multispec_rotated_rtdetr.pretrain_utils import (
    adapt_state_dict_in_channels, adapt_state_dict_stem_conv3d_se,
    expand_conv1_weight)
from projects.multispec_rotated_rtdetr.multispec_rotated_rtdetr.stem_conv3d_se import (
    BandSlotAdaptiveCalibration, DispersionAwareSpectralEvidence,
    FusionQualityConservation, LiquidSpectralSampler, MultispecStemConv3dSE,
    PairAlignedCompactDetailEnhancement)


class TestMultispecPretrainUtils(unittest.TestCase):

    def test_expand_rgbrepeat(self):
        weight = torch.randn(64, 3, 7, 7)
        expanded = expand_conv1_weight(
            weight, in_channels=8, expand_mode='rgbrepeat')
        self.assertEqual(expanded.shape, (64, 8, 7, 7))
        torch.testing.assert_close(expanded[:, :3], weight)
        torch.testing.assert_close(expanded[:, 3:6], weight)
        torch.testing.assert_close(expanded[:, 6:8], weight[:, :2])

    def test_expand_interpolate(self):
        weight = torch.randn(32, 3, 3, 3)
        expanded = expand_conv1_weight(
            weight, in_channels=8, expand_mode='interpolate')
        self.assertEqual(expanded.shape, (32, 8, 3, 3))

    def test_adapt_state_dict(self):
        state_dict = {
            'backbone.conv1.weight': torch.randn(64, 3, 7, 7),
            'backbone.bn1.weight': torch.randn(64),
        }
        adapted = adapt_state_dict_in_channels(state_dict, in_channels=8)
        self.assertEqual(adapted['backbone.conv1.weight'].shape[1], 8)
        self.assertEqual(adapted['backbone.bn1.weight'].shape[0], 64)

    def test_adapt_stem_conv3d_se_skips_layer_conv1(self):
        state_dict = {
            'stem.0.weight': torch.randn(32, 3, 3, 3),
            'layer1.0.conv1.weight': torch.randn(64, 64, 3, 3),
            'layer1.0.conv2.weight': torch.randn(64, 64, 3, 3),
        }
        adapted = adapt_state_dict_stem_conv3d_se(state_dict)
        self.assertIn('stem.0.conv3d.weight', adapted)
        self.assertNotIn('stem.0.weight', adapted)
        self.assertIn('layer1.0.conv1.weight', adapted)
        self.assertNotIn('layer1.0.conv1.conv3d.weight', adapted)
        self.assertEqual(adapted['stem.0.conv3d.weight'].shape, (32, 1, 3, 3, 3))

    def test_liquid_spectral_sampler_initial_windows(self):
        stem = MultispecStemConv3dSE(
            out_channels=16,
            num_spectral=8,
            reduction=2,
            liquid_sampler=dict(embed_dims=32, tau=1.0, hard=True),
        ).eval()
        x = torch.randn(2, 8, 32, 32)
        out, groups, probs = stem(x, return_sampling=True)

        self.assertEqual(out.shape, (2, 16, 16, 16))
        self.assertEqual(groups.shape, (2, 16, 6, 16, 16))
        self.assertEqual(probs.shape, (2, 6, 3, 8))
        expected = torch.tensor([
            [0, 1, 2],
            [1, 2, 3],
            [2, 3, 4],
            [3, 4, 5],
            [4, 5, 6],
            [5, 6, 7],
        ])
        torch.testing.assert_close(probs[0].argmax(dim=-1), expected)

    def test_liquid_block_route_descriptor_shapes_and_gradients(self):
        sampler = LiquidSpectralSampler(
            num_spectral=8,
            spectral_kernel=3,
            num_groups=8,
            embed_dims=16,
            head_weight_std=1e-3,
            hard=False,
            eval_hard=False,
            block_route_descriptor=dict(grid_size=(4, 5)),
            pair_sampler_router=dict(
                hidden_dims=24,
                relation_mode='pair_diff_product'),
        ).train()
        x = torch.randn(4, 8, 16, 20, requires_grad=True)

        sampled, probs = sampler(x, pair_batch_size=2)

        self.assertEqual(sampled.shape, (4, 8, 3, 16, 20))
        self.assertEqual(probs.shape, (4, 8, 3, 8))
        self.assertEqual(
            sampler.last_block_statistics.shape, (4, 8, 20, 3))
        self.assertEqual(sampler.last_block_hidden.shape, (4, 20, 16))
        self.assertEqual(sampler.last_block_summary.shape, (4, 48))

        (sampled.square().mean() + probs.square().mean()).backward()
        for parameter in (
                sampler.desc_proj.weight,
                sampler.w1.weight,
                sampler.w2.weight,
                sampler.block_summary_proj.weight,
                sampler.head.weight,
                sampler.pair_sampler_router.mlp[-1].weight):
            self.assertIsNotNone(parameter.grad)
            self.assertTrue(torch.isfinite(parameter.grad).all())
            self.assertGreater(parameter.grad.abs().sum().item(), 0)
        self.assertTrue(all(
            parameter.grad is not None and torch.isfinite(parameter.grad).all()
            for parameter in sampler.parameters() if parameter.requires_grad))
        self.assertTrue(torch.isfinite(x.grad).all())

    def test_liquid_block_route_descriptor_keeps_spatial_distribution(self):
        sampler = LiquidSpectralSampler(
            num_spectral=8,
            spectral_kernel=3,
            num_groups=8,
            embed_dims=16,
            head_weight_std=1e-3,
            hard=False,
            eval_hard=False,
            block_route_descriptor=dict(grid_size=(2, 2)),
        ).eval()
        concentrated = torch.zeros(1, 8, 8, 8)
        concentrated[:, :, :4, :4] = 1.0
        distributed = torch.zeros_like(concentrated)
        distributed[:, :, ::2, ::2] = 1.0

        torch.testing.assert_close(
            concentrated.mean(dim=(-2, -1)),
            distributed.mean(dim=(-2, -1)))
        torch.testing.assert_close(
            concentrated.amax(dim=(-2, -1)),
            distributed.amax(dim=(-2, -1)))
        torch.testing.assert_close(
            concentrated.flatten(2).std(dim=-1),
            distributed.flatten(2).std(dim=-1))

        concentrated_hidden = sampler._block_route_hidden(concentrated)
        distributed_hidden = sampler._block_route_hidden(distributed)

        self.assertFalse(torch.allclose(
            concentrated_hidden, distributed_hidden))

    def test_band_slot_calibration_is_identity_and_trainable(self):
        calibration = BandSlotAdaptiveCalibration(
            num_spectral=8, spectral_kernel=3)
        sampled = torch.randn(2, 8, 3, 10, 12)
        logits = torch.randn(2, 8, 3, 8, requires_grad=True)
        probs = logits.softmax(dim=-1)

        output = calibration(sampled, probs)

        torch.testing.assert_close(output, sampled)
        output.square().mean().backward()
        self.assertIsNotNone(calibration.band_slot_log_scale.grad)
        self.assertGreater(
            calibration.band_slot_log_scale.grad.abs().sum().item(), 0)
        self.assertTrue(torch.isfinite(
            calibration.band_slot_log_scale.grad).all())

    def test_dispersion_evidence_is_identity_and_learns_rms(self):
        evidence = DispersionAwareSpectralEvidence(num_groups=8)
        groups = torch.randn(2, 16, 8, 9, 11)

        output = evidence(groups)

        torch.testing.assert_close(output, groups.mean(dim=1))
        output.square().mean().backward()
        gradient = evidence.evidence_mixer.weight.grad
        self.assertIsNotNone(gradient)
        self.assertGreater(gradient[:, 1].abs().sum().item(), 0)
        self.assertTrue(torch.isfinite(gradient).all())

    def test_coarse_spectral_preview_router_shapes_and_gradients(self):
        sampler = LiquidSpectralSampler(
            num_spectral=8,
            spectral_kernel=3,
            num_groups=8,
            embed_dims=16,
            head_weight_std=1e-3,
            hard=False,
            eval_hard=False,
            coarse_spectral_preview_router=dict(grid_size=(6, 8)),
            pair_sampler_router=dict(
                hidden_dims=24,
                relation_mode='pair_diff_product'),
        ).train()
        x = torch.randn(4, 8, 18, 22)
        preview_weight = torch.nn.Parameter(torch.randn(12, 1, 3, 3, 3))

        sampled, probs = sampler(
            x, pair_batch_size=2, preview_weight=preview_weight)

        self.assertEqual(sampled.shape, (4, 8, 3, 18, 22))
        self.assertEqual(probs.shape, (4, 8, 3, 8))
        self.assertEqual(
            sampler.last_preview_features.shape, (4, 12, 8, 6, 8))
        self.assertEqual(
            sampler.last_preview_statistics.shape, (4, 8, 3))
        probs.square().mean().backward()
        self.assertIsNone(preview_weight.grad)
        for parameter in (
                sampler.desc_proj.weight,
                sampler.w1.weight,
                sampler.w2.weight,
                sampler.head.weight,
                sampler.pair_sampler_router.mlp[-1].weight):
            self.assertIsNotNone(parameter.grad)
            self.assertTrue(torch.isfinite(parameter.grad).all())
            self.assertGreater(parameter.grad.abs().sum().item(), 0)

    def test_stem_accepts_independent_bsac_dse_and_cspr_options(self):
        options = (
            dict(band_slot_calibration=True),
            dict(dispersion_aware_spectral_evidence=True),
            dict(coarse_spectral_preview_router=dict(grid_size=(6, 8))),
        )
        x = torch.randn(4, 8, 18, 22)
        for option in options:
            stem = MultispecStemConv3dSE(
                out_channels=16,
                num_spectral=8,
                reduction=2,
                liquid_sampler=dict(
                    embed_dims=16,
                    num_groups=8,
                    head_weight_std=1e-3,
                    hard=False,
                    eval_hard=False,
                    **option),
            ).train()
            stem.set_pair_batch_size(2)

            output = stem(x)

            self.assertEqual(output.shape, (4, 16, 9, 11))
            output.square().mean().backward()
            self.assertTrue(all(
                parameter.grad is not None and torch.isfinite(parameter.grad).all()
                for parameter in stem.parameters() if parameter.requires_grad))

    def test_liquid_spectral_sampler_cyclic_initial_windows(self):
        init_patterns = [
            [7, 0, 1],
            [0, 1, 2],
            [1, 2, 3],
            [2, 3, 4],
            [3, 4, 5],
            [4, 5, 6],
            [5, 6, 7],
            [6, 7, 0],
        ]
        stem = MultispecStemConv3dSE(
            out_channels=16,
            num_spectral=8,
            reduction=2,
            liquid_sampler=dict(
                embed_dims=32,
                num_groups=8,
                init_patterns=init_patterns,
                tau=1.0,
                hard=True),
        ).eval()
        x = torch.randn(2, 8, 32, 32)
        out, groups, probs = stem(x, return_sampling=True)

        self.assertEqual(out.shape, (2, 16, 16, 16))
        self.assertEqual(groups.shape, (2, 16, 8, 16, 16))
        self.assertEqual(probs.shape, (2, 8, 3, 8))
        expected = torch.tensor(init_patterns)
        torch.testing.assert_close(probs[0].argmax(dim=-1), expected)

    def test_liquid_sampler_eval_hard_samples_without_replacement(self):
        sampler = LiquidSpectralSampler(
            num_spectral=8,
            spectral_kernel=3,
            num_groups=2,
            init_patterns=[[7, 0, 1], [0, 1, 2]],
            embed_dims=16,
            tau=1.0,
            hard=False,
            eval_hard=True,
        ).eval()
        logits = torch.zeros(2, 2, 3, 8)
        logits[:, :, :, 2] = 10.0

        probs = sampler._sample(logits)
        selected = probs.argmax(dim=-1)

        self.assertEqual(probs.shape, (2, 2, 3, 8))
        for batch_idx in range(selected.size(0)):
            for group_idx in range(selected.size(1)):
                self.assertEqual(
                    len(set(selected[batch_idx, group_idx].tolist())), 3)

    def test_pair_consensus_sampler_shares_route_and_has_gradients(self):
        sampler = LiquidSpectralSampler(
            num_spectral=8,
            spectral_kernel=3,
            num_groups=8,
            init_patterns=[
                [7, 0, 1], [0, 1, 2], [1, 2, 3], [2, 3, 4],
                [3, 4, 5], [4, 5, 6], [5, 6, 7], [6, 7, 0]
            ],
            embed_dims=16,
            hard=True,
            pair_consensus_router=dict(hidden_dims=24),
            soft_group_set_transport=dict(
                initial_strength=0.25,
                confidence_gated=True,
                apply_to_independent_hard=True),
            use_soft_context_after_hard=True,
        ).train()
        x = torch.randn(4, 8, 20, 24, requires_grad=True)
        x.data[2:, 4] += 1.5

        sampled, probs = sampler(x, pair_batch_size=2)
        context = sampler.last_context_probs
        selected = probs.detach().argmax(dim=-1)

        torch.testing.assert_close(probs[:2], probs[2:])
        torch.testing.assert_close(context[:2], context[2:])
        self.assertFalse(torch.equal(sampled[:2], sampled[2:]))
        for batch_selected in selected:
            for group in batch_selected:
                self.assertEqual(len(set(group.tolist())), 3)

        (sampled.square().mean() + context.square().mean()).backward()
        router = sampler.pair_consensus_router
        self.assertGreater(router.mlp[-1].weight.grad.abs().sum().item(), 0)
        self.assertTrue(torch.isfinite(x.grad).all())

    def test_pair_consensus_router_is_symmetric(self):
        sampler = LiquidSpectralSampler(
            num_spectral=8,
            spectral_kernel=3,
            num_groups=8,
            embed_dims=16,
            head_weight_std=1e-3,
            hard=False,
            eval_hard=False,
            pair_consensus_router=dict(hidden_dims=24),
        ).eval()
        x = torch.randn(4, 8, 16, 18)

        _, probs = sampler(x, pair_batch_size=2)
        swapped = torch.cat([x[2:], x[:2]], dim=0)
        _, swapped_probs = sampler(swapped, pair_batch_size=2)

        torch.testing.assert_close(probs[:2], probs[2:])
        torch.testing.assert_close(probs[:2], swapped_probs[:2])

    def test_reliability_weighted_consensus_starts_equal_and_trains(self):
        common = dict(
            num_spectral=8,
            spectral_kernel=3,
            num_groups=8,
            embed_dims=16,
            head_weight_std=1e-3,
            hard=False,
            eval_hard=False)
        baseline = LiquidSpectralSampler(
            **common, pair_consensus_router=dict(hidden_dims=24)).train()
        weighted = LiquidSpectralSampler(
            **common,
            pair_consensus_router=dict(
                hidden_dims=24, reliability_weighted=True)).train()
        weighted.load_state_dict(baseline.state_dict(), strict=False)
        x = torch.randn(4, 8, 16, 18, requires_grad=True)

        torch.manual_seed(17)
        _, baseline_probs = baseline(x.detach(), pair_batch_size=2)
        torch.manual_seed(17)
        _, weighted_probs = weighted(x, pair_batch_size=2)
        torch.testing.assert_close(weighted_probs, baseline_probs)
        torch.testing.assert_close(weighted_probs[:2], weighted_probs[2:])

        weighted_probs.square().mean().backward()
        router = weighted.pair_consensus_router
        self.assertGreater(
            router.reliability_head[-1].weight.grad.abs().sum().item(), 0)
        torch.testing.assert_close(
            router.last_frame_reliability,
            torch.full_like(router.last_frame_reliability, 0.5))

    def test_pair_sampler_router_and_consensus_are_mutually_exclusive(self):
        with self.assertRaisesRegex(AssertionError, 'mutually exclusive'):
            LiquidSpectralSampler(
                num_spectral=8,
                spectral_kernel=3,
                pair_sampler_router={},
                pair_consensus_router={})

    def test_competitive_router_blocks_common_mode_group_collapse(self):
        init_patterns = [
            [7, 0, 1], [0, 1, 2], [1, 2, 3], [2, 3, 4],
            [3, 4, 5], [4, 5, 6], [5, 6, 7], [6, 7, 0]
        ]
        sampler = LiquidSpectralSampler(
            num_spectral=8,
            spectral_kernel=3,
            num_groups=8,
            init_patterns=init_patterns,
            embed_dims=16,
            init_logit=2.0,
            eval_hard=True,
            competitive_router=dict(
                content_dims=16,
                content_strength=0.35,
                common_cap=0.5,
                specific_cap=2.0),
        ).eval()
        router = sampler.competitive_router
        logits = router.anchor_logits.unsqueeze(0).repeat(2, 1, 1, 1)
        # Simulate the observed failure: every group receives an arbitrarily
        # large preference for the same three physical bands.
        logits[..., 1] += 100.0
        logits[..., 2] += 80.0
        logits[..., 4] += 60.0
        stats = torch.randn(2, 8, 3)

        calibrated = router(logits, stats, pair_batch_size=1)
        selected = sampler._dedup_hard_indices(calibrated)
        canonical_sets = {
            tuple(sorted(group.tolist())) for group in selected[0]
        }

        self.assertGreaterEqual(len(canonical_sets), 6)
        self.assertLessEqual(
            router.last_common_correction.abs().max().item(), 0.5)

    def test_competitive_router_is_pair_conditioned_and_has_gradients(self):
        sampler = LiquidSpectralSampler(
            num_spectral=8,
            spectral_kernel=3,
            num_groups=8,
            init_patterns=[
                [7, 0, 1], [0, 1, 2], [1, 2, 3], [2, 3, 4],
                [3, 4, 5], [4, 5, 6], [5, 6, 7], [6, 7, 0]
            ],
            embed_dims=16,
            init_logit=2.0,
            hard=False,
            eval_hard=False,
            pair_sampler_router=dict(
                hidden_dims=16, relation_mode='pair_diff_product'),
            competitive_router=dict(content_dims=16),
        ).train()
        x = torch.randn(4, 8, 24, 24, requires_grad=True)
        x.data[2:, 3] += 2.0

        sampled, probs = sampler(x, pair_batch_size=2)
        loss = sampled.square().mean() + probs.square().mean()
        loss.backward()

        router = sampler.competitive_router
        self.assertEqual(probs.shape, (4, 8, 3, 8))
        self.assertFalse(torch.equal(
            router.last_content_score[0], router.last_content_score[2]))
        self.assertTrue(torch.isfinite(x.grad).all())
        self.assertGreater(
            router.group_slot_query.grad.abs().sum().item(), 0)
        self.assertGreater(
            router.content_encoder[0].weight.grad.abs().sum().item(), 0)

    def test_competitive_router_relaxes_only_aligned_strong_evidence(self):
        sampler = LiquidSpectralSampler(
            num_spectral=8,
            spectral_kernel=3,
            num_groups=8,
            init_patterns=[
                [7, 0, 1], [0, 1, 2], [1, 2, 3], [2, 3, 4],
                [3, 4, 5], [4, 5, 6], [5, 6, 7], [6, 7, 0]
            ],
            embed_dims=16,
            init_logit=2.0,
            competitive_router=dict(
                content_dims=16,
                adaptive_anchor_relax=dict(
                    max_relax=0.45,
                    evidence_threshold=0.08,
                    temperature=0.02)),
        )
        router = sampler.competitive_router
        strong = torch.full((2, 8, 3, 8), 0.2)
        aligned_scale = router._adaptive_anchor_scale(strong, strong)
        opposed_scale = router._adaptive_anchor_scale(strong, -strong)
        weak = torch.full_like(strong, 0.001)
        weak_scale = router._adaptive_anchor_scale(weak, weak)

        self.assertLess(aligned_scale.mean().item(), 0.6)
        torch.testing.assert_close(opposed_scale, torch.ones_like(opposed_scale))
        self.assertGreater(weak_scale.mean().item(), 0.98)

    def test_confidence_preserving_router_keeps_confident_logits_stable(self):
        sampler = LiquidSpectralSampler(
            num_spectral=8,
            spectral_kernel=3,
            num_groups=8,
            init_patterns=[
                [7, 0, 1], [0, 1, 2], [1, 2, 3], [2, 3, 4],
                [3, 4, 5], [4, 5, 6], [5, 6, 7], [6, 7, 0]
            ],
            embed_dims=16,
            confidence_preserving_router=dict(content_dims=16),
        )
        router = sampler.confidence_preserving_router
        stats = torch.randn(2, 8, 3)
        ambiguous = torch.zeros(2, 8, 3, 8)
        confident = ambiguous.clone()
        confident[..., 0] = 4.0

        ambiguous_out = router(ambiguous, stats, pair_batch_size=1)
        ambiguous_delta = (ambiguous_out - ambiguous).abs().mean()
        confident_out = router(confident, stats, pair_batch_size=1)
        confident_delta = (confident_out - confident).abs().mean()

        self.assertGreater(ambiguous_delta.item(), confident_delta.item())
        self.assertLess(router.last_uncertainty_gate.mean().item(), 0.06)
        self.assertTrue(torch.equal(
            confident.argmax(dim=-1), confident_out.argmax(dim=-1)))

    def test_confidence_preserving_router_is_pair_conditioned_and_trainable(self):
        sampler = LiquidSpectralSampler(
            num_spectral=8,
            spectral_kernel=3,
            num_groups=8,
            init_patterns=[
                [7, 0, 1], [0, 1, 2], [1, 2, 3], [2, 3, 4],
                [3, 4, 5], [4, 5, 6], [5, 6, 7], [6, 7, 0]
            ],
            embed_dims=16,
            hard_group_unique_sets=True,
            soft_group_set_transport=dict(initial_strength=1.0),
            confidence_preserving_router=dict(content_dims=16),
        ).train()
        x = torch.randn(4, 8, 24, 24, requires_grad=True)
        x.data[2:, 3] += 2.0

        sampled, probs = sampler(x, pair_batch_size=2)
        (sampled.square().mean() + probs.square().mean()).backward()
        router = sampler.confidence_preserving_router

        self.assertFalse(torch.equal(
            router.last_content_score[0], router.last_content_score[2]))
        self.assertTrue(torch.isfinite(x.grad).all())
        self.assertGreater(
            router.group_slot_query.grad.abs().sum().item(), 0)
        self.assertGreater(
            router.content_encoder[0].weight.grad.abs().sum().item(), 0)
        self.assertGreater(router.last_residual.abs().sum().item(), 0)
        torch.testing.assert_close(
            router.last_content_score.mean(dim=-1),
            torch.zeros_like(router.last_content_score[..., 0]),
            atol=1e-6,
            rtol=0)
        torch.testing.assert_close(
            router.last_content_score.mean(dim=1),
            torch.zeros_like(router.last_content_score[:, 0]),
            atol=1e-6,
            rtol=0)

    def test_liquid_content_router_modes_are_mutually_exclusive(self):
        with self.assertRaisesRegex(AssertionError, 'mutually exclusive'):
            LiquidSpectralSampler(
                num_spectral=8,
                spectral_kernel=3,
                competitive_router={},
                confidence_preserving_router={})

    def test_sparse_evidence_is_initially_exact_and_compact_sensitive(self):
        init_patterns = [
            [7, 0, 1], [0, 1, 2], [1, 2, 3], [2, 3, 4],
            [3, 4, 5], [4, 5, 6], [5, 6, 7], [6, 7, 0]
        ]
        common_sampler = dict(
            embed_dims=16,
            num_groups=8,
            init_patterns=init_patterns,
            tau=1.0,
            hard=False,
            eval_hard=False,
            pair_sampler_router=dict(
                hidden_dims=16, relation_mode='pair_diff_product'),
            liquid_aware_fusion=dict(
                embed_dims=16,
                num_heads=4,
                use_overlap_context=True,
                use_spatial_mixer=True,
                pair_transport=dict(
                    hidden_dims=16,
                    relation_mode='pair_diff_product')),
            liquid_group_modulator=dict(hidden_dims=8),
        )
        baseline = MultispecStemConv3dSE(
            out_channels=16,
            num_spectral=8,
            reduction=2,
            liquid_sampler=common_sampler,
        ).eval()
        sparse_sampler = copy.deepcopy(common_sampler)
        sparse_sampler['sparse_spectral_evidence'] = {}
        sparse = MultispecStemConv3dSE(
            out_channels=16,
            num_spectral=8,
            reduction=2,
            liquid_sampler=sparse_sampler,
        ).eval()
        incompatible = sparse.load_state_dict(
            baseline.state_dict(), strict=False)
        self.assertFalse(incompatible.unexpected_keys)
        self.assertTrue(all(
            'sparse_spectral_evidence' in key or
            'sparse_evidence_proj' in key or
            'sparse_spatial_gain' in key
            for key in incompatible.missing_keys))

        x = 0.01 * torch.randn(4, 8, 32, 28)
        x[:, 3, 12:16, 10:14] += 8.0
        baseline.set_pair_batch_size(2)
        sparse.set_pair_batch_size(2)
        baseline_out, _, baseline_probs = baseline(
            x, return_sampling=True)
        sparse_out, _, sparse_probs = sparse(
            x, return_sampling=True)

        torch.testing.assert_close(sparse_probs, baseline_probs)
        torch.testing.assert_close(sparse_out, baseline_out)
        evidence = sparse.liquid_sampler.sparse_spectral_evidence
        self.assertEqual(evidence.last_contrast.shape, (4, 8, 16, 14))
        self.assertEqual(evidence.last_scale_weights.shape, (4, 8, 2))
        torch.testing.assert_close(
            evidence.last_scale_weights.sum(dim=-1),
            torch.ones_like(evidence.last_scale_weights[..., 0]))
        self.assertGreater(
            evidence.last_contrast[:, 3].square().mean().sqrt().item(),
            evidence.last_contrast[:, 0].square().mean().sqrt().item())
        self.assertEqual(
            sparse.liquid_sampler.last_sparse_group_map.shape,
            (4, 8, 16, 14))

        sparse.train()
        x = x.requires_grad_()
        out, _, probs = sparse(x, return_sampling=True)
        (out.square().mean() + probs.square().mean()).backward()
        sparse_parameters = [
            parameter for name, parameter in sparse.named_parameters()
            if ('sparse_spectral_evidence' in name or
                'sparse_evidence_proj' in name or
                'sparse_spatial_gain' in name)
        ]
        self.assertTrue(sparse_parameters)
        self.assertTrue(all(
            parameter.grad is not None and
            torch.isfinite(parameter.grad).all()
            for parameter in sparse_parameters))
        self.assertGreater(
            evidence.router_gain.grad.abs().sum().item(), 0)
        self.assertGreater(
            sparse.liquid_aware_fusion.sparse_evidence_proj.weight.grad
            .abs().sum().item(), 0)
        self.assertGreater(
            sparse.liquid_aware_fusion.sparse_spatial_gain.grad
            .abs().sum().item(), 0)

    def test_pair_consistent_detail_preservation_identity_and_gradients(self):
        sampler = dict(
            embed_dims=16,
            num_groups=8,
            init_patterns=[
                [7, 0, 1], [0, 1, 2], [1, 2, 3], [2, 3, 4],
                [3, 4, 5], [4, 5, 6], [5, 6, 7], [6, 7, 0],
            ],
            tau=1.0,
            hard=False,
            eval_hard=False,
        )
        baseline = MultispecStemConv3dSE(
            out_channels=16,
            num_spectral=8,
            reduction=2,
            liquid_sampler=sampler,
        ).eval()
        detail_sampler = copy.deepcopy(sampler)
        detail_sampler['pair_consistent_detail_preservation'] = dict(
            hidden_dims=16)
        detail = MultispecStemConv3dSE(
            out_channels=16,
            num_spectral=8,
            reduction=2,
            liquid_sampler=detail_sampler,
        ).eval()
        incompatible = detail.load_state_dict(
            baseline.state_dict(), strict=False)
        self.assertFalse(incompatible.unexpected_keys)
        self.assertTrue(incompatible.missing_keys)
        self.assertTrue(all(
            'pair_consistent_detail_preservation' in key
            for key in incompatible.missing_keys))

        x = torch.randn(4, 8, 32, 28)
        baseline.set_pair_batch_size(2)
        detail.set_pair_batch_size(2)
        baseline_out, _, baseline_probs = baseline(
            x, return_sampling=True)
        detail_out, _, detail_probs = detail(
            x, return_sampling=True)
        torch.testing.assert_close(detail_probs, baseline_probs)
        torch.testing.assert_close(detail_out, baseline_out)

        detail.train()
        x = x.requires_grad_()
        out, _, probs = detail(x, return_sampling=True)
        (out.square().mean() + probs.square().mean()).backward()
        module = detail.pair_consistent_detail_preservation
        parameters = list(module.parameters())
        self.assertTrue(all(
            parameter.grad is not None and
            torch.isfinite(parameter.grad).all()
            for parameter in parameters))
        self.assertGreater(
            module.gain_mlp[-1].weight.grad.abs().sum().item(), 0)
        self.assertIsNotNone(module.last_detail_gain)
        self.assertTrue(torch.isfinite(module.last_detail_gain).all())

    def test_pair_aligned_compact_detail_identity_and_gradients(self):
        sampler = dict(
            embed_dims=16,
            num_groups=8,
            init_patterns=[
                [7, 0, 1], [0, 1, 2], [1, 2, 3], [2, 3, 4],
                [3, 4, 5], [4, 5, 6], [5, 6, 7], [6, 7, 0],
            ],
            tau=1.0,
            hard=False,
            eval_hard=False,
        )
        baseline = MultispecStemConv3dSE(
            out_channels=16,
            num_spectral=8,
            reduction=2,
            liquid_sampler=sampler,
        ).eval()
        enhanced_sampler = copy.deepcopy(sampler)
        enhanced_sampler['pair_aligned_compact_detail_enhancement'] = dict(
            hidden_dims=16)
        enhanced = MultispecStemConv3dSE(
            out_channels=16,
            num_spectral=8,
            reduction=2,
            liquid_sampler=enhanced_sampler,
        ).eval()
        incompatible = enhanced.load_state_dict(
            baseline.state_dict(), strict=False)
        self.assertFalse(incompatible.unexpected_keys)
        self.assertTrue(incompatible.missing_keys)
        self.assertTrue(all(
            'pair_aligned_compact_detail_enhancement' in key
            for key in incompatible.missing_keys))

        x = torch.randn(4, 8, 32, 28)
        x[:, 4, 12:15, 10:13] += 5.0
        baseline.set_pair_batch_size(2)
        enhanced.set_pair_batch_size(2)
        baseline_out, _, baseline_probs = baseline(
            x, return_sampling=True)
        enhanced_out, _, enhanced_probs = enhanced(
            x, return_sampling=True)
        torch.testing.assert_close(enhanced_probs, baseline_probs)
        torch.testing.assert_close(enhanced_out, baseline_out)

        module = enhanced.pair_aligned_compact_detail_enhancement
        descriptor = module._pair_descriptor(
            torch.randn(4, 8, 2), pair_batch_size=2)
        torch.testing.assert_close(descriptor[:2], descriptor[2:])

        enhanced.train()
        x = x.requires_grad_()
        out, _, probs = enhanced(x, return_sampling=True)
        (out.square().mean() + probs.square().mean()).backward()
        parameters = list(module.parameters())
        self.assertTrue(all(
            parameter.grad is not None and
            torch.isfinite(parameter.grad).all()
            for parameter in parameters))
        self.assertGreater(
            module.gain_mlp[-1].weight.grad.abs().sum().item(), 0)
        self.assertGreater(module.last_compact_mask_mean.item(), 0)
        self.assertTrue(torch.isfinite(module.last_uncertainty_mean))

    def test_pair_aligned_compact_detail_fused_reduction_equivalence(self):
        module = PairAlignedCompactDetailEnhancement(
            num_groups=8, hidden_dims=16)
        with torch.no_grad():
            module.gain_mlp[-1].weight.normal_(std=0.1)
            module.gain_mlp[-1].bias.fill_(0.05)

        groups = torch.randn(4, 16, 8, 12, 10, requires_grad=True)
        gate = torch.sigmoid(torch.randn(4, 8, 12, 10))
        response = groups.mean(dim=1)
        detail_gate = module(groups, gate, 2, response)
        legacy = ((groups * gate.unsqueeze(1)).sum(dim=2) +
                  (groups * detail_gate.unsqueeze(1)).sum(dim=2))
        fused = (groups * (gate + detail_gate).unsqueeze(1)).sum(dim=2)
        torch.testing.assert_close(fused, legacy, rtol=2e-5, atol=2e-6)

        fused.square().mean().backward()
        self.assertTrue(torch.isfinite(groups.grad).all())
        self.assertTrue(all(
            parameter.grad is not None and torch.isfinite(parameter.grad).all()
            for parameter in module.parameters()))

    def test_liquid_sampler_eval_hard_assigns_unique_group_sets(self):
        sampler = LiquidSpectralSampler(
            num_spectral=8,
            spectral_kernel=3,
            num_groups=8,
            init_patterns=[[0, 1, 2]] * 8,
            embed_dims=16,
            tau=1.0,
            hard=False,
            eval_hard=True,
            hard_group_unique_sets=True,
        ).eval()
        logits = torch.zeros(2, 8, 3, 8)
        logits[:, :, 0, 4] = 10.0
        logits[:, :, 1, 2] = 10.0
        logits[:, :, 2, 1] = 10.0

        selected = sampler._sample(logits).argmax(dim=-1)

        for batch_selected in selected:
            canonical_sets = {
                tuple(sorted(group.tolist())) for group in batch_selected
            }
            self.assertEqual(len(canonical_sets), 8)
            for group in batch_selected:
                self.assertEqual(len(set(group.tolist())), 3)

    def test_liquid_sampler_group_set_constraint_keeps_soft_sampling(self):
        sampler = LiquidSpectralSampler(
            num_spectral=8,
            spectral_kernel=3,
            num_groups=8,
            init_patterns=[[0, 1, 2]] * 8,
            embed_dims=16,
            tau=1.0,
            hard=False,
            eval_hard=False,
            hard_group_unique_sets=True,
        ).eval()
        logits = torch.randn(2, 1, 3, 8).expand(-1, 8, -1, -1).clone()

        probs = sampler._sample(logits)

        torch.testing.assert_close(probs[:, 0], probs[:, 1])
        torch.testing.assert_close(probs, F.softmax(logits, dim=-1))

    def test_liquid_sampler_group_set_hard_st_has_gradients(self):
        sampler = LiquidSpectralSampler(
            num_spectral=8,
            spectral_kernel=3,
            num_groups=8,
            init_patterns=[[0, 1, 2]] * 8,
            embed_dims=16,
            tau=1.0,
            hard=True,
            hard_group_unique_sets=True,
        ).train()
        logits = torch.randn(2, 8, 3, 8, requires_grad=True)

        probs = sampler._sample(logits)
        weights = torch.randn_like(probs)
        (probs * weights).sum().backward()

        self.assertTrue(torch.isfinite(logits.grad).all())
        self.assertGreater(logits.grad.abs().sum().item(), 0)
        selected = probs.detach().argmax(dim=-1)
        for batch_selected in selected:
            canonical_sets = {
                tuple(sorted(group.tolist())) for group in batch_selected
            }
            self.assertEqual(len(canonical_sets), 8)

    def test_liquid_sampler_hard_sampling_keeps_soft_context(self):
        sampler = LiquidSpectralSampler(
            num_spectral=8,
            spectral_kernel=3,
            num_groups=8,
            init_patterns=[[0, 1, 2]] * 8,
            embed_dims=16,
            tau=1.0,
            hard=True,
            hard_group_unique_sets=True,
            use_soft_context_after_hard=True,
        ).train()
        logits = torch.randn(2, 8, 3, 8, requires_grad=True)

        hard_probs = sampler._sample(logits)
        context_probs = sampler.last_context_probs

        self.assertIsNotNone(context_probs)
        torch.testing.assert_close(
            hard_probs.detach().sum(dim=-1),
            torch.ones_like(hard_probs[..., 0]))
        self.assertTrue(torch.all(
            (hard_probs.detach() == 0) | (hard_probs.detach() == 1)))
        self.assertTrue(torch.all(context_probs > 0))
        self.assertFalse(torch.equal(
            hard_probs.detach(), context_probs.detach()))
        loss = (hard_probs * torch.randn_like(hard_probs)).sum()
        loss = loss + (context_probs * torch.randn_like(context_probs)).sum()
        loss.backward()
        self.assertTrue(torch.isfinite(logits.grad).all())
        self.assertGreater(logits.grad.abs().sum().item(), 0)

    def test_liquid_set_transport_zero_strength_is_exact_identity(self):
        sampler = LiquidSpectralSampler(
            num_spectral=8,
            spectral_kernel=3,
            num_groups=8,
            init_patterns=[[0, 1, 2]] * 8,
            embed_dims=16,
            hard_group_unique_sets=True,
            soft_group_set_transport=dict(initial_strength=0.0),
        )
        raw_probs = F.softmax(torch.randn(2, 8, 3, 8), dim=-1)

        projected = sampler._apply_soft_group_set_transport(raw_probs)

        self.assertIs(projected, raw_probs)
        self.assertIsNone(sampler.last_set_assignment)

    def test_liquid_set_transport_caps_collapsed_set_demand(self):
        sampler = LiquidSpectralSampler(
            num_spectral=8,
            spectral_kernel=3,
            num_groups=8,
            init_patterns=[[0, 1, 2]] * 8,
            embed_dims=16,
            hard_group_unique_sets=True,
            soft_group_set_transport=dict(
                initial_strength=1.0,
                num_iters=16,
                temperature=1.0),
        )
        logits = torch.zeros(2, 8, 3, 8, requires_grad=True)
        with torch.no_grad():
            logits[:, :, 0, 4] = 10.0
            logits[:, :, 1, 2] = 10.0
            logits[:, :, 2, 1] = 10.0

        projected = sampler._apply_soft_group_set_transport(
            F.softmax(logits, dim=-1))
        weights = torch.randn_like(projected)
        (projected * weights).sum().backward()

        torch.testing.assert_close(
            projected.sum(dim=-1),
            torch.ones_like(projected[..., 0]),
            atol=1e-6,
            rtol=1e-6)
        row_mass = sampler.last_set_assignment.sum(dim=-1)
        torch.testing.assert_close(
            row_mass, torch.ones_like(row_mass), atol=1e-5, rtol=1e-5)
        self.assertLessEqual(
            sampler.last_set_assignment.sum(dim=1).max().item(), 1.001)
        self.assertTrue(torch.isfinite(logits.grad).all())
        self.assertGreater(logits.grad.abs().sum().item(), 0)

    def test_liquid_set_transport_hard_st_has_gradients(self):
        sampler = LiquidSpectralSampler(
            num_spectral=8,
            spectral_kernel=3,
            num_groups=8,
            init_patterns=[[0, 1, 2]] * 8,
            embed_dims=16,
            hard=True,
            hard_group_unique_sets=True,
            soft_group_set_transport=dict(initial_strength=1.0),
        ).train()
        logits = torch.randn(2, 8, 3, 8, requires_grad=True)

        probs = sampler._sample(logits)
        (probs * torch.randn_like(probs)).sum().backward()

        self.assertTrue(torch.isfinite(logits.grad).all())
        self.assertGreater(logits.grad.abs().sum().item(), 0)
        self.assertIsNotNone(sampler.last_set_assignment)

    def test_relaxed_set_transport_rewards_but_does_not_force_unique_sets(self):
        sampler = LiquidSpectralSampler(
            num_spectral=8,
            spectral_kernel=3,
            num_groups=8,
            init_patterns=[[0, 1, 2]] * 8,
            embed_dims=16,
            hard=True,
            eval_hard=True,
            hard_group_unique_sets=False,
            use_soft_context_after_hard=True,
            soft_group_set_transport=dict(
                initial_strength=0.25,
                confidence_gated=True,
                margin_threshold=0.35,
                margin_temperature=0.1,
                min_gate=0.05,
                apply_to_independent_hard=True),
        ).eval()
        logits = torch.zeros(2, 8, 3, 8)
        logits[:, :, 0, 0] = 12.0
        logits[:, :, 1, 1] = 10.0
        logits[:, :, 2, 2] = 8.0

        selected = sampler._sample(logits).argmax(dim=-1)
        canonical_sets = {
            tuple(sorted(group.tolist())) for group in selected[0]
        }

        self.assertLess(len(canonical_sets), 8)
        self.assertIsNotNone(sampler.last_set_diversity_gate)
        self.assertLess(sampler.last_set_diversity_gate.mean().item(), 0.1)
        self.assertFalse(torch.equal(
            sampler.last_context_probs.detach(),
            F.one_hot(selected, num_classes=8).to(
                sampler.last_context_probs.dtype)))

    def test_liquid_sampler_band_attention_forward(self):
        sampler = LiquidSpectralSampler(
            num_spectral=8,
            spectral_kernel=3,
            num_groups=8,
            init_patterns=[
                [7, 0, 1],
                [0, 1, 2],
                [1, 2, 3],
                [2, 3, 4],
                [3, 4, 5],
                [4, 5, 6],
                [5, 6, 7],
                [6, 7, 0],
            ],
            embed_dims=16,
            tau=1.0,
            hard=False,
            head_weight_std=1e-3,
            eval_hard=False,
            use_band_attention=True,
            band_attention_heads=4,
        ).train()
        x = torch.randn(2, 8, 16, 20, requires_grad=True)
        sampled, probs = sampler(x)

        self.assertEqual(sampled.shape, (2, 8, 3, 16, 20))
        self.assertEqual(probs.shape, (2, 8, 3, 8))

        loss = sampled.square().mean()
        loss.backward()
        self.assertIsNotNone(sampler.band_attn.in_proj_weight.grad)
        self.assertGreater(
            sampler.band_attn.in_proj_weight.grad.abs().sum().item(), 0)

    def test_liquid_aware_fusion_outputs_se_logit_delta(self):
        init_patterns = [
            [7, 0, 1],
            [0, 1, 2],
            [1, 2, 3],
            [2, 3, 4],
            [3, 4, 5],
            [4, 5, 6],
            [5, 6, 7],
            [6, 7, 0],
        ]
        stem = MultispecStemConv3dSE(
            out_channels=16,
            num_spectral=8,
            reduction=2,
            liquid_sampler=dict(
                embed_dims=32,
                num_groups=8,
                init_patterns=init_patterns,
                tau=1.0,
                hard=False,
                eval_hard=False,
                liquid_aware_fusion=dict(embed_dims=16, num_heads=4)),
        ).train()
        x = torch.randn(2, 8, 32, 32, requires_grad=True)
        out, groups, probs = stem(x, return_sampling=True)

        self.assertEqual(out.shape, (2, 16, 16, 16))
        self.assertEqual(groups.shape, (2, 16, 8, 16, 16))
        self.assertEqual(probs.shape, (2, 8, 3, 8))
        self.assertIsNotNone(stem.last_liquid_aware_delta)
        self.assertEqual(stem.last_liquid_aware_delta.shape, (2, 8, 16, 16))

        loss = out.square().mean()
        loss.backward()
        self.assertIsNotNone(x.grad)
        self.assertIsNotNone(stem.liquid_aware_fusion.out_proj.weight.grad)
        self.assertGreater(
            stem.liquid_aware_fusion.out_proj.weight.grad.abs().sum().item(),
            0)

    def test_liquid_aware_fusion_overlap_pattern_bias(self):
        stem = MultispecStemConv3dSE(
            out_channels=16,
            num_spectral=8,
            reduction=2,
            liquid_sampler=dict(
                embed_dims=16,
                num_groups=8,
                init_patterns=[
                    [7, 0, 1],
                    [0, 1, 2],
                    [1, 2, 3],
                    [2, 3, 4],
                    [3, 4, 5],
                    [4, 5, 6],
                    [5, 6, 7],
                    [6, 7, 0],
                ],
                tau=1.0,
                hard=False,
                eval_hard=False,
                liquid_aware_fusion=dict(
                    embed_dims=16,
                    num_heads=4,
                    use_overlap_context=True,
                    use_spatial_mixer=False)),
        ).train()
        x = torch.randn(2, 8, 32, 32, requires_grad=True)
        out, _, _ = stem(x, return_sampling=True)

        self.assertEqual(out.shape, (2, 16, 16, 16))
        self.assertIsNone(stem.liquid_aware_fusion.spatial_mixer)
        self.assertIsNotNone(stem.liquid_aware_fusion.overlap_proj)
        self.assertEqual(stem.last_liquid_aware_delta.shape, (2, 8, 16, 16))

        out.mean().backward()
        self.assertIsNotNone(stem.liquid_aware_fusion.overlap_proj.weight.grad)

    def test_liquid_group_modulator_forward(self):
        stem = MultispecStemConv3dSE(
            out_channels=16,
            num_spectral=8,
            reduction=2,
            liquid_sampler=dict(
                embed_dims=16,
                num_groups=8,
                init_patterns=[
                    [7, 0, 1],
                    [0, 1, 2],
                    [1, 2, 3],
                    [2, 3, 4],
                    [3, 4, 5],
                    [4, 5, 6],
                    [5, 6, 7],
                    [6, 7, 0],
                ],
                tau=1.0,
                hard=False,
                eval_hard=False,
                liquid_group_modulator=dict(hidden_dims=8)),
        ).train()
        x = torch.randn(2, 8, 32, 32, requires_grad=True)
        out, groups, probs = stem(x, return_sampling=True)

        self.assertEqual(out.shape, (2, 16, 16, 16))
        self.assertEqual(groups.shape, (2, 16, 8, 16, 16))
        self.assertEqual(probs.shape, (2, 8, 3, 8))

        out.square().mean().backward()
        self.assertIsNotNone(stem.liquid_group_modulator.mlp[-1].weight.grad)
        self.assertGreater(
            stem.liquid_group_modulator.mlp[-1].weight.grad.abs().sum().item(),
            0)

    def test_pair_transport_starts_from_wide_groupmod_baseline(self):
        init_patterns = [
            [7, 0, 1],
            [0, 1, 2],
            [1, 2, 3],
            [2, 3, 4],
            [3, 4, 5],
            [4, 5, 6],
            [5, 6, 7],
            [6, 7, 0],
        ]
        common_sampler = dict(
            embed_dims=16,
            num_groups=8,
            init_patterns=init_patterns,
            tau=1.0,
            hard=False,
            eval_hard=False,
            liquid_aware_fusion=dict(
                embed_dims=16,
                num_heads=4,
                use_overlap_context=True,
                use_spatial_mixer=True),
            liquid_group_modulator=dict(hidden_dims=8),
        )
        baseline = MultispecStemConv3dSE(
            out_channels=16,
            num_spectral=8,
            reduction=2,
            liquid_sampler=common_sampler,
        ).eval()
        pair_sampler = dict(common_sampler)
        pair_sampler['liquid_aware_fusion'] = dict(
            common_sampler['liquid_aware_fusion'],
            pair_transport=dict(
                hidden_dims=32,
                temperature=0.25,
                zero_init=True,
                relation_mode='pair'))
        pair_sampler['pair_sampler_router'] = dict(
            hidden_dims=16, zero_init=True, relation_mode='pair')
        paired = MultispecStemConv3dSE(
            out_channels=16,
            num_spectral=8,
            reduction=2,
            liquid_sampler=pair_sampler,
        ).eval()
        incompatible = paired.load_state_dict(
            baseline.state_dict(), strict=False)
        self.assertFalse(incompatible.unexpected_keys)
        self.assertTrue(all(
            'pair_sampler_router' in key or 'pair_transport' in key
            for key in incompatible.missing_keys))
        self.assertEqual(
            paired.liquid_sampler.pair_sampler_router.mlp[0].in_features,
            32)
        self.assertEqual(
            paired.liquid_aware_fusion.pair_transport.mlp[0].in_features,
            32)

        x = torch.randn(4, 8, 24, 20)
        baseline_out, _, baseline_probs = baseline(
            x, return_sampling=True)
        paired.set_pair_batch_size(2)
        paired_out, _, paired_probs = paired(x, return_sampling=True)
        torch.testing.assert_close(paired_probs, baseline_probs)
        torch.testing.assert_close(paired_out, baseline_out)
        transport = paired.liquid_aware_fusion.last_pair_transport
        self.assertEqual(transport.shape, (2, 2, 8, 8))
        torch.testing.assert_close(
            transport.sum(dim=-1), torch.ones_like(transport[..., 0]))

        paired.train()
        x = torch.randn(4, 8, 24, 20, requires_grad=True)
        out = paired(x)
        out.square().mean().backward()
        sampler_grad = paired.liquid_sampler.pair_sampler_router.mlp[-1]
        fusion_grad = paired.liquid_aware_fusion.pair_transport.mlp[-1]
        self.assertGreater(sampler_grad.weight.grad.abs().sum().item(), 0)
        self.assertGreater(fusion_grad.weight.grad.abs().sum().item(), 0)

        paired.eval()
        paired.liquid_sampler.eval_hard = True
        _, _, hard_probs = paired(x.detach(), return_sampling=True)
        selected = hard_probs.argmax(dim=-1)
        for batch_idx in range(selected.size(0)):
            for group_idx in range(selected.size(1)):
                self.assertEqual(
                    len(set(selected[batch_idx, group_idx].tolist())), 3)

    def test_pair_consensus_aligned_fusion_is_initially_exact_and_trainable(self):
        init_patterns = [
            [7, 0, 1], [0, 1, 2], [1, 2, 3], [2, 3, 4],
            [3, 4, 5], [4, 5, 6], [5, 6, 7], [6, 7, 0]
        ]
        common_sampler = dict(
            embed_dims=16,
            num_groups=8,
            init_patterns=init_patterns,
            tau=1.0,
            hard=False,
            eval_hard=False,
            pair_consensus_router=dict(hidden_dims=24),
            liquid_aware_fusion=dict(
                embed_dims=16,
                num_heads=4,
                use_overlap_context=True,
                use_spatial_mixer=True),
            liquid_group_modulator=dict(hidden_dims=8),
        )
        baseline = MultispecStemConv3dSE(
            out_channels=16,
            num_spectral=8,
            reduction=2,
            liquid_sampler=common_sampler,
        ).eval()
        aligned_sampler = copy.deepcopy(common_sampler)
        aligned_sampler['liquid_aware_fusion']['pair_aligned_coupling'] = dict(
            hidden_dims=32,
            zero_init=True,
            relation_mode='pair_diff_product')
        aligned = MultispecStemConv3dSE(
            out_channels=16,
            num_spectral=8,
            reduction=2,
            liquid_sampler=aligned_sampler,
        ).eval()
        incompatible = aligned.load_state_dict(
            baseline.state_dict(), strict=False)
        self.assertFalse(incompatible.unexpected_keys)
        self.assertTrue(all(
            'pair_aligned_coupling' in key
            for key in incompatible.missing_keys))

        x = torch.randn(4, 8, 24, 20)
        baseline.set_pair_batch_size(2)
        aligned.set_pair_batch_size(2)
        baseline_out, _, baseline_probs = baseline(
            x, return_sampling=True)
        aligned_out, _, aligned_probs = aligned(
            x, return_sampling=True)
        torch.testing.assert_close(aligned_probs[:2], aligned_probs[2:])
        torch.testing.assert_close(aligned_probs, baseline_probs)
        torch.testing.assert_close(aligned_out, baseline_out)

        aligned.train()
        x = torch.randn(4, 8, 24, 20, requires_grad=True)
        out = aligned(x)
        out.square().mean().backward()
        coupling = aligned.liquid_aware_fusion.pair_aligned_coupling
        self.assertGreater(
            coupling.mlp[-1].weight.grad.abs().sum().item(), 0)
        self.assertIsNotNone(coupling.last_pair_token_distance)
        self.assertTrue(torch.isfinite(x.grad).all())

    def test_pair_transport_and_aligned_coupling_are_mutually_exclusive(self):
        with self.assertRaisesRegex(AssertionError, 'mutually exclusive'):
            MultispecStemConv3dSE(
                out_channels=16,
                num_spectral=8,
                reduction=2,
                liquid_sampler=dict(
                    liquid_aware_fusion=dict(
                        pair_transport={},
                        pair_aligned_coupling={})))

    def test_pair_band_context_starts_from_wide_groupmod_baseline(self):
        init_patterns = [
            [7, 0, 1],
            [0, 1, 2],
            [1, 2, 3],
            [2, 3, 4],
            [3, 4, 5],
            [4, 5, 6],
            [5, 6, 7],
            [6, 7, 0],
        ]
        common_sampler = dict(
            embed_dims=16,
            num_groups=8,
            init_patterns=init_patterns,
            tau=1.0,
            hard=False,
            eval_hard=False,
            liquid_aware_fusion=dict(
                embed_dims=16,
                num_heads=4,
                use_overlap_context=True,
                use_spatial_mixer=True),
            liquid_group_modulator=dict(hidden_dims=8),
        )
        baseline = MultispecStemConv3dSE(
            out_channels=16,
            num_spectral=8,
            reduction=2,
            liquid_sampler=common_sampler,
        ).eval()
        pair_sampler = dict(common_sampler)
        pair_sampler['pair_band_context'] = dict(
            hidden_dims=32, zero_init=True, relation_mode='pair')
        pair_sampler['liquid_aware_fusion'] = dict(
            common_sampler['liquid_aware_fusion'],
            pair_band_context_fusion=dict(
                context_dims=16, hidden_dims=32, zero_init=True))
        paired = MultispecStemConv3dSE(
            out_channels=16,
            num_spectral=8,
            reduction=2,
            liquid_sampler=pair_sampler,
        ).eval()
        incompatible = paired.load_state_dict(
            baseline.state_dict(), strict=False)
        self.assertFalse(incompatible.unexpected_keys)
        self.assertTrue(all(
            'pair_band_context' in key
            for key in incompatible.missing_keys))

        x = torch.randn(4, 8, 24, 20)
        baseline_out, _, baseline_probs = baseline(
            x, return_sampling=True)
        paired.set_pair_batch_size(2)
        paired_out, _, paired_probs = paired(x, return_sampling=True)
        torch.testing.assert_close(paired_probs, baseline_probs)
        torch.testing.assert_close(paired_out, baseline_out)
        self.assertEqual(
            paired.liquid_sampler.last_pair_band_context.shape, (4, 8, 16))

        paired.train()
        out = paired(torch.randn(4, 8, 24, 20))
        out.square().mean().backward()
        sampler_grad = paired.liquid_sampler.pair_band_context.logit_delta
        fusion = paired.liquid_aware_fusion.pair_band_context_fusion
        self.assertGreater(sampler_grad.weight.grad.abs().sum().item(), 0)
        self.assertGreater(fusion.mlp[-1].weight.grad.abs().sum().item(), 0)

    def test_pair_change_gate_starts_from_wide_groupmod_baseline(self):
        init_patterns = [
            [7, 0, 1],
            [0, 1, 2],
            [1, 2, 3],
            [2, 3, 4],
            [3, 4, 5],
            [4, 5, 6],
            [5, 6, 7],
            [6, 7, 0],
        ]
        common_sampler = dict(
            embed_dims=16,
            num_groups=8,
            init_patterns=init_patterns,
            tau=1.0,
            hard=False,
            eval_hard=False,
            liquid_aware_fusion=dict(
                embed_dims=16,
                num_heads=4,
                use_overlap_context=True,
                use_spatial_mixer=True),
            liquid_group_modulator=dict(hidden_dims=8),
        )
        baseline = MultispecStemConv3dSE(
            out_channels=16,
            num_spectral=8,
            reduction=2,
            liquid_sampler=common_sampler,
        ).eval()
        pair_sampler = dict(common_sampler)
        pair_sampler['liquid_aware_fusion'] = dict(
            common_sampler['liquid_aware_fusion'],
            pair_change_gate=dict(hidden_dims=8, zero_init=True))
        paired = MultispecStemConv3dSE(
            out_channels=16,
            num_spectral=8,
            reduction=2,
            liquid_sampler=pair_sampler,
        ).eval()
        incompatible = paired.load_state_dict(baseline.state_dict(), strict=False)
        self.assertFalse(incompatible.unexpected_keys)
        self.assertTrue(all(
            'pair_change_gate' in key for key in incompatible.missing_keys))

        x = torch.randn(4, 8, 24, 20)
        baseline_out, _, baseline_probs = baseline(x, return_sampling=True)
        paired.set_pair_batch_size(2)
        paired_out, _, paired_probs = paired(x, return_sampling=True)
        torch.testing.assert_close(paired_probs, baseline_probs)
        torch.testing.assert_close(paired_out, baseline_out)
        reliability = paired.liquid_aware_fusion.last_pair_change_reliability
        self.assertEqual(reliability.shape, (4, 8, 1))
        self.assertTrue(torch.all((reliability > 0) & (reliability < 1)))

        paired.train()
        out = paired(torch.randn(4, 8, 24, 20))
        out.square().mean().backward()
        coupling = paired.liquid_aware_fusion.pair_change_gate
        self.assertGreater(coupling.out_proj.weight.grad.abs().sum().item(), 0)
        self.assertTrue(all(
            parameter.grad is not None
            for parameter in coupling.parameters() if parameter.requires_grad))

        paired.set_pair_batch_size(None)
        paired(torch.randn(2, 8, 24, 20))
        self.assertIsNone(paired.liquid_sampler.last_pair_band_context)

    def test_liquid_aware_output_residual_forward(self):
        stem = MultispecStemConv3dSE(
            out_channels=16,
            num_spectral=8,
            reduction=2,
            liquid_sampler=dict(
                embed_dims=16,
                num_groups=8,
                init_patterns=[
                    [7, 0, 1],
                    [0, 1, 2],
                    [1, 2, 3],
                    [2, 3, 4],
                    [3, 4, 5],
                    [4, 5, 6],
                    [5, 6, 7],
                    [6, 7, 0],
                ],
                tau=1.0,
                hard=False,
                eval_hard=False,
                liquid_aware_fusion=dict(
                    embed_dims=16,
                    num_heads=4,
                    use_overlap_context=True,
                    use_spatial_mixer=True,
                    output_residual=dict(init_value=0.05))),
        ).train()
        x = torch.randn(2, 8, 32, 32, requires_grad=True)
        out, _, _ = stem(x, return_sampling=True)

        self.assertEqual(out.shape, (2, 16, 16, 16))
        self.assertIsNotNone(stem.liquid_output_residual_scale)
        self.assertIsNotNone(stem.last_liquid_aware_delta)

        out.mean().backward()
        self.assertIsNotNone(stem.liquid_output_residual_scale.grad)
        self.assertIsNotNone(stem.liquid_aware_fusion.out_proj.weight.grad)

    def test_fusion_quality_conservation_constraints_and_gradients(self):
        base_logits = torch.randn(2, 8, 6, 5)
        response = torch.randn_like(base_logits)
        for mode in ('gate_mass', 'response_mass', 'dual_moment'):
            delta = torch.randn_like(base_logits, requires_grad=True)
            module = FusionQualityConservation(mode=mode)
            corrected = module(base_logits, delta, response)

            base_gate = base_logits.sigmoid()
            sensitivity = base_gate * (1.0 - base_gate)
            response_scale = response.abs()
            response_scale = response_scale / response_scale.mean(
                dim=1, keepdim=True).clamp_min(1e-4)
            weight = (sensitivity * response_scale
                      if mode == 'response_mass' else sensitivity)
            weight = weight.clamp_min(1e-4)
            constraint = (weight * corrected).sum(dim=1)
            torch.testing.assert_close(
                constraint, torch.zeros_like(constraint), atol=2e-5, rtol=2e-5)

            if mode == 'dual_moment':
                response_mean = (weight * response_scale).sum(
                    dim=1, keepdim=True) / weight.sum(
                        dim=1, keepdim=True)
                centered = response_scale - response_mean
                response_constraint = (
                    weight * centered * corrected).sum(dim=1)
                torch.testing.assert_close(
                    response_constraint,
                    torch.zeros_like(response_constraint),
                    atol=2e-5,
                    rtol=2e-5)

            corrected.square().mean().backward()
            self.assertTrue(torch.isfinite(delta.grad).all())
            self.assertGreater(delta.grad.abs().sum().item(), 0)

    def test_liquid_sampler_lowres_grad_correction(self):
        sampler = LiquidSpectralSampler(
            num_spectral=8,
            spectral_kernel=3,
            embed_dims=16,
            tau=1.0,
            hard=False,
            lowres_grad_size=4,
        ).train()
        x = torch.randn(2, 8, 16, 20, requires_grad=True)
        sampled, probs = sampler(x)

        expected = torch.bmm(
            probs.reshape(2, 18, 8).detach(),
            x.flatten(2)).view(2, 6, 3, 16, 20)
        torch.testing.assert_close(sampled, expected)

        loss = sampled.square().mean()
        loss.backward()
        self.assertIsNotNone(x.grad)
        self.assertGreater(x.grad.abs().sum().item(), 0)
        self.assertIsNotNone(sampler.head.bias.grad)
        self.assertGreater(sampler.head.bias.grad.abs().sum().item(), 0)

    def test_liquid_sampler_bilinear_expand_matches_interpolate(self):
        source = torch.randn(2, 5, 4, 7, requires_grad=True)
        reference_source = source.detach().clone().requires_grad_(True)

        actual = LiquidSpectralSampler._bilinear_expand(source, (15, 22))
        expected = F.interpolate(
            reference_source,
            size=(15, 22),
            mode='bilinear',
            align_corners=False)
        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)

        gradient = torch.randn_like(actual)
        actual.backward(gradient)
        expected.backward(gradient)
        torch.testing.assert_close(
            source.grad, reference_source.grad, rtol=1e-5, atol=1e-6)

        bf16_output = LiquidSpectralSampler._bilinear_expand(
            source.detach().to(torch.bfloat16), (15, 22))
        self.assertEqual(bf16_output.dtype, torch.bfloat16)


if __name__ == '__main__':
    unittest.main()
