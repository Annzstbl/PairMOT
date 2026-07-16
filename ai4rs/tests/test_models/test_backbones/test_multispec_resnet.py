# Copyright (c) AI4RS. All rights reserved.
import tempfile
import unittest

import numpy as np
import torch
import torch.nn.functional as F

from projects.multispec_rotated_rtdetr.multispec_rotated_rtdetr.pretrain_utils import (
    adapt_state_dict_in_channels, adapt_state_dict_stem_conv3d_se,
    expand_conv1_weight)
from projects.multispec_rotated_rtdetr.multispec_rotated_rtdetr.stem_conv3d_se import (
    LiquidSpectralSampler, MultispecStemConv3dSE)


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
