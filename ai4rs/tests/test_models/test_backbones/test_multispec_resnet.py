# Copyright (c) AI4RS. All rights reserved.
import tempfile
import unittest

import numpy as np
import torch

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


if __name__ == '__main__':
    unittest.main()
