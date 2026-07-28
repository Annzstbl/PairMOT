"""Unit tests for pair temporal adapter in RT-DETR hybrid encoder."""

import unittest

import torch

from projects.rotated_rtdetr.rotated_rtdetr.rtdetr_layers import (
    PairTemporalAdapter,
    PairTemporalPoolGateAdapter,
    PairTemporalPyramidCommonDetailAdapter,
    PairTemporalPyramidDualEvidenceAdapter,
    PairTemporalPyramidLocalAdapter,
    RTDETRHybridEncoder,
)


class TestPairTemporalAdapter(unittest.TestCase):

    def test_gamma_zero_is_exact_identity(self):
        torch.manual_seed(1)
        adapter = PairTemporalAdapter(embed_dims=8, num_heads=2)
        feat = torch.randn(4, 8, 5, 3)

        out = adapter(feat)

        self.assertTrue(torch.equal(out, feat))
        self.assertEqual(float(adapter.gamma.detach()), 0.0)

    def test_nonzero_gamma_changes_output_and_backpropagates(self):
        torch.manual_seed(2)
        adapter = PairTemporalAdapter(embed_dims=8, num_heads=2)
        adapter.gamma.data.fill_(1.0)
        feat = torch.randn(4, 8, 5, 3, requires_grad=True)

        out = adapter(feat)
        loss = out.square().mean()
        loss.backward()

        self.assertFalse(torch.allclose(out.detach(), feat.detach()))
        self.assertIsNotNone(feat.grad)
        self.assertGreater(float(feat.grad.abs().sum()), 0.0)
        self.assertIsNotNone(adapter.gamma.grad)
        self.assertGreater(float(adapter.gamma.grad.abs()), 0.0)
        self.assertIsNotNone(adapter.attn.in_proj_weight.grad)
        self.assertGreater(
            float(adapter.attn.in_proj_weight.grad.abs().sum()), 0.0)

    def test_odd_batch_raises(self):
        adapter = PairTemporalAdapter(embed_dims=8, num_heads=2)
        with self.assertRaisesRegex(ValueError, 'even batch'):
            adapter(torch.randn(3, 8, 5, 3))

    def test_hybrid_encoder_builds_adapter_on_last_encoded_level(self):
        encoder = RTDETRHybridEncoder(
            layer_cfg=dict(
                self_attn_cfg=dict(embed_dims=8, num_heads=2, dropout=0.0),
                ffn_cfg=dict(
                    embed_dims=8, feedforward_channels=16, ffn_drop=0.0)),
            in_channels=[8, 8, 8],
            use_encoder_idx=[-1],
            num_encoder_layers=1,
            pair_temporal_adapter_cfg=dict(num_heads=2),
            fpn_cfg=None,
        )

        self.assertIsInstance(encoder.pair_temporal_adapter,
                              PairTemporalAdapter)
        self.assertEqual(encoder.pair_temporal_adapter_idx, -1)
        self.assertEqual(encoder.pair_temporal_adapter.embed_dims, 8)

    def test_pool_gate_gamma_zero_is_exact_identity(self):
        torch.manual_seed(3)
        adapter = PairTemporalPoolGateAdapter(embed_dims=8, reduction=2)
        feat = torch.randn(4, 8, 5, 3)

        out = adapter(feat)

        self.assertTrue(torch.equal(out, feat))
        self.assertEqual(float(adapter.gamma.detach()), 0.0)

    def test_pool_gate_nonzero_gamma_backpropagates(self):
        torch.manual_seed(4)
        adapter = PairTemporalPoolGateAdapter(embed_dims=8, reduction=2)
        adapter.gamma.data.fill_(1.0)
        feat = torch.randn(4, 8, 5, 3, requires_grad=True)

        out = adapter(feat)
        loss = out.square().mean()
        loss.backward()

        self.assertFalse(torch.allclose(out.detach(), feat.detach()))
        self.assertIsNotNone(feat.grad)
        self.assertGreater(float(feat.grad.abs().sum()), 0.0)
        self.assertIsNotNone(adapter.gamma.grad)
        self.assertGreater(float(adapter.gamma.grad.abs()), 0.0)
        self.assertIsNotNone(adapter.delta_conv[-1].weight.grad)
        self.assertGreater(
            float(adapter.delta_conv[-1].weight.grad.abs().sum()), 0.0)

    def test_hybrid_encoder_builds_pool_gate_adapter(self):
        encoder = RTDETRHybridEncoder(
            layer_cfg=dict(
                self_attn_cfg=dict(embed_dims=8, num_heads=2, dropout=0.0),
                ffn_cfg=dict(
                    embed_dims=8, feedforward_channels=16, ffn_drop=0.0)),
            in_channels=[8, 8, 8],
            use_encoder_idx=[-1],
            num_encoder_layers=1,
            pair_temporal_adapter_cfg=dict(type='pool_gate', reduction=2),
            fpn_cfg=None,
        )

        self.assertIsInstance(encoder.pair_temporal_adapter,
                              PairTemporalPoolGateAdapter)
        self.assertEqual(encoder.pair_temporal_adapter_idx, -1)
        self.assertEqual(encoder.pair_temporal_adapter.embed_dims, 8)

    def test_pyramid_local_gamma_zero_is_exact_identity(self):
        torch.manual_seed(5)
        adapter = PairTemporalPyramidLocalAdapter(
            in_channels=[8, 8, 8], pointwise_groups=2)
        feats = (
            torch.randn(4, 8, 12, 12),
            torch.randn(4, 8, 6, 6),
            torch.randn(4, 8, 3, 3),
        )

        outs = adapter(feats)

        for out, feat in zip(outs, feats):
            self.assertTrue(torch.equal(out, feat))
        self.assertTrue(torch.equal(adapter.gamma.detach(),
                                    torch.zeros(3)))

    def test_pyramid_local_nonzero_gamma_backpropagates(self):
        torch.manual_seed(6)
        adapter = PairTemporalPyramidLocalAdapter(
            in_channels=[8, 8, 8], level_indices=[0, 2], pointwise_groups=2)
        adapter.gamma.data.fill_(1.0)
        feats = (
            torch.randn(4, 8, 12, 12, requires_grad=True),
            torch.randn(4, 8, 6, 6, requires_grad=True),
            torch.randn(4, 8, 3, 3, requires_grad=True),
        )

        outs = adapter(feats)
        loss = sum(out.square().mean() for out in outs)
        loss.backward()

        self.assertFalse(torch.allclose(outs[0].detach(), feats[0].detach()))
        self.assertTrue(torch.equal(outs[1].detach(), feats[1].detach()))
        self.assertFalse(torch.allclose(outs[2].detach(), feats[2].detach()))
        self.assertIsNotNone(feats[0].grad)
        self.assertGreater(float(feats[0].grad.abs().sum()), 0.0)
        self.assertIsNotNone(adapter.gamma.grad)
        self.assertGreater(float(adapter.gamma.grad.abs().sum()), 0.0)
        self.assertIsNotNone(adapter.local_blocks[0][-1].weight.grad)
        self.assertGreater(
            float(adapter.local_blocks[0][-1].weight.grad.abs().sum()), 0.0)

    def test_hybrid_encoder_builds_post_pyramid_local_adapter(self):
        encoder = RTDETRHybridEncoder(
            layer_cfg=dict(
                self_attn_cfg=dict(embed_dims=8, num_heads=2, dropout=0.0),
                ffn_cfg=dict(
                    embed_dims=8, feedforward_channels=16, ffn_drop=0.0)),
            in_channels=[8, 8, 8],
            use_encoder_idx=[-1],
            num_encoder_layers=1,
            pair_temporal_adapter_cfg=dict(num_heads=2),
            post_pair_temporal_adapter_cfg=dict(
                type='pyramid_local', pointwise_groups=2),
            fpn_cfg=None,
        )

        self.assertIsInstance(encoder.pair_temporal_adapter,
                              PairTemporalAdapter)
        self.assertIsInstance(encoder.post_pair_temporal_adapter,
                              PairTemporalPyramidLocalAdapter)
        self.assertEqual(
            encoder.post_pair_temporal_adapter.level_indices, [0, 1, 2])

    def test_common_detail_zero_gate_is_exact_identity(self):
        torch.manual_seed(7)
        adapter = PairTemporalPyramidCommonDetailAdapter(
            in_channels=[8, 8, 8], pointwise_groups=2)
        feats = (
            torch.randn(4, 8, 12, 12),
            torch.randn(4, 8, 6, 6),
            torch.randn(4, 8, 3, 3),
        )

        outs = adapter(feats)

        for out, feat in zip(outs, feats):
            self.assertTrue(torch.equal(out, feat))

    def test_common_detail_preserves_mean_and_is_swap_equivariant(self):
        torch.manual_seed(8)
        adapter = PairTemporalPyramidCommonDetailAdapter(
            in_channels=[8], pointwise_groups=2)
        adapter.gamma.data.fill_(0.5)
        feat = torch.randn(4, 8, 7, 5, requires_grad=True)

        out = adapter((feat, ))[0]
        pair_batch = feat.shape[0] // 2
        input_mean = (feat[:pair_batch] + feat[pair_batch:]) * 0.5
        output_mean = (out[:pair_batch] + out[pair_batch:]) * 0.5
        self.assertTrue(torch.allclose(output_mean, input_mean, atol=1e-7))

        swapped = torch.cat([feat[pair_batch:], feat[:pair_batch]], dim=0)
        swapped_out = adapter((swapped, ))[0]
        expected = torch.cat([out[pair_batch:], out[:pair_batch]], dim=0)
        self.assertTrue(torch.allclose(swapped_out, expected, atol=1e-6))

        out.square().mean().backward()
        self.assertGreater(float(feat.grad.abs().sum()), 0.0)
        self.assertGreater(float(adapter.gamma.grad.abs().sum()), 0.0)
        self.assertGreater(
            float(adapter.local_blocks[0][0].weight.grad.abs().sum()), 0.0)
        self.assertGreater(
            float(adapter.local_blocks[0][-1].weight.grad.abs().sum()), 0.0)

    def test_hybrid_encoder_builds_common_detail_adapter(self):
        encoder = RTDETRHybridEncoder(
            layer_cfg=dict(
                self_attn_cfg=dict(embed_dims=8, num_heads=2, dropout=0.0),
                ffn_cfg=dict(
                    embed_dims=8, feedforward_channels=16, ffn_drop=0.0)),
            in_channels=[8, 8, 8],
            use_encoder_idx=[-1],
            num_encoder_layers=1,
            pair_temporal_adapter_cfg=dict(num_heads=2),
            post_pair_temporal_adapter_cfg=dict(
                type='pyramid_common_detail', pointwise_groups=2),
            fpn_cfg=None,
        )

        self.assertIsInstance(
            encoder.post_pair_temporal_adapter,
            PairTemporalPyramidCommonDetailAdapter)

    def test_common_detail_energy_conservation_caps_update(self):
        torch.manual_seed(12)
        adapter = PairTemporalPyramidCommonDetailAdapter(
            in_channels=[8],
            pointwise_groups=2,
            conserve_detail_energy=True)
        adapter.gamma.data.fill_(100.0)
        feat = torch.randn(4, 8, 7, 5)

        out = adapter((feat, ))[0]
        pair_batch = feat.shape[0] // 2
        prev = feat[:pair_batch]
        curr = feat[pair_batch:]
        detail = (curr - prev) * 0.5
        update = out[:pair_batch] - prev
        detail_rms = detail.square().mean(dim=(-2, -1)).sqrt()
        update_rms = update.square().mean(dim=(-2, -1)).sqrt()

        self.assertTrue(torch.all(update_rms <= detail_rms + 1e-6))
        input_mean = (prev + curr) * 0.5
        output_mean = (
            out[:pair_batch] + out[pair_batch:]) * 0.5
        self.assertTrue(torch.allclose(output_mean, input_mean, atol=1e-6))

    def test_common_detail_spatial_reliability_is_equivariant_and_trainable(
            self):
        torch.manual_seed(15)
        adapter = PairTemporalPyramidCommonDetailAdapter(
            in_channels=[8],
            pointwise_groups=2,
            use_spatial_reliability=True)
        adapter.gamma.data.fill_(0.5)
        feat = torch.randn(4, 8, 7, 5, requires_grad=True)

        out = adapter((feat, ))[0]
        pair_batch = feat.shape[0] // 2
        input_mean = (feat[:pair_batch] + feat[pair_batch:]) * 0.5
        output_mean = (out[:pair_batch] + out[pair_batch:]) * 0.5
        self.assertTrue(torch.allclose(output_mean, input_mean, atol=1e-7))

        swapped = torch.cat([feat[pair_batch:], feat[:pair_batch]], dim=0)
        swapped_out = adapter((swapped, ))[0]
        expected = torch.cat([out[pair_batch:], out[:pair_batch]], dim=0)
        self.assertTrue(torch.allclose(swapped_out, expected, atol=1e-6))

        out.square().mean().backward()
        spatial_weight = adapter.spatial_gates[0].weight
        self.assertIsNotNone(spatial_weight.grad)
        self.assertGreater(float(spatial_weight.grad.abs().sum()), 0.0)

    def test_common_detail_spatial_reliability_starts_as_parent(self):
        torch.manual_seed(16)
        parent = PairTemporalPyramidCommonDetailAdapter(
            in_channels=[8], pointwise_groups=2)
        spatial = PairTemporalPyramidCommonDetailAdapter(
            in_channels=[8],
            pointwise_groups=2,
            use_spatial_reliability=True)
        spatial.load_state_dict(parent.state_dict(), strict=False)
        parent.gamma.data.fill_(0.75)
        spatial.gamma.data.copy_(parent.gamma.data)
        feat = torch.randn(4, 8, 7, 5)

        parent_out = parent((feat, ))[0]
        spatial_out = spatial((feat, ))[0]

        self.assertTrue(torch.equal(spatial_out, parent_out))

    def test_common_detail_shared_scalar_gain_starts_as_parent(self):
        torch.manual_seed(17)
        parent = PairTemporalPyramidCommonDetailAdapter(
            in_channels=[8], pointwise_groups=2)
        scalar = PairTemporalPyramidCommonDetailAdapter(
            in_channels=[8],
            pointwise_groups=2,
            use_shared_scalar_gain=True)
        scalar.load_state_dict(parent.state_dict(), strict=False)
        parent.gamma.data.fill_(0.75)
        scalar.gamma.data.copy_(parent.gamma.data)
        feat = torch.randn(4, 8, 7, 5)

        self.assertTrue(torch.equal(
            scalar((feat, ))[0], parent((feat, ))[0]))

    def test_common_detail_shared_scalar_gain_preserves_pair_geometry(self):
        torch.manual_seed(18)
        adapter = PairTemporalPyramidCommonDetailAdapter(
            in_channels=[8],
            pointwise_groups=2,
            use_shared_scalar_gain=True)
        adapter.shared_gain_gates[0].weight.data.fill_(0.1)
        feat = torch.randn(4, 8, 7, 5, requires_grad=True)

        out = adapter((feat, ))[0]
        pair_batch = feat.shape[0] // 2
        prev = feat[:pair_batch]
        curr = feat[pair_batch:]
        common = (prev + curr) * 0.5
        detail = (curr - prev) * 0.5
        common_energy = common.abs().mean(dim=1, keepdim=True)
        detail_energy = detail.abs().mean(dim=1, keepdim=True)
        spatial_descriptor = torch.cat(
            [
                common_energy / common_energy.detach().mean(
                    dim=(-2, -1), keepdim=True).clamp_min(1e-6),
                detail_energy / detail_energy.detach().mean(
                    dim=(-2, -1), keepdim=True).clamp_min(1e-6),
            ],
            dim=1)
        expected_gain = 1.0 + torch.tanh(
            adapter.shared_gain_gates[0](spatial_descriptor))
        self.assertTrue(torch.allclose(
            out[:pair_batch], prev * expected_gain, atol=1e-6))
        self.assertTrue(torch.allclose(
            out[pair_batch:], curr * expected_gain, atol=1e-6))
        input_detail = (
            feat[pair_batch:] - feat[:pair_batch]) * 0.5
        output_detail = (
            out[pair_batch:] - out[:pair_batch]) * 0.5
        self.assertTrue(torch.allclose(
            output_detail,
            input_detail * expected_gain,
            atol=1e-5,
            rtol=1e-5))

        swapped = torch.cat(
            [feat[pair_batch:], feat[:pair_batch]], dim=0)
        swapped_out = adapter((swapped, ))[0]
        expected = torch.cat([out[pair_batch:], out[:pair_batch]], dim=0)
        self.assertTrue(torch.allclose(swapped_out, expected, atol=1e-6))

        out.square().mean().backward()
        weight = adapter.shared_gain_gates[0].weight
        self.assertIsNotNone(weight.grad)
        self.assertGreater(float(weight.grad.abs().sum()), 0.0)

    def test_dual_evidence_zero_gate_is_exact_identity(self):
        torch.manual_seed(9)
        adapter = PairTemporalPyramidDualEvidenceAdapter(
            in_channels=[8, 8, 8], pointwise_groups=2)
        feats = (
            torch.randn(4, 8, 12, 12),
            torch.randn(4, 8, 6, 6),
            torch.randn(4, 8, 3, 3),
        )

        outs = adapter(feats)

        for out, feat in zip(outs, feats):
            self.assertTrue(torch.equal(out, feat))

    def test_dual_evidence_is_swap_equivariant_and_backpropagates(self):
        torch.manual_seed(10)
        adapter = PairTemporalPyramidDualEvidenceAdapter(
            in_channels=[8],
            pointwise_groups=2,
            use_spatial_evidence=True)
        adapter.gamma.data.fill_(0.5)
        feat = torch.randn(4, 8, 7, 5, requires_grad=True)

        out = adapter((feat, ))[0]
        pair_batch = feat.shape[0] // 2
        swapped = torch.cat([feat[pair_batch:], feat[:pair_batch]], dim=0)
        swapped_out = adapter((swapped, ))[0]
        expected = torch.cat([out[pair_batch:], out[:pair_batch]], dim=0)
        self.assertTrue(torch.allclose(swapped_out, expected, atol=1e-6))

        out.square().mean().backward()
        self.assertGreater(float(feat.grad.abs().sum()), 0.0)
        self.assertGreater(float(adapter.gamma.grad.abs().sum()), 0.0)
        self.assertGreater(
            float(adapter.local_blocks[0][-1].weight.grad.abs().sum()), 0.0)
        self.assertGreater(
            float(adapter.detail_blocks[0][-1].weight.grad.abs().sum()), 0.0)
        self.assertGreater(
            float(adapter.spatial_gates[0].weight.grad.abs().sum()), 0.0)

    def test_dual_evidence_branch_energy_is_bounded_and_differentiable(self):
        torch.manual_seed(12)
        adapter = PairTemporalPyramidDualEvidenceAdapter(
            in_channels=[8],
            pointwise_groups=2,
            use_spatial_evidence=True,
            conserve_branch_energy=True)
        adapter.gamma.data.fill_(100.0)
        feat = torch.randn(4, 8, 7, 5, requires_grad=True)
        pair_batch = feat.shape[0] // 2
        prev = feat[:pair_batch]
        curr = feat[pair_batch:]
        common = (prev + curr) * 0.5
        detail = (curr - prev) * 0.5

        out = adapter((feat, ))[0]
        out_prev = out[:pair_batch]
        out_curr = out[pair_batch:]
        shared_update = (out_prev + out_curr) * 0.5 - common
        signed_update = (out_curr - out_prev) * 0.5 - detail

        common_rms = common.square().mean(dim=(-2, -1)).sqrt()
        shared_rms = shared_update.square().mean(dim=(-2, -1)).sqrt()
        detail_rms = detail.square().mean(dim=(-2, -1)).sqrt()
        signed_rms = signed_update.square().mean(dim=(-2, -1)).sqrt()
        self.assertTrue(torch.all(shared_rms <= common_rms + 1e-5))
        self.assertTrue(torch.all(signed_rms <= detail_rms + 1e-5))
        stats = adapter.latest_branch_energy_stats
        self.assertEqual(
            set(stats),
            {'common_scale', 'detail_scale', 'common_clip', 'detail_clip'})
        for value in stats.values():
            self.assertGreaterEqual(float(value), 0.0)
            self.assertLessEqual(float(value), 1.0)
        self.assertGreater(float(stats['common_clip']), 0.0)
        self.assertGreater(float(stats['detail_clip']), 0.0)

        swapped = torch.cat([curr, prev], dim=0)
        swapped_out = adapter((swapped, ))[0]
        expected = torch.cat([out_curr, out_prev], dim=0)
        self.assertTrue(torch.allclose(swapped_out, expected, atol=1e-5))

        out.square().mean().backward()
        self.assertGreater(float(feat.grad.abs().sum()), 0.0)
        self.assertTrue(all(
            param.grad is not None
            for param in adapter.parameters()
            if param.requires_grad))

    def test_dual_evidence_branch_energy_does_not_raise_near_zero_cap(self):
        torch.manual_seed(13)
        adapter = PairTemporalPyramidDualEvidenceAdapter(
            in_channels=[8],
            pointwise_groups=2,
            use_spatial_evidence=True,
            conserve_branch_energy=True)
        adapter.gamma.data.fill_(1000.0)
        common = torch.randn(2, 8, 5, 5) * 1e-5
        detail = torch.randn(2, 8, 5, 5)
        feat = torch.cat([common - detail, common + detail], dim=0)

        out = adapter((feat, ))[0]
        out_prev, out_curr = out.chunk(2, dim=0)
        shared_update = (out_prev + out_curr) * 0.5 - common
        common_rms = common.square().mean(dim=(-2, -1)).sqrt()
        shared_rms = shared_update.square().mean(dim=(-2, -1)).sqrt()

        self.assertTrue(torch.all(shared_rms <= common_rms + 1e-7))

    def test_dual_evidence_moment_competitive_gate_is_budgeted(self):
        torch.manual_seed(16)
        adapter = PairTemporalPyramidDualEvidenceAdapter(
            in_channels=[8],
            pointwise_groups=2,
            moment_competitive_gating=True)
        adapter.gamma.data.fill_(0.5)
        feat = torch.randn(4, 8, 7, 5, requires_grad=True)
        captured = {}

        def capture_logits(_module, _inputs, output):
            captured['logits'] = output

        hook = adapter.gate_mlps[0].register_forward_hook(capture_logits)
        out = adapter((feat, ))[0]
        hook.remove()

        gates = captured['logits'].view(2, 2, 8).softmax(dim=1)
        self.assertTrue(torch.allclose(
            gates.sum(dim=1),
            torch.ones_like(gates[:, 0]),
            atol=1e-6,
            rtol=0.0))
        self.assertEqual(tuple(adapter.moment_mix[0].shape), (2, 8))
        out.square().mean().backward()
        self.assertGreater(float(feat.grad.abs().sum()), 0.0)
        self.assertGreater(
            float(adapter.moment_mix[0].grad.abs().sum()), 0.0)
        self.assertTrue(all(
            param.grad is not None
            for param in adapter.parameters()
            if param.requires_grad))

    def test_dual_evidence_moment_competitive_gate_is_swap_equivariant(self):
        torch.manual_seed(17)
        adapter = PairTemporalPyramidDualEvidenceAdapter(
            in_channels=[8],
            pointwise_groups=2,
            moment_competitive_gating=True)
        adapter.gamma.data.fill_(0.5)
        adapter.moment_mix[0].data.normal_(std=0.2)
        feat = torch.randn(4, 8, 7, 5)
        prev, curr = feat.chunk(2, dim=0)

        out = adapter((feat, ))[0]
        swapped_out = adapter((torch.cat([curr, prev], dim=0), ))[0]

        self.assertTrue(torch.allclose(
            swapped_out,
            torch.cat(out.chunk(2, dim=0)[::-1], dim=0),
            atol=1e-6,
            rtol=0.0))

    def test_dual_evidence_moment_competitive_zero_gate_is_identity(self):
        torch.manual_seed(18)
        adapter = PairTemporalPyramidDualEvidenceAdapter(
            in_channels=[8, 8, 8],
            pointwise_groups=2,
            moment_competitive_gating=True)
        feats = (
            torch.randn(4, 8, 12, 12),
            torch.randn(4, 8, 6, 6),
            torch.randn(4, 8, 3, 3),
        )

        outs = adapter(feats)

        for out, feat in zip(outs, feats):
            self.assertTrue(torch.equal(out, feat))

    def test_dual_evidence_cross_scale_budget_starts_as_parent(self):
        torch.manual_seed(19)
        parent = PairTemporalPyramidDualEvidenceAdapter(
            in_channels=[8, 8, 8], pointwise_groups=2)
        adapter = PairTemporalPyramidDualEvidenceAdapter(
            in_channels=[8, 8, 8],
            pointwise_groups=2,
            cross_scale_evidence_budget=True,
            cross_scale_hidden_dims=4)
        incompatible = adapter.load_state_dict(
            parent.state_dict(), strict=False)
        self.assertTrue(all(
            key.startswith('cross_scale_')
            for key in incompatible.missing_keys))
        self.assertEqual(incompatible.unexpected_keys, [])
        parent.gamma.data.fill_(0.5)
        adapter.gamma.data.copy_(parent.gamma.data)
        feats = (
            torch.randn(4, 8, 12, 12),
            torch.randn(4, 8, 6, 6),
            torch.randn(4, 8, 3, 3),
        )

        parent_out = parent(feats)
        out = adapter(feats)

        for actual, expected in zip(out, parent_out):
            self.assertTrue(torch.equal(actual, expected))
        budget = adapter.latest_cross_scale_budget
        self.assertIsNotNone(budget)
        self.assertTrue(torch.allclose(
            budget.sum(dim=1),
            torch.full_like(budget[:, 0], 3.0),
            atol=1e-6,
            rtol=0.0))

    def test_dual_evidence_cross_scale_budget_is_swap_equivariant(self):
        torch.manual_seed(20)
        adapter = PairTemporalPyramidDualEvidenceAdapter(
            in_channels=[8, 8, 8],
            pointwise_groups=2,
            cross_scale_evidence_budget=True,
            cross_scale_hidden_dims=4)
        adapter.gamma.data.fill_(0.5)
        torch.nn.init.normal_(adapter.cross_scale_out.weight, std=0.1)
        feats = (
            torch.randn(4, 8, 12, 12, requires_grad=True),
            torch.randn(4, 8, 6, 6, requires_grad=True),
            torch.randn(4, 8, 3, 3, requires_grad=True),
        )

        out = adapter(feats)
        swapped = tuple(
            torch.cat(feat.chunk(2, dim=0)[::-1], dim=0)
            for feat in feats)
        swapped_out = adapter(swapped)

        for actual, expected in zip(swapped_out, out):
            expected = torch.cat(expected.chunk(2, dim=0)[::-1], dim=0)
            self.assertTrue(torch.allclose(
                actual, expected, atol=1e-6, rtol=0.0))
        sum(value.square().mean() for value in out).backward()
        self.assertTrue(all(
            param.grad is not None
            for param in adapter.parameters()
            if param.requires_grad))
        self.assertGreater(
            float(adapter.cross_scale_token_proj.weight.grad.abs().sum()),
            0.0)
        self.assertGreater(
            float(adapter.cross_scale_out.weight.grad.abs().sum()), 0.0)

    def test_dual_evidence_cross_scale_budget_validates_channels(self):
        with self.assertRaisesRegex(
                ValueError, 'equal channel dimensions'):
            PairTemporalPyramidDualEvidenceAdapter(
                in_channels=[8, 16],
                cross_scale_evidence_budget=True)

    def test_hybrid_encoder_builds_dual_evidence_adapter(self):
        encoder = RTDETRHybridEncoder(
            layer_cfg=dict(
                self_attn_cfg=dict(embed_dims=8, num_heads=2, dropout=0.0),
                ffn_cfg=dict(
                    embed_dims=8, feedforward_channels=16, ffn_drop=0.0)),
            in_channels=[8, 8, 8],
            use_encoder_idx=[-1],
            num_encoder_layers=1,
            pair_temporal_adapter_cfg=dict(num_heads=2),
            post_pair_temporal_adapter_cfg=dict(
                type='pyramid_dual_evidence',
                pointwise_groups=2,
                use_spatial_evidence=True),
            fpn_cfg=None,
        )

        self.assertIsInstance(
            encoder.post_pair_temporal_adapter,
            PairTemporalPyramidDualEvidenceAdapter)

    def test_dual_evidence_detail_spatial_gate_preserves_parent_at_init(self):
        torch.manual_seed(14)
        parent = PairTemporalPyramidDualEvidenceAdapter(
            in_channels=[8], pointwise_groups=2)
        adapter = PairTemporalPyramidDualEvidenceAdapter(
            in_channels=[8],
            pointwise_groups=2,
            use_spatial_evidence=True,
            spatial_common_evidence=False,
            spatial_detail_evidence=True,
            spatial_unit_init=True,
            spatial_detach_descriptor=True,
            spatial_preserve_mean=True)
        incompatible = adapter.load_state_dict(parent.state_dict(), strict=False)
        self.assertEqual(
            incompatible.missing_keys,
            ['spatial_gates.0.weight', 'spatial_gates.0.bias'])
        self.assertEqual(incompatible.unexpected_keys, [])
        parent.gamma.data.fill_(0.5)
        adapter.gamma.data.copy_(parent.gamma.data)
        feat = torch.randn(4, 8, 7, 5, requires_grad=True)

        parent_out = parent((feat, ))[0]
        out = adapter((feat, ))[0]

        self.assertTrue(torch.equal(out, parent_out))
        self.assertEqual(adapter.spatial_gates[0].out_channels, 1)
        out.square().mean().backward()
        self.assertTrue(all(
            param.grad is not None
            for param in adapter.parameters()
            if param.requires_grad))
        self.assertGreater(
            float(adapter.spatial_gates[0].weight.grad.abs().sum()), 0.0)

    def test_dual_evidence_spatial_gate_redistributes_detached_evidence(self):
        torch.manual_seed(15)
        adapter = PairTemporalPyramidDualEvidenceAdapter(
            in_channels=[8],
            pointwise_groups=2,
            use_spatial_evidence=True,
            spatial_common_evidence=False,
            spatial_detail_evidence=True,
            spatial_unit_init=True,
            spatial_detach_descriptor=True,
            spatial_preserve_mean=True)
        adapter.gamma.data.fill_(0.5)
        torch.nn.init.normal_(adapter.spatial_gates[0].weight, std=0.2)
        torch.nn.init.normal_(adapter.spatial_gates[0].bias, std=0.1)
        captured = {}

        def capture_descriptor(_module, inputs):
            captured['descriptor'] = inputs[0]

        hook = adapter.spatial_gates[0].register_forward_pre_hook(
            capture_descriptor)
        feat = torch.randn(4, 8, 7, 5, requires_grad=True)
        out = adapter((feat, ))[0]
        hook.remove()

        descriptor = captured['descriptor']
        self.assertFalse(descriptor.requires_grad)
        raw_gate = 2.0 * torch.sigmoid(
            adapter.spatial_gates[0](descriptor))
        spatial_gate = raw_gate / raw_gate.mean(
            dim=(-2, -1), keepdim=True)
        self.assertTrue(torch.allclose(
            spatial_gate.mean(dim=(-2, -1)),
            torch.ones_like(spatial_gate[:, :, 0, 0]),
            atol=1e-6,
            rtol=0.0))
        out.square().mean().backward()
        self.assertIsNotNone(feat.grad)
        self.assertGreater(float(feat.grad.abs().sum()), 0.0)
        self.assertGreater(
            float(adapter.spatial_gates[0].weight.grad.abs().sum()), 0.0)

    def test_dual_evidence_scale_split_has_complete_gradients(self):
        torch.manual_seed(11)
        adapter = PairTemporalPyramidDualEvidenceAdapter(
            in_channels=[8, 8, 8],
            pointwise_groups=2,
            common_level_indices=[0, 1, 2],
            detail_level_indices=[1, 2])
        adapter.gamma.data.fill_(0.5)
        feats = (
            torch.randn(4, 8, 12, 12, requires_grad=True),
            torch.randn(4, 8, 6, 6, requires_grad=True),
            torch.randn(4, 8, 3, 3, requires_grad=True),
        )

        outs = adapter(feats)
        sum(out.square().mean() for out in outs).backward()

        self.assertIsInstance(adapter.detail_blocks[0], torch.nn.Identity)
        self.assertTrue(all(
            param.grad is not None
            for param in adapter.parameters()
            if param.requires_grad))
        pair_batch = feats[0].shape[0] // 2
        input_detail = (
            feats[0][pair_batch:] - feats[0][:pair_batch]) * 0.5
        output_detail = (
            outs[0][pair_batch:] - outs[0][:pair_batch]) * 0.5
        self.assertTrue(torch.allclose(
            output_detail, input_detail, atol=1e-6))


if __name__ == '__main__':
    unittest.main()
