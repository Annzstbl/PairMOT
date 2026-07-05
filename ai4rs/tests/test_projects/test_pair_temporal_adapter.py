"""Unit tests for pair temporal adapter in RT-DETR hybrid encoder."""

import unittest

import torch

from projects.rotated_rtdetr.rotated_rtdetr.rtdetr_layers import (
    PairTemporalAdapter,
    PairTemporalPoolGateAdapter,
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


if __name__ == '__main__':
    unittest.main()
