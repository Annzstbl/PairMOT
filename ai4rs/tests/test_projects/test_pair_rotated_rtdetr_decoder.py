# Copyright (c) AI4RS. All rights reserved.
"""Unit tests for PairRotatedRTDETRTransformerDecoder (M3j / M3-2)."""

import copy
import os.path as osp
import sys
import unittest

import torch
from mmengine.config import Config

_AI4RS_ROOT = osp.abspath(osp.join(osp.dirname(__file__), '../..'))
if _AI4RS_ROOT not in sys.path:
    sys.path.insert(0, _AI4RS_ROOT)

from mmrotate.utils import register_all_modules
from projects.multispec_pair_rotated_rtdetr.multispec_pair_rotated_rtdetr.pair_rotated_rtdetr_layers import (  # noqa: E501
    PairRotatedRTDETRTransformerDecoder,
    PairRotatedRTDETRTransformerDecoderLayer,
)


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
                   tristate_zero_init_coupling: bool = False):
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

    def test_amp_fp16_forward(self):
        if not torch.cuda.is_available():
            self.skipTest('CUDA required for autocast smoke test')
        device = torch.device('cuda')
        decoder, reg_prev, reg_curr = _build_decoder(device=device)
        spatial_shapes, level_start_index, num_value = _spatial_meta(device)
        memory_prev = torch.randn(
            1, num_value, decoder.embed_dims, device=device)
        memory_curr = torch.randn(
            1, num_value, decoder.embed_dims, device=device)
        with torch.autocast(device_type='cuda', dtype=torch.float16):
            hidden, refs_prev, refs_curr = decoder(
                memory_prev=memory_prev,
                memory_curr=memory_curr,
                spatial_shapes=spatial_shapes,
                level_start_index=level_start_index,
                reg_branches_prev=reg_prev,
                reg_branches_curr=reg_curr,
            )
        self.assertFalse(torch.isnan(hidden[-1]).any().item())
        self.assertFalse(torch.isnan(refs_prev[-1]).any().item())
        self.assertFalse(torch.isnan(refs_curr[-1]).any().item())

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
