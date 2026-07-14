# Copyright (c) AI4RS. All rights reserved.
"""Equivalence tests for MultispecPairRotatedRTDETR (M2)."""

import copy
import os.path as osp
import sys
import unittest
from types import MethodType

import torch
from mmengine.config import Config

_AI4RS_ROOT = osp.abspath(osp.join(osp.dirname(__file__), '../..'))
if _AI4RS_ROOT not in sys.path:
    sys.path.insert(0, _AI4RS_ROOT)

from mmrotate.utils import register_all_modules
from mmrotate.registry import MODELS
from projects.multispec_pair_rotated_rtdetr.multispec_pair_rotated_rtdetr import (
    MultispecPairRotatedRTDETR)


def _build_minimal_models(device: torch.device):
    """Build small single-frame and pair RT-DETR models for unit tests."""
    cfg_path = osp.join(
        _AI4RS_ROOT,
        'projects/multispec_rotated_rtdetr/configs/'
        'o2_rtdetr_r18vd_1xb1_1e_hsmot_debug.py')
    cfg = Config.fromfile(cfg_path)
    model_cfg = copy.deepcopy(cfg.model)
    model_cfg['backbone']['init_cfg'] = None
    model_cfg['backbone']['frozen_stages'] = 4
    model_cfg['backbone']['norm_eval'] = True
    model_cfg['num_queries'] = 20
    model_cfg['dn_cfg']['group_cfg']['num_dn_queries'] = 8

    single_model = MODELS.build(copy.deepcopy(model_cfg))
    pair_cfg = copy.deepcopy(model_cfg)
    pair_cfg['type'] = MultispecPairRotatedRTDETR
    pair_model = MODELS.build(pair_cfg)
    pair_model.load_state_dict(single_model.state_dict(), strict=True)

    single_model = single_model.to(device).eval()
    pair_model = pair_model.to(device).eval()
    return single_model, pair_model


def _assert_head_outputs_close(reference,
                               candidate,
                               branch: str,
                               atol: float = 1e-5) -> None:
    ref_cls, ref_coord = reference
    cand_cls, cand_coord = candidate
    for layer_idx, (ref_c, cand_c) in enumerate(zip(ref_cls, cand_cls)):
        torch.testing.assert_close(
            cand_c,
            ref_c,
            atol=atol,
            rtol=0,
            msg=f'{branch} cls layer {layer_idx} mismatch')
    for layer_idx, (ref_c, cand_c) in enumerate(zip(ref_coord, cand_coord)):
        torch.testing.assert_close(
            cand_c,
            ref_c,
            atol=atol,
            rtol=0,
            msg=f'{branch} coord layer {layer_idx} mismatch')


class TestMultispecPairRotatedRTDETREquivalence(unittest.TestCase):
    """Numerical equivalence checks (run on CPU for strict atol=1e-5).

    RT-DETR hybrid encoder FPN is slightly batch-size sensitive under CUDA
    cuDNN; CPU eval matches single-frame ``RotatedRTDETR`` within 1e-5.
    """

    @classmethod
    def setUpClass(cls):
        register_all_modules()
        cls.equiv_device = torch.device('cpu')

    def test_identical_pair_matches_single_frame(self):
        """When input is (I, I) in eval mode, both branches match single model."""
        single_model, pair_model = _build_minimal_models(self.equiv_device)
        torch.manual_seed(0)
        img = torch.randn(1, 8, 256, 320, device=self.equiv_device)
        pair_input = torch.cat([img, img], dim=0).unsqueeze(0)

        with torch.no_grad():
            single_out = single_model._forward(img, None)
            pair_out = pair_model._forward(pair_input, None)

        _assert_head_outputs_close(single_out, pair_out['prev'], 'prev')
        _assert_head_outputs_close(single_out, pair_out['curr'], 'curr')

    def test_shared_encoder_splits_memory(self):
        """Shared 2B encoder memory splits into equal prev/curr slices."""
        _, pair_model = _build_minimal_models(self.equiv_device)
        torch.manual_seed(1)
        img = torch.randn(1, 8, 256, 320, device=self.equiv_device)
        pair_input = torch.cat([img, img], dim=0).unsqueeze(0)
        pair_batch = pair_input.shape[0]

        with torch.no_grad():
            feats = pair_model.extract_feat(pair_input)
            enc_in, _ = pair_model.pre_transformer(feats, None)
            enc_out = pair_model.forward_encoder(**enc_in)
            memory = enc_out['memory']
            memory_prev = memory[:pair_batch]
            memory_curr = memory[pair_batch:2 * pair_batch]

        self.assertEqual(memory.shape[0], 2 * pair_batch)
        self.assertEqual(memory_prev.shape, memory_curr.shape)
        torch.testing.assert_close(
            memory_prev,
            memory_curr,
            atol=1e-3,
            rtol=0,
            msg='memory_prev and memory_curr should match for (I, I)')

    def test_pair_input_shape(self):
        """Pair model accepts (B, 2, C, H, W) and produces dual outputs."""
        _, pair_model = _build_minimal_models(self.equiv_device)
        pair_input = torch.randn(2, 2, 8, 256, 320, device=self.equiv_device)
        with torch.no_grad():
            out = pair_model._forward(pair_input, None)

        self.assertIn('prev', out)
        self.assertIn('curr', out)
        prev_cls, prev_coord = out['prev']
        curr_cls, curr_coord = out['curr']
        self.assertEqual(prev_cls[0].shape[0], 2)
        self.assertEqual(curr_cls[0].shape[0], 2)
        self.assertEqual(prev_coord[0].shape[-1], 5)
        self.assertEqual(curr_coord[0].shape[-1], 5)

    def test_pair_extract_feat_groups_prev_then_curr(self):
        """Flattened pair batches must match the later memory[:B] split."""

        class EchoBackbone(torch.nn.Module):

            def forward(self, x):
                return (x, )

        pair_model = object.__new__(MultispecPairRotatedRTDETR)
        torch.nn.Module.__init__(pair_model)
        pair_model.backbone = EchoBackbone()
        pair_model.debug_shapes = False

        pair_input = torch.arange(4 * 2, dtype=torch.float32).reshape(
            4, 2, 1, 1, 1)
        feats = pair_model.extract_feat(pair_input)
        flat = feats[0].flatten().tolist()

        self.assertEqual(flat[:4], [0.0, 2.0, 4.0, 6.0])
        self.assertEqual(flat[4:], [1.0, 3.0, 5.0, 7.0])

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA is required')
    def test_bf16_encoder_fp32_post_boundary_preserves_gradient(self):
        """Encoder may use BF16, but every later stage must see FP32."""
        model = object.__new__(MultispecPairRotatedRTDETR)
        torch.nn.Module.__init__(model)
        model.fp32_after_encoder_loss = True
        model._active_timer = None
        model.encoder_weight = torch.nn.Parameter(
            torch.randn(4, 4, device='cuda'))
        observed = {}

        def split_pair_batch(self, flat_batch):
            return flat_batch // 2, None

        def pre_transformer(self, img_feats, batch_data_samples):
            return dict(source=img_feats[0]), dict()

        def forward_encoder(self, source):
            memory = source @ self.encoder_weight
            observed['encoder_dtype'] = memory.dtype
            return dict(
                memory=memory,
                memory_mask=None,
                spatial_shapes=torch.tensor(
                    [[1, source.shape[1]]], device=source.device))

        def forward_after_encoder(self, encoder_outputs_dict,
                                  decoder_inputs_dict, batch_data_samples,
                                  pair_batch, timer):
            memory = encoder_outputs_dict['memory']
            observed['post_dtype'] = memory.dtype
            observed['post_autocast'] = torch.is_autocast_enabled()
            return memory.square().mean()

        model._split_pair_batch = MethodType(split_pair_batch, model)
        model.pre_transformer = MethodType(pre_transformer, model)
        model.forward_encoder = MethodType(forward_encoder, model)
        model._forward_after_encoder = MethodType(
            forward_after_encoder, model)

        source = torch.randn(
            2, 3, 4, device='cuda', requires_grad=True)
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            loss = model.forward_transformer((source, ), None)
        loss.backward()

        self.assertEqual(observed['encoder_dtype'], torch.bfloat16)
        self.assertEqual(observed['post_dtype'], torch.float32)
        self.assertFalse(observed['post_autocast'])
        self.assertEqual(loss.dtype, torch.float32)
        self.assertIsNotNone(source.grad)
        self.assertTrue(torch.isfinite(source.grad).all().item())
        self.assertIsNotNone(model.encoder_weight.grad)
        self.assertTrue(
            torch.isfinite(model.encoder_weight.grad).all().item())


class TestMultispecPairRotatedRTDETRDebugShapes(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        register_all_modules()
        cls.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')

    def test_debug_shapes_runs(self):
        cfg_path = osp.join(
            _AI4RS_ROOT,
            'projects/multispec_rotated_rtdetr/configs/'
            'o2_rtdetr_r18vd_1xb1_1e_hsmot_debug.py')
        cfg = Config.fromfile(cfg_path)
        model_cfg = copy.deepcopy(cfg.model)
        model_cfg['backbone']['init_cfg'] = None
        model_cfg['backbone']['frozen_stages'] = 4
        model_cfg['num_queries'] = 10
        model_cfg['type'] = MultispecPairRotatedRTDETR
        model_cfg['debug_shapes'] = True

        model = MODELS.build(model_cfg).to(self.device).eval()
        pair_input = torch.randn(1, 2, 8, 128, 160, device=self.device)
        with torch.no_grad():
            model._forward(pair_input, None)


if __name__ == '__main__':
    unittest.main()
