# Copyright (c) AI4RS. All rights reserved.
import tempfile
import unittest

import numpy as np
import torch

from projects.multispec_rotated_rtdetr.multispec_rotated_rtdetr.pretrain_utils import (
    adapt_state_dict_in_channels, adapt_state_dict_stem_conv3d_se,
    expand_conv1_weight)


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


if __name__ == '__main__':
    unittest.main()
