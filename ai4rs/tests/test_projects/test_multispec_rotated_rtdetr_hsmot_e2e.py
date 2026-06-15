# Copyright (c) AI4RS. All rights reserved.
import os
import os.path as osp
import shutil
import sys
import tempfile
import unittest

import torch

_AI4RS_ROOT = osp.abspath(osp.join(osp.dirname(__file__), '../..'))
if _AI4RS_ROOT not in sys.path:
    sys.path.insert(0, _AI4RS_ROOT)

from mmengine.config import Config

from mmrotate.utils import register_all_modules
from projects.multispec_rotated_rtdetr.multispec_rotated_rtdetr.data_preprocessor import (
    MultispecDetDataPreprocessor)
from projects.multispec_rotated_rtdetr.tools.create_hsmot_debug_data import (
    create_minimal_hsmot)
from projects.multispec_rotated_rtdetr.tools.run_hsmot_debug_e2e import (
    get_hsmot_debug_tmp_root, run_debug_e2e)


class TestMultispecDetDataPreprocessor(unittest.TestCase):

    def test_8ch_mean_std(self):
        mean = [1.0] * 8
        std = [2.0] * 8
        pre = MultispecDetDataPreprocessor(mean=mean, std=std)
        self.assertEqual(pre.mean.shape[0], 8)
        self.assertEqual(pre.std.shape[0], 8)


@unittest.skipUnless(
    torch.cuda.is_available(), 'CUDA is required for HSMOT e2e smoke test')
class TestMultispecRotatedRTDETRHsmotE2E(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        register_all_modules()

    def test_config_loads(self):
        cfg_path = osp.join(
            _AI4RS_ROOT,
            'projects/multispec_rotated_rtdetr/configs/'
            'o2_rtdetr_r18vd_1xb1_1e_hsmot_debug.py')
        cfg = Config.fromfile(cfg_path)
        self.assertEqual(cfg.max_epochs, 1)
        self.assertEqual(cfg.model.backbone.in_channels, 8)

    def test_minimal_train_test_eval(self):
        tmp = get_hsmot_debug_tmp_root('hsmot_e2e_test')
        if osp.isdir(tmp):
            shutil.rmtree(tmp)
        os.makedirs(tmp, exist_ok=True)
        data_root = osp.join(tmp, 'HSMOT_mini')
        work_dir = osp.join(tmp, 'work_dir')
        config = osp.join(
            _AI4RS_ROOT,
            'projects/multispec_rotated_rtdetr/configs/'
            'o2_rtdetr_r18vd_1xb1_1e_hsmot_debug.py')
        run_debug_e2e(config, data_root, work_dir)
        self.assertTrue(
            osp.exists(osp.join(work_dir, 'epoch_1.pth')))


class TestCreateHsmotDebugData(unittest.TestCase):

    def test_create_dataset_layout(self):
        with tempfile.TemporaryDirectory() as tmp:
            create_minimal_hsmot(tmp)
            train_list = osp.join(tmp, 'train', 'ImageSets', 'train.txt')
            test_npy = osp.join(tmp, 'test', 'npy', 'mini-1', '000001.npy')
            self.assertTrue(osp.isfile(train_list))
            self.assertTrue(osp.isfile(test_npy))


if __name__ == '__main__':
    unittest.main()
