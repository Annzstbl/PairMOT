#!/usr/bin/env python3
"""Smoke test on synthetic minimal HSMOT: train 1 epoch, then test/eval."""
import argparse
import glob
import os
import os.path as osp
import shutil
import sys

_AI4RS_ROOT = osp.abspath(osp.join(osp.dirname(__file__), '../../..'))
_PAIRMMOT_ROOT = osp.abspath(osp.join(_AI4RS_ROOT, '..'))
if _AI4RS_ROOT not in sys.path:
    sys.path.insert(0, _AI4RS_ROOT)

from mmengine.config import Config
from mmengine.runner import Runner

from mmrotate.utils import register_all_modules

from projects.multispec_rotated_rtdetr.tools.create_hsmot_debug_data import (
    create_minimal_hsmot)


def get_hsmot_debug_tmp_root(name: str = 'hsmot_debug_e2e') -> str:
    tmp_root = osp.join(_PAIRMMOT_ROOT, 'tmp', name)
    os.makedirs(tmp_root, exist_ok=True)
    return tmp_root


def _latest_checkpoint(work_dir: str) -> str:
    ckpts = sorted(glob.glob(osp.join(work_dir, '*.pth')))
    if not ckpts:
        raise FileNotFoundError(f'No checkpoint found in {work_dir}')
    return ckpts[-1]


def _apply_eval_prefix(cfg, work_dir: str) -> None:
    eval_prefix = osp.join(work_dir, 'eval')
    os.makedirs(eval_prefix, exist_ok=True)
    cfg.val_evaluator['outfile_prefix'] = eval_prefix
    cfg.test_evaluator['outfile_prefix'] = eval_prefix


def run_debug_e2e(config_path: str, data_root: str, work_dir: str) -> None:
    register_all_modules()
    os.chdir(_AI4RS_ROOT)

    create_minimal_hsmot(data_root)

    cfg = Config.fromfile(config_path)
    cfg.work_dir = work_dir
    cfg.train_dataloader.dataset.data_root = osp.join(data_root, 'train')
    cfg.val_dataloader.dataset.data_root = osp.join(data_root, 'test')
    cfg.test_dataloader.dataset.data_root = osp.join(data_root, 'test')
    _apply_eval_prefix(cfg, work_dir)

    print('[1/3] Debug training ...')
    train_runner = Runner.from_cfg(cfg)
    train_runner.train()

    ckpt = _latest_checkpoint(work_dir)
    print(f'[2/3] Testing with checkpoint: {ckpt}')
    cfg.load_from = ckpt
    _apply_eval_prefix(cfg, work_dir)
    test_runner = Runner.from_cfg(cfg)
    metrics = test_runner.test()
    print(f'[3/3] Metrics: {metrics}')
    assert metrics is not None, 'Test returned no metrics'
    print('HSMOT debug e2e test PASSED.')


def parse_args():
    parser = argparse.ArgumentParser(
        description='Run synthetic HSMOT debug e2e test')
    parser.add_argument(
        '--config',
        default='projects/multispec_rotated_rtdetr/configs/'
        'o2_rtdetr_r18vd_1xb1_1e_hsmot_debug.py')
    parser.add_argument(
        '--data-root',
        default='data/HSMOT_mini',
        help='Where to create/read synthetic HSMOT data')
    parser.add_argument(
        '--work-dir',
        default='work_dirs/o2_rtdetr_r18vd_1xb1_1e_hsmot_debug')
    parser.add_argument(
        '--tmpdir',
        action='store_true',
        help='Use PairMmot/tmp/hsmot_debug_e2e for data and work_dir')
    return parser.parse_args()


def main():
    args = parse_args()
    if args.tmpdir:
        tmp = get_hsmot_debug_tmp_root()
        if osp.isdir(tmp):
            shutil.rmtree(tmp)
        os.makedirs(tmp, exist_ok=True)
        data_root = osp.join(tmp, 'HSMOT_mini')
        work_dir = osp.join(tmp, 'work_dir')
        print(f'Using workspace tmp dir: {tmp}')
        run_debug_e2e(args.config, data_root, work_dir)
    else:
        run_debug_e2e(args.config, args.data_root, args.work_dir)


if __name__ == '__main__':
    main()
