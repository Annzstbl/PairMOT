#!/usr/bin/env python3
"""Integration test on real HSMOT subset (outputs outside dataset tree)."""
import argparse
import glob
import os
import os.path as osp
import sys

_AI4RS_ROOT = osp.abspath(osp.join(osp.dirname(__file__), '../../..'))
_PAIRMMOT_ROOT = osp.abspath(osp.join(_AI4RS_ROOT, '..'))
if _AI4RS_ROOT not in sys.path:
    sys.path.insert(0, _AI4RS_ROOT)

from mmengine.config import Config
from mmengine.runner import Runner

from mmrotate.utils import register_all_modules

from projects.multispec_rotated_rtdetr.tools.prepare_hsmot_test_splits import (
    ensure_hsmot_test_splits)


def get_hsmot_test_tmp_root(name: str = 'hsmot_test_e2e') -> str:
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


def run_test_e2e(config_path: str, work_dir: str,
                 test_seqs=None, train_seqs=None) -> None:
    register_all_modules()
    os.chdir(_AI4RS_ROOT)

    splits = ensure_hsmot_test_splits(
        test_seqs=test_seqs, train_seqs=train_seqs)
    hsmot_root = splits['hsmot_root']
    if not osp.isdir(osp.join(hsmot_root, 'train', 'npy2jpg')):
        raise FileNotFoundError(
            f'HSMOT not found at {hsmot_root}. '
            f'Expected train/npy2jpg under dataset root.')

    cfg = Config.fromfile(config_path)
    cfg.work_dir = work_dir
    cfg.train_dataloader.dataset.data_root = osp.join(hsmot_root, 'train')
    cfg.train_dataloader.dataset.ann_file = splits['train_ann_file']
    cfg.val_dataloader.dataset.data_root = osp.join(hsmot_root, 'test')
    cfg.val_dataloader.dataset.ann_file = splits['test_ann_file']
    cfg.test_dataloader.dataset.data_root = osp.join(hsmot_root, 'test')
    cfg.test_dataloader.dataset.ann_file = splits['test_ann_file']
    _apply_eval_prefix(cfg, work_dir)

    print('[1/3] Test training on real HSMOT subset ...')
    print(f'  train ann: {splits["train_ann_file"]}')
    print(f'  test ann : {splits["test_ann_file"]}')
    print(f'  work dir : {work_dir}')
    train_runner = Runner.from_cfg(cfg)
    train_runner.train()

    ckpt = _latest_checkpoint(work_dir)
    print(f'[2/3] Testing with checkpoint: {ckpt}')
    cfg.load_from = ckpt
    _apply_eval_prefix(cfg, work_dir)
    metrics = Runner.from_cfg(cfg).test()
    print(f'[3/3] Metrics: {metrics}')
    assert metrics is not None, 'Test returned no metrics'
    print('HSMOT test e2e PASSED.')


def parse_args():
    parser = argparse.ArgumentParser(
        description='Run real HSMOT subset integration test')
    parser.add_argument(
        '--config',
        default='projects/multispec_rotated_rtdetr/configs/'
        'o2_rtdetr_r18vd_1xb1_1e_hsmot_test.py')
    parser.add_argument(
        '--work-dir',
        default='work_dirs/o2_rtdetr_r18vd_1xb1_1e_hsmot_test')
    parser.add_argument(
        '--use-tmp-workdir',
        action='store_true',
        help='Write outputs to PairMmot/tmp/hsmot_test_e2e/work_dir')
    return parser.parse_args()


def main():
    args = parse_args()
    work_dir = args.work_dir
    if args.use_tmp_workdir:
        work_dir = osp.join(get_hsmot_test_tmp_root(), 'work_dir')
    os.makedirs(work_dir, exist_ok=True)
    run_test_e2e(args.config, osp.abspath(work_dir))


if __name__ == '__main__':
    main()
