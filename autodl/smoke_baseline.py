#!/usr/bin/env python3
"""Run two baseline train iterations and one inference batch."""

from __future__ import annotations

import argparse
import copy
import os
from pathlib import Path

import torch
from mmdet.utils import register_all_modules as register_all_modules_mmdet
from mmengine.config import Config
from mmengine.registry import RUNNERS
from mmengine.runner import Runner
from mmengine.utils import import_modules_from_strings
from mmrotate.utils import register_all_modules


def set_resize(dataset_cfg: dict, scale: tuple[int, int]) -> None:
    for transform in dataset_cfg['pipeline']:
        if transform.get('type') == 'PairSharedResize':
            transform.update(scale=scale, keep_ratio=True)


def finite_predictions(sample) -> int:
    if hasattr(sample, 'pred_pair_instances'):
        pred = sample.pred_pair_instances
    elif hasattr(sample, 'pred_instances'):
        pred = sample.pred_instances
    else:
        raise RuntimeError(
            f'No prediction field found; available fields: {sample.all_keys()}')
    for key, value in pred.items():
        if torch.is_tensor(value) and not torch.isfinite(value).all():
            raise RuntimeError(f'Non-finite inference output: {key}')
    return len(pred)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--work-dir', required=True)
    parser.add_argument('--scale', default='640,480')
    args = parser.parse_args()
    scale = tuple(int(item) for item in args.scale.split(','))
    if len(scale) != 2:
        raise ValueError('--scale must be WIDTH,HEIGHT')

    register_all_modules_mmdet(init_default_scope=False)
    register_all_modules(init_default_scope=False)
    cfg = Config.fromfile(args.config)
    if cfg.get('custom_imports', None):
        import_modules_from_strings(**cfg.custom_imports)
    inference_dataloader = copy.deepcopy(cfg.test_dataloader)
    for dataloader in (cfg.train_dataloader, inference_dataloader):
        dataloader.update(batch_size=1, num_workers=0, persistent_workers=False)
        set_resize(dataloader['dataset'], scale)

    cfg.work_dir = args.work_dir
    cfg.train_dataloader.update(batch_size=1, num_workers=0,
                                persistent_workers=False)
    cfg.train_cfg = dict(type='IterBasedTrainLoop', max_iters=2,
                         val_interval=999999)
    cfg.val_dataloader = None
    cfg.val_cfg = None
    cfg.val_evaluator = None
    cfg.test_dataloader = None
    cfg.test_cfg = None
    cfg.test_evaluator = None
    cfg.param_scheduler = []
    cfg.custom_hooks = []
    cfg.default_hooks['logger'].update(interval=1)
    cfg.default_hooks['checkpoint'].update(
        by_epoch=False, interval=999999, max_keep_ckpts=1)
    if 'visualization' in cfg.default_hooks:
        cfg.default_hooks['visualization'].update(draw=False)
    cfg.randomness = dict(seed=3407, diff_rank_seed=False,
                          deterministic=False)
    cfg.launcher = 'none'
    cfg.resume = False

    Path(args.work_dir).mkdir(parents=True, exist_ok=True)
    if 'runner_type' in cfg:
        runner = RUNNERS.build(cfg)
    else:
        runner = Runner.from_cfg(cfg)
    print(f'train_samples={len(runner.train_dataloader.dataset)}')
    runner.train()
    if runner.iter != 2:
        raise RuntimeError(f'Expected 2 iterations, completed {runner.iter}')
    runner.save_checkpoint(args.work_dir, 'smoke_baseline.pth',
                           save_optimizer=False,
                           save_param_scheduler=False)

    loader = Runner.build_dataloader(
        inference_dataloader, seed=3407, diff_rank_seed=False)
    batch = next(iter(loader))
    runner.model.eval()
    with torch.no_grad():
        outputs = runner.model.test_step(batch)
    if not isinstance(outputs, list) or len(outputs) != 1:
        raise RuntimeError(f'Unexpected inference output: {type(outputs)}')
    count = finite_predictions(outputs[0])
    print(f'inference_predictions={count}')
    print(f'SMOKE_TEST_PASSED work_dir={args.work_dir}')


if __name__ == '__main__':
    main()
