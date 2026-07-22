#!/usr/bin/env python3
"""Profile HSMOT pair dataset stages without constructing a model."""
import argparse
import random
import time
from collections import defaultdict

import cv2
import numpy as np
import torch
from mmengine.config import Config
from mmengine.registry import init_default_scope
from mmengine.runner import Runner
from mmengine.utils import import_modules_from_strings

from mmrotate.registry import DATASETS
from mmrotate.utils import register_all_modules


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('config')
    parser.add_argument('--samples', type=int, default=200)
    parser.add_argument('--warmup', type=int, default=10)
    parser.add_argument('--seed', type=int, default=3407)
    parser.add_argument(
        '--mode', choices=('pipeline', 'dataloader'), default='pipeline')
    parser.add_argument('--workers', type=int)
    parser.add_argument('--prefetch-factor', type=int)
    return parser.parse_args()


def percentile(values, q):
    return float(np.percentile(np.asarray(values, dtype=np.float64), q))


def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(1)
    cv2.setNumThreads(0)

    register_all_modules(init_default_scope=False)
    cfg = Config.fromfile(args.config)
    if cfg.get('custom_imports'):
        import_modules_from_strings(**cfg.custom_imports)
    init_default_scope(cfg.get('default_scope', 'mmrotate'))
    if args.mode == 'dataloader':
        if args.workers is not None:
            cfg.train_dataloader.num_workers = args.workers
        if args.prefetch_factor is not None:
            cfg.train_dataloader.prefetch_factor = args.prefetch_factor
        loader = Runner.build_dataloader(
            cfg.train_dataloader, seed=args.seed, diff_rank_seed=False)
        iterator = iter(loader)
        values = []
        for batch_index in range(args.samples + args.warmup):
            start = time.perf_counter()
            next(iterator)
            elapsed = (time.perf_counter() - start) * 1000.0
            if batch_index >= args.warmup:
                values.append(elapsed)
        print(f'batches={args.samples} seed={args.seed} '
              f'workers={cfg.train_dataloader.num_workers} '
              f'batch_size={cfg.train_dataloader.batch_size}')
        print('mean_ms\tp50_ms\tp90_ms\tp99_ms\tmax_ms')
        print(f'{np.mean(values):.3f}\t{percentile(values, 50):.3f}'
              f'\t{percentile(values, 90):.3f}'
              f'\t{percentile(values, 99):.3f}\t{max(values):.3f}')
        return

    dataset = DATASETS.build(cfg.train_dataloader.dataset)

    rng = np.random.default_rng(args.seed)
    indices = rng.integers(0, len(dataset), args.samples + args.warmup)
    records = defaultdict(list)
    for sample_index, dataset_index in enumerate(indices):
        start = time.perf_counter()
        data = dataset.get_data_info(int(dataset_index))
        info_done = time.perf_counter()
        stage_times = [('GetDataInfo', info_done - start)]
        for transform in dataset.pipeline.transforms:
            stage_start = time.perf_counter()
            data = transform(data)
            stage_times.append((transform.__class__.__name__,
                                time.perf_counter() - stage_start))
            if data is None:
                break
        if sample_index < args.warmup:
            continue
        for name, elapsed in stage_times:
            records[name].append(elapsed * 1000.0)
        records['TOTAL'].append((time.perf_counter() - start) * 1000.0)

    print(f'samples={args.samples} seed={args.seed}')
    print('stage\tmean_ms\tp50_ms\tp90_ms\tp99_ms\tmax_ms')
    for name, values in records.items():
        print(f'{name}\t{np.mean(values):.3f}\t{percentile(values, 50):.3f}'
              f'\t{percentile(values, 90):.3f}'
              f'\t{percentile(values, 99):.3f}\t{max(values):.3f}')


if __name__ == '__main__':
    main()
