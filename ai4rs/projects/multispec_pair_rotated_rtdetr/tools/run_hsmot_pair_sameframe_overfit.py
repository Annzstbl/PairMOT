#!/usr/bin/env python3
"""Same-frame pair overfit: prev/curr identical images + InfiniteSampler.

Sanity-check training where temporal motion is removed; both sides see the
same image and GT. Reuses acceptance logic from
``run_hsmot_pair_overfit_acceptance``.
"""
from __future__ import annotations

import argparse
import os
import os.path as osp
import sys

_AI4RS_ROOT = osp.abspath(osp.join(osp.dirname(__file__), '../../..'))
_PAIRMMOT_ROOT = osp.abspath(osp.join(_AI4RS_ROOT, '..'))
if _AI4RS_ROOT not in sys.path:
    sys.path.insert(0, _AI4RS_ROOT)

from projects.multispec_pair_rotated_rtdetr.tools.run_hsmot_pair_overfit_acceptance import (
    _is_dist_launched,
    _local_rank,
    run_acceptance,
)

_DEFAULT_CONFIG = (
    'projects/multispec_pair_rotated_rtdetr/configs/'
    'o2_pair_rtdetr_r18vd_overfit_sameframe.py')
_DEFAULT_TMP = 'hsmot_pair_sameframe_overfit_accept'


def parse_args():
    parser = argparse.ArgumentParser(
        description='HSMOT Pair same-frame overfit (prev=curr, InfiniteSampler)')
    parser.add_argument('--config', default=_DEFAULT_CONFIG)
    parser.add_argument('--data-root', default='data/HSMOT_pair_overfit')
    parser.add_argument(
        '--work-dir',
        default='work_dirs/o2_pair_rtdetr_r18vd_overfit_sameframe_accept')
    parser.add_argument('--max-iters', type=int, default=3000)
    parser.add_argument('--val-interval', type=int, default=500)
    parser.add_argument('--num-frames', type=int, default=10)
    parser.add_argument('--source-frame-interval', type=int, default=1)
    parser.add_argument('--from-real', action='store_true')
    parser.add_argument('--src-root', default='../data/hsmot/train')
    parser.add_argument('--ann-file', default='../data/hsmot/train_half.txt')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--min-independent-ap50', type=float, default=0.90)
    parser.add_argument('--min-independent-map50-95', type=float, default=0.40)
    parser.add_argument('--min-pair-ap50', type=float, default=0.80)
    parser.add_argument('--min-pair-map50-95', type=float, default=0.30)
    parser.add_argument('--skip-train', action='store_true')
    parser.add_argument('--checkpoint')
    parser.add_argument('--force-recreate-data', action='store_true')
    parser.add_argument(
        '--tmpdir',
        action='store_true',
        help=f'Use PairMmot/tmp/{_DEFAULT_TMP}')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none')
    parser.add_argument('--local-rank', '--local_rank', type=int, default=0)
    return parser.parse_args()


def main():
    args = parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    data_root = osp.abspath(args.data_root)
    work_dir = osp.abspath(args.work_dir)
    if args.tmpdir:
        tmp = osp.join(_PAIRMMOT_ROOT, 'tmp', _DEFAULT_TMP)
        os.makedirs(tmp, exist_ok=True)
        data_root = osp.join(tmp, 'data')
        work_dir = osp.join(tmp, 'work_dir')
        if _local_rank() == 0:
            print(f'Using tmp workspace: {tmp}')

    device = args.device
    if _is_dist_launched(args.launcher) and device == 'cuda:0':
        device = f'cuda:{_local_rank()}'

    report = run_acceptance(
        config_path=args.config,
        data_root=data_root,
        work_dir=work_dir,
        max_iters=args.max_iters,
        num_frames=args.num_frames,
        source_frame_interval=args.source_frame_interval,
        min_independent_ap50=args.min_independent_ap50,
        min_independent_map50_95=args.min_independent_map50_95,
        min_pair_ap50=args.min_pair_ap50,
        min_pair_map50_95=args.min_pair_map50_95,
        source_seq=None,
        source_start_frame=None,
        skip_train=args.skip_train,
        device=device,
        launcher=args.launcher,
        from_real=args.from_real,
        src_root=args.src_root,
        ann_file=args.ann_file,
        seed=args.seed,
        reuse_data=not args.force_recreate_data,
        val_interval=args.val_interval,
        checkpoint_path=args.checkpoint,
    )
    sys.exit(0 if report.passed else 1)


if __name__ == '__main__':
    main()
