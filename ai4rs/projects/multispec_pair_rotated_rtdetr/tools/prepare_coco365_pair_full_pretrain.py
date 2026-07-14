#!/usr/bin/env python3
"""Create COCO+Objects365 adapted PairMOT pretrain checkpoint."""
from __future__ import annotations

import argparse
from pathlib import Path

from projects.multispec_pair_rotated_rtdetr.tools.load_pair_pretrain import (
    ensure_coco365_pair_adapted_checkpoint,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--src',
        default=(
            '/data4/litianhao/PairMmot/pretrained_weights/'
            'rtdetr_r18vd_5x_coco_objects365_from_paddle.pth'))
    parser.add_argument(
        '--target-config',
        default=(
            'projects/multispec_pair_rotated_rtdetr/configs/'
            'o2_pair_rtdetr_r18vd_2xb4_72e_hsmot_pairdn_gap1train_'
            'dualcls_nopres_pairtopk_v2_unique_pairdn_allgt_'
            'coco365_full_252.py'))
    parser.add_argument(
        '--cache-dir',
        default=(
            '/data4/litianhao/PairMmot/pretrained_weights/'
            'rtdetr_r18vd_5x_coco_objects365_pair_unique_allgt_full'))
    parser.add_argument('--force', action='store_true')
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = ensure_coco365_pair_adapted_checkpoint(
        src_ckpt=str(Path(args.src)),
        target_config=str(Path(args.target_config)),
        cache_dir=str(Path(args.cache_dir)),
        force=args.force,
        output_name='pair_coco365_full_adapted_pretrain.pth')
    print(output)


if __name__ == '__main__':
    main()
