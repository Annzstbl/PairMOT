#!/usr/bin/env python3
"""Create synthetic HSMOT pair data for overfit acceptance.

Produces one sequence with ``num_frames`` frames (default 10 → 9 pairs).
Tracks:
  - id=1: persistent (both sides always visible)
  - id=2: new target (appears from frame 5)
  - id=3: disappear target (visible until frame 7)
"""
import argparse
import os
import os.path as osp

import numpy as np


def _qbox_line(frame_id: int, track_id: int, cx: float, cy: float,
               w: float, h: float, cls_id: int) -> str:
    x1, y1 = cx - w / 2, cy - h / 2
    x2, y2 = cx + w / 2, cy - h / 2
    x3, y3 = cx + w / 2, cy + h / 2
    x4, y4 = cx - w / 2, cy + h / 2
    return (
        f'{frame_id},{track_id},'
        f'{x1:.1f},{y1:.1f},{x2:.1f},{y2:.1f},'
        f'{x3:.1f},{y3:.1f},{x4:.1f},{y4:.1f},'
        f'-1,{cls_id},0\n')


def _write_mot(path: str, num_frames: int) -> None:
    lines = []
    for frame_id in range(1, num_frames + 1):
        lines.append(_qbox_line(frame_id, 1, 40, 40, 30, 30, 0))
        if frame_id >= 5:
            lines.append(_qbox_line(frame_id, 2, 90, 90, 28, 28, 1))
        if frame_id <= 7:
            lines.append(_qbox_line(frame_id, 3, 60, 100, 26, 26, 2))
    with open(path, 'w', encoding='utf-8') as f:
        f.writelines(lines)


def _write_npy(img_dir: str, num_frames: int, img_size: int) -> None:
    os.makedirs(img_dir, exist_ok=True)
    rng = np.random.RandomState(42)
    for frame_id in range(1, num_frames + 1):
        img = rng.randint(0, 255, (img_size, img_size, 8), dtype=np.uint8)
        np.save(osp.join(img_dir, f'{frame_id:06d}.npy'), img)


def create_hsmot_pair_overfit_data(
    root: str,
    num_frames: int = 10,
    img_size: int = 128,
) -> str:
    """Create train split under ``root`` (test uses the same split)."""
    seq_name = 'pair-overfit-1'
    split_root = osp.join(root, 'train')
    mot_dir = osp.join(split_root, 'mot')
    img_dir = osp.join(split_root, 'npy', seq_name)
    imagesets_dir = osp.join(split_root, 'ImageSets')
    os.makedirs(mot_dir, exist_ok=True)
    os.makedirs(imagesets_dir, exist_ok=True)
    _write_mot(osp.join(mot_dir, f'{seq_name}.txt'), num_frames)
    _write_npy(img_dir, num_frames, img_size)
    with open(osp.join(imagesets_dir, 'train.txt'), 'w',
              encoding='utf-8') as f:
        f.write(f'{seq_name}\n')
    num_pairs = max(0, num_frames - 1)
    print(f'Created pair overfit data: {root} '
          f'({num_frames} frames, {num_pairs} pairs, 3 tracks)')
    return root


def parse_args():
    parser = argparse.ArgumentParser(
        description='Create synthetic HSMOT pair overfit dataset')
    parser.add_argument(
        '--root',
        default='data/HSMOT_pair_overfit',
        help='Dataset root (contains train/)')
    parser.add_argument('--num-frames', type=int, default=10)
    parser.add_argument('--img-size', type=int, default=128)
    return parser.parse_args()


def main():
    args = parse_args()
    create_hsmot_pair_overfit_data(
        osp.abspath(args.root),
        num_frames=args.num_frames,
        img_size=args.img_size,
    )


if __name__ == '__main__':
    main()
