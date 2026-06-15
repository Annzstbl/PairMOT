#!/usr/bin/env python3
"""Create synthetic minimal HSMOT data for debug smoke tests."""
import argparse
import os
import os.path as osp

import numpy as np


def _write_mot(path: str, num_frames: int = 2) -> None:
    lines = []
    for frame_id in range(1, num_frames + 1):
        lines.append(
            f'{frame_id},1,10,10,40,10,40,40,10,40,-1,0,0\n')
        lines.append(
            f'{frame_id},2,50,50,80,50,80,80,50,80,-1,2,0\n')
    with open(path, 'w', encoding='utf-8') as f:
        f.writelines(lines)


def _write_npy(img_dir: str, num_frames: int = 2, size: int = 128) -> None:
    os.makedirs(img_dir, exist_ok=True)
    rng = np.random.RandomState(0)
    for frame_id in range(1, num_frames + 1):
        img = rng.randint(0, 255, (size, size, 8), dtype=np.uint8)
        np.save(osp.join(img_dir, f'{frame_id:06d}.npy'), img)


def create_minimal_hsmot(root: str, num_frames: int = 2, img_size: int = 128) -> str:
    """Create train/test splits under ``root``."""
    seq_name = 'mini-1'
    for split in ('train', 'test'):
        split_root = osp.join(root, split)
        mot_dir = osp.join(split_root, 'mot')
        img_dir = osp.join(split_root, 'npy', seq_name)
        imagesets_dir = osp.join(split_root, 'ImageSets')
        os.makedirs(mot_dir, exist_ok=True)
        os.makedirs(imagesets_dir, exist_ok=True)
        _write_mot(osp.join(mot_dir, f'{seq_name}.txt'), num_frames)
        _write_npy(img_dir, num_frames, img_size)
        split_file = 'train.txt' if split == 'train' else 'test.txt'
        with open(osp.join(imagesets_dir, split_file), 'w',
                  encoding='utf-8') as f:
            f.write(f'{seq_name}\n')
    return root


def parse_args():
    parser = argparse.ArgumentParser(
        description='Create synthetic HSMOT debug dataset')
    parser.add_argument(
        '--root',
        default='data/HSMOT_mini',
        help='dataset root containing train/ and test/')
    parser.add_argument('--num-frames', type=int, default=2)
    parser.add_argument('--img-size', type=int, default=128)
    return parser.parse_args()


def main():
    args = parse_args()
    root = osp.abspath(args.root)
    create_minimal_hsmot(root, args.num_frames, args.img_size)
    print(f'Created HSMOT debug dataset at {root}')


if __name__ == '__main__':
    main()
