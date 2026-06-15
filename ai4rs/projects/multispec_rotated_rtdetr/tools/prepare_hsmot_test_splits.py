#!/usr/bin/env python3
"""Prepare HSMOT test splits under PairMmot/tmp (outside dataset tree)."""
import os
import os.path as osp

_PAIRMMOT_ROOT = osp.abspath(osp.join(osp.dirname(__file__), '../../../..'))
_HSMOT_ROOT = osp.join(_PAIRMMOT_ROOT, 'data', 'hsmot')
_SPLIT_DIR = osp.join(_PAIRMMOT_ROOT, 'tmp', 'hsmot_splits')


def ensure_hsmot_test_splits(
        test_seqs=None,
        train_seqs=None,
        split_dir: str = _SPLIT_DIR) -> dict:
    """Write sequence lists under ``PairMmot/tmp/hsmot_splits/`` only."""
    os.makedirs(split_dir, exist_ok=True)
    if test_seqs is None:
        test_seqs = ['data23-1', 'data24-1']
    if train_seqs is None:
        train_seqs = ['data23-3', 'data25-1']

    test_ann = osp.join(split_dir, 'test_mini.txt')
    train_ann = osp.join(split_dir, 'train_mini.txt')
    with open(test_ann, 'w', encoding='utf-8') as f:
        f.write('\n'.join(test_seqs) + '\n')
    with open(train_ann, 'w', encoding='utf-8') as f:
        f.write('\n'.join(train_seqs) + '\n')

    return {
        'hsmot_root': _HSMOT_ROOT,
        'train_ann_file': train_ann,
        'test_ann_file': test_ann,
        'split_dir': split_dir,
    }


if __name__ == '__main__':
    paths = ensure_hsmot_test_splits()
    for key, value in paths.items():
        print(f'{key}: {value}')
