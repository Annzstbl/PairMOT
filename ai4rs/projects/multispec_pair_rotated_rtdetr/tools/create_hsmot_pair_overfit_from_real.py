#!/usr/bin/env python3
"""Sample one contiguous clip from real HSMOT for overfit acceptance.

Extracts ``num_frames`` consecutive frames from a single source sequence into
one mini-sequence (``pair-overfit-real``), producing ``num_frames - 1`` pairs.
Images are symlinked; MOT frame ids are remapped to ``1 .. num_frames``.
"""
from __future__ import annotations

import argparse
import json
import os
import os.path as osp
import random
import sys
from collections import Counter, defaultdict
from typing import Dict, List, Sequence, Tuple

import numpy as np

_AI4RS_ROOT = osp.abspath(osp.join(osp.dirname(__file__), '../../..'))
if _AI4RS_ROOT not in sys.path:
    sys.path.insert(0, _AI4RS_ROOT)

from mmengine.fileio import list_from_file
from mmrotate.datasets.hsmot import load_hsmot_sequence_ann
from mmrotate.datasets.pair_gt import build_pair_gt_from_instances
from mmrotate.utils import register_all_modules

DEFAULT_SEQ_NAME = 'pair-overfit-real'


def _instances_from_frame(frame_anns: dict, frame_id: int) -> List[dict]:
    instances = []
    for ann in frame_anns.get(frame_id, []):
        instances.append({
            'bbox': np.array(ann['polygon'], dtype=np.float32),
            'bbox_label': ann['class_id'],
            'ignore_flag': ann['ignore_flag'],
            'track_id': ann['track_id'],
        })
    return instances


def _categorize_pair(instances_prev: Sequence[dict],
                     instances_curr: Sequence[dict]) -> str:
    pair_gt = build_pair_gt_from_instances(instances_prev, instances_curr)
    if len(pair_gt['labels']) == 0:
        return 'empty'
    valid_prev = pair_gt['valid_prev']
    valid_curr = pair_gt['valid_curr']
    has_new = bool(np.any(~valid_prev & valid_curr))
    has_dis = bool(np.any(valid_prev & ~valid_curr))
    has_both = bool(np.any(valid_prev & valid_curr))
    if has_new and has_dis:
        return 'mixed'
    if has_new:
        return 'new'
    if has_dis:
        return 'disappear'
    if has_both:
        return 'persistent'
    return 'empty'


def _mot_line(frame_id: int, inst: dict) -> str:
    track_id = int(inst['track_id'])
    poly = inst['bbox']
    if hasattr(poly, 'tolist'):
        poly = poly.tolist()
    cls_id = int(inst['bbox_label'])
    trunc = int(inst.get('ignore_flag', 0))
    coords = ','.join(f'{float(v):.1f}' for v in poly)
    return f'{frame_id},{track_id},{coords},-1,{cls_id},{trunc}\n'


def _write_sequence_mot(path: str, frame_anns: dict,
                        src_frame_ids: List[int]) -> None:
    lines = []
    for dst_fid, src_fid in enumerate(src_frame_ids, start=1):
        for inst in _instances_from_frame(frame_anns, src_fid):
            lines.append(_mot_line(dst_fid, inst))
    with open(path, 'w', encoding='utf-8') as f:
        f.writelines(lines)


def _symlink_3jpg(src_seq_dir: str, dst_seq_dir: str, src_frame: int,
                  dst_frame: int) -> None:
    os.makedirs(dst_seq_dir, exist_ok=True)
    for part in ('p1', 'p2', 'p3'):
        src = osp.join(src_seq_dir, f'{src_frame:06d}_{part}.jpg')
        dst = osp.join(dst_seq_dir, f'{dst_frame:06d}_{part}.jpg')
        if not osp.isfile(src):
            raise FileNotFoundError(f'Missing 3jpg part: {src}')
        if osp.lexists(dst):
            os.remove(dst)
        os.symlink(osp.abspath(src), dst)


def _frame_has_images(img_seq_dir: str, frame_id: int) -> bool:
    return osp.isfile(osp.join(img_seq_dir, f'{frame_id:06d}_p1.jpg'))


def _load_sequence_list(src_root: str, ann_file: str) -> List[str]:
    if osp.isdir(ann_file):
        import glob
        txt_files = sorted(glob.glob(osp.join(ann_file, '*.txt')))
        return [osp.splitext(osp.basename(p))[0] for p in txt_files]
    seq_list = list_from_file(ann_file)
    return [seq.strip() for seq in seq_list if seq.strip()]


def _score_window(frame_anns: dict, src_frame_ids: List[int]) -> Tuple[int, dict]:
    """Score a contiguous clip; higher prefers new/disappear/mixed + GT density."""
    pair_cats: Counter = Counter()
    total_gt = 0
    for i in range(1, len(src_frame_ids)):
        prev = _instances_from_frame(frame_anns, src_frame_ids[i - 1])
        curr = _instances_from_frame(frame_anns, src_frame_ids[i])
        cat = _categorize_pair(prev, curr)
        pair_cats[cat] += 1
        total_gt += len(build_pair_gt_from_instances(prev, curr)['labels'])

    num_pairs = len(src_frame_ids) - 1
    score = total_gt
    score += pair_cats.get('new', 0) * 500
    score += pair_cats.get('disappear', 0) * 500
    score += pair_cats.get('mixed', 0) * 800
    score += pair_cats.get('persistent', 0) * 50
    stats = {
        'num_pairs': num_pairs,
        'pair_categories': dict(pair_cats),
        'total_gt_rows': total_gt,
    }
    return score, stats


def _collect_windows(
    src_root: str,
    ann_file: str,
    img_subdir: str,
    num_frames: int,
    source_frame_interval: int,
    source_seq: str | None = None,
    source_start_frame: int | None = None,
) -> List[dict]:
    mot_dir = osp.join(src_root, 'mot')
    img_root = osp.join(src_root, img_subdir)
    seq_list = _load_sequence_list(src_root, ann_file)
    windows: List[dict] = []

    for seq_name in seq_list:
        if source_seq is not None and seq_name != source_seq:
            continue
        ann_path = osp.join(mot_dir, f'{seq_name}.txt')
        if not osp.isfile(ann_path):
            continue
        frame_anns = load_hsmot_sequence_ann(ann_path)
        img_seq_dir = osp.join(img_root, seq_name)
        valid_frames = sorted(
            fid for fid in frame_anns.keys()
            if _frame_has_images(img_seq_dir, fid))
        required_span = (num_frames - 1) * source_frame_interval
        if len(valid_frames) <= required_span:
            continue

        for start_idx in range(len(valid_frames) - required_span):
            src_ids = valid_frames[
                start_idx:start_idx + required_span + 1:source_frame_interval]
            if len(src_ids) != num_frames:
                continue
            if (source_start_frame is not None
                    and src_ids[0] != source_start_frame):
                continue
            if any(b - a != source_frame_interval
                   for a, b in zip(src_ids, src_ids[1:])):
                continue
            if not all(_frame_has_images(img_seq_dir, fid) for fid in src_ids):
                continue
            score, stats = _score_window(frame_anns, src_ids)
            if stats['total_gt_rows'] == 0:
                continue
            windows.append({
                'source_seq': seq_name,
                'src_frame_ids': src_ids,
                'score': score,
                'stats': stats,
            })
    return windows


def _pick_window(windows: List[dict], seed: int) -> dict:
    """Pick best-scoring window; shuffle among top candidates for seed diversity."""
    if not windows:
        raise RuntimeError('No valid contiguous windows found.')
    rng = random.Random(seed)
    windows = sorted(windows, key=lambda w: w['score'], reverse=True)
    top_k = min(20, len(windows))
    pool = windows[:top_k]
    rng.shuffle(pool)
    return pool[0]


def create_hsmot_pair_overfit_from_real(
    dst_root: str,
    src_root: str,
    ann_file: str,
    img_subdir: str = 'npy2jpg',
    num_frames: int = 10,
    source_frame_interval: int = 1,
    source_seq: str | None = None,
    source_start_frame: int | None = None,
    seed: int = 42,
    seq_name: str = DEFAULT_SEQ_NAME,
) -> str:
    """Build a sampled ``num_frames``-frame clip under ``dst_root/train``."""
    register_all_modules()

    if num_frames < 2:
        raise ValueError(f'num_frames must be >= 2, got {num_frames}')
    if source_frame_interval < 1:
        raise ValueError(
            'source_frame_interval must be >= 1, got '
            f'{source_frame_interval}')

    if not osp.isabs(ann_file):
        ann_file = osp.abspath(osp.join(_AI4RS_ROOT, ann_file))
    if not osp.isabs(src_root):
        src_root = osp.abspath(osp.join(_AI4RS_ROOT, src_root))

    windows = _collect_windows(
        src_root, ann_file, img_subdir, num_frames, source_frame_interval,
        source_seq, source_start_frame)
    chosen = _pick_window(windows, seed)

    src_seq = chosen['source_seq']
    src_frame_ids = chosen['src_frame_ids']
    ann_path = osp.join(src_root, 'mot', f'{src_seq}.txt')
    frame_anns = load_hsmot_sequence_ann(ann_path)

    split_root = osp.join(dst_root, 'train')
    mot_dir = osp.join(split_root, 'mot')
    img_root = osp.join(split_root, img_subdir)
    imagesets_dir = osp.join(split_root, 'ImageSets')
    os.makedirs(mot_dir, exist_ok=True)
    os.makedirs(imagesets_dir, exist_ok=True)
    os.makedirs(img_root, exist_ok=True)

    # Remove stale sequences from previous extractions.
    for old_mot in os.listdir(mot_dir):
        if old_mot.endswith('.txt'):
            os.remove(osp.join(mot_dir, old_mot))
    for old_seq in os.listdir(img_root):
        old_path = osp.join(img_root, old_seq)
        if osp.isdir(old_path):
            for name in os.listdir(old_path):
                part = osp.join(old_path, name)
                if osp.islink(part) or osp.isfile(part):
                    os.remove(part)
            os.rmdir(old_path)

    dst_seq_dir = osp.join(img_root, seq_name)
    os.makedirs(dst_seq_dir, exist_ok=True)

    src_img_seq = osp.join(src_root, img_subdir, src_seq)
    for dst_fid, src_fid in enumerate(src_frame_ids, start=1):
        _symlink_3jpg(src_img_seq, dst_seq_dir, src_fid, dst_fid)

    _write_sequence_mot(osp.join(mot_dir, f'{seq_name}.txt'), frame_anns,
                        src_frame_ids)

    with open(osp.join(imagesets_dir, 'train.txt'), 'w',
              encoding='utf-8') as f:
        f.write(f'{seq_name}\n')

    pair_manifest = []
    for i in range(1, len(src_frame_ids)):
        prev = _instances_from_frame(frame_anns, src_frame_ids[i - 1])
        curr = _instances_from_frame(frame_anns, src_frame_ids[i])
        pair_gt = build_pair_gt_from_instances(
            prev, curr,
            video_id=src_seq,
            frame_id_prev=src_frame_ids[i - 1],
            frame_id_curr=src_frame_ids[i],
        )
        pair_manifest.append({
            'pair_index': i,
            'dst_frame_id_prev': i,
            'dst_frame_id_curr': i + 1,
            'source_frame_id_prev': src_frame_ids[i - 1],
            'source_frame_id_curr': src_frame_ids[i],
            'category': _categorize_pair(prev, curr),
            'num_gt': int(len(pair_gt['labels'])),
            'num_new': int(np.sum(~pair_gt['valid_prev'] & pair_gt['valid_curr'])),
            'num_disappear': int(
                np.sum(pair_gt['valid_prev'] & ~pair_gt['valid_curr'])),
            'num_persistent': int(
                np.sum(pair_gt['valid_prev'] & pair_gt['valid_curr'])),
        })

    num_pairs = num_frames - 1
    meta = {
        'src_root': osp.abspath(src_root),
        'dst_root': osp.abspath(dst_root),
        'ann_file': ann_file,
        'img_subdir': img_subdir,
        'seq_name': seq_name,
        'num_frames': num_frames,
        'num_pairs': num_pairs,
        'source_frame_interval': source_frame_interval,
        'requested_source_seq': source_seq,
        'requested_source_start_frame': source_start_frame,
        'seed': seed,
        'source_seq': src_seq,
        'source_frame_ids': src_frame_ids,
        'window_score': chosen['score'],
        'window_stats': chosen['stats'],
        'pairs': pair_manifest,
    }
    with open(osp.join(dst_root, 'pairs_manifest.json'), 'w',
              encoding='utf-8') as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    print(
        f'Created real overfit clip: {dst_root} '
        f'({num_frames} sampled frames, {num_pairs} pairs, seq={seq_name})')
    print(f'  source: {src_seq} frames {src_frame_ids[0]}..{src_frame_ids[-1]} '
          f'(gap={source_frame_interval})')
    print('  pair categories:', dict(Counter(p['category'] for p in pair_manifest)))
    return dst_root


def parse_args():
    parser = argparse.ArgumentParser(
        description='Extract one contiguous HSMOT clip for overfit acceptance')
    parser.add_argument(
        '--dst-root',
        default='data/HSMOT_pair_overfit',
        help='Output dataset root (contains train/)')
    parser.add_argument(
        '--src-root',
        default='../data/hsmot/train',
        help='Real HSMOT train root')
    parser.add_argument(
        '--ann-file',
        default='../data/hsmot/train_half.txt',
        help='Sequence list relative to ai4rs cwd or absolute path')
    parser.add_argument('--img-subdir', default='npy2jpg')
    parser.add_argument(
        '--num-frames',
        type=int,
        default=10,
        help='Sampled frames in one sequence (pairs = num_frames - 1)')
    parser.add_argument(
        '--source-frame-interval',
        type=int,
        default=1,
        help='Original-sequence frame gap between consecutive sampled frames')
    parser.add_argument(
        '--source-seq', help='Optionally pin the original HSMOT sequence')
    parser.add_argument(
        '--source-start-frame',
        type=int,
        help='Optionally pin the first original frame of the sampled clip')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument(
        '--seq-name',
        default=DEFAULT_SEQ_NAME,
        help='Output sequence name under train/')
    return parser.parse_args()


def main():
    args = parse_args()
    create_hsmot_pair_overfit_from_real(
        dst_root=osp.abspath(args.dst_root),
        src_root=osp.abspath(args.src_root),
        ann_file=args.ann_file,
        img_subdir=args.img_subdir,
        num_frames=args.num_frames,
        source_frame_interval=args.source_frame_interval,
        source_seq=args.source_seq,
        source_start_frame=args.source_start_frame,
        seed=args.seed,
        seq_name=args.seq_name,
    )


if __name__ == '__main__':
    main()
