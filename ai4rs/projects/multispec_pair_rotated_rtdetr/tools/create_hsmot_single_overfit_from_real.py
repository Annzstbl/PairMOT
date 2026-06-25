#!/usr/bin/env python3
"""Sample one contiguous clip from real HSMOT for single-frame overfit acceptance.

Extracts ``num_frames`` consecutive frames from a single source sequence into
one mini-sequence (``single-overfit-real``). Images are symlinked; MOT frame ids
are remapped to ``1 .. num_frames``.
"""
from __future__ import annotations

import argparse
import json
import os
import os.path as osp
import random
import sys
from typing import Dict, List, Optional

import numpy as np

_AI4RS_ROOT = osp.abspath(osp.join(osp.dirname(__file__), '../../..'))
if _AI4RS_ROOT not in sys.path:
    sys.path.insert(0, _AI4RS_ROOT)

from mmengine.fileio import list_from_file
from mmrotate.datasets.hsmot import load_hsmot_sequence_ann
from mmrotate.utils import register_all_modules

DEFAULT_SEQ_NAME = 'single-overfit-real'


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


def _score_window(frame_anns: dict, src_frame_ids: List[int]) -> tuple[int, dict]:
    """Score a contiguous clip; higher prefers denser GT and frame diversity."""
    per_frame = []
    for src_fid in src_frame_ids:
        insts = _instances_from_frame(frame_anns, src_fid)
        per_frame.append(len(insts))
    total_gt = sum(per_frame)
    min_gt = min(per_frame) if per_frame else 0
    max_gt = max(per_frame) if per_frame else 0
    score = total_gt * 10 + min_gt * 50 + (max_gt - min_gt) * 20
    stats = {
        'num_frames': len(src_frame_ids),
        'per_frame_gt': per_frame,
        'total_gt': total_gt,
        'min_gt': min_gt,
        'max_gt': max_gt,
    }
    return score, stats


def _collect_windows(
    src_root: str,
    ann_file: str,
    img_subdir: str,
    num_frames: int,
) -> List[dict]:
    mot_dir = osp.join(src_root, 'mot')
    img_root = osp.join(src_root, img_subdir)
    seq_list = _load_sequence_list(src_root, ann_file)
    windows: List[dict] = []

    for seq_name in seq_list:
        ann_path = osp.join(mot_dir, f'{seq_name}.txt')
        if not osp.isfile(ann_path):
            continue
        frame_anns = load_hsmot_sequence_ann(ann_path)
        img_seq_dir = osp.join(img_root, seq_name)
        valid_frames = sorted(
            fid for fid in frame_anns.keys()
            if _frame_has_images(img_seq_dir, fid))
        if len(valid_frames) < num_frames:
            continue

        for start_idx in range(len(valid_frames) - num_frames + 1):
            src_ids = valid_frames[start_idx:start_idx + num_frames]
            if src_ids[-1] - src_ids[0] != num_frames - 1:
                continue
            if not all(_frame_has_images(img_seq_dir, fid) for fid in src_ids):
                continue
            score, stats = _score_window(frame_anns, src_ids)
            if stats['total_gt'] == 0:
                continue
            windows.append({
                'source_seq': seq_name,
                'src_frame_ids': src_ids,
                'score': score,
                'stats': stats,
            })
    return windows


def _pick_window(windows: List[dict], seed: int) -> dict:
    if not windows:
        raise RuntimeError('No valid contiguous windows found.')
    rng = random.Random(seed)
    windows = sorted(windows, key=lambda w: w['score'], reverse=True)
    top_k = min(20, len(windows))
    pool = windows[:top_k]
    rng.shuffle(pool)
    return pool[0]


def _resolve_chosen_window(
    src_root: str,
    img_subdir: str,
    num_frames: int,
    seed: int,
    ann_file: str,
    source_seq: Optional[str] = None,
    source_frame_start: Optional[int] = None,
) -> dict:
    """Pick a clip automatically or use an explicit source sequence / frames."""
    if source_seq is None:
        windows = _collect_windows(src_root, ann_file, img_subdir, num_frames)
        return _pick_window(windows, seed)

    if source_frame_start is None:
        raise ValueError(
            'source_frame_start is required when source_seq is set.')
    src_frame_ids = list(
        range(source_frame_start, source_frame_start + num_frames))
    ann_path = osp.join(src_root, 'mot', f'{source_seq}.txt')
    if not osp.isfile(ann_path):
        raise FileNotFoundError(f'Missing MOT annotation: {ann_path}')

    frame_anns = load_hsmot_sequence_ann(ann_path)
    img_seq_dir = osp.join(src_root, img_subdir, source_seq)
    missing_frames = [
        fid for fid in src_frame_ids
        if fid not in frame_anns or not _frame_has_images(img_seq_dir, fid)
    ]
    if missing_frames:
        raise RuntimeError(
            f'Invalid clip for {source_seq}: missing GT or images at '
            f'frames {missing_frames}')

    score, stats = _score_window(frame_anns, src_frame_ids)
    if stats['total_gt'] == 0:
        raise RuntimeError(
            f'Clip {source_seq} frames {src_frame_ids} has zero GT instances.')
    return {
        'source_seq': source_seq,
        'src_frame_ids': src_frame_ids,
        'score': score,
        'stats': stats,
    }


def create_hsmot_single_overfit_from_real(
    dst_root: str,
    src_root: str,
    ann_file: str,
    img_subdir: str = 'npy2jpg',
    num_frames: int = 10,
    seed: int = 42,
    seq_name: str = DEFAULT_SEQ_NAME,
    source_seq: Optional[str] = None,
    source_frame_start: Optional[int] = None,
) -> str:
    """Build one ``num_frames``-frame clip under ``dst_root/train``."""
    register_all_modules()

    if num_frames < 1:
        raise ValueError(f'num_frames must be >= 1, got {num_frames}')

    if not osp.isabs(ann_file):
        ann_file = osp.abspath(osp.join(_AI4RS_ROOT, ann_file))
    if not osp.isabs(src_root):
        src_root = osp.abspath(osp.join(_AI4RS_ROOT, src_root))

    chosen = _resolve_chosen_window(
        src_root,
        img_subdir,
        num_frames,
        seed,
        ann_file,
        source_seq=source_seq,
        source_frame_start=source_frame_start,
    )

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

    frame_manifest = []
    for dst_fid, src_fid in enumerate(src_frame_ids, start=1):
        insts = _instances_from_frame(frame_anns, src_fid)
        frame_manifest.append({
            'dst_frame_id': dst_fid,
            'source_frame_id': src_fid,
            'num_gt': len(insts),
            'class_ids': sorted({int(i['bbox_label']) for i in insts}),
        })

    meta = {
        'src_root': osp.abspath(src_root),
        'dst_root': osp.abspath(dst_root),
        'ann_file': ann_file,
        'img_subdir': img_subdir,
        'seq_name': seq_name,
        'num_frames': num_frames,
        'seed': seed,
        'source_seq': src_seq,
        'source_frame_ids': src_frame_ids,
        'window_score': chosen['score'],
        'window_stats': chosen['stats'],
        'frames': frame_manifest,
    }
    with open(osp.join(dst_root, 'frames_manifest.json'), 'w',
              encoding='utf-8') as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    print(
        f'Created real single-frame overfit clip: {dst_root} '
        f'({num_frames} frames, seq={seq_name})')
    print(f'  source: {src_seq} frames {src_frame_ids[0]}..{src_frame_ids[-1]}')
    print('  per-frame GT:', [f['num_gt'] for f in frame_manifest])
    return dst_root


def parse_args():
    parser = argparse.ArgumentParser(
        description='Extract one contiguous HSMOT clip for single-frame overfit')
    parser.add_argument(
        '--dst-root',
        default='data/HSMOT_single_overfit',
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
    parser.add_argument('--num-frames', type=int, default=10)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument(
        '--seq-name',
        default=DEFAULT_SEQ_NAME,
        help='Output sequence name under train/')
    parser.add_argument(
        '--source-seq',
        default=None,
        help='Use a fixed source sequence instead of auto window selection')
    parser.add_argument(
        '--source-frame-start',
        type=int,
        default=None,
        help='First source frame id (contiguous num_frames clip; with --source-seq)')
    return parser.parse_args()


def main():
    args = parse_args()
    create_hsmot_single_overfit_from_real(
        dst_root=osp.abspath(args.dst_root),
        src_root=osp.abspath(args.src_root),
        ann_file=args.ann_file,
        img_subdir=args.img_subdir,
        num_frames=args.num_frames,
        seed=args.seed,
        seq_name=args.seq_name,
        source_seq=args.source_seq,
        source_frame_start=args.source_frame_start,
    )


if __name__ == '__main__':
    main()
