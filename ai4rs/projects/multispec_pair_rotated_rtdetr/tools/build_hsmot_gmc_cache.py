#!/usr/bin/env python3
"""Build reusable HSMOT GMC cache for adjacent or configured frame pairs.

The implementation follows the Bot-SORT GMC idea: estimate global camera
motion from background-like sparse image correspondences, cache a 3x3
prev-to-current matrix, and keep training purely cache-reading.
"""

from __future__ import annotations

import argparse
import json
import os
import os.path as osp
from typing import Iterable, List, Sequence, Tuple

import cv2
import mmcv
import numpy as np
from mmengine.fileio import list_from_file


def _seqs(data_root: str, ann_file: str, ann_subdir: str) -> List[str]:
    if ann_file:
        path = ann_file
        if not osp.isabs(path):
            path = osp.normpath(osp.join(data_root, ann_file))
        return [x.strip() for x in list_from_file(path) if x.strip()]
    ann_dir = osp.join(data_root, ann_subdir)
    return [
        osp.splitext(name)[0] for name in sorted(os.listdir(ann_dir))
        if name.endswith('.txt')
    ]


def _frame_ids(img_dir: str, img_format: str) -> List[int]:
    suffix = '.npy' if img_format == 'npy' else '_p1.jpg'
    ids = []
    for name in os.listdir(img_dir):
        if not name.endswith(suffix):
            continue
        stem = name[:-len(suffix)]
        try:
            ids.append(int(stem))
        except ValueError:
            continue
    return sorted(set(ids))


def _img_path(img_root: str, seq: str, frame_id: int, img_format: str) -> str:
    if img_format == 'npy':
        return osp.join(img_root, seq, f'{frame_id:06d}.npy')
    return osp.join(img_root, seq, f'{frame_id:06d}_p1.jpg')


def _read_gray(path: str, img_format: str) -> np.ndarray:
    if img_format == 'npy':
        img = np.load(path)
        if img.ndim == 3:
            img = img[..., 0]
        return _to_u8(img)
    img = mmcv.imread(path, channel_order='rgb')
    if img is None:
        raise FileNotFoundError(path)
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)


def _to_u8(img: np.ndarray) -> np.ndarray:
    img = np.asarray(img)
    if img.dtype == np.uint8:
        return img
    finite = np.isfinite(img)
    if not finite.any():
        return np.zeros(img.shape[:2], dtype=np.uint8)
    lo, hi = np.percentile(img[finite], [1, 99])
    if hi <= lo:
        hi = lo + 1.0
    out = (np.clip(img, lo, hi) - lo) * (255.0 / (hi - lo))
    return out.astype(np.uint8)


def _estimate_sparse(prev: np.ndarray, curr: np.ndarray,
                     max_corners: int = 2000) -> Tuple[np.ndarray, dict]:
    prev_small = prev
    curr_small = curr
    scale = 1.0
    max_side = max(prev.shape[:2])
    if max_side > 960:
        scale = 960.0 / float(max_side)
        prev_small = cv2.resize(prev, None, fx=scale, fy=scale,
                                interpolation=cv2.INTER_AREA)
        curr_small = cv2.resize(curr, None, fx=scale, fy=scale,
                                interpolation=cv2.INTER_AREA)

    pts = cv2.goodFeaturesToTrack(
        prev_small,
        maxCorners=max_corners,
        qualityLevel=0.01,
        minDistance=8,
        blockSize=3)
    if pts is None or len(pts) < 8:
        return np.eye(3, dtype=np.float32), {
            'method': 'sparse_lk',
            'ok': False,
            'num_points': 0,
            'inliers': 0,
        }

    nxt, status, _ = cv2.calcOpticalFlowPyrLK(
        prev_small, curr_small, pts, None,
        winSize=(21, 21),
        maxLevel=3,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))
    status = status.reshape(-1).astype(bool)
    src = pts.reshape(-1, 2)[status]
    dst = nxt.reshape(-1, 2)[status]
    if len(src) < 8:
        return np.eye(3, dtype=np.float32), {
            'method': 'sparse_lk',
            'ok': False,
            'num_points': int(len(src)),
            'inliers': 0,
        }
    affine, inlier_mask = cv2.estimateAffinePartial2D(
        src, dst, method=cv2.RANSAC, ransacReprojThreshold=3.0,
        maxIters=2000, confidence=0.99)
    if affine is None:
        return np.eye(3, dtype=np.float32), {
            'method': 'sparse_lk',
            'ok': False,
            'num_points': int(len(src)),
            'inliers': 0,
        }
    if scale != 1.0:
        s = np.array([[scale, 0, 0], [0, scale, 0], [0, 0, 1]],
                     dtype=np.float32)
        h_small = np.eye(3, dtype=np.float32)
        h_small[:2] = affine.astype(np.float32)
        h = np.linalg.inv(s) @ h_small @ s
    else:
        h = np.eye(3, dtype=np.float32)
        h[:2] = affine.astype(np.float32)
    inliers = int(inlier_mask.sum()) if inlier_mask is not None else 0
    return h.astype(np.float32), {
        'method': 'sparse_lk',
        'ok': True,
        'num_points': int(len(src)),
        'inliers': inliers,
    }


def _pairs(frame_ids: Sequence[int],
           gaps: Sequence[int]) -> Iterable[Tuple[int, int]]:
    frame_set = set(frame_ids)
    for curr in frame_ids:
        for gap in gaps:
            prev = curr - int(gap)
            if prev in frame_set:
                yield prev, curr


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-root', required=True)
    parser.add_argument('--ann-file', default='')
    parser.add_argument('--ann-subdir', default='mot')
    parser.add_argument('--img-subdir', default='npy2jpg')
    parser.add_argument('--img-format', choices=('3jpg', 'npy'), default='3jpg')
    parser.add_argument('--out-dir', required=True)
    parser.add_argument('--gaps', default='1')
    parser.add_argument('--overwrite', action='store_true')
    args = parser.parse_args()

    gaps = [int(x) for x in args.gaps.split(',') if x.strip()]
    img_root = osp.join(args.data_root, args.img_subdir)
    seqs = _seqs(args.data_root, args.ann_file, args.ann_subdir)
    total = 0
    for seq_idx, seq in enumerate(seqs):
        seq_img_dir = osp.join(img_root, seq)
        ids = _frame_ids(seq_img_dir, args.img_format)
        out_seq = osp.join(args.out_dir, seq)
        os.makedirs(out_seq, exist_ok=True)
        done = 0
        for prev_id, curr_id in _pairs(ids, gaps):
            out_path = osp.join(out_seq, f'{prev_id:06d}_{curr_id:06d}.json')
            if osp.isfile(out_path) and not args.overwrite:
                done += 1
                continue
            prev = _read_gray(_img_path(img_root, seq, prev_id,
                                        args.img_format), args.img_format)
            curr = _read_gray(_img_path(img_root, seq, curr_id,
                                        args.img_format), args.img_format)
            matrix, meta = _estimate_sparse(prev, curr)
            payload = {
                'seq_name': seq,
                'frame_id_prev': int(prev_id),
                'frame_id_curr': int(curr_id),
                'matrix': matrix.tolist(),
                **meta,
            }
            with open(out_path, 'w', encoding='utf-8') as f:
                json.dump(payload, f)
            done += 1
        total += done
        print(f'[{seq_idx + 1}/{len(seqs)}] {seq}: {done} cached',
              flush=True)
    print(f'Done. cached pairs={total} out_dir={args.out_dir}')


if __name__ == '__main__':
    main()
