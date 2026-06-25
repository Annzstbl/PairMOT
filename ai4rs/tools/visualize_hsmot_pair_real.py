#!/usr/bin/env python
# Copyright (c) AI4RS. All rights reserved.
"""Randomly visualize and validate HSMOT image pairs on the real dataset."""
import argparse
import copy
import json
import os.path as osp
import random
import sys
from pathlib import Path

import numpy as np

# ai4rs root on sys.path when run as script
_AI4RS_ROOT = Path(__file__).resolve().parents[1]
if str(_AI4RS_ROOT) not in sys.path:
    sys.path.insert(0, str(_AI4RS_ROOT))

from mmengine.dataset import Compose

from mmrotate.datasets.hsmot_pair import HSMOTPairDataset
from mmrotate.datasets.transforms.loading_hsmot_pair import (
    ConvertPairBoxType,
    HSMOTPairLoadAnnotations,
    LoadHSMOTPairImages,
)
from mmrotate.datasets.transforms.transforms_hsmot_pair import (
    PairSharedRandomFlip,
    PairSharedRandomRotate,
    PairSharedResize,
)
from mmrotate.datasets.transforms.validate_hsmot_pair import (
    compute_raw_rbox_angles,
    validate_bbox_in_bounds,
    validate_pair_results,
)
from mmrotate.datasets.transforms.visualize_hsmot_pair import visualize_hsmot_pair


def _default_out_dir() -> Path:
    return _AI4RS_ROOT.parent / 'tmp' / 'hsmot_pair_real_vis'


def build_pair_pipeline(
        scale=(800, 1200),
        flip_prob=0.5,
        rotate_prob=0.5,
        angle_range=180,
        backend_args=None):
    """Training-like pair pipeline without Pack (keeps numpy images for vis)."""
    return Compose([
        LoadHSMOTPairImages(to_float32=False, backend_args=backend_args),
        HSMOTPairLoadAnnotations(box_type='qbox'),
        ConvertPairBoxType(dst_box_type='rbox'),
        PairSharedResize(scale=scale, keep_ratio=True),
        PairSharedRandomFlip(prob=flip_prob, direction=['horizontal', 'vertical']),
        PairSharedRandomRotate(prob=rotate_prob, angle_range=angle_range),
    ])


def _bbox_angles_and_oob(results):
    """Per-row rbox angle (deg) and out-of-bounds flag for drawing."""
    h, w = results['img_shape'][:2]
    angles_prev, angles_curr = [], []
    oob_prev, oob_curr = [], []

    def _per_side(key, valid_key, angles_out, oob_out):
        boxes = results[key]
        valid = results[valid_key]
        if hasattr(valid, 'cpu'):
            valid = valid.cpu().numpy()
        tensor = boxes.tensor if hasattr(boxes, 'tensor') else boxes
        polys = None
        if tensor.numel() > 0 and tensor.size(-1) == 5:
            from mmrotate.structures.bbox import rbox2qbox
            polys = rbox2qbox(tensor).reshape(-1, 4, 2).cpu().numpy()
        for i, ok in enumerate(valid):
            if not ok:
                angles_out.append(0.0)
                oob_out.append(False)
                continue
            if tensor.size(-1) == 5:
                angles_out.append(float(tensor[i, 4]))
            else:
                angles_out.append(0.0)
            if polys is not None:
                poly = polys[i]
                oob = (
                    poly[:, 0].min() < -2 or poly[:, 1].min() < -2
                    or poly[:, 0].max() > w + 2 or poly[:, 1].max() > h + 2
                    or float(tensor[i, 2]) <= 0 or float(tensor[i, 3]) <= 0)
                oob_out.append(oob)
            else:
                oob_out.append(False)

    _per_side('gt_bboxes_prev', 'pair_valid_prev', angles_prev, oob_prev)
    _per_side('gt_bboxes_curr', 'pair_valid_curr', angles_curr, oob_curr)
    return angles_prev, angles_curr, oob_prev, oob_curr


def process_one_sample(data_info, pipeline, idx: int) -> dict:
    """Run pipeline and validation; return record for report."""
    raw_info = copy.deepcopy(data_info)
    raw_angle_prev = compute_raw_rbox_angles(raw_info['instances_prev'])
    raw_angle_curr = compute_raw_rbox_angles(raw_info['instances_curr'])

    results = pipeline(raw_info)
    report = validate_pair_results(
        results, raw_angle_prev=raw_angle_prev, raw_angle_curr=raw_angle_curr)

    angles_prev, angles_curr, oob_prev, oob_curr = _bbox_angles_and_oob(results)

    vp = results['pair_valid_prev']
    vc = results['pair_valid_curr']
    if hasattr(vp, 'cpu'):
        vp = vp.cpu().numpy().tolist()
        vc = vc.cpu().numpy().tolist()
    else:
        vp = list(vp)
        vc = list(vc)

    track_ids = results['pair_track_ids']
    if hasattr(track_ids, 'cpu'):
        track_ids = track_ids.cpu().numpy().tolist()
    else:
        track_ids = list(track_ids)

    meta = (
        f'{results.get("video_id")} prev={results.get("frame_id_prev")} '
        f'curr={results.get("frame_id")} | flip={results.get("flip")} '
        f'{results.get("flip_direction")} scale={results.get("scale_factor")}'
    )
    check_line = 'PASS' if report.ok else 'FAIL'
    failed = [c.name for c in report.checks if not c.ok]
    if failed:
        check_line += f': {", ".join(failed)}'

    return {
        'index': idx,
        'img_id': data_info.get('img_id', ''),
        'video_id': data_info.get('video_id', ''),
        'frame_id_prev': data_info.get('frame_id_prev'),
        'frame_id': data_info.get('frame_id'),
        'ok': report.ok,
        'checks': [
            {'name': c.name, 'ok': c.ok, 'detail': c.detail}
            for c in report.checks
        ],
        'report_text': report.summary(),
        'results': results,
        'track_ids': track_ids,
        'valid_prev': vp,
        'valid_curr': vc,
        'angles_prev': angles_prev,
        'angles_curr': angles_curr,
        'oob_prev': oob_prev,
        'oob_curr': oob_curr,
        'meta_line': meta,
        'check_line': check_line,
    }


def _json_safe(obj):
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, (np.integer, np.int64)):
        return int(obj)
    if isinstance(obj, (np.floating, np.float64)):
        return float(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    return obj


def export_random_pairs(
        num_samples: int = 100,
        seed: int = 42,
        out_dir: Path = None,
        data_root: str = '../data/hsmot/train',
        ann_file: str = '../train_half.txt',
        img_subdir: str = 'npy2jpg',
        frame_interval: int = 1,
        scale=(800, 1200),
        flip_prob: float = 0.5,
        rotate_prob: float = 0.5,
) -> dict:
    out_dir = Path(out_dir or _default_out_dir())
    out_dir.mkdir(parents=True, exist_ok=True)

    dataset = HSMOTPairDataset(
        data_root=data_root,
        ann_subdir='mot',
        ann_file=ann_file,
        data_prefix=dict(img_path=img_subdir),
        img_format='3jpg',
        frame_interval=frame_interval,
        pipeline=[],
        lazy_init=True,
    )
    dataset.full_init()
    total = len(dataset)
    if num_samples > total:
        num_samples = total

    rng = random.Random(seed)
    indices = rng.sample(range(total), num_samples)

    pipeline = build_pair_pipeline(
        scale=scale, flip_prob=flip_prob, rotate_prob=rotate_prob)

    records = []
    fail_count = 0
    for i, ds_idx in enumerate(indices):
        data_info = dataset.get_data_info(ds_idx)
        rec = process_one_sample(data_info, pipeline, ds_idx)
        records.append({
            k: v for k, v in rec.items() if k != 'results'
        })

        fname = (
            f'{i+1:03d}_{rec["video_id"]}_'
            f'p{rec["frame_id_prev"]:06d}_c{rec["frame_id"]:06d}.jpg'
        )
        results = rec['results']
        visualize_hsmot_pair(
            results['img'][0],
            results['img'][1],
            results['gt_bboxes_prev'],
            results['gt_bboxes_curr'],
            track_ids=rec['track_ids'],
            valid_prev=rec['valid_prev'],
            valid_curr=rec['valid_curr'],
            angles_prev=rec['angles_prev'],
            angles_curr=rec['angles_curr'],
            oob_prev=rec['oob_prev'],
            oob_curr=rec['oob_curr'],
            meta_line=rec['meta_line'],
            check_summary=rec['check_line'],
            save_path=str(out_dir / fname),
        )
        if not rec['ok']:
            fail_count += 1
        print(f'[{i+1}/{num_samples}] {fname} {rec["check_line"]}')

    summary = {
        'num_samples': num_samples,
        'seed': seed,
        'dataset_total': total,
        'data_root': osp.abspath(data_root),
        'out_dir': str(out_dir.resolve()),
        'pass_count': num_samples - fail_count,
        'fail_count': fail_count,
        'indices': indices,
        'records': records,
    }
    with open(out_dir / 'report.json', 'w', encoding='utf-8') as f:
        json.dump(_json_safe(summary), f, indent=2, ensure_ascii=False)

    with open(out_dir / 'report.txt', 'w', encoding='utf-8') as f:
        f.write(
            f'HSMOT pair real-data validation\n'
            f'samples={num_samples} pass={summary["pass_count"]} '
            f'fail={fail_count}\n'
            f'data_root={summary["data_root"]}\n\n')
        for rec in records:
            f.write(
                f'--- {rec["img_id"]} ok={rec["ok"]}\n'
                f'{rec["report_text"]}\n\n')

    return summary


def main():
    parser = argparse.ArgumentParser(
        description='Visualize and validate random HSMOT image pairs')
    parser.add_argument('--num', type=int, default=100,
                        help='Number of random pairs (default 100)')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--out-dir', type=str, default=None)
    parser.add_argument('--data-root', type=str, default='../data/hsmot/train')
    parser.add_argument('--ann-file', type=str, default='../train_half.txt')
    parser.add_argument('--img-subdir', type=str, default='npy2jpg')
    parser.add_argument('--frame-interval', type=int, default=1)
    parser.add_argument('--flip-prob', type=float, default=0.5)
    parser.add_argument('--rotate-prob', type=float, default=0.5)
    args = parser.parse_args()

    summary = export_random_pairs(
        num_samples=args.num,
        seed=args.seed,
        out_dir=Path(args.out_dir) if args.out_dir else None,
        data_root=args.data_root,
        ann_file=args.ann_file,
        img_subdir=args.img_subdir,
        frame_interval=args.frame_interval,
        flip_prob=args.flip_prob,
        rotate_prob=args.rotate_prob,
    )
    print(
        f'Done: {summary["pass_count"]}/{summary["num_samples"]} passed. '
        f'Output: {summary["out_dir"]}')


if __name__ == '__main__':
    main()
