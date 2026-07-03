#!/usr/bin/env python3
"""Compare pair AP on high new/disappear HSMOT pairs.

This script selects image pairs with many one-frame-only track ids, runs two
pair RT-DETR checkpoints, computes AP on all union GT and both-visible-only GT,
and writes qualitative visualizations.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import os
import os.path as osp
import sys
from dataclasses import dataclass
from typing import Dict, List, Sequence

import cv2
import numpy as np
import torch
from mmengine.config import Config
from mmengine.dataset import Compose
from mmengine.runner import load_checkpoint
from mmengine.structures import InstanceData
from mmengine.utils import mkdir_or_exist
from mmrotate.registry import MODELS
from mmrotate.structures.bbox import rbox2qbox
from mmrotate.utils import register_all_modules

_AI4RS_ROOT = osp.abspath(osp.join(osp.dirname(__file__), '../../..'))
if _AI4RS_ROOT not in sys.path:
    sys.path.insert(0, _AI4RS_ROOT)

import projects.multispec_pair_rotated_rtdetr.multispec_pair_rotated_rtdetr  # noqa: E402,F401
from mmrotate.datasets.hsmot import load_hsmot_sequence_ann  # noqa: E402
from projects.multispec_pair_rotated_rtdetr.multispec_pair_rotated_rtdetr.pair_ap import (  # noqa: E402,E501
    pair_and_independent_ap_metrics,
    serialize_pair_sample,
)
from projects.multispec_pair_rotated_rtdetr.tools.run_pair_mot import (  # noqa: E402
    _frame_ids_from_images,
    _make_pair_info,
    _sequence_list,
)


@dataclass
class PairCandidate:
    seq_name: str
    prev_frame_id: int
    curr_frame_id: int
    num_prev: int
    num_curr: int
    num_survival: int
    num_new: int
    num_disappear: int
    score: int


@dataclass
class Experiment:
    name: str
    config: str
    checkpoint: str
    device: str
    cfg: Config
    model: torch.nn.Module
    preprocessor: torch.nn.Module
    pipeline: Compose
    torch_device: torch.device


DEFAULT_FIXED_CONFIG = (
    '/data/users/litianhao01/PairMmot/workdir/'
    'o2_pair_rtdetr_r18vd_2xb4_72e_hsmot_half_pairdn_gap1to2_fixed_20260628/'
    'o2_pair_rtdetr_r18vd_2xb4_72e_hsmot_pairdn_gap1to2train.py')
DEFAULT_FIXED_CKPT = (
    '/data/users/litianhao01/PairMmot/workdir/'
    'o2_pair_rtdetr_r18vd_2xb4_72e_hsmot_half_pairdn_gap1to2_fixed_20260628/'
    'epoch_72.pth')
DEFAULT_0702_CONFIG = (
    '/data/users/litianhao01/PairMmot/workdir/'
    '0702_baseline_o2_pair_rtdetr_r18vd_2xb4_72e_hsmot_half_pairdn_'
    'gap1train_bothvis_dualcls_nopres_pairtopk_v2_unique_pairdn/'
    'o2_pair_rtdetr_r18vd_2xb4_72e_hsmot_pairdn_gap1train_bothvis_'
    'dualcls_nopres_pairtopk_v2_unique_pairdn.py')
DEFAULT_0702_CKPT = (
    '/data/users/litianhao01/PairMmot/workdir/'
    '0702_baseline_o2_pair_rtdetr_r18vd_2xb4_72e_hsmot_half_pairdn_'
    'gap1train_bothvis_dualcls_nopres_pairtopk_v2_unique_pairdn/'
    'epoch_40.pth')


def _instances_by_track(frame_anns: Dict[int, List[dict]], frame_id: int) -> Dict[int, dict]:
    return {
        int(ann['track_id']): ann
        for ann in frame_anns.get(frame_id, [])
        if int(ann.get('ignore_flag', 0)) == 0
    }


def select_pairs(args) -> List[PairCandidate]:
    img_root = osp.join(args.data_root, args.img_subdir)
    ann_dir = osp.join(args.data_root, args.ann_subdir)
    candidates: List[PairCandidate] = []
    for seq_name in _sequence_list(args.data_root, args.ann_file, args.ann_subdir):
        frame_anns = load_hsmot_sequence_ann(osp.join(ann_dir, f'{seq_name}.txt'))
        frame_ids = _frame_ids_from_images(
            osp.join(img_root, seq_name), args.img_format)
        frame_ids = [fid for fid in frame_ids if fid in frame_anns]
        for prev_id, curr_id in zip(frame_ids[:-1], frame_ids[1:]):
            if curr_id - prev_id != args.frame_gap:
                continue
            prev_map = _instances_by_track(frame_anns, prev_id)
            curr_map = _instances_by_track(frame_anns, curr_id)
            prev_ids = set(prev_map)
            curr_ids = set(curr_map)
            num_new = len(curr_ids - prev_ids)
            num_disappear = len(prev_ids - curr_ids)
            num_survival = len(prev_ids & curr_ids)
            score = num_new + num_disappear
            if score < args.min_change:
                continue
            candidates.append(PairCandidate(
                seq_name=seq_name,
                prev_frame_id=prev_id,
                curr_frame_id=curr_id,
                num_prev=len(prev_ids),
                num_curr=len(curr_ids),
                num_survival=num_survival,
                num_new=num_new,
                num_disappear=num_disappear,
                score=score,
            ))
    candidates.sort(
        key=lambda x: (x.score, x.num_new, x.num_disappear, x.num_survival),
        reverse=True)
    return candidates[:args.num_pairs]


def build_experiment(name: str, config: str, checkpoint: str, device: str) -> Experiment:
    register_all_modules()
    cfg = Config.fromfile(config)
    model = MODELS.build(cfg.model)
    load_checkpoint(model, checkpoint, map_location='cpu')
    torch_device = torch.device(device if torch.cuda.is_available() else 'cpu')
    model = model.to(torch_device).eval()
    preprocessor = MODELS.build(cfg.model.data_preprocessor).to(torch_device)
    pipeline = Compose(cfg.val_pipeline)
    return Experiment(
        name=name,
        config=config,
        checkpoint=checkpoint,
        device=device,
        cfg=cfg,
        model=model,
        preprocessor=preprocessor,
        pipeline=pipeline,
        torch_device=torch_device,
    )


def _filter_gt(gt, mode: str):
    if mode == 'all':
        return gt
    valid_prev = gt.valid_prev.bool()
    valid_curr = gt.valid_curr.bool()
    if mode == 'both':
        keep = valid_prev & valid_curr
    elif mode == 'new':
        keep = (~valid_prev) & valid_curr
    elif mode == 'disappear':
        keep = valid_prev & (~valid_curr)
    else:
        raise ValueError(f'Unknown GT filter mode: {mode}')
    filtered = InstanceData()
    for key in gt.keys():
        value = getattr(gt, key)
        try:
            should_filter = hasattr(value, '__getitem__') and len(value) == len(keep)
        except TypeError:
            should_filter = False
        setattr(filtered, key, value[keep] if should_filter else value)
    return filtered


def _predict_one(exp: Experiment, pair_info: dict):
    packed = exp.pipeline(pair_info)
    data = exp.preprocessor(
        {
            'inputs': packed['inputs'].unsqueeze(0),
            'data_samples': [packed['data_samples']],
        },
        training=False)
    with torch.inference_mode():
        outputs = exp.model.predict(
            data['inputs'].to(exp.torch_device),
            data['data_samples'],
            rescale=False)
    return outputs[0]


def _sample_counts(gt) -> dict:
    valid_prev = gt.valid_prev.detach().cpu().bool()
    valid_curr = gt.valid_curr.detach().cpu().bool()
    return {
        'gt_all': int(valid_prev.numel()),
        'gt_both': int((valid_prev & valid_curr).sum()),
        'gt_new': int(((~valid_prev) & valid_curr).sum()),
        'gt_disappear': int((valid_prev & (~valid_curr)).sum()),
    }


def _metric_row(exp_name: str, mode: str, samples: Sequence[dict]) -> dict:
    metrics = pair_and_independent_ap_metrics(samples)
    row = {'experiment': exp_name, 'gt_filter': mode}
    for key in (
            'pair_AP50', 'pair_AP75', 'pair_mAP50_95',
            'independent_AP50', 'independent_AP75', 'independent_mAP50_95',
            'independent_prev_AP50', 'independent_curr_AP50',
            'association_gap_AP50'):
        row[key] = metrics.get(key, float('nan'))
    return row


def _scale_factor(meta: dict) -> np.ndarray:
    sf = meta.get('scale_factor', (1.0, 1.0))
    if isinstance(sf, torch.Tensor):
        sf = sf.detach().cpu().numpy()
    sf = np.asarray(sf, dtype=np.float32).reshape(-1)
    if sf.size == 1:
        sf = np.repeat(sf, 2)
    if sf.size >= 4:
        return np.asarray([sf[0], sf[1], sf[2], sf[3], 1.0], dtype=np.float32)
    return np.asarray([sf[0], sf[1], sf[0], sf[1], 1.0], dtype=np.float32)


def _rboxes_to_original(rboxes: torch.Tensor, meta: dict) -> np.ndarray:
    if rboxes.numel() == 0:
        return np.zeros((0, 5), dtype=np.float32)
    scale = rboxes.new_tensor(_scale_factor(meta)).clamp_min(1e-6)
    return (rboxes.detach().cpu().float() / scale.cpu()).numpy()


def _qboxes_from_rboxes(rboxes: np.ndarray) -> np.ndarray:
    if rboxes.size == 0:
        return np.zeros((0, 8), dtype=np.float32)
    tensor = torch.as_tensor(rboxes, dtype=torch.float32)
    return rbox2qbox(tensor).detach().cpu().numpy()


def _draw_poly(img: np.ndarray, qbox: Sequence[float], color, thickness: int = 2):
    pts = np.asarray(qbox, dtype=np.float32).reshape(4, 2).round().astype(np.int32)
    cv2.polylines(img, [pts], isClosed=True, color=color, thickness=thickness)


def _draw_text(img: np.ndarray, text: str, org, color):
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1,
                cv2.LINE_AA)


def visualize_pair(pair_info: dict, outputs: Dict[str, object], out_path: str,
                   max_preds: int = 30) -> None:
    prev = cv2.imread(pair_info['img_path_prev'], cv2.IMREAD_COLOR)
    curr = cv2.imread(pair_info['img_path'], cv2.IMREAD_COLOR)
    if prev is None or curr is None:
        return
    h = min(prev.shape[0], curr.shape[0])
    if prev.shape[0] != h:
        prev = cv2.resize(prev, (int(prev.shape[1] * h / prev.shape[0]), h))
    if curr.shape[0] != h:
        curr = cv2.resize(curr, (int(curr.shape[1] * h / curr.shape[0]), h))
    panel_prev = prev.copy()
    panel_curr = curr.copy()

    prev_ids = {int(x['track_id']): x for x in pair_info['instances_prev']}
    curr_ids = {int(x['track_id']): x for x in pair_info['instances_curr']}
    all_ids = sorted(set(prev_ids) | set(curr_ids))
    for tid in all_ids:
        in_prev = tid in prev_ids
        in_curr = tid in curr_ids
        if in_prev and in_curr:
            color = (0, 220, 0)
            tag = 'S'
        elif in_prev:
            color = (0, 165, 255)
            tag = 'D'
        else:
            color = (255, 0, 255)
            tag = 'N'
        if in_prev:
            _draw_poly(panel_prev, prev_ids[tid]['bbox'], color, 2)
        if in_curr:
            _draw_poly(panel_curr, curr_ids[tid]['bbox'], color, 2)
        if in_prev:
            x, y = np.asarray(prev_ids[tid]['bbox']).reshape(4, 2).mean(0).astype(int)
            _draw_text(panel_prev, f'{tag}{tid}', (x, y), color)
        if in_curr:
            x, y = np.asarray(curr_ids[tid]['bbox']).reshape(4, 2).mean(0).astype(int)
            _draw_text(panel_curr, f'{tag}{tid}', (x, y), color)

    pred_colors = [(255, 255, 0), (0, 0, 255)]
    for exp_idx, (exp_name, sample) in enumerate(outputs.items()):
        pred = sample.pred_pair_instances
        meta = sample.metainfo
        k = min(max_preds, int(pred.scores.numel()))
        q_prev = _qboxes_from_rboxes(
            _rboxes_to_original(pred.bboxes_prev[:k], meta))
        q_curr = _qboxes_from_rboxes(
            _rboxes_to_original(pred.bboxes_curr[:k], meta))
        color = pred_colors[exp_idx % len(pred_colors)]
        for rank in range(k):
            _draw_poly(panel_prev, q_prev[rank], color, 1)
            _draw_poly(panel_curr, q_curr[rank], color, 1)
        _draw_text(panel_prev, f'pred {exp_name}', (8, 22 + 18 * exp_idx), color)
        _draw_text(panel_curr, f'pred {exp_name}', (8, 22 + 18 * exp_idx), color)

    header = np.zeros((34, panel_prev.shape[1] + panel_curr.shape[1], 3),
                      dtype=np.uint8)
    _draw_text(
        header,
        f"{pair_info['seq_name']} {pair_info['frame_id_prev']}->{pair_info['frame_id']} "
        'GT: green=survival orange=disappear magenta=new',
        (8, 22),
        (255, 255, 255))
    canvas = np.vstack([header, np.hstack([panel_prev, panel_curr])])
    mkdir_or_exist(osp.dirname(out_path))
    cv2.imwrite(out_path, canvas)


def write_csv(path: str, rows: Sequence[dict]) -> None:
    mkdir_or_exist(osp.dirname(path))
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with open(path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-root',
                        default='/data/users/litianhao01/PairMmot/data/hsmot/test')
    parser.add_argument('--ann-subdir', default='mot')
    parser.add_argument('--img-subdir', default='npy2jpg')
    parser.add_argument('--img-format', default='3jpg')
    parser.add_argument('--ann-file', default=None)
    parser.add_argument('--frame-gap', type=int, default=1)
    parser.add_argument('--num-pairs', type=int, default=100)
    parser.add_argument('--min-change', type=int, default=1)
    parser.add_argument('--max-dets', type=int, default=100)
    parser.add_argument('--vis-count', type=int, default=30)
    parser.add_argument('--vis-max-preds', type=int, default=25)
    parser.add_argument('--out-dir',
                        default='/data/users/litianhao01/PairMmot/workdir/_analysis/'
                        '20260703_new_disappear_pair_ap')
    parser.add_argument('--fixed-config', default=DEFAULT_FIXED_CONFIG)
    parser.add_argument('--fixed-checkpoint', default=DEFAULT_FIXED_CKPT)
    parser.add_argument('--baseline-config', default=DEFAULT_0702_CONFIG)
    parser.add_argument('--baseline-checkpoint', default=DEFAULT_0702_CKPT)
    parser.add_argument('--devices', default='cuda:0,cuda:1')
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    mkdir_or_exist(args.out_dir)
    devices = [x.strip() for x in args.devices.split(',') if x.strip()]
    if not devices:
        devices = ['cuda:0']
    pair_candidates = select_pairs(args)
    if len(pair_candidates) < args.num_pairs:
        print(
            f'Warning: selected only {len(pair_candidates)} pairs with '
            f'min_change={args.min_change}',
            flush=True)
    selected_rows = [candidate.__dict__ for candidate in pair_candidates]
    write_csv(osp.join(args.out_dir, 'selected_pairs.csv'), selected_rows)
    with open(osp.join(args.out_dir, 'args.json'), 'w', encoding='utf-8') as f:
        json.dump(vars(args), f, indent=2)

    experiments = [
        build_experiment('fixed_20260628_epoch72', args.fixed_config,
                         args.fixed_checkpoint, devices[0]),
        build_experiment('baseline_0702_epoch40', args.baseline_config,
                         args.baseline_checkpoint,
                         devices[min(1, len(devices) - 1)]),
    ]

    img_root = osp.join(args.data_root, args.img_subdir)
    ann_dir = osp.join(args.data_root, args.ann_subdir)
    ann_cache: Dict[str, dict] = {}
    samples_by_exp = {
        exp.name: {'all': [], 'both': [], 'new': [], 'disappear': []}
        for exp in experiments
    }
    per_pair_rows = []
    count_rows = []

    for idx, candidate in enumerate(pair_candidates):
        if candidate.seq_name not in ann_cache:
            ann_cache[candidate.seq_name] = load_hsmot_sequence_ann(
                osp.join(ann_dir, f'{candidate.seq_name}.txt'))
        pair_info = _make_pair_info(
            candidate.seq_name,
            img_root,
            args.img_format,
            ann_cache[candidate.seq_name],
            candidate.prev_frame_id,
            candidate.curr_frame_id)
        output_for_vis = {}
        for exp in experiments:
            sample = _predict_one(exp, pair_info)
            output_for_vis[exp.name] = sample
            gt = sample.pair_gt_instances
            pred = sample.pred_pair_instances
            counts = _sample_counts(gt)
            count_rows.append({
                'experiment': exp.name,
                'seq_name': candidate.seq_name,
                'prev_frame_id': candidate.prev_frame_id,
                'curr_frame_id': candidate.curr_frame_id,
                **counts,
            })
            for mode in ('all', 'both', 'new', 'disappear'):
                filtered_gt = _filter_gt(gt, mode)
                ap_sample = serialize_pair_sample(
                    filtered_gt,
                    pred,
                    pres_thr=0.5,
                    max_dets=args.max_dets)
                samples_by_exp[exp.name][mode].append(ap_sample)
        if idx < args.vis_count:
            name = f'{idx:03d}_{candidate.seq_name}_{candidate.prev_frame_id:06d}_{candidate.curr_frame_id:06d}.jpg'
            visualize_pair(
                pair_info,
                output_for_vis,
                osp.join(args.out_dir, 'vis', name),
                max_preds=args.vis_max_preds)
        if (idx + 1) % 10 == 0:
            print(f'processed {idx + 1}/{len(pair_candidates)} pairs', flush=True)

    metric_rows = []
    for exp in experiments:
        for mode in ('all', 'both', 'new', 'disappear'):
            metric_rows.append(_metric_row(exp.name, mode,
                                           samples_by_exp[exp.name][mode]))
    write_csv(osp.join(args.out_dir, 'metrics.csv'), metric_rows)
    write_csv(osp.join(args.out_dir, 'per_pair_gt_counts.csv'), count_rows)
    summary = {
        'selected_pairs': len(pair_candidates),
        'total_new': int(sum(x.num_new for x in pair_candidates)),
        'total_disappear': int(sum(x.num_disappear for x in pair_candidates)),
        'total_survival': int(sum(x.num_survival for x in pair_candidates)),
        'metrics': metric_rows,
    }
    with open(osp.join(args.out_dir, 'summary.json'), 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    main()
