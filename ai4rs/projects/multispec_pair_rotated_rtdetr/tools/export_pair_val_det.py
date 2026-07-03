#!/usr/bin/env python3
"""Export pair detector validation predictions as per-sequence txt files.

The first exported pair is 01-02. Same-frame bootstrap pairs are intentionally
not generated here.
"""
from __future__ import annotations

import argparse
import csv
import json
import multiprocessing as mp
import os
import os.path as osp
import re
import sys
from typing import Dict, List, Sequence, Tuple

import torch
from mmrotate.structures.bbox import qbox2rbox

_AI4RS_ROOT = osp.abspath(osp.join(osp.dirname(__file__), '../../..'))
if _AI4RS_ROOT not in sys.path:
    sys.path.insert(0, _AI4RS_ROOT)

from mmrotate.datasets.hsmot import load_hsmot_sequence_ann  # noqa: E402
from mmrotate.datasets.pair_gt import build_pair_gt_from_instances  # noqa: E402
from projects.multispec_pair_rotated_rtdetr.multispec_pair_rotated_rtdetr.pair_ap import (  # noqa: E402,E501
    pair_and_independent_ap_metrics_with_gt_filters,
)
from projects.multispec_pair_rotated_rtdetr.tools.run_pair_mot import (  # noqa: E402,E501
    _build_model_and_pipeline,
    _frame_ids_from_images,
    _instances_from_frame,
    _make_pair_info,
    _predict_batch,
    _sequence_list,
)
from projects.multispec_pair_rotated_rtdetr.multispec_pair_rotated_rtdetr.pair_mot_tracker import (  # noqa: E402,E501
    write_pair_det_txt,
)


def _epoch_name(checkpoint: str) -> str:
    match = re.search(r'epoch_(\d+)\.pth$', osp.basename(checkpoint))
    if match:
        return f'epoch_{int(match.group(1)):02d}'
    stem = osp.splitext(osp.basename(checkpoint))[0]
    return stem.replace('/', '_')


def _maybe_float(value, default: float = 1.0) -> float:
    return default if value is None else float(value)


def _records_to_txt(path: str, records) -> Tuple[int, int]:
    write_pair_det_txt(path, records)
    valid_records = [
        rec for rec in records if rec.prev_frame_id != rec.curr_frame_id
    ]
    return len(valid_records), sum(len(rec.detections) for rec in valid_records)


def _pair_gt_to_ap_tensors(pair_gt: dict) -> dict:
    gt_prev_q = torch.as_tensor(pair_gt['bboxes_prev'], dtype=torch.float32)
    gt_curr_q = torch.as_tensor(pair_gt['bboxes_curr'], dtype=torch.float32)
    return dict(
        gt_labels=torch.as_tensor(pair_gt['labels'], dtype=torch.long),
        gt_prev=qbox2rbox(gt_prev_q) if gt_prev_q.numel() else torch.zeros((0, 5)),
        gt_curr=qbox2rbox(gt_curr_q) if gt_curr_q.numel() else torch.zeros((0, 5)),
        gt_valid_prev=torch.as_tensor(pair_gt['valid_prev'], dtype=torch.bool),
        gt_valid_curr=torch.as_tensor(pair_gt['valid_curr'], dtype=torch.bool),
    )


def _record_to_ap_sample(rec, pair_gt: dict, pres_thr: float) -> dict:
    out = _pair_gt_to_ap_tensors(pair_gt)
    dets = sorted(rec.detections, key=lambda det: det.index)
    if dets:
        score_prev = torch.as_tensor(
            [det.prev_side_score() for det in dets], dtype=torch.float32)
        score_curr = torch.as_tensor(
            [det.curr_side_score() for det in dets], dtype=torch.float32)
        has_presence = any(
            det.presence_prev is not None and det.presence_curr is not None
            for det in dets)
        pres_prev = torch.as_tensor(
            [_maybe_float(det.presence_prev) for det in dets],
            dtype=torch.float32)
        pres_curr = torch.as_tensor(
            [_maybe_float(det.presence_curr) for det in dets],
            dtype=torch.float32)
        if has_presence:
            valid_prev = pres_prev >= pres_thr
            valid_curr = pres_curr >= pres_thr
            pair_scores = torch.as_tensor([det.pair_score() for det in dets],
                                          dtype=torch.float32)
        else:
            valid_prev = score_prev >= pres_thr
            valid_curr = score_curr >= pres_thr
            pair_scores = torch.where(
                valid_prev & valid_curr,
                torch.sqrt(score_prev.clamp(min=1e-6) *
                           score_curr.clamp(min=1e-6)),
                torch.where(
                    valid_prev,
                    score_prev * (1 - score_curr),
                    (1 - score_prev) * score_curr))
            pres_prev = score_prev
            pres_curr = score_curr
        keep = valid_prev | valid_curr
        out.update(
            pred_labels=torch.as_tensor(
                [det.label for det in dets], dtype=torch.long)[keep],
            pred_prev=torch.as_tensor([det.prev_bbox for det in dets],
                                      dtype=torch.float32)[keep],
            pred_curr=torch.as_tensor([det.curr_bbox for det in dets],
                                      dtype=torch.float32)[keep],
            pred_valid_prev=valid_prev[keep],
            pred_valid_curr=valid_curr[keep],
            pred_scores=pair_scores[keep],
            pred_cls_scores=torch.as_tensor(
                [det.cls_score for det in dets], dtype=torch.float32)[keep],
            pred_presence_prev=pres_prev[keep],
            pred_presence_curr=pres_curr[keep],
            pred_score_prev=score_prev[keep],
            pred_score_curr=score_curr[keep],
        )
    else:
        out.update(
            pred_labels=torch.zeros((0,), dtype=torch.long),
            pred_prev=torch.zeros((0, 5), dtype=torch.float32),
            pred_curr=torch.zeros((0, 5), dtype=torch.float32),
            pred_valid_prev=torch.zeros((0,), dtype=torch.bool),
            pred_valid_curr=torch.zeros((0,), dtype=torch.bool),
            pred_scores=torch.zeros((0,), dtype=torch.float32),
            pred_cls_scores=torch.zeros((0,), dtype=torch.float32),
            pred_presence_prev=torch.zeros((0,), dtype=torch.float32),
            pred_presence_curr=torch.zeros((0,), dtype=torch.float32),
            pred_score_prev=torch.zeros((0,), dtype=torch.float32),
            pred_score_curr=torch.zeros((0,), dtype=torch.float32),
        )
    return out


def _detect_worker(worker_id: int, seqs: Sequence[str], args_dict: dict,
                   device: str) -> None:
    cfg, model, preprocessor, pipeline, torch_device = _build_model_and_pipeline(
        args_dict['config'], args_dict['checkpoint'], device)
    del cfg
    img_root = osp.join(args_dict['data_root'], args_dict['img_subdir'])
    ann_dir = osp.join(args_dict['data_root'], args_dict['ann_subdir'])
    val_det_dir = args_dict['val_det_dir']
    batch_size = max(1, int(args_dict['batch_size']))
    num_workers = max(0, int(args_dict['num_workers']))
    ap_samples = []
    summaries = []
    for seq_idx, seq_name in enumerate(seqs):
        txt_path = osp.join(val_det_dir, f'{seq_name}.txt')
        if osp.isfile(txt_path) and not args_dict['force']:
            print(f'[worker {worker_id}] exists, skip {seq_name}: {txt_path}',
                  flush=True)
            continue
        ann_path = osp.join(ann_dir, f'{seq_name}.txt')
        frame_anns = load_hsmot_sequence_ann(ann_path)
        frame_ids = _frame_ids_from_images(
            osp.join(img_root, seq_name), args_dict['img_format'])
        pair_infos = [
            _make_pair_info(seq_name, img_root, args_dict['img_format'],
                            frame_anns, prev_id, curr_id)
            for prev_id, curr_id in zip(frame_ids[:-1], frame_ids[1:])
        ]
        records = []
        for start in range(0, len(pair_infos), batch_size):
            batch_infos = pair_infos[start:start + batch_size]
            batch_records = _predict_batch(
                model, preprocessor, pipeline, torch_device, batch_infos,
                args_dict['score_mode'], num_workers=num_workers)
            records.extend(batch_records)
            for pair_info, rec in zip(batch_infos, batch_records):
                pair_gt = build_pair_gt_from_instances(
                    pair_info['instances_prev'],
                    pair_info['instances_curr'],
                    video_id=seq_name,
                    frame_id_prev=int(pair_info['frame_id_prev']),
                    frame_id_curr=int(pair_info['frame_id']),
                )
                ap_samples.append(_record_to_ap_sample(
                    rec, pair_gt, float(args_dict['pres_thr'])))
            done = start + len(batch_infos)
            if args_dict['log_interval'] and done % args_dict['log_interval'] == 0:
                print(
                    f'[worker {worker_id}] {seq_name} pair '
                    f'{done}/{len(pair_infos)}',
                    flush=True)
        num_pairs, num_dets = _records_to_txt(txt_path, records)
        summaries.append(dict(seq_name=seq_name, pairs=num_pairs,
                              detections=num_dets, txt_path=txt_path))
        print(
            f'[worker {worker_id}] wrote {seq_idx + 1}/{len(seqs)} '
            f'{seq_name}: pairs={num_pairs} dets={num_dets}',
            flush=True)
    torch.save(ap_samples, osp.join(val_det_dir, f'_ap_samples_worker{worker_id}.pt'))
    with open(osp.join(val_det_dir, f'_summary_worker{worker_id}.json'),
              'w', encoding='utf-8') as f:
        json.dump(summaries, f, indent=2)


def _write_summary_csv(path: str, summaries: Sequence[dict]) -> None:
    fields = ['seq_name', 'pairs', 'detections', 'txt_path']
    with open(path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(summaries)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--work-dir', required=True)
    parser.add_argument('--data-root', default='../data/hsmot/test')
    parser.add_argument('--ann-file', default=None)
    parser.add_argument('--ann-subdir', default='mot')
    parser.add_argument('--img-subdir', default='npy2jpg')
    parser.add_argument('--img-format', choices=['npy', '3jpg'], default='3jpg')
    parser.add_argument('--score-mode', choices=['cls', 'cls_min_presence', 'auto'],
                        default='cls')
    parser.add_argument('--pres-thr', type=float, default=0.5)
    parser.add_argument('--devices', default='cuda:0')
    parser.add_argument('--num-procs', type=int, default=1)
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--num-workers', type=int, default=8)
    parser.add_argument('--max-seqs', type=int, default=0)
    parser.add_argument('--force', action='store_true')
    parser.add_argument('--log-interval', type=int, default=100)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.data_root = osp.abspath(args.data_root)
    args.work_dir = osp.abspath(args.work_dir)
    args.val_det_dir = osp.join(args.work_dir, 'val_det', _epoch_name(args.checkpoint))
    os.makedirs(args.val_det_dir, exist_ok=True)
    seqs = _sequence_list(args.data_root, args.ann_file, args.ann_subdir)
    if args.max_seqs:
        seqs = seqs[:args.max_seqs]
    devices = [item.strip() for item in args.devices.split(',') if item.strip()]
    if not devices:
        devices = ['cuda:0']
    num_procs = max(1, int(args.num_procs))
    worker_specs = []
    args_dict = vars(args).copy()
    for proc_idx in range(num_procs):
        shard = seqs[proc_idx::num_procs]
        if shard:
            worker_specs.append((proc_idx, shard, args_dict,
                                 devices[proc_idx % len(devices)]))
    if len(worker_specs) == 1:
        _detect_worker(*worker_specs[0])
    else:
        ctx = mp.get_context('spawn')
        procs = []
        for spec in worker_specs:
            proc = ctx.Process(target=_detect_worker, args=spec)
            proc.start()
            procs.append(proc)
        failed = False
        for proc in procs:
            proc.join()
            failed = failed or proc.exitcode != 0
        if failed:
            raise RuntimeError('At least one val_det worker failed')

    ap_samples = []
    summaries = []
    for proc_idx, _, _, _ in worker_specs:
        sample_path = osp.join(args.val_det_dir, f'_ap_samples_worker{proc_idx}.pt')
        if osp.isfile(sample_path):
            ap_samples.extend(torch.load(sample_path, map_location='cpu'))
        summary_path = osp.join(args.val_det_dir, f'_summary_worker{proc_idx}.json')
        if osp.isfile(summary_path):
            with open(summary_path, 'r', encoding='utf-8') as f:
                summaries.extend(json.load(f))
    metrics = pair_and_independent_ap_metrics_with_gt_filters(
        ap_samples, gt_filters=('both', 'new', 'disappear'))
    metrics_path = osp.join(args.val_det_dir, 'metrics.json')
    with open(metrics_path, 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=2)
    _write_summary_csv(osp.join(args.val_det_dir, 'summary.csv'), summaries)
    print(f'wrote val_det: {args.val_det_dir}', flush=True)
    print(f'wrote metrics: {metrics_path}', flush=True)
    print(json.dumps({
        key: metrics[key]
        for key in ('pair_AP50', 'pair_mAP50_95', 'both_pair_mAP50_95',
                    'new_pair_mAP50_95', 'disappear_pair_mAP50_95')
        if key in metrics
    }, indent=2), flush=True)


if __name__ == '__main__':
    main()
