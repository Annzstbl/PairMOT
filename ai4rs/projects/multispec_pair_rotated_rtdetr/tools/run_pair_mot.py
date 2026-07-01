#!/usr/bin/env python3
"""Run pair-detection based online MOT for HSMOT.

Stages:
  detect: sequentially run pair detector per video and write jsonl caches.
  track:  read caches, run Rotated BoT-SORT style tracker, write TrackEval txt.
  eval:   call ../TrackEval HSMOT_8ch evaluation on tracker txt files.
  all:    detect + track + eval.
"""
from __future__ import annotations

import argparse
import csv
import itertools
import json
import multiprocessing as mp
import os
import os.path as osp
import shutil
import subprocess
import sys
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import torch
from mmengine.config import Config
from mmengine.dataset import Compose
from mmengine.runner import load_checkpoint
from mmrotate.registry import MODELS
from mmrotate.utils import register_all_modules

_AI4RS_ROOT = osp.abspath(osp.join(osp.dirname(__file__), '../../..'))
if _AI4RS_ROOT not in sys.path:
    sys.path.insert(0, _AI4RS_ROOT)

import projects.multispec_pair_rotated_rtdetr.multispec_pair_rotated_rtdetr  # noqa: E402,F401
from mmrotate.datasets.hsmot import load_hsmot_sequence_ann  # noqa: E402
from projects.multispec_pair_rotated_rtdetr.multispec_pair_rotated_rtdetr.pair_mot_tracker import (  # noqa: E402,E501
    PairDetection,
    PairFrameRecord,
    PairMOTTracker,
    detection_score,
    read_cache,
    rbox_to_qbox_list,
    write_cache,
    write_trackeval_txt,
)


def _sequence_list(data_root: str, ann_file: str | None,
                   ann_subdir: str) -> List[str]:
    if ann_file:
        ann_path = ann_file
        if not osp.isabs(ann_path):
            ann_path = osp.abspath(ann_path)
        if osp.isdir(ann_path):
            return sorted(
                osp.splitext(name)[0] for name in os.listdir(ann_path)
                if name.endswith('.txt'))
        with open(ann_path, 'r', encoding='utf-8') as f:
            return [line.strip() for line in f if line.strip()]
    ann_dir = osp.join(data_root, ann_subdir)
    return sorted(
        osp.splitext(name)[0] for name in os.listdir(ann_dir)
        if name.endswith('.txt'))


def _img_filename(frame_id: int, img_format: str) -> str:
    if img_format == 'npy':
        return f'{frame_id:06d}.npy'
    if img_format == '3jpg':
        return f'{frame_id:06d}_p1.jpg'
    raise ValueError(f'Unsupported img_format: {img_format}')


def _frame_ids_from_images(img_dir: str, img_format: str) -> List[int]:
    if not osp.isdir(img_dir):
        raise FileNotFoundError(f'Image sequence directory not found: {img_dir}')
    frame_ids = []
    suffix = '.npy' if img_format == 'npy' else '_p1.jpg'
    for name in os.listdir(img_dir):
        if not name.endswith(suffix):
            continue
        stem = name[:-len(suffix)]
        if stem.isdigit():
            frame_ids.append(int(stem))
    frame_ids = sorted(set(frame_ids))
    expected = list(range(1, len(frame_ids) + 1))
    if frame_ids != expected:
        raise ValueError(
            f'Non-continuous frame ids in {img_dir}: first ids={frame_ids[:8]}, '
            f'last ids={frame_ids[-8:] if frame_ids else []}, '
            f'expected 1..{len(frame_ids)}')
    return frame_ids


def _instances_from_frame(frame_anns: Dict[int, List[dict]],
                          frame_id: int) -> List[dict]:
    instances = []
    for ann in frame_anns.get(frame_id, []):
        instances.append({
            'bbox': np.asarray(ann['polygon'], dtype=np.float32),
            'bbox_label': ann['class_id'],
            'ignore_flag': ann['ignore_flag'],
            'track_id': ann['track_id'],
        })
    return instances


def _make_pair_info(seq_name: str, img_root: str, img_format: str,
                    frame_anns: Dict[int, List[dict]],
                    prev_frame_id: int, curr_frame_id: int) -> dict:
    prev_img = osp.join(img_root, seq_name, _img_filename(prev_frame_id, img_format))
    curr_img = osp.join(img_root, seq_name, _img_filename(curr_frame_id, img_format))
    if not osp.isfile(prev_img):
        raise FileNotFoundError(f'Previous frame image missing: {prev_img}')
    if not osp.isfile(curr_img):
        raise FileNotFoundError(f'Current frame image missing: {curr_img}')
    return {
        'img_id': f'{seq_name}_{curr_frame_id:06d}_p{prev_frame_id:06d}',
        'video_id': seq_name,
        'seq_name': seq_name,
        'frame_id': curr_frame_id,
        'frame_id_prev': prev_frame_id,
        'frame_gap': curr_frame_id - prev_frame_id,
        'anchor_frame_id': curr_frame_id,
        'img_path': curr_img,
        'img_path_prev': prev_img,
        'file_name': _img_filename(curr_frame_id, img_format),
        'file_name_prev': _img_filename(prev_frame_id, img_format),
        'instances_prev': _instances_from_frame(frame_anns, prev_frame_id),
        'instances_curr': _instances_from_frame(frame_anns, curr_frame_id),
    }


def _build_model_and_pipeline(config: str, checkpoint: str, device: str):
    register_all_modules()
    cfg = Config.fromfile(config)
    model = MODELS.build(cfg.model)
    load_checkpoint(model, checkpoint, map_location='cpu')
    torch_device = torch.device(device if torch.cuda.is_available() else 'cpu')
    model = model.to(torch_device)
    model.eval()
    preprocessor = MODELS.build(cfg.model.data_preprocessor).to(torch_device)
    pipeline = Compose(cfg.val_pipeline)
    return cfg, model, preprocessor, pipeline, torch_device


def _scale_factor_list(meta: dict) -> List[float]:
    sf = meta.get('scale_factor', (1.0, 1.0))
    if isinstance(sf, torch.Tensor):
        sf = sf.detach().cpu().numpy()
    sf = np.asarray(sf, dtype=np.float32).reshape(-1)
    if sf.size == 1:
        sf = np.repeat(sf, 2)
    if sf.size >= 2:
        return [float(sf[0]), float(sf[1])]
    return [1.0, 1.0]


def _rboxes_to_original_image(rboxes: torch.Tensor, meta: dict) -> torch.Tensor:
    if rboxes.numel() == 0:
        return rboxes
    sx, sy = _scale_factor_list(meta)
    scale = rboxes.new_tensor([sx, sy, sx, sy, 1.0]).clamp(min=1e-6)
    return rboxes / scale


def _predict_one(model, preprocessor, pipeline, device, pair_info: dict,
                 score_mode: str) -> PairFrameRecord:
    packed = pipeline(pair_info)
    inputs = packed['inputs'].unsqueeze(0)
    data_sample = packed['data_samples']
    preprocessed = preprocessor(
        {'inputs': inputs, 'data_samples': [data_sample]}, training=False)
    with torch.no_grad():
        outputs = model.predict(
            preprocessed['inputs'].to(device),
            preprocessed['data_samples'],
            rescale=False)
    sample = outputs[0]
    meta = sample.metainfo
    pred = sample.pred_pair_instances
    scores = pred.scores.detach().cpu().float()
    labels = pred.labels.detach().cpu().long()
    bboxes_prev = _rboxes_to_original_image(
        pred.bboxes_prev.detach().cpu().float(), meta)
    bboxes_curr = _rboxes_to_original_image(
        pred.bboxes_curr.detach().cpu().float(), meta)
    pres_prev = getattr(pred, 'presence_prev', None)
    pres_curr = getattr(pred, 'presence_curr', None)
    score_prev = getattr(pred, 'scores_prev', None)
    score_curr = getattr(pred, 'scores_curr', None)
    label_prev = getattr(pred, 'labels_prev', None)
    label_curr = getattr(pred, 'labels_curr', None)
    if pres_prev is not None:
        pres_prev = pres_prev.detach().cpu().float()
    if pres_curr is not None:
        pres_curr = pres_curr.detach().cpu().float()
    if score_prev is not None:
        score_prev = score_prev.detach().cpu().float()
    if score_curr is not None:
        score_curr = score_curr.detach().cpu().float()
    if label_prev is not None:
        label_prev = label_prev.detach().cpu().long()
    if label_curr is not None:
        label_curr = label_curr.detach().cpu().long()

    detections = []
    for idx in range(int(scores.numel())):
        pp = float(pres_prev[idx]) if pres_prev is not None else None
        pc = float(pres_curr[idx]) if pres_curr is not None else None
        cls_score = float(scores[idx])
        detections.append(PairDetection(
            index=idx,
            prev_bbox=[float(x) for x in bboxes_prev[idx].tolist()],
            curr_bbox=[float(x) for x in bboxes_curr[idx].tolist()],
            score=detection_score(cls_score, pp, pc, score_mode),
            cls_score=cls_score,
            label=int(labels[idx]),
            presence_prev=pp,
            presence_curr=pc,
            score_prev=float(score_prev[idx]) if score_prev is not None else None,
            score_curr=float(score_curr[idx]) if score_curr is not None else None,
            label_prev=int(label_prev[idx]) if label_prev is not None else None,
            label_curr=int(label_curr[idx]) if label_curr is not None else None,
        ))

    return PairFrameRecord(
        seq_name=str(pair_info['seq_name']),
        prev_frame_id=int(pair_info['frame_id_prev']),
        curr_frame_id=int(pair_info['frame_id']),
        frame_gap=int(pair_info['frame_gap']),
        prev_img_path=str(pair_info['img_path_prev']),
        curr_img_path=str(pair_info['img_path']),
        img_shape=list(meta.get('img_shape', [])),
        ori_shape=list(meta.get('ori_shape', [])),
        scale_factor=_scale_factor_list(meta),
        detections=detections,
        is_first_pair=int(pair_info['frame_id']) == int(pair_info['frame_id_prev']),
    )


def _detect_sequences_worker(worker_id: int, seqs: Sequence[str], args_dict: dict,
                             device: str) -> None:
    cfg, model, preprocessor, pipeline, torch_device = _build_model_and_pipeline(
        args_dict['config'], args_dict['checkpoint'], device)
    img_root = osp.join(args_dict['data_root'], args_dict['img_subdir'])
    ann_dir = osp.join(args_dict['data_root'], args_dict['ann_subdir'])
    cache_dir = osp.join(args_dict['out_dir'], 'pair_cache')
    for seq_idx, seq_name in enumerate(seqs):
        cache_path = osp.join(cache_dir, f'{seq_name}.jsonl')
        if osp.isfile(cache_path) and not args_dict['force']:
            print(f'[worker {worker_id}] cache exists, skip {seq_name}: {cache_path}', flush=True)
            continue
        ann_path = osp.join(ann_dir, f'{seq_name}.txt')
        if not osp.isfile(ann_path):
            raise FileNotFoundError(f'Annotation not found: {ann_path}')
        frame_anns = load_hsmot_sequence_ann(ann_path)
        frame_ids = _frame_ids_from_images(
            osp.join(img_root, seq_name), args_dict['img_format'])
        records = []
        # First frame bootstrap: Frame1 + Frame1.
        first = frame_ids[0]
        pair_infos = [_make_pair_info(
            seq_name, img_root, args_dict['img_format'], frame_anns, first, first)]
        pair_infos.extend(
            _make_pair_info(seq_name, img_root, args_dict['img_format'],
                            frame_anns, prev_id, curr_id)
            for prev_id, curr_id in zip(frame_ids[:-1], frame_ids[1:]))
        for pair_idx, pair_info in enumerate(pair_infos):
            rec = _predict_one(
                model, preprocessor, pipeline, torch_device, pair_info,
                args_dict['score_mode'])
            records.append(rec)
            if args_dict['log_interval'] and (pair_idx + 1) % args_dict['log_interval'] == 0:
                print(
                    f'[worker {worker_id}] {seq_name} pair '
                    f'{pair_idx + 1}/{len(pair_infos)}',
                    flush=True)
        meta = {
            'seq_name': seq_name,
            'config': osp.abspath(args_dict['config']),
            'checkpoint': osp.abspath(args_dict['checkpoint']),
            'data_root': osp.abspath(args_dict['data_root']),
            'ann_subdir': args_dict['ann_subdir'],
            'img_subdir': args_dict['img_subdir'],
            'img_format': args_dict['img_format'],
            'score_mode': args_dict['score_mode'],
            'num_frames': len(frame_ids),
            'num_pair_records': len(records),
            'box_format': 'rbox_cxcywha_original_image',
        }
        write_cache(cache_path, meta, records)
        print(
            f'[worker {worker_id}] wrote cache {seq_idx + 1}/{len(seqs)} '
            f'{seq_name}: {cache_path}',
            flush=True)


def run_detect(args) -> None:
    seqs = _sequence_list(args.data_root, args.ann_file, args.ann_subdir)
    if args.max_seqs:
        seqs = seqs[:args.max_seqs]
    devices = [item.strip() for item in args.devices.split(',') if item.strip()]
    if not devices:
        devices = [args.device]
    num_procs = max(1, int(args.num_procs))
    worker_specs = []
    for proc_idx in range(num_procs):
        shard = seqs[proc_idx::num_procs]
        if shard:
            worker_specs.append((proc_idx, shard, devices[proc_idx % len(devices)]))
    args_dict = vars(args).copy()
    if len(worker_specs) == 1:
        proc_idx, shard, device = worker_specs[0]
        _detect_sequences_worker(proc_idx, shard, args_dict, device)
        return
    ctx = mp.get_context('spawn')
    procs = []
    for proc_idx, shard, device in worker_specs:
        proc = ctx.Process(
            target=_detect_sequences_worker,
            args=(proc_idx, shard, args_dict, device))
        proc.start()
        procs.append(proc)
    failed = False
    for proc in procs:
        proc.join()
        failed = failed or proc.exitcode != 0
    if failed:
        raise RuntimeError('At least one detect worker failed')


def _run_tracker_on_cache(cache_path: str, args, tracker_name: str) -> dict:
    meta, records = read_cache(cache_path)
    if not records:
        raise ValueError(f'Empty cache: {cache_path}')
    seq_name = str(meta['seq_name'])
    save_events = bool(args.save_debug_matches or args.save_vis)
    tracker = PairMOTTracker(
        new_born_th=args.new_born_th,
        track_th=args.track_th,
        match_iou_th=args.match_iou_th,
        new_birth_iou_th=args.new_birth_iou_th,
        max_age=args.max_age,
        init_same_iou_th=args.init_same_iou_th,
        class_aware=args.class_aware,
    )
    event_rows = []
    first = True
    for rec in sorted(records, key=lambda x: (x.curr_frame_id, x.prev_frame_id)):
        if first:
            if not rec.is_first_pair or rec.prev_frame_id != rec.curr_frame_id:
                raise ValueError(
                    f'First cache record must be Frame1+Frame1 for {seq_name}')
            tracker.init_first_frame(rec)
            if save_events:
                event_rows.extend(tracker.last_events)
            first = False
            continue
        if rec.frame_gap != 1:
            raise ValueError(
                f'MOT inference expects sequential gap=1, got {rec.frame_gap} '
                f'in {cache_path}')
        tracker.update_pair(rec)
        if save_events:
            event_rows.extend(tracker.last_events)
    pred_path = osp.join(
        args.out_dir, 'trackers', tracker_name, args.tracker_sub_folder,
        f'{seq_name}.txt')
    rows = tracker.all_history()
    write_trackeval_txt(pred_path, rows)
    out = {
        'seq_name': seq_name,
        'tracks': len({row[1] for row in rows}),
        'detections': len(rows),
        'pred_path': pred_path,
    }
    if save_events:
        event_path = osp.join(
            args.out_dir, 'trackers', tracker_name, 'debug_matches',
            f'{seq_name}.jsonl')
        os.makedirs(osp.dirname(event_path), exist_ok=True)
        with open(event_path, 'w', encoding='utf-8') as f:
            for event in event_rows:
                f.write(json.dumps(event, ensure_ascii=False) + '\n')
        out['event_path'] = event_path
    return out


def _tracker_name(args, suffix: str = '') -> str:
    if args.tracker_name:
        return args.tracker_name + suffix
    base = (
        f'pairmot_nb{args.new_born_th:g}_tr{args.track_th:g}_'
        f'iou{args.match_iou_th:g}_age{args.max_age}')
    return base + suffix


def run_track(args, suffix: str = '') -> Tuple[str, List[dict]]:
    cache_dir = osp.join(args.out_dir, 'pair_cache')
    seqs = _sequence_list(args.data_root, args.ann_file, args.ann_subdir)
    if args.max_seqs:
        seqs = seqs[:args.max_seqs]
    tracker_name = _tracker_name(args, suffix)
    summaries = []
    for seq_name in seqs:
        cache_path = osp.join(cache_dir, f'{seq_name}.jsonl')
        if not osp.isfile(cache_path):
            raise FileNotFoundError(f'Pair cache missing: {cache_path}')
        summaries.append(_run_tracker_on_cache(cache_path, args, tracker_name))
    summary_path = osp.join(args.out_dir, 'trackers', tracker_name, 'track_summary.json')
    os.makedirs(osp.dirname(summary_path), exist_ok=True)
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump({
            'tracker_name': tracker_name,
            'params': {
                'new_born_th': args.new_born_th,
                'track_th': args.track_th,
                'match_iou_th': args.match_iou_th,
                'new_birth_iou_th': args.new_birth_iou_th,
                'max_age': args.max_age,
                'init_same_iou_th': args.init_same_iou_th,
                'class_aware': args.class_aware,
                'save_debug_matches': args.save_debug_matches,
            },
            'sequences': summaries,
        }, f, indent=2)
    print(f'wrote tracker results: {summary_path}', flush=True)
    return tracker_name, summaries


def run_eval(args, tracker_name: str | None = None) -> None:
    tracker_name = tracker_name or _tracker_name(args)
    trackeval_root = osp.abspath(args.trackeval_root)
    gt_folder = osp.abspath(osp.join(args.data_root, args.ann_subdir))
    img_folder = osp.abspath(osp.join(args.data_root, args.img_subdir))
    if args.max_seqs:
        seqs = _sequence_list(args.data_root, args.ann_file, args.ann_subdir)
        seqs = seqs[:args.max_seqs]
        subset_root = osp.join(args.out_dir, '_trackeval_subset')
        gt_subset = osp.join(subset_root, 'mot')
        img_subset = osp.join(subset_root, args.img_subdir)
        if osp.isdir(subset_root):
            shutil.rmtree(subset_root)
        os.makedirs(gt_subset, exist_ok=True)
        os.makedirs(img_subset, exist_ok=True)
        for seq in seqs:
            os.symlink(osp.join(gt_folder, f'{seq}.txt'),
                       osp.join(gt_subset, f'{seq}.txt'))
            os.symlink(osp.join(img_folder, seq), osp.join(img_subset, seq))
        gt_folder = osp.abspath(gt_subset)
        img_folder = osp.abspath(img_subset)
    cmd = [
        sys.executable,
        osp.join(trackeval_root, 'scripts/run_hsmot_8ch.py'),
        '--USE_PARALLEL', 'False',
        '--METRICS', 'HOTA', 'CLEAR', 'Identity',
        '--TRACKERS_TO_EVAL', tracker_name,
        '--TRACKER_SUB_FOLDER', args.tracker_sub_folder,
        '--GT_FOLDER', gt_folder,
        '--IMG_FOLDER', img_folder,
        '--TRACKERS_FOLDER', osp.abspath(osp.join(args.out_dir, 'trackers')),
        '--OUTPUT_FOLDER', osp.abspath(osp.join(args.out_dir, 'trackers')),
    ]
    print('running TrackEval:', ' '.join(cmd), flush=True)
    env = os.environ.copy()
    pairmmot_root = osp.abspath(osp.join(_AI4RS_ROOT, '..'))
    extra_pythonpath = [
        _AI4RS_ROOT,
        pairmmot_root,
        osp.join(pairmmot_root, 'hsmot'),
    ]
    env['PYTHONPATH'] = os.pathsep.join(
        extra_pythonpath + ([env['PYTHONPATH']] if env.get('PYTHONPATH') else []))
    subprocess.run(cmd, cwd=trackeval_root, check=True, env=env)


def _float_list(text: str) -> List[float]:
    return [float(x) for x in text.split(',') if x.strip()]


def _int_list(text: str) -> List[int]:
    return [int(x) for x in text.split(',') if x.strip()]


def _read_trackeval_summary(args, tracker_name: str) -> dict:
    path = osp.join(
        args.out_dir, 'trackers', tracker_name, 'eval',
        'cls_comb_det_av_summary.csv')
    if not osp.isfile(path):
        return {}
    with open(path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        row = next(reader, None)
    if row is None:
        return {}
    keys = ('HOTA', 'DetA', 'AssA', 'MOTA', 'IDF1', 'Dets', 'GT_Dets',
            'IDs', 'GT_IDs')
    out = {}
    for key in keys:
        if key in row:
            try:
                out[key] = float(row[key])
            except ValueError:
                out[key] = row[key]
    return out


def _load_track_txt(path: str) -> Dict[int, List[Tuple[int, np.ndarray, float, int]]]:
    by_frame: Dict[int, List[Tuple[int, np.ndarray, float, int]]] = {}
    if not osp.isfile(path):
        return by_frame
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split(',')
            if len(parts) < 12:
                continue
            frame_id = int(float(parts[0]))
            track_id = int(float(parts[1]))
            poly = np.asarray([float(x) for x in parts[2:10]],
                              dtype=np.float32).reshape(4, 2)
            score = float(parts[10])
            label = int(float(parts[11]))
            by_frame.setdefault(frame_id, []).append((track_id, poly, score, label))
    return by_frame


def _read_image(path: str) -> np.ndarray:
    import cv2

    if path.endswith('.npy'):
        arr = np.load(path)
        if arr.ndim == 2:
            arr = arr[:, :, None]
        rgb = arr[:, :, :3].astype(np.float32)
        lo, hi = np.percentile(rgb, [1, 99])
        rgb = np.clip((rgb - lo) / max(hi - lo, 1e-6) * 255, 0, 255)
        return rgb.astype(np.uint8)[:, :, ::-1].copy()
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f'Failed to read image: {path}')
    return img


def _color_for_id(track_id: int) -> Tuple[int, int, int]:
    rng = np.random.default_rng(int(track_id) * 7919)
    color = rng.integers(64, 255, size=3)
    return int(color[0]), int(color[1]), int(color[2])


def rbox_to_qbox_for_vis(rbox: Sequence[float]) -> np.ndarray:
    return np.asarray(rbox_to_qbox_list(rbox), dtype=np.float32).reshape(4, 2)


def _draw_poly(img: np.ndarray, poly: np.ndarray, color: Tuple[int, int, int],
               text: str) -> None:
    import cv2

    pts = np.round(poly).astype(np.int32).reshape(-1, 1, 2)
    cv2.polylines(img, [pts], isClosed=True, color=color, thickness=2)
    x, y = int(poly[:, 0].min()), int(poly[:, 1].min())
    cv2.putText(
        img, text, (max(x, 0), max(y - 4, 12)),
        cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)


def _events_by_frame(path: str) -> Dict[int, List[dict]]:
    by_frame: Dict[int, List[dict]] = {}
    if not osp.isfile(path):
        return by_frame
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            event = json.loads(line)
            by_frame.setdefault(int(event['frame_id']), []).append(event)
    return by_frame


def _records_by_curr_frame(cache_path: str) -> Dict[int, PairFrameRecord]:
    _, records = read_cache(cache_path)
    return {rec.curr_frame_id: rec for rec in records}


def _records_for_vis(cache_path: str) -> List[PairFrameRecord]:
    _, records = read_cache(cache_path)
    return records


def _draw_frame_tracks(img: np.ndarray,
                       rows: Sequence[Tuple[int, np.ndarray, float, int]],
                       score_thr: float) -> None:
    for track_id, poly, score, label in rows:
        if score < score_thr:
            continue
        color = _color_for_id(track_id)
        _draw_poly(img, poly, color, f'id{track_id}:{label}:{score:.2f}')


def _draw_canvas_label(img: np.ndarray, text: str) -> None:
    import cv2

    cv2.putText(img, text, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.75,
                (255, 255, 255), 3, cv2.LINE_AA)
    cv2.putText(img, text, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.75,
                (20, 20, 20), 1, cv2.LINE_AA)


def _draw_pair_link(canvas: np.ndarray, prev_poly: np.ndarray,
                    curr_poly: np.ndarray,
                    color: Tuple[int, int, int]) -> None:
    import cv2

    p0 = tuple(np.round(prev_poly.mean(axis=0)).astype(np.int32).tolist())
    p1 = tuple(np.round(curr_poly.mean(axis=0)).astype(np.int32).tolist())
    cv2.line(canvas, p0, p1, color, 1, cv2.LINE_AA)


def _draw_diag_legend(img: np.ndarray) -> None:
    import cv2

    items = [
        ((255, 80, 40), 'track'),
        ((0, 190, 255), 'prev_det'),
        ((60, 230, 80), 'curr_det'),
    ]
    x = 8
    y = 54
    for color, text in items:
        cv2.rectangle(img, (x, y - 12), (x + 18, y + 2), color, -1)
        cv2.putText(img, text, (x + 24, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(img, text, (x + 24, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    (20, 20, 20), 1, cv2.LINE_AA)
        x += 105


def save_track_visualizations(args, tracker_name: str,
                              summaries: Sequence[dict]) -> None:
    import cv2

    vis_root = osp.join(args.out_dir, 'trackers', tracker_name, 'vis')
    for summary in summaries:
        seq = summary['seq_name']
        pred_path = summary['pred_path']
        event_path = summary.get('event_path', '')
        cache_path = osp.join(args.out_dir, 'pair_cache', f'{seq}.jsonl')
        by_frame = _load_track_txt(pred_path)
        events = _events_by_frame(event_path)
        pair_records = _records_for_vis(cache_path)
        seq_vis_dir = osp.join(vis_root, seq)
        if osp.isdir(seq_vis_dir):
            shutil.rmtree(seq_vis_dir)
        final_dir = osp.join(seq_vis_dir, 'final_tracks')
        raw_dir = osp.join(seq_vis_dir, 'raw_pair')
        match_dir = osp.join(seq_vis_dir, 'matched_pair')
        diag_dir = osp.join(seq_vis_dir, 'track_match_diag')
        for path in (final_dir, raw_dir, match_dir, diag_dir):
            os.makedirs(path, exist_ok=True)
        saved = 0
        for rec in pair_records:
            if args.vis_stride > 1 and (rec.prev_frame_id - 1) % args.vis_stride != 0:
                continue
            if saved >= args.vis_max_frames:
                break
            prev_img = _read_image(rec.prev_img_path)
            curr_img = _read_image(rec.curr_img_path)
            w_prev = prev_img.shape[1]
            name = f'{rec.prev_frame_id:06d}_to_{rec.curr_frame_id:06d}.jpg'

            final_canvas = np.concatenate([prev_img.copy(), curr_img.copy()], axis=1)
            _draw_canvas_label(final_canvas[:, :w_prev], f'frame {rec.prev_frame_id}')
            _draw_canvas_label(final_canvas[:, w_prev:], f'frame {rec.curr_frame_id}')
            _draw_frame_tracks(
                final_canvas[:, :w_prev],
                by_frame.get(rec.prev_frame_id, []),
                args.vis_score_thr)
            curr_rows = []
            for track_id, poly, score, label in by_frame.get(rec.curr_frame_id, []):
                shifted = poly.copy()
                shifted[:, 0] += w_prev
                curr_rows.append((track_id, shifted, score, label))
            _draw_frame_tracks(final_canvas, curr_rows, args.vis_score_thr)
            cv2.imwrite(osp.join(final_dir, name), final_canvas)

            raw_canvas = np.concatenate([prev_img.copy(), curr_img.copy()], axis=1)
            _draw_canvas_label(raw_canvas[:, :w_prev], f'prev {rec.prev_frame_id}')
            _draw_canvas_label(raw_canvas[:, w_prev:], f'curr {rec.curr_frame_id}')
            for det in rec.detections:
                if max(det.prev_side_score(), det.curr_side_score()) < args.vis_score_thr:
                    continue
                color = _color_for_id(det.index + 1)
                prev_poly = rbox_to_qbox_for_vis(det.prev_bbox)
                curr_poly = rbox_to_qbox_for_vis(det.curr_bbox)
                curr_poly[:, 0] += w_prev
                _draw_poly(raw_canvas, prev_poly, color,
                           f'q{det.index} p={det.prev_side_score():.2f}')
                _draw_poly(raw_canvas, curr_poly, color,
                           f'q{det.index} c={det.curr_side_score():.2f}')
                _draw_pair_link(raw_canvas, prev_poly, curr_poly, color)
            cv2.imwrite(osp.join(raw_dir, name), raw_canvas)

            match_canvas = np.concatenate([prev_img.copy(), curr_img.copy()], axis=1)
            _draw_canvas_label(match_canvas[:, :w_prev], f'prev {rec.prev_frame_id}')
            _draw_canvas_label(match_canvas[:, w_prev:], f'curr {rec.curr_frame_id}')
            det_by_index = {det.index: det for det in rec.detections}
            frame_events = events.get(rec.curr_frame_id, [])
            for event in frame_events:
                if event.get('event') not in (
                        'birth', 'match', 'matched_prev_curr_low'):
                    continue
                det = det_by_index.get(int(event['det_index']))
                if det is None:
                    continue
                track_id = int(event.get('track_id', det.index + 1))
                color = _color_for_id(track_id)
                prev_poly = rbox_to_qbox_for_vis(det.prev_bbox)
                curr_poly = rbox_to_qbox_for_vis(det.curr_bbox)
                curr_poly[:, 0] += w_prev
                label = event['event']
                text = (
                    f'{label} id{track_id} '
                    f'p={event.get("prev_score", det.prev_side_score()):.2f} '
                    f'c={event.get("curr_score", det.curr_side_score()):.2f}')
                _draw_poly(match_canvas, prev_poly, color, text)
                _draw_poly(match_canvas, curr_poly, color, text)
                _draw_pair_link(match_canvas, prev_poly, curr_poly, color)
            cv2.imwrite(osp.join(match_dir, name), match_canvas)

            diag_canvas = np.concatenate([prev_img.copy(), curr_img.copy()], axis=1)
            _draw_canvas_label(
                diag_canvas[:, :w_prev],
                f'track objects @ {rec.prev_frame_id}')
            _draw_canvas_label(
                diag_canvas[:, w_prev:],
                f'pair detections {rec.prev_frame_id}->{rec.curr_frame_id}')
            _draw_diag_legend(diag_canvas)
            for event in frame_events:
                if event.get('event') != 'match_diag':
                    continue
                det = det_by_index.get(int(event['best_det_index']))
                if det is None:
                    continue
                iou = float(event['best_iou'])
                track_id = int(event['track_id'])
                reason = str(event.get('reason', 'unmatched'))
                track_poly = rbox_to_qbox_for_vis(event['track_bbox'])
                det_prev_poly = rbox_to_qbox_for_vis(det.prev_bbox)
                det_curr_poly = rbox_to_qbox_for_vis(det.curr_bbox)
                det_curr_poly[:, 0] += w_prev
                _draw_poly(diag_canvas, track_poly, (255, 80, 40),
                           f'track id{track_id}')
                _draw_poly(diag_canvas, det_prev_poly, (0, 190, 255),
                           f'prev q{det.index} IoU={iou:.2f} '
                           f'p={event.get("best_det_prev_score", det.prev_side_score()):.2f} '
                           f'{reason}')
                _draw_poly(diag_canvas, det_curr_poly, (60, 230, 80),
                           f'curr q{det.index} '
                           f'c={event.get("best_det_curr_score", det.curr_side_score()):.2f}')
                _draw_pair_link(diag_canvas, det_prev_poly, det_curr_poly,
                                (0, 190, 255))
            cv2.imwrite(osp.join(diag_dir, name), diag_canvas)
            saved += 1
    print(f'wrote track visualizations: {vis_root}', flush=True)


def run_sweep(args) -> None:
    rows = []
    base_tracker_name = args.tracker_name
    if base_tracker_name is None:
        args.tracker_name = 'pairmot_sweep'
    for nb, tr, iou, birth_iou, age in itertools.product(
            _float_list(args.sweep_new_born_th),
            _float_list(args.sweep_track_th),
            _float_list(args.sweep_match_iou_th),
            _float_list(args.sweep_new_birth_iou_th),
            _int_list(args.sweep_max_age)):
        args.new_born_th = nb
        args.track_th = tr
        args.match_iou_th = iou
        args.new_birth_iou_th = birth_iou
        args.max_age = age
        suffix = (
            f'_nb{nb:g}_tr{tr:g}_iou{iou:g}_'
            f'birthiou{birth_iou:g}_age{age}')
        tracker_name, summaries = run_track(args, suffix=suffix)
        if args.eval:
            run_eval(args, tracker_name)
        row = {
            'tracker_name': tracker_name,
            'new_born_th': nb,
            'track_th': tr,
            'match_iou_th': iou,
            'new_birth_iou_th': birth_iou,
            'max_age': age,
            'detections': sum(item['detections'] for item in summaries),
            'tracks': sum(item['tracks'] for item in summaries),
        }
        row.update(_read_trackeval_summary(args, tracker_name))
        rows.append(row)
    args.tracker_name = base_tracker_name
    csv_path = osp.join(args.out_dir, 'sweep_summary.csv')
    base_fields = [
        'tracker_name', 'new_born_th', 'track_th', 'match_iou_th',
        'new_birth_iou_th', 'max_age', 'detections', 'tracks'
    ]
    extra_fields = sorted({key for row in rows for key in row} - set(base_fields))
    fieldnames = base_fields + extra_fields
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f'wrote sweep summary: {csv_path}', flush=True)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', choices=['detect', 'track', 'eval', 'all', 'sweep'],
                        default='all')
    parser.add_argument('--config', required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--data-root', default='../data/hsmot/test')
    parser.add_argument('--ann-file', default=None)
    parser.add_argument('--ann-subdir', default='mot')
    parser.add_argument('--img-subdir', default='npy2jpg')
    parser.add_argument('--img-format', choices=['npy', '3jpg'], default='3jpg')
    parser.add_argument('--out-dir', required=True)
    parser.add_argument('--score-mode', choices=['cls', 'cls_min_presence', 'auto'],
                        default='cls')
    parser.add_argument('--new-born-th', type=float, default=0.5)
    parser.add_argument('--track-th', type=float, default=0.2)
    parser.add_argument('--match-iou-th', type=float, default=0.3)
    parser.add_argument('--new-birth-iou-th', type=float, default=0.6)
    parser.add_argument('--max-age', type=int, default=30)
    parser.add_argument('--init-same-iou-th', type=float, default=0.3)
    parser.add_argument('--class-aware', action='store_true')
    parser.add_argument('--tracker-name', default=None)
    parser.add_argument('--tracker-sub-folder', default='preds')
    parser.add_argument('--trackeval-root', default='../TrackEval')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--devices', default='')
    parser.add_argument('--num-procs', type=int, default=1)
    parser.add_argument('--max-seqs', type=int, default=0)
    parser.add_argument('--force', action='store_true')
    parser.add_argument('--log-interval', type=int, default=100)
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--save-vis', action='store_true')
    parser.add_argument('--save-debug-matches', action='store_true')
    parser.add_argument('--vis-max-frames', type=int, default=50)
    parser.add_argument('--vis-stride', type=int, default=10)
    parser.add_argument('--vis-score-thr', type=float, default=0.0)
    parser.add_argument('--sweep-new-born-th', default='0.5,0.6,0.7')
    parser.add_argument('--sweep-track-th', default='0.15,0.2,0.25')
    parser.add_argument('--sweep-match-iou-th', default='0.2,0.3,0.4')
    parser.add_argument('--sweep-new-birth-iou-th', default='0.5,0.6,0.7')
    parser.add_argument('--sweep-max-age', default='15,30')
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.stage in ('detect', 'all'):
        run_detect(args)
    tracker_name = None
    summaries = None
    if args.stage in ('track', 'all'):
        tracker_name, summaries = run_track(args)
        if args.save_vis:
            save_track_visualizations(args, tracker_name, summaries)
    if args.stage == 'eval' or (args.stage == 'all' and args.eval):
        run_eval(args, tracker_name)
    if args.stage == 'sweep':
        run_sweep(args)


if __name__ == '__main__':
    main()
