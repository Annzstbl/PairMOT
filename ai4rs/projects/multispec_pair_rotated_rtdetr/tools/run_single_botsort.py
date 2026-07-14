#!/usr/bin/env python3
"""Run single-frame detection followed by RMMOT rotated BoT-SORT on HSMOT."""
from __future__ import annotations

import argparse
import csv
import json
import os
import os.path as osp
import shutil
import subprocess
import sys
from types import SimpleNamespace
from typing import Dict, Iterable, List, Sequence, Tuple

import cv2
import numpy as np
import torch
from mmengine.config import Config
from mmengine.dataset import Compose
from mmengine.runner import load_checkpoint
from mmrotate.registry import MODELS
from mmrotate.structures.bbox import rbox2qbox
from mmrotate.utils import register_all_modules

_AI4RS_ROOT = osp.abspath(osp.join(osp.dirname(__file__), '../../..'))
if _AI4RS_ROOT not in sys.path:
    sys.path.insert(0, _AI4RS_ROOT)

import projects.multispec_pair_rotated_rtdetr.multispec_pair_rotated_rtdetr  # noqa: E402,F401
import projects.multispec_rotated_rtdetr.multispec_rotated_rtdetr  # noqa: E402,F401
import projects.rotated_rtdetr.rotated_rtdetr  # noqa: E402,F401
from mmrotate.datasets.hsmot import load_hsmot_sequence_ann  # noqa: E402
from projects.multispec_pair_rotated_rtdetr.multispec_pair_rotated_rtdetr.botsort_tracker.bot_sort_rotate import BoTSORT  # noqa: E402,E501


BOT_SORT_DEFAULTS = dict(
    track_high_thresh=0.6,
    track_low_thresh=0.1,
    new_track_thresh=0.7,
    track_buffer=30,
    match_thresh=0.8,
    with_reid=False,
    cmc_method='sparseOptFlow',
    name=None,
    ablation=None,
    proximity_thresh=0.5,
    appearance_thresh=0.25,
)


def _sequence_list(data_root: str, ann_file: str | None,
                   ann_subdir: str) -> List[str]:
    if ann_file:
        path = ann_file if osp.isabs(ann_file) else osp.join(data_root, ann_file)
        with open(path, 'r', encoding='utf-8') as f:
            return [line.strip() for line in f if line.strip()]
    ann_dir = osp.join(data_root, ann_subdir)
    return sorted(osp.splitext(name)[0] for name in os.listdir(ann_dir)
                  if name.endswith('.txt'))


def _img_filename(frame_id: int, img_format: str) -> str:
    if img_format == 'npy':
        return f'{frame_id:06d}.npy'
    if img_format == '3jpg':
        return f'{frame_id:06d}_p1.jpg'
    raise ValueError(f'Unsupported img_format: {img_format}')


def _frame_ids_from_images(img_dir: str, img_format: str) -> List[int]:
    suffix = '.npy' if img_format == 'npy' else '_p1.jpg'
    frame_ids = []
    for name in os.listdir(img_dir):
        if name.endswith(suffix):
            stem = name[:-len(suffix)]
            if stem.isdigit():
                frame_ids.append(int(stem))
    return sorted(set(frame_ids))


def _instances_from_frame(frame_anns: Dict[int, List[dict]],
                          frame_id: int) -> List[dict]:
    return [{
        'bbox': np.asarray(ann['polygon'], dtype=np.float32),
        'bbox_label': int(ann['class_id']),
        'ignore_flag': int(ann['ignore_flag']),
        'track_id': int(ann['track_id']),
    } for ann in frame_anns.get(frame_id, [])]


def _make_data_info(seq_name: str, frame_id: int, img_root: str,
                    img_format: str,
                    frame_anns: Dict[int, List[dict]]) -> dict:
    img_path = osp.join(img_root, seq_name, _img_filename(frame_id, img_format))
    return dict(
        img_id=f'{seq_name}_{frame_id:06d}',
        video_id=seq_name,
        seq_name=seq_name,
        frame_id=frame_id,
        file_name=_img_filename(frame_id, img_format),
        img_path=img_path,
        instances=_instances_from_frame(frame_anns, frame_id),
    )


def _scale_factor(meta: dict) -> Tuple[float, float]:
    sf = meta.get('scale_factor', (1.0, 1.0))
    if isinstance(sf, torch.Tensor):
        sf = sf.detach().cpu().numpy()
    sf = np.asarray(sf, dtype=np.float32).reshape(-1)
    if sf.size == 1:
        return float(sf[0]), float(sf[0])
    if sf.size >= 2:
        return float(sf[0]), float(sf[1])
    return 1.0, 1.0


def _to_tensor_boxes(boxes) -> torch.Tensor:
    if hasattr(boxes, 'tensor'):
        return boxes.tensor
    return torch.as_tensor(boxes)


def _build_model(config: str, checkpoint: str, device: str):
    register_all_modules()
    cfg = Config.fromfile(config)
    model = MODELS.build(cfg.model)
    load_checkpoint(model, checkpoint, map_location='cpu')
    torch_device = torch.device(device if torch.cuda.is_available() else 'cpu')
    model = model.to(torch_device).eval()
    preprocessor = MODELS.build(cfg.model.data_preprocessor).to(torch_device)
    pipeline_cfg = cfg.get('test_pipeline', None) or cfg.get('val_pipeline')
    pipeline = Compose(pipeline_cfg)
    return cfg, model, preprocessor, pipeline, torch_device


def _predict_batch(model, preprocessor, pipeline, device,
                   infos: Sequence[dict]) -> List[Tuple[dict, np.ndarray]]:
    packed = [pipeline(info) for info in infos]
    inputs = [item['inputs'] for item in packed]
    samples = [item['data_samples'] for item in packed]
    if len(inputs) == 1:
        inputs = inputs[0].unsqueeze(0)
    with torch.inference_mode():
        processed = preprocessor(
            {'inputs': inputs, 'data_samples': samples}, training=False)
        outputs = model.predict(
            processed['inputs'].to(device),
            processed['data_samples'],
            rescale=False)

    records = []
    for info, sample in zip(infos, outputs):
        meta = sample.metainfo
        pred = sample.pred_instances
        bboxes = _to_tensor_boxes(pred.bboxes).detach().cpu().float()
        scores = pred.scores.detach().cpu().float()
        labels = pred.labels.detach().cpu().long()
        if bboxes.numel():
            sx, sy = _scale_factor(meta)
            scale = bboxes.new_tensor([sx, sy, sx, sy, 1.0]).clamp(min=1e-6)
            bboxes = bboxes / scale
            qboxes = rbox2qbox(bboxes).detach().cpu().numpy()
            rows = np.concatenate([
                np.full((qboxes.shape[0], 1), int(info['frame_id'])),
                qboxes,
                labels.numpy().reshape(-1, 1).astype(np.float32),
                scores.numpy().reshape(-1, 1),
            ], axis=1)
        else:
            rows = np.zeros((0, 11), dtype=np.float32)
        records.append((info, rows))
    return records


def write_det_cache(path: str, rows: Iterable[np.ndarray]) -> Tuple[int, int]:
    rows = [row for row in rows if len(row)]
    os.makedirs(osp.dirname(path), exist_ok=True)
    num_rows = 0
    with open(path, 'w', encoding='utf-8') as f:
        f.write('# frame,x1,y1,x2,y2,x3,y3,x4,y4,cls,score\n')
        for row in rows:
            for vals in row:
                num_rows += 1
                f.write(','.join(
                    [str(int(vals[0]))] +
                    [f'{float(v):.3f}' for v in vals[1:9]] +
                    [str(int(vals[9])), f'{float(vals[10]):.6f}']) + '\n')
    return len(rows), num_rows


def read_det_cache(path: str) -> Dict[int, np.ndarray]:
    by_frame: Dict[int, List[List[float]]] = {}
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            vals = [float(x) for x in line.split(',')]
            frame_id = int(vals[0])
            by_frame.setdefault(frame_id, []).append(vals[1:])
    return {
        frame_id: np.asarray(rows, dtype=np.float32)
        for frame_id, rows in by_frame.items()
    }


def _load_gmc_image(img_root: str, seq_name: str, frame_id: int,
                    img_format: str) -> np.ndarray:
    if img_format == 'npy':
        arr = np.load(osp.join(img_root, seq_name, f'{frame_id:06d}.npy'))
        if arr.ndim == 3 and arr.shape[2] >= 5:
            arr = arr[:, :, [1, 2, 4]]
        elif arr.ndim == 3:
            arr = arr[:, :, :3]
        else:
            arr = np.repeat(arr[:, :, None], 3, axis=2)
        if arr.dtype != np.uint8:
            arr = np.clip(arr, 0, 255).astype(np.uint8)
        return np.ascontiguousarray(arr)
    path = osp.join(img_root, seq_name, f'{frame_id:06d}_p1.jpg')
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(path)
    return img


def result2str(frame: int, track_id: int, xyxyxyxy: Sequence[float],
               score: float, cls: int) -> str:
    vals = [int(frame), int(track_id)]
    vals += [f'{float(v):.3f}' for v in xyxyxyxy]
    vals += [f'{float(score):.3f}', int(cls), -1]
    return ','.join(map(str, vals))


def run_detect(args) -> None:
    cfg, model, preprocessor, pipeline, device = _build_model(
        args.config, args.checkpoint, args.device)
    del cfg
    img_root = osp.join(args.data_root, args.img_subdir)
    ann_dir = osp.join(args.data_root, args.ann_subdir)
    seqs = _sequence_list(args.data_root, args.ann_file, args.ann_subdir)
    if args.max_seqs:
        seqs = seqs[:args.max_seqs]
    det_dir = osp.join(args.out_dir, 'single_det')
    os.makedirs(det_dir, exist_ok=True)
    summaries = []
    for seq_idx, seq_name in enumerate(seqs):
        out_path = osp.join(det_dir, f'{seq_name}.txt')
        if osp.isfile(out_path) and not args.force:
            print(f'[detect] exists, skip {seq_name}: {out_path}', flush=True)
            continue
        frame_anns = load_hsmot_sequence_ann(osp.join(ann_dir, f'{seq_name}.txt'))
        frame_ids = _frame_ids_from_images(osp.join(img_root, seq_name),
                                           args.img_format)
        all_rows = []
        for start in range(0, len(frame_ids), args.batch_size):
            infos = [
                _make_data_info(seq_name, frame_id, img_root, args.img_format,
                                frame_anns)
                for frame_id in frame_ids[start:start + args.batch_size]
            ]
            all_rows.extend(
                rows for _, rows in _predict_batch(
                    model, preprocessor, pipeline, device, infos))
        frames, detections = write_det_cache(out_path, all_rows)
        summaries.append(dict(seq_name=seq_name, frames=frames,
                              detections=detections, txt_path=out_path))
        print(f'[detect] {seq_idx + 1}/{len(seqs)} {seq_name}: '
              f'frames={frames} dets={detections}', flush=True)
    with open(osp.join(det_dir, 'summary.json'), 'w', encoding='utf-8') as f:
        json.dump(summaries, f, indent=2)


def _botsort_args(args, seq_name: str) -> SimpleNamespace:
    data = dict(BOT_SORT_DEFAULTS)
    data.update(
        track_high_thresh=args.bot_track_high_thresh,
        track_low_thresh=args.bot_track_low_thresh,
        new_track_thresh=args.bot_new_track_thresh,
        track_buffer=args.bot_track_buffer,
        match_thresh=args.bot_match_thresh,
        with_reid=args.bot_with_reid,
        cmc_method=args.bot_cmc_method,
        name=seq_name,
        ablation=None,
        proximity_thresh=args.bot_proximity_thresh,
        appearance_thresh=args.bot_appearance_thresh,
    )
    return SimpleNamespace(**data)


def run_track(args) -> str:
    seqs = _sequence_list(args.data_root, args.ann_file, args.ann_subdir)
    if args.max_seqs:
        seqs = seqs[:args.max_seqs]
    img_root = osp.join(args.data_root, args.img_subdir)
    tracker_name = args.tracker_name or (
        'single_botsort_h'
        f'{args.bot_track_high_thresh:g}_l{args.bot_track_low_thresh:g}_'
        f'n{args.bot_new_track_thresh:g}_m{args.bot_match_thresh:g}_'
        f'buf{args.bot_track_buffer}_{args.bot_cmc_method}')
    pred_dir = osp.join(args.out_dir, 'trackers', tracker_name,
                        args.tracker_sub_folder)
    os.makedirs(pred_dir, exist_ok=True)
    summaries = []
    for seq_name in seqs:
        cache = read_det_cache(osp.join(args.out_dir, 'single_det',
                                        f'{seq_name}.txt'))
        frame_ids = _frame_ids_from_images(osp.join(img_root, seq_name),
                                           args.img_format)
        tracker = BoTSORT(_botsort_args(args, seq_name))
        lines = []
        for frame_id in frame_ids:
            dets = cache.get(frame_id, np.zeros((0, 10), dtype=np.float32))
            img = _load_gmc_image(img_root, seq_name, frame_id, args.img_format)
            tracks = tracker.update(dets, img)
            for track in tracks:
                lines.append(result2str(frame_id, track.track_id,
                                        track.xyxyxyxy, track.score, track.cls))
        out_path = osp.join(pred_dir, f'{seq_name}.txt')
        with open(out_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(lines))
        summaries.append(dict(seq_name=seq_name, frames=len(frame_ids),
                              tracks=len(lines), txt_path=out_path))
        print(f'[track] {seq_name}: frames={len(frame_ids)} tracks={len(lines)}',
              flush=True)
    summary_path = osp.join(args.out_dir, 'trackers', tracker_name,
                            'track_summary.json')
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(dict(tracker_name=tracker_name, params=_botsort_args(
            args, '').__dict__, sequences=summaries), f, indent=2)
    return tracker_name


def run_eval(args, tracker_name: str | None = None) -> None:
    tracker_name = tracker_name or args.tracker_name
    if not tracker_name:
        raise ValueError('tracker_name is required for eval-only stage')
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
    env = os.environ.copy()
    pairmot_root = osp.abspath(osp.join(_AI4RS_ROOT, '..'))
    extra_pythonpath = [_AI4RS_ROOT, pairmot_root, osp.join(pairmot_root, 'hsmot')]
    env['PYTHONPATH'] = os.pathsep.join(
        extra_pythonpath + ([env['PYTHONPATH']] if env.get('PYTHONPATH') else []))
    print('running TrackEval:', ' '.join(cmd), flush=True)
    subprocess.run(cmd, cwd=trackeval_root, check=True, env=env)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', choices=['detect', 'track', 'eval', 'all'],
                        default='all')
    parser.add_argument('--config', default=None)
    parser.add_argument('--checkpoint', default=None)
    parser.add_argument('--data-root', default='../data/hsmot/test')
    parser.add_argument('--ann-file', default=None)
    parser.add_argument('--ann-subdir', default='mot')
    parser.add_argument('--img-subdir', default='npy2jpg')
    parser.add_argument('--img-format', choices=['npy', '3jpg'], default='3jpg')
    parser.add_argument('--out-dir', required=True)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--max-seqs', type=int, default=0)
    parser.add_argument('--force', action='store_true')
    parser.add_argument('--tracker-name', default=None)
    parser.add_argument('--tracker-sub-folder', default='preds')
    parser.add_argument('--trackeval-root', default='../TrackEval')
    parser.add_argument('--eval', action='store_true')

    parser.add_argument('--bot-track-high-thresh', type=float, default=0.6)
    parser.add_argument('--bot-track-low-thresh', type=float, default=0.1)
    parser.add_argument('--bot-new-track-thresh', type=float, default=0.7)
    parser.add_argument('--bot-track-buffer', type=int, default=30)
    parser.add_argument('--bot-match-thresh', type=float, default=0.8)
    parser.add_argument('--bot-with-reid', action='store_true')
    parser.add_argument('--bot-cmc-method', default='sparseOptFlow')
    parser.add_argument('--bot-proximity-thresh', type=float, default=0.5)
    parser.add_argument('--bot-appearance-thresh', type=float, default=0.25)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.stage in ('detect', 'all') and (
            not args.config or not args.checkpoint):
        raise ValueError('--config and --checkpoint are required for detect/all')
    args.data_root = osp.abspath(args.data_root)
    args.out_dir = osp.abspath(args.out_dir)
    os.makedirs(args.out_dir, exist_ok=True)
    tracker_name = args.tracker_name
    if args.stage in ('detect', 'all'):
        run_detect(args)
    if args.stage in ('track', 'all'):
        tracker_name = run_track(args)
    if args.stage == 'eval' or (args.stage == 'all' and args.eval):
        run_eval(args, tracker_name)


if __name__ == '__main__':
    main()
