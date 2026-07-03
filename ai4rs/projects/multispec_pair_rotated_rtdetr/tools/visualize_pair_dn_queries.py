#!/usr/bin/env python3
"""Visualize PairDN noisy reference boxes on HSMOT pair samples."""

from __future__ import annotations

import argparse
import json
import os.path as osp
import sys
from typing import Sequence, Tuple

import cv2
import numpy as np
import torch
from mmengine.config import Config
from mmengine.dataset import Compose
from mmengine.utils import mkdir_or_exist
from mmrotate.registry import MODELS
from mmrotate.structures.bbox import rbox2qbox
from mmrotate.utils import register_all_modules

_AI4RS_ROOT = osp.abspath(osp.join(osp.dirname(__file__), '../../..'))
if _AI4RS_ROOT not in sys.path:
    sys.path.insert(0, _AI4RS_ROOT)

import projects.multispec_pair_rotated_rtdetr.multispec_pair_rotated_rtdetr  # noqa: E402,F401
from mmrotate.datasets.hsmot import load_hsmot_sequence_ann  # noqa: E402
from projects.multispec_pair_rotated_rtdetr.multispec_pair_rotated_rtdetr.pair_cdn_query_generator import (  # noqa: E402,E501
    PairCdnQueryGenerator,
)
from projects.multispec_pair_rotated_rtdetr.tools.run_pair_mot import (  # noqa: E402
    _frame_ids_from_images,
    _make_pair_info,
    _sequence_list,
)


def _to_preview(img: torch.Tensor) -> np.ndarray:
    """Convert one packed 8-channel frame to a BGR preview."""
    arr = img.detach().cpu().numpy()
    if arr.ndim == 3 and arr.shape[0] >= 3:
        arr = np.transpose(arr[:3], (1, 2, 0))
    arr = np.clip(arr, 0, 255).astype(np.uint8)
    return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)


def _to_rbox_tensor(boxes) -> torch.Tensor:
    return boxes.tensor if hasattr(boxes, 'tensor') else boxes


def _normalized_to_pixels(refs: torch.Tensor, img_shape: Tuple[int, int],
                          angle_factor: float) -> torch.Tensor:
    img_h, img_w = img_shape
    factor = refs.new_tensor([img_w, img_h, img_w, img_h, angle_factor])
    return refs * factor


def _draw_rboxes(canvas: np.ndarray, boxes: torch.Tensor,
                 color: Tuple[int, int, int], prefix: str,
                 thickness: int = 1, limit: int | None = None) -> None:
    if boxes.numel() == 0:
        return
    if limit is not None:
        boxes = boxes[:limit]
    polys = rbox2qbox(boxes.detach().cpu().float()).reshape(-1, 4, 2).numpy()
    for idx, poly in enumerate(polys):
        pts = np.round(poly).astype(np.int32).reshape(-1, 1, 2)
        cv2.polylines(canvas, [pts], True, color, thickness, cv2.LINE_AA)
        center = np.round(poly.mean(axis=0)).astype(np.int32)
        x = int(np.clip(center[0], 0, canvas.shape[1] - 1))
        y = int(np.clip(center[1], 12, canvas.shape[0] - 1))
        cv2.putText(canvas, f'{prefix}{idx}', (x, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1, cv2.LINE_AA)


def _side_by_side(prev: np.ndarray, curr: np.ndarray, title: str,
                  lines: Sequence[str]) -> np.ndarray:
    h = max(prev.shape[0], curr.shape[0])
    canvas = np.full((h, prev.shape[1] + curr.shape[1], 3), 255, np.uint8)
    canvas[:prev.shape[0], :prev.shape[1]] = prev
    canvas[:curr.shape[0], prev.shape[1]:prev.shape[1] + curr.shape[1]] = curr
    cv2.line(canvas, (prev.shape[1], 0), (prev.shape[1], h), (255, 255, 255), 3)
    band_h = 34 + 20 * len(lines)
    band = np.full((band_h, canvas.shape[1], 3), 245, np.uint8)
    cv2.putText(band, title, (10, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.62,
                (30, 30, 30), 2, cv2.LINE_AA)
    for i, line in enumerate(lines):
        cv2.putText(band, line, (10, 49 + 20 * i),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, (60, 60, 60), 1,
                    cv2.LINE_AA)
    return np.vstack([band, canvas])


def _draw_view(inputs: torch.Tensor, gt_prev: torch.Tensor, gt_curr: torch.Tensor,
               dn_prev: torch.Tensor, dn_curr: torch.Tensor, title: str,
               lines: Sequence[str], dn_color: Tuple[int, int, int],
               out_path: str, limit: int | None) -> None:
    prev = _to_preview(inputs[0])
    curr = _to_preview(inputs[1])
    _draw_rboxes(prev, gt_prev, (0, 220, 0), 'g', thickness=2, limit=limit)
    _draw_rboxes(curr, gt_curr, (0, 220, 0), 'g', thickness=2, limit=limit)
    _draw_rboxes(prev, dn_prev, dn_color, 'd', thickness=1, limit=limit)
    _draw_rboxes(curr, dn_curr, dn_color, 'd', thickness=1, limit=limit)
    cv2.imwrite(out_path, _side_by_side(prev, curr, title, lines))


def _build_generator(cfg: Config) -> PairCdnQueryGenerator:
    model_cfg = cfg.model
    head_cfg = model_cfg['bbox_head']
    dn_cfg = model_cfg.get('pair_dn_cfg', None)
    if dn_cfg is None:
        raise RuntimeError(f'{cfg.filename} has no model.pair_dn_cfg')
    model = MODELS.build(model_cfg)
    return PairCdnQueryGenerator(
        num_classes=head_cfg['num_classes'],
        embed_dims=model.decoder.embed_dims,
        num_matching_queries=model_cfg['num_queries'],
        angle_factor=model.decoder.angle_factor,
        **dn_cfg)


def visualize_one(cfg: Config, pipeline: Compose, generator: PairCdnQueryGenerator,
                  pair_info: dict, out_dir: str, max_groups: int,
                  draw_limit: int | None) -> dict:
    packed = pipeline(pair_info)
    inputs = packed['inputs']
    data_sample = packed['data_samples']
    generator.train()
    with torch.no_grad():
        dn_query, dn_prev_unact, dn_curr_unact, _attn_mask, dn_meta = generator(
            [data_sample])
    del dn_query
    dn_prev = torch.sigmoid(dn_prev_unact[0])
    dn_curr = torch.sigmoid(dn_curr_unact[0])
    img_shape = data_sample.metainfo['img_shape']
    angle_factor = generator.angle_factor
    dn_prev = _normalized_to_pixels(dn_prev, img_shape, angle_factor)
    dn_curr = _normalized_to_pixels(dn_curr, img_shape, angle_factor)

    gt = data_sample.pair_gt_instances
    gt_prev = _to_rbox_tensor(gt.bboxes_prev).detach().cpu().float()
    gt_curr = _to_rbox_tensor(gt.bboxes_curr).detach().cpu().float()
    valid_prev = gt.valid_prev.detach().cpu().bool()
    valid_curr = gt.valid_curr.detach().cpu().bool()
    gt_prev = gt_prev[valid_prev]
    gt_curr = gt_curr[valid_curr]

    seq_dir = osp.join(
        out_dir, str(pair_info['seq_name']),
        f'{int(pair_info["frame_id_prev"]):06d}_{int(pair_info["frame_id"]):06d}')
    mkdir_or_exist(seq_dir)
    max_targets = int(dn_meta['max_num_dn_targets'])
    num_groups = int(dn_meta['num_denoising_groups'])
    num_draw_groups = min(max_groups, 2 * num_groups)

    manifest = dict(
        seq_name=pair_info['seq_name'],
        prev_frame_id=int(pair_info['frame_id_prev']),
        curr_frame_id=int(pair_info['frame_id']),
        img_shape=list(img_shape),
        num_gt=int(len(gt.labels)),
        num_denoising_groups=num_groups,
        max_num_dn_targets=max_targets,
        num_denoising_queries=int(dn_meta['num_denoising_queries']),
        files=[],
    )

    gt_path = osp.join(seq_dir, '00_gt.jpg')
    _draw_view(inputs, gt_prev, gt_curr, gt_prev.new_zeros((0, 5)),
               gt_curr.new_zeros((0, 5)), 'GT pair boxes',
               ['green: GT boxes after val resize'], (255, 255, 255),
               gt_path, draw_limit)
    manifest['files'].append(gt_path)

    for group_idx in range(num_draw_groups):
        start = group_idx * max_targets
        end = start + max_targets
        negative = bool(group_idx % 2)
        color = (255, 0, 255) if negative else (255, 255, 0)
        name = 'negative' if negative else 'positive'
        path = osp.join(seq_dir, f'{group_idx + 1:02d}_dn_group{group_idx:02d}_{name}.jpg')
        _draw_view(
            inputs, gt_prev, gt_curr, dn_prev[start:end], dn_curr[start:end],
            f'DN group {group_idx:02d} ({name})',
            [
                'green: GT; cyan: positive DN; magenta: negative DN',
                f'max_targets={max_targets}, group={group_idx}/{2 * num_groups - 1}',
            ],
            color, path, draw_limit)
        manifest['files'].append(path)

    if num_draw_groups >= 2:
        prev = _to_preview(inputs[0])
        curr = _to_preview(inputs[1])
        _draw_rboxes(prev, gt_prev, (0, 220, 0), 'g', thickness=2,
                     limit=draw_limit)
        _draw_rboxes(curr, gt_curr, (0, 220, 0), 'g', thickness=2,
                     limit=draw_limit)
        _draw_rboxes(prev, dn_prev[:max_targets], (255, 255, 0), 'p',
                     thickness=1, limit=draw_limit)
        _draw_rboxes(curr, dn_curr[:max_targets], (255, 255, 0), 'p',
                     thickness=1, limit=draw_limit)
        _draw_rboxes(prev, dn_prev[max_targets:2 * max_targets],
                     (255, 0, 255), 'n', thickness=1, limit=draw_limit)
        _draw_rboxes(curr, dn_curr[max_targets:2 * max_targets],
                     (255, 0, 255), 'n', thickness=1, limit=draw_limit)
        path = osp.join(seq_dir, 'overlay_first_pos_neg.jpg')
        cv2.imwrite(path, _side_by_side(
            prev, curr, 'First positive/negative DN groups',
            ['green: GT; cyan: positive DN group 0; magenta: negative DN group 1']))
        manifest['files'].append(path)

    json_path = osp.join(seq_dir, 'summary.json')
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(manifest, f, indent=2)
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--config',
        default='projects/multispec_pair_rotated_rtdetr/configs/'
        'o2_pair_rtdetr_r18vd_2xb4_72e_hsmot_pairdn_gap1train_bothvis_dualcls_nopres_pairdn.py')
    parser.add_argument(
        '--data-root',
        default='/data/users/litianhao01/PairMmot/data/hsmot/test')
    parser.add_argument('--ann-file', default=None)
    parser.add_argument('--ann-subdir', default='mot')
    parser.add_argument('--img-subdir', default='npy2jpg')
    parser.add_argument('--img-format', default='3jpg')
    parser.add_argument(
        '--out-dir',
        default='/data/users/litianhao01/PairMmot/workdir/_analysis/'
        'pairdn_query_vis_20260702')
    parser.add_argument('--max-seqs', type=int, default=3)
    parser.add_argument('--max-groups', type=int, default=4)
    parser.add_argument('--draw-limit', type=int, default=40)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    register_all_modules()
    cfg = Config.fromfile(args.config)
    pipeline = Compose(cfg.val_pipeline)
    generator = _build_generator(cfg)
    mkdir_or_exist(args.out_dir)
    seqs = _sequence_list(args.data_root, args.ann_file, args.ann_subdir)
    if args.max_seqs > 0:
        seqs = seqs[:args.max_seqs]
    img_root = osp.join(args.data_root, args.img_subdir)
    ann_dir = osp.join(args.data_root, args.ann_subdir)
    manifests = []
    for seq_name in seqs:
        frame_anns = load_hsmot_sequence_ann(osp.join(ann_dir, f'{seq_name}.txt'))
        frame_ids = _frame_ids_from_images(
            osp.join(img_root, seq_name), args.img_format)
        if len(frame_ids) < 2:
            continue
        pair_info = _make_pair_info(
            seq_name, img_root, args.img_format, frame_anns,
            frame_ids[0], frame_ids[1])
        print(f'[vis] {seq_name} {frame_ids[0]:06d}->{frame_ids[1]:06d}', flush=True)
        manifests.append(visualize_one(
            cfg, pipeline, generator, pair_info, args.out_dir, args.max_groups,
            args.draw_limit if args.draw_limit > 0 else None))

    index_path = osp.join(args.out_dir, 'index.json')
    with open(index_path, 'w', encoding='utf-8') as f:
        json.dump(manifests, f, indent=2)
    readme_path = osp.join(args.out_dir, 'README.md')
    with open(readme_path, 'w', encoding='utf-8') as f:
        f.write('# PairDN Query Visualization\n\n')
        f.write('- `00_gt.jpg`: GT pair boxes after val resize.\n')
        f.write('- `*_positive.jpg`: positive DN group, small box noise.\n')
        f.write('- `*_negative.jpg`: negative DN group, larger contrastive noise.\n')
        f.write('- `overlay_first_pos_neg.jpg`: GT + first positive/negative group.\n\n')
        f.write('Colors: GT green, positive DN cyan, negative DN magenta.\n')
    print(f'[done] wrote {len(manifests)} samples to {args.out_dir}', flush=True)


if __name__ == '__main__':
    main()
