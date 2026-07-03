#!/usr/bin/env python3
"""Visualize typed proposals and PairDN single-visible targets."""

from __future__ import annotations

import argparse
import json
import os.path as osp
import sys
from typing import Dict, List, Sequence, Tuple

import cv2
import numpy as np
import torch
from mmengine.config import Config
from mmengine.dataset import Compose
from mmengine.runner import load_checkpoint
from mmengine.utils import mkdir_or_exist
from mmrotate.registry import MODELS
from mmrotate.structures.bbox import rbox2qbox
from mmrotate.utils import register_all_modules

_AI4RS_ROOT = osp.abspath(osp.join(osp.dirname(__file__), '../../..'))
if _AI4RS_ROOT not in sys.path:
    sys.path.insert(0, _AI4RS_ROOT)

import projects.multispec_pair_rotated_rtdetr.multispec_pair_rotated_rtdetr  # noqa: E402,F401
from mmrotate.datasets.hsmot import load_hsmot_sequence_ann  # noqa: E402
from projects.multispec_pair_rotated_rtdetr.multispec_pair_rotated_rtdetr.multispec_pair_rotated_rtdetr import (  # noqa: E402,E501
    QUERY_TYPE_CURR_ONLY,
    QUERY_TYPE_PREV_ONLY,
    QUERY_TYPE_SURVIVAL,
)
from projects.multispec_pair_rotated_rtdetr.multispec_pair_rotated_rtdetr.pair_cdn_query_generator import (  # noqa: E402,E501
    PairCdnQueryGenerator,
)
from projects.multispec_pair_rotated_rtdetr.tools.run_pair_mot import (  # noqa: E402
    _frame_ids_from_images,
    _make_pair_info,
    _sequence_list,
)


TYPE_NAMES = {
    QUERY_TYPE_SURVIVAL: 'survival',
    QUERY_TYPE_CURR_ONLY: 'curr_only',
    QUERY_TYPE_PREV_ONLY: 'prev_only',
}
COLORS = {
    'survival': (0, 255, 255),
    'curr_only': (255, 128, 0),
    'prev_only': (255, 0, 255),
    'both_gt': (0, 220, 0),
    'new_gt': (255, 128, 0),
    'disappear_gt': (255, 0, 255),
    'invalid': (160, 160, 160),
}


def _to_preview(img: torch.Tensor) -> np.ndarray:
    arr = img.detach().cpu().numpy()
    if arr.ndim == 3 and arr.shape[0] >= 3:
        arr = np.transpose(arr[:3], (1, 2, 0))
    arr = np.clip(arr, 0, 255).astype(np.uint8)
    return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)


def _to_rbox_tensor(boxes) -> torch.Tensor:
    return boxes.tensor if hasattr(boxes, 'tensor') else boxes


def _rbox_to_poly(box: Sequence[float]) -> np.ndarray:
    tensor = torch.tensor(box, dtype=torch.float32).reshape(1, 5)
    return rbox2qbox(tensor).reshape(4, 2).cpu().numpy()


def _draw_box(img: np.ndarray, box: Sequence[float], color: Tuple[int, int, int],
              label: str, thickness: int = 1) -> None:
    poly = _rbox_to_poly(box)
    pts = np.round(poly).astype(np.int32).reshape(-1, 1, 2)
    cv2.polylines(img, [pts], True, color, thickness, cv2.LINE_AA)
    if label:
        center = np.round(poly.mean(axis=0)).astype(np.int32)
        x = int(np.clip(center[0], 0, img.shape[1] - 1))
        y = int(np.clip(center[1], 12, img.shape[0] - 1))
        cv2.putText(img, label, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.36,
                    color, 1, cv2.LINE_AA)


def _draw_invalid_marker(img: np.ndarray) -> None:
    center = (img.shape[1] // 2, img.shape[0] // 2)
    cv2.drawMarker(img, center, COLORS['invalid'], cv2.MARKER_CROSS, 18, 2,
                   cv2.LINE_AA)


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


def _normalized_to_pixels(refs: torch.Tensor, img_shape: Tuple[int, int],
                          angle_factor: float) -> torch.Tensor:
    img_h, img_w = img_shape
    factor = refs.new_tensor([img_w, img_h, img_w, img_h, angle_factor])
    return refs * factor


def _build_model_and_pipeline(config: str, checkpoint: str, device: str):
    register_all_modules()
    cfg = Config.fromfile(config)
    model = MODELS.build(cfg.model)
    load_checkpoint(model, checkpoint, map_location='cpu')
    torch_device = torch.device(device if torch.cuda.is_available() else 'cpu')
    model = model.to(torch_device).eval()
    preprocessor = MODELS.build(cfg.model.data_preprocessor).to(torch_device)
    pipeline = Compose(cfg.val_pipeline)
    return cfg, model, preprocessor, pipeline, torch_device


def _prepare_pair(preprocessor, pipeline, pair_info: dict, device: torch.device):
    packed = pipeline(pair_info)
    inputs = packed['inputs'].unsqueeze(0)
    data_sample = packed['data_samples']
    data = preprocessor(
        {'inputs': inputs, 'data_samples': [data_sample]}, training=False)
    return packed['inputs'], data['inputs'].to(device), data['data_samples']


def _encoder_memory(model, batch_inputs: torch.Tensor, data_samples: list):
    img_feats = model.extract_feat(batch_inputs)
    encoder_inputs, _decoder_inputs = model.pre_transformer(
        img_feats, data_samples)
    encoder_outputs = model.forward_encoder(**encoder_inputs)
    memory = encoder_outputs['memory']
    pair_batch, _ = model._split_pair_batch(img_feats[0].shape[0])
    return (
        memory[:pair_batch],
        memory[pair_batch:],
        encoder_outputs['memory_mask'],
        encoder_outputs['spatial_shapes'],
    )


def _gt_type_masks(gt) -> Dict[str, torch.Tensor]:
    valid_prev = gt.valid_prev.detach().cpu().bool()
    valid_curr = gt.valid_curr.detach().cpu().bool()
    return {
        'both': valid_prev & valid_curr,
        'new': (~valid_prev) & valid_curr,
        'disappear': valid_prev & (~valid_curr),
    }


def _draw_gt(prev: np.ndarray, curr: np.ndarray, gt, limit: int | None = None) -> Dict[str, int]:
    boxes_prev = _to_rbox_tensor(gt.bboxes_prev).detach().cpu().float()
    boxes_curr = _to_rbox_tensor(gt.bboxes_curr).detach().cpu().float()
    masks = _gt_type_masks(gt)
    counts = {k: int(v.sum()) for k, v in masks.items()}
    for name, mask in masks.items():
        color = COLORS[f'{name}_gt'] if name != 'both' else COLORS['both_gt']
        idxs = torch.nonzero(mask, as_tuple=False).flatten().tolist()
        if limit is not None:
            idxs = idxs[:limit]
        for local_rank, idx in enumerate(idxs):
            if name in ('both', 'disappear'):
                _draw_box(prev, boxes_prev[idx].tolist(), color,
                          f'g{name[0]}{local_rank}', 2)
            if name in ('both', 'new'):
                _draw_box(curr, boxes_curr[idx].tolist(), color,
                          f'g{name[0]}{local_rank}', 2)
    return counts


def _draw_typed_proposals(model, packed_inputs: torch.Tensor,
                          batch_inputs: torch.Tensor, data_samples: list,
                          out_path: str, draw_limit: int) -> dict:
    data_sample = data_samples[0]
    with torch.no_grad():
        memory_prev, memory_curr, memory_mask, spatial_shapes = _encoder_memory(
            model, batch_inputs, data_samples)
        (query, ref_prev, ref_curr, _dn_meta, _attn_mask, _enc_cls_prev,
         _enc_cls_curr, _enc_bbox_prev, _enc_bbox_curr) = (
             model._init_pair_decoder_queries(
                 memory_prev,
                 memory_curr,
                 memory_mask,
                 spatial_shapes,
                 batch_data_samples=data_samples,
             ))
    del query
    meta = data_samples[0].metainfo
    query_types = meta['pair_query_types'].detach().cpu().long()
    ref_prev_px = _normalized_to_pixels(
        ref_prev[0].detach().cpu(), meta['img_shape'], model.decoder.angle_factor)
    ref_curr_px = _normalized_to_pixels(
        ref_curr[0].detach().cpu(), meta['img_shape'], model.decoder.angle_factor)

    gt = data_sample.pair_gt_instances
    counts = {TYPE_NAMES[k]: int((query_types == k).sum())
              for k in TYPE_NAMES}
    selected_counts = {}
    for qtype, name in TYPE_NAMES.items():
        prev = _to_preview(packed_inputs[0])
        curr = _to_preview(packed_inputs[1])
        gt_counts = _draw_gt(prev, curr, gt, limit=60)
        idxs = torch.nonzero(query_types == qtype, as_tuple=False).flatten()
        idxs = idxs[:draw_limit].tolist()
        selected_counts[name] = len(idxs)
        color = COLORS[name]
        for rank, idx in enumerate(idxs):
            if name != 'curr_only':
                _draw_box(prev, ref_prev_px[idx].tolist(), color,
                          f'{rank}', 1)
            else:
                _draw_invalid_marker(prev)
            if name != 'prev_only':
                _draw_box(curr, ref_curr_px[idx].tolist(), color,
                          f'{rank}', 1)
            else:
                _draw_invalid_marker(curr)
        canvas = _side_by_side(
            prev, curr,
            f'Typed proposals: {name}',
            [
                f'query counts={counts}, drawn {name}={len(idxs)}',
                f'GT both/new/disappear={gt_counts}',
                'green: survival GT; orange: new/curr-only; magenta: disappear/prev-only; gray: null side',
            ])
        sub_path = out_path.replace('.jpg', f'_{name}.jpg')
        cv2.imwrite(sub_path, canvas)
    return {'query_counts': counts, 'drawn_counts': selected_counts}


def _build_generator(cfg: Config, model) -> PairCdnQueryGenerator:
    head_cfg = cfg.model['bbox_head']
    dn_cfg = cfg.model.get('pair_dn_cfg', None)
    if dn_cfg is None:
        raise RuntimeError(f'{cfg.filename} has no model.pair_dn_cfg')
    device = next(model.parameters()).device
    generator = PairCdnQueryGenerator(
        num_classes=head_cfg['num_classes'],
        embed_dims=model.decoder.embed_dims,
        num_matching_queries=cfg.model['num_queries'],
        angle_factor=model.decoder.angle_factor,
        **dn_cfg)
    return generator.to(device)


def _draw_dn_single_visible(cfg: Config, model, generator: PairCdnQueryGenerator,
                            packed_inputs: torch.Tensor, data_sample,
                            out_path: str, draw_limit: int) -> dict:
    generator.train()
    with torch.no_grad():
        _dn_query, dn_prev_unact, dn_curr_unact, _attn_mask, dn_meta = (
            generator([data_sample]))
    dn_prev = torch.sigmoid(dn_prev_unact[0]).detach().cpu()
    dn_curr = torch.sigmoid(dn_curr_unact[0]).detach().cpu()
    img_shape = data_sample.metainfo['img_shape']
    dn_prev_px = _normalized_to_pixels(dn_prev, img_shape, generator.angle_factor)
    dn_curr_px = _normalized_to_pixels(dn_curr, img_shape, generator.angle_factor)
    gt = data_sample.pair_gt_instances
    masks = _gt_type_masks(gt)
    max_targets = int(dn_meta['max_num_dn_targets'])
    num_groups = int(dn_meta['num_denoising_groups'])
    manifest = {
        'num_gt': int(len(gt.labels)),
        'gt_counts': {k: int(v.sum()) for k, v in masks.items()},
        'num_denoising_groups': num_groups,
        'max_num_dn_targets': max_targets,
        'num_denoising_queries': int(dn_meta['num_denoising_queries']),
        'files': [],
    }
    for mode in ('both', 'new', 'disappear'):
        target_idxs = torch.nonzero(masks[mode], as_tuple=False).flatten().tolist()
        if not target_idxs:
            continue
        for group_idx in range(min(2 * num_groups, 4)):
            prev = _to_preview(packed_inputs[0])
            curr = _to_preview(packed_inputs[1])
            _draw_gt(prev, curr, gt, limit=60)
            negative = bool(group_idx % 2)
            color = (255, 0, 255) if negative else (255, 255, 0)
            name = 'negative' if negative else 'positive'
            for local_rank, tgt_idx in enumerate(target_idxs[:draw_limit]):
                dn_idx = group_idx * max_targets + tgt_idx
                if mode in ('both', 'disappear'):
                    _draw_box(prev, dn_prev_px[dn_idx].tolist(), color,
                              f'd{local_rank}', 1)
                else:
                    _draw_invalid_marker(prev)
                if mode in ('both', 'new'):
                    _draw_box(curr, dn_curr_px[dn_idx].tolist(), color,
                              f'd{local_rank}', 1)
                else:
                    _draw_invalid_marker(curr)
            canvas = _side_by_side(
                prev, curr,
                f'PairDN {mode} targets: group {group_idx} ({name})',
                [
                    f'gt_counts={manifest["gt_counts"]}, drawn_targets={min(draw_limit, len(target_idxs))}',
                    f'max_targets={max_targets}, num_groups={num_groups}',
                    'green/orange/magenta: GT; cyan: positive DN; magenta: negative DN; gray cross: invalid side',
                ])
            sub_path = out_path.replace('.jpg', f'_{mode}_g{group_idx}_{name}.jpg')
            cv2.imwrite(sub_path, canvas)
            manifest['files'].append(sub_path)
    return manifest


def _sample_has_single_visible(pair_info: dict, pipeline: Compose) -> bool:
    packed = pipeline(pair_info)
    gt = packed['data_samples'].pair_gt_instances
    masks = _gt_type_masks(gt)
    return bool(masks['new'].any() or masks['disappear'].any())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--data-root',
                        default='/data/users/litianhao01/PairMmot/data/hsmot/test')
    parser.add_argument('--ann-file', default=None)
    parser.add_argument('--ann-subdir', default='mot')
    parser.add_argument('--img-subdir', default='npy2jpg')
    parser.add_argument('--img-format', default='3jpg')
    parser.add_argument('--out-dir', required=True)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--max-samples', type=int, default=6)
    parser.add_argument('--draw-proposals', type=int, default=40)
    parser.add_argument('--draw-dn-targets', type=int, default=40)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg, model, preprocessor, pipeline, device = _build_model_and_pipeline(
        args.config, args.checkpoint, args.device)
    generator = _build_generator(cfg, model)
    mkdir_or_exist(args.out_dir)
    seqs = _sequence_list(args.data_root, args.ann_file, args.ann_subdir)
    img_root = osp.join(args.data_root, args.img_subdir)
    ann_dir = osp.join(args.data_root, args.ann_subdir)
    manifests: List[dict] = []

    for seq_name in seqs:
        frame_anns = load_hsmot_sequence_ann(osp.join(ann_dir, f'{seq_name}.txt'))
        frame_ids = _frame_ids_from_images(
            osp.join(img_root, seq_name), args.img_format)
        for prev_id, curr_id in zip(frame_ids[:-1], frame_ids[1:]):
            pair_info = _make_pair_info(
                seq_name, img_root, args.img_format, frame_anns, prev_id,
                curr_id)
            if not _sample_has_single_visible(pair_info, pipeline):
                continue
            packed_inputs, batch_inputs, data_samples = _prepare_pair(
                preprocessor, pipeline, pair_info, device)
            gt = data_samples[0].pair_gt_instances
            masks = _gt_type_masks(gt)
            sample_dir = osp.join(args.out_dir, seq_name,
                                  f'{prev_id:06d}_{curr_id:06d}')
            mkdir_or_exist(sample_dir)
            print(f'[vis] {seq_name} {prev_id:06d}->{curr_id:06d} '
                  f'new={int(masks["new"].sum())} '
                  f'disappear={int(masks["disappear"].sum())}',
                  flush=True)
            prop_stats = _draw_typed_proposals(
                model, packed_inputs, batch_inputs, data_samples,
                osp.join(sample_dir, 'typed_proposals.jpg'),
                args.draw_proposals)
            dn_stats = _draw_dn_single_visible(
                cfg, model, generator, packed_inputs, data_samples[0],
                osp.join(sample_dir, 'pairdn.jpg'), args.draw_dn_targets)
            manifest = {
                'seq_name': seq_name,
                'prev_frame_id': int(prev_id),
                'curr_frame_id': int(curr_id),
                'proposal': prop_stats,
                'dn': dn_stats,
            }
            with open(osp.join(sample_dir, 'summary.json'), 'w',
                      encoding='utf-8') as f:
                json.dump(manifest, f, indent=2)
            manifests.append(manifest)
            if len(manifests) >= args.max_samples:
                break
        if len(manifests) >= args.max_samples:
            break

    with open(osp.join(args.out_dir, 'index.json'), 'w', encoding='utf-8') as f:
        json.dump(manifests, f, indent=2)
    readme = osp.join(args.out_dir, 'README.md')
    with open(readme, 'w', encoding='utf-8') as f:
        f.write('# Typed Proposal and PairDN Single-visible Visualization\n\n')
        f.write('- `typed_proposals_survival.jpg`: survival query proposals.\n')
        f.write('- `typed_proposals_curr_only.jpg`: curr-only query proposals; prev side is gray/null.\n')
        f.write('- `typed_proposals_prev_only.jpg`: prev-only query proposals; curr side is gray/null.\n')
        f.write('- `pairdn_*new*`: DN boxes for current-only GT.\n')
        f.write('- `pairdn_*disappear*`: DN boxes for previous-only GT.\n')
        f.write('- `summary.json`: counts and DN metadata per sample.\n')
    print(f'[done] wrote {len(manifests)} samples to {args.out_dir}', flush=True)


if __name__ == '__main__':
    main()
