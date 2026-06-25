#!/usr/bin/env python3
"""Run validation predict on a single-frame overfit checkpoint and save val_vis."""
from __future__ import annotations

import argparse
import os
import os.path as osp
import sys

import torch
from mmengine.config import Config
from mmengine.dataset import pseudo_collate
from mmengine.runner import load_checkpoint
from torch.utils.data import DataLoader

_AI4RS_ROOT = osp.abspath(osp.join(osp.dirname(__file__), '../../..'))
if _AI4RS_ROOT not in sys.path:
    sys.path.insert(0, _AI4RS_ROOT)

from mmrotate.registry import DATASETS, MODELS
from mmrotate.utils import register_all_modules
from projects.multispec_pair_rotated_rtdetr.multispec_pair_rotated_rtdetr.single_val_visualization_hook import (
    visualize_hsmot_single_pred_gt,
)
from projects.multispec_pair_rotated_rtdetr.tools.run_hsmot_single_overfit_acceptance import (
    _prepare_det_batch,
    _sync_single_dataset_cfg,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description='Visualize single-frame overfit checkpoint on val set')
    parser.add_argument(
        '--config',
        default='projects/multispec_pair_rotated_rtdetr/configs/'
        'o2_rtdetr_r18vd_overfit.py')
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--data-root', required=True)
    parser.add_argument(
        '--out-dir',
        default=None,
        help='Output directory; default: <work_dir>/val_vis/iter_<iter_tag>')
    parser.add_argument('--iter-tag', default='3000')
    parser.add_argument('--score-thr', type=float, default=0.35)
    parser.add_argument('--device', default='cuda:0')
    return parser.parse_args()


@torch.no_grad()
def main():
    args = parse_args()
    register_all_modules()
    os.chdir(_AI4RS_ROOT)

    cfg = Config.fromfile(args.config)
    data_root = osp.abspath(args.data_root)
    is_real_layout = osp.isdir(osp.join(data_root, 'train', 'npy2jpg'))
    _sync_single_dataset_cfg(cfg, data_root, is_real_layout)

    out_dir = args.out_dir
    if out_dir is None:
        work_dir = osp.dirname(osp.abspath(args.checkpoint))
        out_dir = osp.join(work_dir, 'val_vis', f'iter_{int(args.iter_tag):06d}')
    os.makedirs(out_dir, exist_ok=True)

    device = torch.device(
        args.device if torch.cuda.is_available() else 'cpu')
    preprocessor = MODELS.build(cfg.model.data_preprocessor)
    model = MODELS.build(cfg.model)
    load_checkpoint(model, args.checkpoint, map_location='cpu')
    model = model.to(device).eval()
    preprocessor = preprocessor.to(device)

    dataset = DATASETS.build(cfg.val_dataloader.dataset)
    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        collate_fn=pseudo_collate,
    )

    frame_idx = 0
    for batch in loader:
        inputs, batch_samples = _prepare_det_batch(
            batch, preprocessor, device, training=False)
        outputs = model.predict(inputs, batch_samples, rescale=False)
        for sample, raw_input in zip(outputs, batch['inputs']):
            meta = sample.metainfo
            seq = meta.get('seq_name', meta.get('video_id', 'seq'))
            frame_id = meta.get('frame_id', frame_idx + 1)
            frame_name = f'{seq}_{frame_id:06d}'
            save_path = osp.join(out_dir, f'{frame_idx:04d}_{frame_name}.jpg')
            meta_line = f'{seq} frame={frame_id} ckpt={osp.basename(args.checkpoint)}'
            visualize_hsmot_single_pred_gt(
                raw_input,
                sample.gt_instances,
                sample.pred_instances,
                score_thr=args.score_thr,
                save_path=save_path,
                meta_line=meta_line,
                img_meta=meta,
            )
            print(f'Wrote {save_path}')
            frame_idx += 1

    print(f'Done: {frame_idx} frames -> {out_dir}')


if __name__ == '__main__':
    main()
