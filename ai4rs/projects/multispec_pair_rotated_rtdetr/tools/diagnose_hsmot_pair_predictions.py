#!/usr/bin/env python3
"""Inspect pair overfit predictions against GT pairs."""
from __future__ import annotations

import argparse
import os
import os.path as osp
import sys

import torch
from mmengine.config import Config
from mmengine.structures import InstanceData
from mmrotate.registry import DATASETS, MODELS
from torch.utils.data import DataLoader

_AI4RS_ROOT = osp.abspath(osp.join(osp.dirname(__file__), '../../..'))
if _AI4RS_ROOT not in sys.path:
    sys.path.insert(0, _AI4RS_ROOT)

import projects.multispec_pair_rotated_rtdetr.multispec_pair_rotated_rtdetr  # noqa: F401,E402
from projects.multispec_pair_rotated_rtdetr.tools.run_hsmot_pair_overfit_acceptance import (  # noqa: E501
    _build_preprocessor,
    _pair_loss_sum,
    _prepare_pair_batch,
    _rbox_iou,
    _sync_pair_dataset_cfg,
    _to_rbox_tensor,
    collate_pair_batch,
)
from mmrotate.utils import register_all_modules


def _mean_valid_iou(pred, gt, gi: int, qi: int) -> float:
    ious = []
    if bool(gt.valid_prev[gi].item()):
        ious.append(_rbox_iou(pred.bboxes_prev[qi].cpu(), gt.bboxes_prev[gi].cpu()))
    if bool(gt.valid_curr[gi].item()):
        ious.append(_rbox_iou(pred.bboxes_curr[qi].cpu(), gt.bboxes_curr[gi].cpu()))
    return sum(ious) / len(ious) if ious else 0.0


def _greedy_unique(
    gt: InstanceData,
    pred: InstanceData,
    *,
    score_thr: float | None,
) -> dict:
    candidates = []
    num_gt = len(gt.labels)
    has_candidate = [False] * num_gt
    for gi in range(num_gt):
        label = int(gt.labels[gi].item())
        for qi in range(len(pred.scores)):
            if int(pred.labels[qi].item()) != label:
                continue
            if score_thr is not None and float(pred.scores[qi].item()) < score_thr:
                continue
            has_candidate[gi] = True
            candidates.append((_mean_valid_iou(pred, gt, gi, qi), gi, qi))

    gt_to_q = {}
    used_gt = set()
    used_q = set()
    for mean_iou, gi, qi in sorted(candidates, reverse=True):
        if gi in used_gt or qi in used_q:
            continue
        used_gt.add(gi)
        used_q.add(qi)
        gt_to_q[gi] = (qi, mean_iou)

    return {
        'matched': len(gt_to_q),
        'total': num_gt,
        'no_candidate': sum(not item for item in has_candidate),
        'unmatched_with_candidate': num_gt - len(gt_to_q) -
        sum(not item for item in has_candidate),
        'ious': [item[1] for item in gt_to_q.values()],
    }


def _summarize(name: str, values: list[float]) -> None:
    if not values:
        print(f'{name}: empty')
        return
    tensor = torch.tensor(values)
    print(
        f'{name}: mean={tensor.mean().item():.4f} '
        f'min={tensor.min().item():.4f} p10={tensor.quantile(0.1).item():.4f} '
        f'p50={tensor.quantile(0.5).item():.4f} '
        f'>=.5={(tensor >= 0.5).float().mean().item():.4f}')


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--config',
        default='projects/multispec_pair_rotated_rtdetr/configs/'
        'o2_pair_rtdetr_r18vd_overfit_sameframe.py')
    parser.add_argument(
        '--data-root',
        default='/data/users/litianhao01/PairMmot/tmp/'
        'hsmot_pair_sameframe_alignedtopk_accept/data')
    parser.add_argument(
        '--checkpoint',
        default='/data/users/litianhao01/PairMmot/tmp/'
        'hsmot_pair_sameframe_alignedtopk_accept/work_dir/iter_3000.pth')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--score-thr', type=float, default=0.35)
    parser.add_argument('--max-samples', type=int, default=0)
    args = parser.parse_args()

    register_all_modules()
    cfg = Config.fromfile(args.config)
    _sync_pair_dataset_cfg(cfg, osp.abspath(args.data_root), is_real_layout=True)

    preprocessor = _build_preprocessor(cfg)
    model = MODELS.build(cfg.model)
    checkpoint = torch.load(args.checkpoint, map_location='cpu')
    state = checkpoint.get('state_dict', checkpoint)
    missing, unexpected = model.load_state_dict(state, strict=False)
    print(f'load_state missing={len(missing)} unexpected={len(unexpected)}')

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    preprocessor = preprocessor.to(device)
    model.eval()

    dataset = DATASETS.build(cfg.train_dataloader.dataset)
    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_pair_batch)

    best_same_all = []
    top_score_same = []
    best_scores = []
    greedy_all = {'matched': 0, 'total': 0, 'no_candidate': 0,
                  'unmatched_with_candidate': 0, 'ious': []}
    greedy_thr = {'matched': 0, 'total': 0, 'no_candidate': 0,
                  'unmatched_with_candidate': 0, 'ious': []}
    score_ranges = []
    losses = []
    examples = []

    for sample_idx, batch in enumerate(loader):
        if args.max_samples and sample_idx >= args.max_samples:
            break
        inputs, samples = _prepare_pair_batch(
            batch, preprocessor, device, training=False)
        with torch.no_grad():
            model.train()
            train_inputs, train_samples = _prepare_pair_batch(
                batch, preprocessor, device, training=True)
            losses.append(_pair_loss_sum(model.loss(train_inputs, train_samples)))
            model.eval()
            outputs = model.predict(inputs, samples, rescale=False)

        sample = outputs[0]
        gt = sample.pair_gt_instances
        pred = sample.pred_pair_instances
        gt.bboxes_prev = _to_rbox_tensor(gt.bboxes_prev).cpu()
        gt.bboxes_curr = _to_rbox_tensor(gt.bboxes_curr).cpu()
        gt.labels = gt.labels.cpu()
        gt.valid_prev = gt.valid_prev.cpu().bool()
        gt.valid_curr = gt.valid_curr.cpu().bool()
        pred.bboxes_prev = pred.bboxes_prev.cpu()
        pred.bboxes_curr = pred.bboxes_curr.cpu()
        pred.labels = pred.labels.cpu()
        pred.scores = pred.scores.cpu()
        score_ranges.append((float(pred.scores.min()), float(pred.scores.max())))

        for target, source in (
                (_greedy_unique(gt, pred, score_thr=None), greedy_all),
                (_greedy_unique(gt, pred, score_thr=args.score_thr), greedy_thr)):
            for key in ('matched', 'total', 'no_candidate',
                        'unmatched_with_candidate'):
                source[key] += target[key]
            source['ious'].extend(target['ious'])

        for gi in range(len(gt.labels)):
            same = pred.labels == int(gt.labels[gi].item())
            if not bool(same.any()):
                best_same_all.append(0.0)
                top_score_same.append(0.0)
                best_scores.append(0.0)
                continue

            same_idx = torch.nonzero(same, as_tuple=False).flatten()
            same_scores = pred.scores[same_idx]
            top_q = int(same_idx[int(same_scores.argmax())].item())
            top_iou = _mean_valid_iou(pred, gt, gi, top_q)
            top_score_same.append(top_iou)

            best_iou = -1.0
            best_q = -1
            for qi in same_idx.tolist():
                iou = _mean_valid_iou(pred, gt, gi, qi)
                if iou > best_iou:
                    best_iou = iou
                    best_q = qi
            best_same_all.append(best_iou)
            best_scores.append(float(pred.scores[best_q].item()))
            if (best_iou < 0.5 or float(pred.scores[best_q].item()) <
                    args.score_thr) and len(examples) < 16:
                examples.append({
                    'sample': sample_idx,
                    'gt': gi,
                    'label': int(gt.labels[gi].item()),
                    'best_q': best_q,
                    'best_iou': round(best_iou, 4),
                    'best_score': round(float(pred.scores[best_q].item()), 4),
                    'top_score_iou': round(top_iou, 4),
                    'top_score': round(float(pred.scores[top_q].item()), 4),
                })

    print(f'samples={len(losses)} mean_loss={sum(losses) / max(len(losses), 1):.4f}')
    print('score range min/max:',
          min(item[0] for item in score_ranges),
          max(item[1] for item in score_ranges))
    _summarize('best same-class IoU, no score threshold', best_same_all)
    _summarize('top-score same-class IoU', top_score_same)
    _summarize('best-match score for same-class IoU', best_scores)
    for name, stats in (('greedy unique, no score threshold', greedy_all),
                        (f'greedy unique, score>={args.score_thr}', greedy_thr)):
        ratio = stats['matched'] / max(stats['total'], 1)
        print(
            f'{name}: matched={stats["matched"]}/{stats["total"]} '
            f'ratio={ratio:.4f} no_candidate={stats["no_candidate"]} '
            f'unmatched_with_candidate={stats["unmatched_with_candidate"]}')
        _summarize(f'{name} matched IoU', stats['ious'])
    print('examples:')
    for item in examples:
        print(item)


if __name__ == '__main__':
    main()
