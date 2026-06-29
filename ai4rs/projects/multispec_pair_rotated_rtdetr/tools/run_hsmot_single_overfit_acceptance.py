#!/usr/bin/env python3
"""Single-frame HSMOT overfit acceptance: train on fixed mini frames and verify.

Checks:
  1. Final detection loss sum below threshold after thousands of iterations.
  2. Each GT instance has exactly one high-score query.
  3. Predicted boxes IoU > threshold vs GT.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import os.path as osp
import sys
import time
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import torch
from mmengine.config import Config
from mmengine.dist import barrier, get_dist_info
from mmengine.runner import Runner
from mmengine.structures import InstanceData
from torch.utils.data import DataLoader

_AI4RS_ROOT = osp.abspath(osp.join(osp.dirname(__file__), '../../..'))
_PAIRMMOT_ROOT = osp.abspath(osp.join(_AI4RS_ROOT, '..'))
if _AI4RS_ROOT not in sys.path:
    sys.path.insert(0, _AI4RS_ROOT)

from mmengine.dataset import pseudo_collate
from mmrotate.registry import DATASETS, METRICS, MODELS
from mmrotate.structures.bbox import qbox2rbox
from mmrotate.utils import register_all_modules

import projects.multispec_pair_rotated_rtdetr.multispec_pair_rotated_rtdetr  # noqa: F401

from projects.multispec_pair_rotated_rtdetr.tools.create_hsmot_single_overfit_data import (
    create_hsmot_single_overfit_data,
)
from projects.multispec_pair_rotated_rtdetr.tools.create_hsmot_single_overfit_from_real import (
    create_hsmot_single_overfit_from_real,
)


@dataclass
class AcceptanceReport:
    passed: bool = True
    messages: List[str] = field(default_factory=list)
    metrics: Dict[str, float] = field(default_factory=dict)

    def fail(self, msg: str) -> None:
        self.passed = False
        self.messages.append(f'FAIL: {msg}')

    def ok(self, msg: str) -> None:
        self.messages.append(f'OK: {msg}')


def _latest_checkpoint(work_dir: str) -> str:
    work_dir = osp.abspath(work_dir)
    last_file = osp.join(work_dir, 'last_checkpoint')
    if osp.isfile(last_file):
        with open(last_file, encoding='utf-8') as f:
            ckpt = f.read().strip()
        if ckpt:
            if osp.isfile(ckpt):
                return osp.abspath(ckpt)
            joined = osp.join(work_dir, ckpt)
            if osp.isfile(joined):
                return osp.abspath(joined)

    ckpts = glob.glob(osp.join(work_dir, '*.pth'))
    if not ckpts:
        raise FileNotFoundError(f'No training checkpoint in {work_dir}')
    return max(ckpts, key=osp.getmtime)


def _finalize_dist_for_eval(launcher: str) -> bool:
    if not _is_dist_launched(launcher):
        return True
    if not torch.distributed.is_initialized():
        return _local_rank() == 0
    barrier()
    rank, _ = get_dist_info()
    torch.distributed.destroy_process_group()
    return rank == 0


def _overfit_dataset_exists(data_root: str) -> bool:
    train_root = osp.join(data_root, 'train')
    imageset = osp.join(train_root, 'ImageSets', 'train.txt')
    mot_dir = osp.join(train_root, 'mot')
    if not osp.isfile(imageset) or osp.getsize(imageset) == 0:
        return False
    if not osp.isdir(mot_dir):
        return False
    mot_files = [name for name in os.listdir(mot_dir) if name.endswith('.txt')]
    if not mot_files:
        return False
    for img_subdir in ('npy2jpg', 'npy'):
        img_root = osp.join(train_root, img_subdir)
        if not osp.isdir(img_root):
            continue
        for seq_name in os.listdir(img_root):
            seq_dir = osp.join(img_root, seq_name)
            if osp.isdir(seq_dir) and os.listdir(seq_dir):
                return True
    return False


def _detect_real_layout(data_root: str) -> bool:
    return osp.isdir(osp.join(data_root, 'train', 'npy2jpg'))


def _local_rank() -> int:
    return int(os.environ.get('LOCAL_RANK', '0'))


def _is_dist_launched(launcher: str) -> bool:
    return launcher != 'none'


def _wait_for_overfit_dataset(data_root: str, timeout_sec: float = 600.0) -> None:
    deadline = time.time() + timeout_sec
    while time.time() < deadline:
        if _overfit_dataset_exists(data_root):
            return
        time.sleep(0.5)
    raise TimeoutError(
        f'Timed out waiting for overfit dataset at {data_root}')


def _ensure_overfit_dataset(
    data_root: str,
    *,
    from_real: bool,
    src_root: str,
    ann_file: str,
    num_frames: int,
    seed: int,
    reuse_data: bool,
) -> bool:
    if reuse_data and _overfit_dataset_exists(data_root):
        is_real = _detect_real_layout(data_root)
        print(f'[0/3] Reuse existing dataset at {data_root} '
              f'(layout={"3jpg" if is_real else "npy"})')
        return is_real

    if from_real:
        create_hsmot_single_overfit_from_real(
            dst_root=data_root,
            src_root=osp.abspath(src_root),
            ann_file=ann_file,
            num_frames=num_frames,
            seed=seed,
        )
        return True

    create_hsmot_single_overfit_data(data_root, num_frames=num_frames)
    return False


def _sync_single_dataset_cfg(cfg: Config, data_root: str,
                             is_real_layout: bool) -> None:
    train_root = osp.join(data_root, 'train')
    img_loader = dict(
        type='LoadMultichannelImageFrom3JPG' if is_real_layout
        else 'LoadMultichannelImageFromNpy',
        backend_args=None)
    for key in ('train_dataloader', 'val_dataloader', 'test_dataloader'):
        loader = cfg.get(key)
        if loader is None or loader.get('dataset') is None:
            continue
        ds = loader.dataset
        ds.data_root = train_root
        if is_real_layout:
            ds.img_format = '3jpg'
            ds.data_prefix = dict(img_path='npy2jpg')
        pipeline = ds.get('pipeline')
        if pipeline:
            pipeline[0] = img_loader


def _build_preprocessor(cfg: Config):
    return MODELS.build(cfg.model.data_preprocessor)


def _prepare_det_batch(
    batch: dict,
    preprocessor,
    device: torch.device,
    training: bool = False,
) -> Tuple[torch.Tensor, list]:
    from mmdet.structures import DetDataSample

    data_samples = batch['data_samples']
    if not isinstance(data_samples, list):
        data_samples = [data_samples]

    samples = []
    for ds in data_samples:
        sample = DetDataSample()
        sample.gt_instances = ds.gt_instances
        sample.set_metainfo(ds.metainfo)
        samples.append(sample)

    preprocessed = preprocessor(
        {'inputs': batch['inputs'], 'data_samples': samples},
        training=training)
    inputs = preprocessed['inputs'].to(device)
    return inputs, preprocessed['data_samples']


def _det_loss_sum(losses: Dict[str, torch.Tensor]) -> float:
    total = 0.0
    for key, val in losses.items():
        if 'loss' in key and isinstance(val, torch.Tensor):
            total += float(val.detach().cpu())
    return total


@torch.no_grad()
def evaluate_det_ap(model, dataloader: DataLoader, preprocessor,
                    device: torch.device, dataset) -> Dict[str, float]:
    """Evaluate the final checkpoint with the standard rotated-box AP metric."""
    metric = METRICS.build(dict(type='HSMOTDetMetric', eval_mode='area'))
    metric.dataset_meta = dataset.metainfo
    model.eval()
    for batch in dataloader:
        inputs, batch_samples = _prepare_det_batch(
            batch, preprocessor, device, training=False)
        outputs = model.predict(inputs, batch_samples, rescale=False)
        # DOTAMetric predates BaseDataElement-based test loops and expects
        # mapping-style samples. The overfit dataset has no ignored boxes,
        # while DOTAMetric expects that field to be present.
        metric_samples = []
        for output in outputs:
            sample = output.to_dict()
            if 'ignored_instances' not in sample:
                boxes = sample['gt_instances']['bboxes']
                box_dim = boxes.tensor.size(-1) if hasattr(boxes, 'tensor') else boxes.size(-1)
                sample['ignored_instances'] = dict(
                    bboxes=torch.empty((0, box_dim)),
                    labels=torch.empty((0,), dtype=torch.long))
            metric_samples.append(sample)
        metric.process({}, metric_samples)
    raw = metric.compute_metrics(metric.results)
    return dict(
        AP50=float(raw.get('AP50', 0.0)),
        AP75=float(raw.get('AP75', 0.0)),
        mAP50_95=float(raw.get('mAP', 0.0)))


def _to_rbox_tensor(boxes) -> torch.Tensor:
    if hasattr(boxes, 'tensor'):
        t = boxes.tensor
    else:
        t = boxes
    if t.size(-1) == 8:
        return qbox2rbox(t)
    return t


def _rbox_iou(a: torch.Tensor, b: torch.Tensor) -> float:
    from mmrotate.structures.bbox import rbbox_overlaps
    if a.numel() == 0 or b.numel() == 0:
        return 0.0
    return float(rbbox_overlaps(
        a.unsqueeze(0), b.unsqueeze(0), is_aligned=True)[0].item())


@torch.no_grad()
def evaluate_det_predictions(
    model,
    dataloader: DataLoader,
    preprocessor,
    device: torch.device,
    score_thr: float = 0.35,
    iou_thr: float = 0.5,
) -> AcceptanceReport:
    """Run predict on training frames and verify matching quality."""
    report = AcceptanceReport()
    model.eval()

    total_gt = 0
    matched_queries = 0
    iou_sum = 0.0
    iou_count = 0
    duplicate_match = 0

    for batch in dataloader:
        inputs, batch_samples = _prepare_det_batch(
            batch, preprocessor, device, training=False)
        outputs = model.predict(inputs, batch_samples, rescale=False)
        for sample in outputs:
            gt: InstanceData = sample.gt_instances
            pred = sample.pred_instances
            num_gt = len(gt.labels)
            total_gt += num_gt

            gt_labels = gt.labels.cpu()
            gt_boxes = _to_rbox_tensor(gt.bboxes).cpu()

            pred_scores = pred.scores.cpu()
            pred_labels = pred.labels.cpu()
            pred_boxes = _to_rbox_tensor(pred.bboxes).cpu()

            candidates = []
            has_score_candidate = [False] * num_gt
            for gi in range(num_gt):
                label = int(gt_labels[gi].item())
                for qi in range(len(pred_scores)):
                    if int(pred_labels[qi].item()) != label:
                        continue
                    if float(pred_scores[qi].item()) < score_thr:
                        continue
                    has_score_candidate[gi] = True
                    iou = _rbox_iou(pred_boxes[qi], gt_boxes[gi])
                    candidates.append((iou, gi, qi))

            gt_to_match = {}
            used_gt = set()
            used_queries = set()
            for iou, gi, qi in sorted(candidates, reverse=True):
                if gi in used_gt or qi in used_queries:
                    continue
                used_gt.add(gi)
                used_queries.add(qi)
                gt_to_match[gi] = (qi, iou)

            duplicate_match += len(used_queries) - len(set(used_queries))
            for gi in range(num_gt):
                if gi not in gt_to_match:
                    if not has_score_candidate[gi]:
                        report.fail(
                            f'GT#{gi} label={int(gt_labels[gi].item())} '
                            f'has no query above score_thr={score_thr}')
                    else:
                        report.fail(
                            f'GT#{gi} has no unique high-score query match')
                    continue

                best_q, iou = gt_to_match[gi]
                matched_queries += 1
                iou_sum += iou
                iou_count += 1
                if iou < iou_thr:
                    report.fail(f'GT#{gi} IoU={iou:.3f} < {iou_thr}')

    if total_gt == 0:
        report.fail('No GT instances in evaluation set')
    else:
        match_ratio = matched_queries / total_gt
        report.metrics['gt_instances'] = float(total_gt)
        report.metrics['matched_queries'] = float(matched_queries)
        report.metrics['match_ratio'] = match_ratio
        if match_ratio < 1.0:
            report.fail(
                f'Only {matched_queries}/{total_gt} GT instances matched '
                f'(score>={score_thr})')
        else:
            report.ok(
                f'Each GT instance has one high-score query '
                f'({matched_queries}/{total_gt})')

        if duplicate_match > 0:
            report.fail(f'{duplicate_match} duplicate query assignments')

        if iou_count > 0:
            mean_iou = iou_sum / iou_count
            report.metrics['mean_iou'] = mean_iou
            report.ok(f'mean IoU={mean_iou:.3f} ({iou_count} boxes)')

    return report


def run_training(cfg: Config, launcher: str = 'none') -> Runner:
    register_all_modules()
    os.chdir(_AI4RS_ROOT)
    cfg.launcher = launcher
    runner = Runner.from_cfg(cfg)
    runner.train()
    return runner


def collate_det_batch(batch):
    return pseudo_collate(batch)


def run_acceptance(
    config_path: str,
    data_root: str,
    work_dir: str,
    max_iters: int,
    num_frames: int,
    min_ap50: float,
    min_map50_95: float,
    skip_train: bool = False,
    device: str = 'cuda:0',
    launcher: str = 'none',
    from_real: bool = False,
    src_root: str = '../data/hsmot/train',
    ann_file: str = '../data/hsmot/train_half.txt',
    seed: int = 42,
    reuse_data: bool = True,
    val_interval: int = 500,
) -> AcceptanceReport:
    register_all_modules()
    os.chdir(_AI4RS_ROOT)

    local_rank = _local_rank()
    if local_rank == 0:
        is_real_layout = _ensure_overfit_dataset(
            data_root,
            from_real=from_real,
            src_root=src_root,
            ann_file=ann_file,
            num_frames=num_frames,
            seed=seed,
            reuse_data=reuse_data,
        )
    else:
        _wait_for_overfit_dataset(data_root)
        is_real_layout = _detect_real_layout(data_root)

    cfg = Config.fromfile(config_path)
    cfg.work_dir = work_dir
    cfg.train_cfg.max_iters = max_iters
    cfg.train_cfg.val_interval = val_interval
    _sync_single_dataset_cfg(cfg, data_root, is_real_layout)
    cfg.default_hooks.logger.interval = min(10, max_iters // 20)

    if not skip_train:
        if local_rank == 0:
            print(f'[1/3] Training {max_iters} iters ...')
        run_training(cfg, launcher=launcher)
    elif local_rank == 0:
        print('[1/3] Skip training (--skip-train)')

    if not _finalize_dist_for_eval(launcher):
        return AcceptanceReport(passed=True)

    ckpt = _latest_checkpoint(work_dir)
    print(f'[2/3] Evaluating checkpoint: {ckpt}')

    preprocessor = _build_preprocessor(cfg)
    model = MODELS.build(cfg.model)
    checkpoint = torch.load(ckpt, map_location='cpu')
    state = checkpoint.get('state_dict', checkpoint)
    model.load_state_dict(state, strict=False)
    dev = torch.device(device if torch.cuda.is_available() else 'cpu')
    model = model.to(dev)
    preprocessor = preprocessor.to(dev)

    dataset = DATASETS.build(cfg.train_dataloader.dataset)
    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_det_batch,
    )

    report = AcceptanceReport()
    ap_metrics = evaluate_det_ap(model, loader, preprocessor, dev, dataset)
    report.metrics.update(ap_metrics)
    if ap_metrics['AP50'] < min_ap50:
        report.fail(f"AP50 {ap_metrics['AP50']:.4f} < {min_ap50:.4f}")
    else:
        report.ok(f"AP50 {ap_metrics['AP50']:.4f} >= {min_ap50:.4f}")
    if ap_metrics['mAP50_95'] < min_map50_95:
        report.fail(
            f"mAP50:95 {ap_metrics['mAP50_95']:.4f} < {min_map50_95:.4f}")
    else:
        report.ok(
            f"mAP50:95 {ap_metrics['mAP50_95']:.4f} >= {min_map50_95:.4f}")

    print('[3/3] Acceptance summary')
    for msg in report.messages:
        print(' ', msg)
    print('RESULT:', 'PASSED' if report.passed else 'FAILED')
    print('METRICS:', json.dumps(report.metrics, indent=2))
    return report


def parse_args():
    parser = argparse.ArgumentParser(
        description='HSMOT single-frame overfit acceptance script')
    parser.add_argument(
        '--config',
        default='projects/multispec_pair_rotated_rtdetr/configs/'
        'o2_rtdetr_r18vd_overfit.py')
    parser.add_argument(
        '--data-root',
        default='data/HSMOT_single_overfit')
    parser.add_argument(
        '--work-dir',
        default='work_dirs/o2_rtdetr_r18vd_overfit_accept')
    parser.add_argument('--max-iters', type=int, default=3000)
    parser.add_argument(
        '--val-interval',
        type=int,
        default=500,
        help='Run validation every N iters')
    parser.add_argument(
        '--num-frames',
        type=int,
        default=10,
        help='Frames in one clip: synthetic uses this directly; '
        'real (--from-real) extracts one contiguous clip')
    data_group = parser.add_mutually_exclusive_group()
    data_group.add_argument(
        '--from-real',
        dest='from_real',
        action='store_true',
        default=True,
        help='Extract one contiguous clip from real HSMOT (default)')
    data_group.add_argument(
        '--synthetic',
        dest='from_real',
        action='store_false',
        help='Create the small synthetic overfit dataset')
    parser.add_argument(
        '--src-root',
        default='../data/hsmot/train',
        help='Real HSMOT train root (with --from-real)')
    parser.add_argument(
        '--ann-file',
        default='../data/hsmot/train_half.txt',
        help='Sequence split file for real HSMOT')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--min-ap50', type=float, default=0.90,
                        help='Relaxed AP50 acceptance for tiny HSMOT targets')
    parser.add_argument('--min-map50-95', type=float, default=0.40,
                        help='Relaxed rotated mAP50:95 acceptance threshold')
    parser.add_argument('--skip-train', action='store_true')
    parser.add_argument(
        '--force-recreate-data',
        action='store_true',
        help='Recreate dataset even when data-root already contains a split')
    parser.add_argument(
        '--tmpdir',
        action='store_true',
        help='Use PairMmot/tmp/hsmot_single_overfit_accept')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='Job launcher. Use ``pytorch`` with torchrun for multi-GPU training.')
    parser.add_argument('--local-rank', '--local_rank', type=int, default=0)
    return parser.parse_args()


def main():
    args = parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    data_root = osp.abspath(args.data_root)
    work_dir = osp.abspath(args.work_dir)
    if args.tmpdir:
        tmp = osp.join(_PAIRMMOT_ROOT, 'tmp', 'hsmot_single_overfit_accept')
        os.makedirs(tmp, exist_ok=True)
        data_root = osp.join(tmp, 'data')
        work_dir = osp.join(tmp, 'work_dir')
        if _local_rank() == 0:
            print(f'Using tmp workspace: {tmp}')

    device = args.device
    if _is_dist_launched(args.launcher) and device == 'cuda:0':
        device = f'cuda:{_local_rank()}'

    report = run_acceptance(
        config_path=args.config,
        data_root=data_root,
        work_dir=work_dir,
        max_iters=args.max_iters,
        num_frames=args.num_frames,
        min_ap50=args.min_ap50,
        min_map50_95=args.min_map50_95,
        skip_train=args.skip_train,
        device=device,
        launcher=args.launcher,
        from_real=args.from_real,
        src_root=args.src_root,
        ann_file=args.ann_file,
        seed=args.seed,
        reuse_data=not args.force_recreate_data,
        val_interval=args.val_interval,
    )
    sys.exit(0 if report.passed else 1)


if __name__ == '__main__':
    main()
