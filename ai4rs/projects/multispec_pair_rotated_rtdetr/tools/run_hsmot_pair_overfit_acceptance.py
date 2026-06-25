#!/usr/bin/env python3
"""Pair MOT overfit acceptance: train on fixed mini pairs and verify convergence.

Checks:
  1. Final pair loss sum below threshold after thousands of iterations.
  2. Each GT pair has exactly one high-score query.
  3. Visible-side boxes IoU > threshold vs GT.
  4. Presence logits match valid_prev / valid_curr (new / disappear).
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
from typing import Dict, List, Optional, Sequence, Tuple

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
from mmrotate.registry import DATASETS, MODELS
from mmrotate.structures.bbox import qbox2rbox
from mmrotate.utils import register_all_modules

import projects.multispec_pair_rotated_rtdetr.multispec_pair_rotated_rtdetr  # noqa: F401
from projects.multispec_pair_rotated_rtdetr.tools.create_hsmot_pair_overfit_data import (
    create_hsmot_pair_overfit_data,
)
from projects.multispec_pair_rotated_rtdetr.tools.create_hsmot_pair_overfit_from_real import (
    create_hsmot_pair_overfit_from_real,
)
from projects.multispec_pair_rotated_rtdetr.tools.load_pair_pretrain import (
    ensure_pair_adapted_checkpoint,
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


_PRETRAIN_CKPT_NAMES = frozenset({'pair_adapted_pretrain.pth'})


def _latest_checkpoint(work_dir: str) -> str:
    """Return the latest *training* checkpoint, not pretrain-adapt artifacts."""
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

    ckpts = [
        p for p in glob.glob(osp.join(work_dir, '*.pth'))
        if osp.basename(p) not in _PRETRAIN_CKPT_NAMES
    ]
    if not ckpts:
        raise FileNotFoundError(
            f'No training checkpoint in {work_dir} '
            f'(excluding {_PRETRAIN_CKPT_NAMES})')
    return max(ckpts, key=osp.getmtime)


def _finalize_dist_for_eval(launcher: str) -> bool:
    """Sync ranks, tear down DDP; return True only on rank 0 for single-GPU eval."""
    if not _is_dist_launched(launcher):
        return True
    if not torch.distributed.is_initialized():
        return _local_rank() == 0
    barrier()
    rank, _ = get_dist_info()
    torch.distributed.destroy_process_group()
    return rank == 0


def _overfit_dataset_exists(data_root: str) -> bool:
    """Return True if ``data_root/train`` looks like a ready overfit split."""
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
    """Guess whether existing overfit data uses real HSMOT ``3jpg`` layout."""
    return osp.isdir(osp.join(data_root, 'train', 'npy2jpg'))


def _local_rank() -> int:
    return int(os.environ.get('LOCAL_RANK', '0'))


def _is_dist_launched(launcher: str) -> bool:
    return launcher != 'none'


def _wait_for_overfit_dataset(data_root: str, timeout_sec: float = 600.0) -> None:
    """Block until rank 0 finishes creating the overfit split."""
    deadline = time.time() + timeout_sec
    while time.time() < deadline:
        if _overfit_dataset_exists(data_root):
            return
        time.sleep(0.5)
    raise TimeoutError(
        f'Timed out waiting for overfit dataset at {data_root}')


def _pair_adapted_ckpt_path(cfg: Config) -> str:
    return osp.join(osp.abspath(cfg.work_dir), 'pair_adapted_pretrain.pth')


def _sync_pair_pretrain(cfg: Config, timeout_sec: float = 600.0) -> None:
    """Wait for rank 0 to finish pair checkpoint adaptation."""
    if not cfg.get('pair_pretrain_adapt', False):
        return
    src_ckpt = cfg.get('load_from')
    if not src_ckpt:
        return
    dst_ckpt = _pair_adapted_ckpt_path(cfg)
    deadline = time.time() + timeout_sec
    while time.time() < deadline:
        if osp.isfile(dst_ckpt):
            cfg.load_from = dst_ckpt
            return
        time.sleep(0.5)
    raise TimeoutError(
        f'Timed out waiting for adapted checkpoint at {dst_ckpt}')


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
    """Create or reuse overfit dataset; return whether layout is real ``3jpg``."""
    if reuse_data and _overfit_dataset_exists(data_root):
        is_real = _detect_real_layout(data_root)
        print(f'[0/3] Reuse existing dataset at {data_root} '
              f'(layout={"3jpg" if is_real else "npy"})')
        return is_real

    if from_real:
        create_hsmot_pair_overfit_from_real(
            dst_root=data_root,
            src_root=osp.abspath(src_root),
            ann_file=ann_file,
            num_frames=num_frames,
            seed=seed,
        )
        return True

    create_hsmot_pair_overfit_data(data_root, num_frames=num_frames)
    return False


def _build_preprocessor(cfg: Config):
    """Build pair data preprocessor from config (normalize + pad)."""
    return MODELS.build(cfg.model.data_preprocessor)


def _prepare_pair_batch(
    batch: dict,
    preprocessor,
    device: torch.device,
    training: bool = False,
) -> Tuple[torch.Tensor, list]:
    """Collate raw dataloader batch and run ``PairMultispecDetDataPreprocessor``."""
    from mmdet.structures import DetDataSample

    data_samples = batch['data_samples']
    if not isinstance(data_samples, list):
        data_samples = [data_samples]

    samples = []
    for ds in data_samples:
        sample = DetDataSample()
        sample.pair_gt_instances = ds.pair_gt_instances
        sample.set_metainfo(ds.metainfo)
        samples.append(sample)

    preprocessed = preprocessor(
        {'inputs': batch['inputs'], 'data_samples': samples},
        training=training)
    inputs = preprocessed['inputs'].to(device)
    return inputs, preprocessed['data_samples']


def _pair_loss_sum(losses: Dict[str, torch.Tensor]) -> float:
    total = 0.0
    for key, val in losses.items():
        if 'loss' in key and isinstance(val, torch.Tensor):
            total += float(val.detach().cpu())
    return total


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
def evaluate_pair_predictions(
    model,
    dataloader: DataLoader,
    preprocessor,
    device: torch.device,
    score_thr: float = 0.35,
    iou_thr: float = 0.5,
    pres_thr: float = 0.5,
) -> AcceptanceReport:
    """Run predict on training pairs and verify matching quality."""
    report = AcceptanceReport()
    model.eval()

    total_gt = 0
    matched_queries = 0
    iou_prev_sum = 0.0
    iou_curr_sum = 0.0
    iou_prev_count = 0
    iou_curr_count = 0
    presence_ok = 0
    presence_total = 0
    duplicate_match = 0

    for batch in dataloader:
        inputs, batch_samples = _prepare_pair_batch(
            batch, preprocessor, device, training=False)
        outputs = model.predict(inputs, batch_samples, rescale=False)
        for sample in outputs:
            gt: InstanceData = sample.pair_gt_instances
            pred = sample.pred_pair_instances
            num_gt = len(gt.labels)
            total_gt += num_gt

            gt_labels = gt.labels.cpu()
            gt_prev = _to_rbox_tensor(gt.bboxes_prev).cpu()
            gt_curr = _to_rbox_tensor(gt.bboxes_curr).cpu()
            valid_prev = gt.valid_prev.cpu().bool()
            valid_curr = gt.valid_curr.cpu().bool()

            pred_scores = pred.scores.cpu()
            pred_labels = pred.labels.cpu()
            pred_prev = pred.bboxes_prev.cpu()
            pred_curr = pred.bboxes_curr.cpu()
            pred_pres_p = pred.presence_prev.cpu()
            pred_pres_c = pred.presence_curr.cpu()

            used_queries = set()
            for gi in range(num_gt):
                label = int(gt_labels[gi].item())
                cls_mask = pred_labels == label
                cand_scores = pred_scores.clone()
                cand_scores[~cls_mask] = -1.0
                cand_scores[list(used_queries)] = -1.0
                best_q = int(cand_scores.argmax().item())
                best_score = float(cand_scores[best_q].item())

                if best_score < score_thr:
                    report.fail(
                        f'GT#{gi} label={label} has no query above '
                        f'score_thr={score_thr} (best={best_score:.3f})')
                    continue

                if best_q in used_queries:
                    duplicate_match += 1
                used_queries.add(best_q)
                matched_queries += 1

                if valid_prev[gi]:
                    iou_p = _rbox_iou(pred_prev[best_q], gt_prev[gi])
                    iou_prev_sum += iou_p
                    iou_prev_count += 1
                    if iou_p < iou_thr:
                        report.fail(
                            f'GT#{gi} prev IoU={iou_p:.3f} < {iou_thr}')
                else:
                    if pred_pres_p[best_q].item() > pres_thr:
                        report.fail(
                            f'GT#{gi} invalid prev but '
                            f'presence_prev={pred_pres_p[best_q]:.3f}')
                presence_total += 1
                presence_ok += int(
                    (pred_pres_p[best_q].item() > pres_thr) == bool(
                        valid_prev[gi].item()))

                if valid_curr[gi]:
                    iou_c = _rbox_iou(pred_curr[best_q], gt_curr[gi])
                    iou_curr_sum += iou_c
                    iou_curr_count += 1
                    if iou_c < iou_thr:
                        report.fail(
                            f'GT#{gi} curr IoU={iou_c:.3f} < {iou_thr}')
                else:
                    if pred_pres_c[best_q].item() > pres_thr:
                        report.fail(
                            f'GT#{gi} invalid curr but '
                            f'presence_curr={pred_pres_c[best_q]:.3f}')
                presence_total += 1
                presence_ok += int(
                    (pred_pres_c[best_q].item() > pres_thr) == bool(
                        valid_curr[gi].item()))

    if total_gt == 0:
        report.fail('No GT pairs in evaluation set')
    else:
        match_ratio = matched_queries / total_gt
        report.metrics['gt_pairs'] = float(total_gt)
        report.metrics['matched_queries'] = float(matched_queries)
        report.metrics['match_ratio'] = match_ratio
        if match_ratio < 1.0:
            report.fail(
                f'Only {matched_queries}/{total_gt} GT pairs matched '
                f'(score>={score_thr})')
        else:
            report.ok(
                f'Each GT pair has one high-score query '
                f'({matched_queries}/{total_gt})')

        if duplicate_match > 0:
            report.fail(f'{duplicate_match} duplicate query assignments')

        if iou_prev_count > 0:
            mean_iou_p = iou_prev_sum / iou_prev_count
            report.metrics['mean_iou_prev'] = mean_iou_p
            report.ok(f'mean prev IoU={mean_iou_p:.3f} ({iou_prev_count} boxes)')
        if iou_curr_count > 0:
            mean_iou_c = iou_curr_sum / iou_curr_count
            report.metrics['mean_iou_curr'] = mean_iou_c
            report.ok(f'mean curr IoU={mean_iou_c:.3f} ({iou_curr_count} boxes)')

        if presence_total > 0:
            pres_acc = presence_ok / presence_total
            report.metrics['presence_acc'] = pres_acc
            if pres_acc < 0.95:
                report.fail(f'presence accuracy {pres_acc:.3f} < 0.95')
            else:
                report.ok(
                    f'presence accuracy={pres_acc:.3f} '
                    f'(new/disappear states)')

    return report


def _prepare_pair_pretrain(cfg: Config) -> None:
    """Adapt single-frame HSMOT checkpoint for pair model loading."""
    if not cfg.get('pair_pretrain_adapt', False):
        return
    src_ckpt = cfg.get('load_from')
    if not src_ckpt:
        return
    adapted = ensure_pair_adapted_checkpoint(
        src_ckpt, cfg.work_dir, force=False)
    cfg.load_from = adapted


def _sync_pair_dataset_cfg(cfg: Config, data_root: str,
                           is_real_layout: bool) -> None:
    """Point train/val/test dataloaders at the overfit split."""
    train_root = osp.join(data_root, 'train')
    for key in ('train_dataloader', 'val_dataloader', 'test_dataloader'):
        loader = cfg.get(key)
        if loader is None or loader.get('dataset') is None:
            continue
        loader.dataset.data_root = train_root
        if is_real_layout:
            loader.dataset.img_format = '3jpg'
            loader.dataset.data_prefix = dict(img_path='npy2jpg')


def run_training(cfg: Config, launcher: str = 'none') -> Runner:
    register_all_modules()
    os.chdir(_AI4RS_ROOT)
    cfg.launcher = launcher
    runner = Runner.from_cfg(cfg)
    runner.train()
    return runner


def collate_pair_batch(batch):
    return pseudo_collate(batch)


def run_acceptance(
    config_path: str,
    data_root: str,
    work_dir: str,
    max_iters: int,
    num_frames: int,
    loss_thr: float,
    score_thr: float,
    iou_thr: float,
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
    _sync_pair_dataset_cfg(cfg, data_root, is_real_layout)
    cfg.default_hooks.logger.interval = min(10, max_iters // 20)

    if local_rank == 0:
        _prepare_pair_pretrain(cfg)
    else:
        _sync_pair_pretrain(cfg)

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
        collate_fn=collate_pair_batch,
    )

    # final train-mode loss on one batch (sanity)
    model.train()
    batch = next(iter(loader))
    inputs, samples = _prepare_pair_batch(
        batch, preprocessor, dev, training=True)
    losses = model.loss(inputs, samples)
    loss_sum = _pair_loss_sum(losses)
    print(f'Final batch loss sum: {loss_sum:.4f} (threshold {loss_thr})')

    report = AcceptanceReport()
    report.metrics['final_loss_sum'] = loss_sum
    if loss_sum > loss_thr:
        report.fail(f'loss sum {loss_sum:.4f} > {loss_thr}')
    else:
        report.ok(f'loss sum {loss_sum:.4f} <= {loss_thr}')

    eval_report = evaluate_pair_predictions(
        model,
        loader,
        preprocessor,
        dev,
        score_thr=score_thr,
        iou_thr=iou_thr,
    )
    report.metrics.update(eval_report.metrics)
    report.messages.extend(eval_report.messages)
    if not eval_report.passed:
        report.passed = False

    print('[3/3] Acceptance summary')
    for msg in report.messages:
        print(' ', msg)
    print('RESULT:', 'PASSED' if report.passed else 'FAILED')
    print('METRICS:', json.dumps(report.metrics, indent=2))
    return report


def parse_args():
    parser = argparse.ArgumentParser(
        description='HSMOT Pair overfit acceptance script')
    parser.add_argument(
        '--config',
        default='projects/multispec_pair_rotated_rtdetr/configs/'
        'o2_pair_rtdetr_r18vd_overfit.py')
    parser.add_argument(
        '--data-root',
        default='data/HSMOT_pair_overfit')
    parser.add_argument(
        '--work-dir',
        default='work_dirs/o2_pair_rtdetr_r18vd_overfit_accept')
    parser.add_argument('--max-iters', type=int, default=3000)
    parser.add_argument(
        '--val-interval',
        type=int,
        default=500,
        help='Run validation and save pair visualizations every N iters')
    parser.add_argument(
        '--num-frames',
        type=int,
        default=10,
        help='Frames in one clip: synthetic uses this directly; '
        'real (--from-real) extracts one contiguous clip (pairs = frames - 1)')
    parser.add_argument(
        '--from-real',
        action='store_true',
        help='Extract one contiguous clip from real HSMOT instead of synthetic')
    parser.add_argument(
        '--src-root',
        default='../data/hsmot/train',
        help='Real HSMOT train root (with --from-real)')
    parser.add_argument(
        '--ann-file',
        default='../data/hsmot/train_half.txt',
        help='Sequence split file for real HSMOT')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--loss-thr', type=float, default=2.0)
    parser.add_argument('--score-thr', type=float, default=0.35)
    parser.add_argument('--iou-thr', type=float, default=0.5)
    parser.add_argument('--skip-train', action='store_true')
    parser.add_argument(
        '--force-recreate-data',
        action='store_true',
        help='Recreate dataset even when data-root already contains a split')
    parser.add_argument(
        '--tmpdir',
        action='store_true',
        help='Use PairMmot/tmp/hsmot_pair_overfit_accept')
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
        tmp = osp.join(_PAIRMMOT_ROOT, 'tmp', 'hsmot_pair_overfit_accept')
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
        loss_thr=args.loss_thr,
        score_thr=args.score_thr,
        iou_thr=args.iou_thr,
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
