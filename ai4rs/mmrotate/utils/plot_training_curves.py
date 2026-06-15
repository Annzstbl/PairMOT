# Copyright (c) AI4RS. All rights reserved.
"""Plot training curves from MMEngine LocalVisBackend scalars.json."""
from __future__ import annotations

import json
import os
import os.path as osp
from collections import defaultdict
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402


_META_KEYS = frozenset({
    'data_time', 'time', 'epoch', 'iter', 'step', 'memory',
})

_TRAIN_PRIMARY = (
    'lr',
    'loss',
    'loss_cls',
    'loss_bbox',
    'loss_iou',
    'grad_norm',
)


def load_scalars(scalars_path: str) -> List[dict]:
    records = []
    with open(scalars_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def split_train_val(records: Sequence[dict]) -> Tuple[List[dict], List[dict]]:
    train_records, val_records = [], []
    for record in records:
        if 'loss' in record:
            train_records.append(record)
        elif any('/' in key for key in record):
            val_records.append(record)
    return train_records, val_records


def _series(records: Sequence[dict], key: str) -> Tuple[List[float], List[float]]:
    xs, ys = [], []
    for record in records:
        if key in record:
            xs.append(float(record.get('step', record.get('iter', len(xs)))))
            ys.append(float(record[key]))
    return xs, ys


def _epoch_mean_series(
        records: Sequence[dict], key: str) -> Tuple[List[float], List[float]]:
    """Average a train metric within each epoch."""
    buckets: Dict[int, List[float]] = defaultdict(list)
    for record in records:
        if key in record and 'epoch' in record:
            buckets[int(record['epoch'])].append(float(record[key]))
    epochs = sorted(buckets)
    means = [sum(buckets[e]) / len(buckets[e]) for e in epochs]
    return [float(e) for e in epochs], means


def _val_epoch_series(
        val_records: Sequence[dict], key: str) -> Tuple[List[float], List[float]]:
    """Val metrics are logged sparsely; ``step`` stores the epoch index."""
    xs, ys = [], []
    for record in val_records:
        if key not in record:
            continue
        epoch = record.get('epoch', record.get('step'))
        if epoch is None:
            continue
        xs.append(float(epoch))
        ys.append(float(record[key]))
    return xs, ys


def _plot_loss_vs_val_by_epoch(
        ax,
        train_records: Sequence[dict],
        val_records: Sequence[dict],
        val_keys: Sequence[str]) -> None:
    """Plot epoch-mean train loss and sparse val metrics on aligned epoch axis."""
    epochs, loss_means = _epoch_mean_series(train_records, 'loss')
    if not epochs:
        ax.set_visible(False)
        return

    ax.plot(epochs, loss_means, color='#1f77b4', linewidth=1.5,
            label='train loss (epoch mean)')
    ax.set_xlabel('epoch')
    ax.set_ylabel('loss', color='#1f77b4')
    ax.tick_params(axis='y', labelcolor='#1f77b4')
    ax.grid(True, alpha=0.3)

    if not val_keys:
        ax.legend(loc='best', fontsize=8)
        return

    ax2 = ax.twinx()
    val_colors = ('#d62728', '#2ca02c', '#ff7f0e', '#9467bd')
    for idx, key in enumerate(val_keys):
        xs, ys = _val_epoch_series(val_records, key)
        if not xs:
            continue
        color = val_colors[idx % len(val_colors)]
        ax2.plot(xs, ys, '-o', color=color, linewidth=1.2, markersize=5,
                 label=key.split('/')[-1])
    ax2.set_ylabel('val metric', color='#d62728')
    ax2.tick_params(axis='y', labelcolor='#d62728')

    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc='best', fontsize=7)


def _collect_train_loss_keys(records: Sequence[dict]) -> List[str]:
    keys = set()
    for record in records:
        for key, value in record.items():
            if key in _META_KEYS or key == 'base_lr':
                continue
            if key == 'lr' or not isinstance(value, (int, float)):
                continue
            if 'loss' in key:
                keys.add(key)
    primary = [k for k in _TRAIN_PRIMARY if k in keys and k != 'lr']
    secondary = sorted(k for k in keys if k not in _TRAIN_PRIMARY)
    return primary + secondary


def _collect_val_metric_keys(records: Sequence[dict]) -> List[str]:
    keys = set()
    for record in records:
        for key in record:
            if '/' in key:
                keys.add(key)
    return sorted(keys)


def _split_val_metric_keys(
        val_keys: Sequence[str]) -> Tuple[List[str], List[str], List[str]]:
    """Split val keys into mAP/APxx, class AP, and class recall groups."""
    map_keys, class_ap_keys, class_recall_keys = [], [], []
    for key in val_keys:
        name = key.split('/')[-1]
        if name in ('mAP', 'mean_recall') or (
                name.startswith('AP') and name[2:].isdigit()):
            map_keys.append(key)
        elif name.endswith('_AP'):
            class_ap_keys.append(key)
        elif name.endswith('_recall'):
            class_recall_keys.append(key)
    return map_keys, class_ap_keys, class_recall_keys


def _sort_ap_keys(keys: Sequence[str]) -> List[str]:
    def _ap_order(key: str) -> tuple:
        name = key.split('/')[-1]
        if name == 'mAP':
            return (0, 0)
        if name.startswith('AP') and name[2:].isdigit():
            return (1, int(name[2:]))
        return (2, name)

    return sorted(keys, key=_ap_order)


def _plot_series(ax, xs, ys, label, color=None):
    if not xs:
        ax.set_visible(False)
        return
    ax.plot(xs, ys, label=label, linewidth=1.2, color=color)
    ax.set_xlabel('step')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best', fontsize=8)


def plot_training_curves(
        scalars_path: str,
        out_dir: Optional[str] = None,
        dpi: int = 150) -> Dict[str, str]:
    """Generate training curve PNGs from scalars.json.

    Returns:
        dict mapping plot name to saved file path.
    """
    if not osp.isfile(scalars_path):
        raise FileNotFoundError(f'scalars.json not found: {scalars_path}')

    if out_dir is None:
        out_dir = osp.join(osp.dirname(scalars_path), 'curves')
    os.makedirs(out_dir, exist_ok=True)

    records = load_scalars(scalars_path)
    train_records, val_records = split_train_val(records)
    if not train_records and not val_records:
        raise ValueError(f'No plottable records in {scalars_path}')

    saved: Dict[str, str] = {}

    # --- overview figure ---
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    fig.suptitle('Training Overview', fontsize=14)

    _plot_series(axes[0, 0], *_series(train_records, 'lr'), 'lr')
    axes[0, 0].set_title('Learning Rate')

    _plot_series(axes[0, 1], *_series(train_records, 'loss'), 'loss')
    axes[0, 1].set_title('Total Loss')

    ax = axes[0, 2]
    for key, color in zip(
            ('loss_cls', 'loss_bbox', 'loss_iou'),
            ('#1f77b4', '#ff7f0e', '#2ca02c')):
        xs, ys = _series(train_records, key)
        if xs:
            ax.plot(xs, ys, label=key, linewidth=1.0, color=color)
    ax.set_title('Main Loss Components')
    ax.set_xlabel('step')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best', fontsize=8)

    _plot_series(axes[1, 0], *_series(train_records, 'grad_norm'), 'grad_norm')
    axes[1, 0].set_title('Grad Norm')

    val_keys = _collect_val_metric_keys(val_records)
    map_keys, class_ap_keys, class_recall_keys = _split_val_metric_keys(
        val_keys)
    overview_map_keys = [
        k for k in _sort_ap_keys(map_keys)
        if k.split('/')[-1] in ('mAP', 'AP50', 'AP75', 'AP95')
    ] or map_keys

    ax = axes[1, 1]
    if overview_map_keys:
        for key in overview_map_keys:
            xs, ys = _val_epoch_series(val_records, key)
            if xs:
                ax.plot(xs, ys, label=key.split('/')[-1], marker='o',
                        linewidth=1.2, markersize=4)
        ax.set_title('Validation mAP / AP')
        ax.set_xlabel('epoch')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='best', fontsize=8)
    else:
        ax.set_visible(False)

    ax = axes[1, 2]
    _plot_loss_vs_val_by_epoch(
        ax, train_records, val_records, overview_map_keys or map_keys)
    ax.set_title('Loss vs Val (by epoch)')

    fig.tight_layout()
    overview_path = osp.join(out_dir, 'overview.png')
    fig.savefig(overview_path, dpi=dpi, bbox_inches='tight')
    plt.close(fig)
    saved['overview'] = overview_path

    # --- individual primary plots ---
    single_plots = {
        'learning_rate': 'lr',
        'total_loss': 'loss',
        'grad_norm': 'grad_norm',
    }
    for name, key in single_plots.items():
        xs, ys = _series(train_records, key)
        if not xs:
            continue
        fig, ax = plt.subplots(figsize=(8, 4))
        _plot_series(ax, xs, ys, key)
        ax.set_title(key)
        path = osp.join(out_dir, f'{name}.png')
        fig.savefig(path, dpi=dpi, bbox_inches='tight')
        plt.close(fig)
        saved[name] = path

    # --- all loss components ---
    loss_keys = _collect_train_loss_keys(train_records)
    if loss_keys:
        n = len(loss_keys)
        ncols = 3
        nrows = (n + ncols - 1) // ncols
        fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 3.5 * nrows))
        axes_flat = axes.flatten() if n > 1 else [axes]
        for ax, key in zip(axes_flat, loss_keys):
            _plot_series(ax, *_series(train_records, key), key)
            ax.set_title(key, fontsize=9)
        for ax in axes_flat[len(loss_keys):]:
            ax.set_visible(False)
        fig.suptitle('Training Loss Components', fontsize=13)
        fig.tight_layout()
        path = osp.join(out_dir, 'loss_components.png')
        fig.savefig(path, dpi=dpi, bbox_inches='tight')
        plt.close(fig)
        saved['loss_components'] = path

    # --- validation metrics ---
    if val_keys:
        if map_keys:
            fig, ax = plt.subplots(figsize=(9, 4))
            for key in _sort_ap_keys(map_keys):
                xs, ys = _val_epoch_series(val_records, key)
                if xs:
                    ax.plot(xs, ys, label=key.split('/')[-1], marker='o',
                            linewidth=1.2, markersize=5)
            ax.set_title('Validation mAP / AP')
            ax.set_xlabel('epoch')
            ax.grid(True, alpha=0.3)
            ax.legend(loc='best', fontsize=8)
            fig.tight_layout()
            path = osp.join(out_dir, 'validation_map.png')
            fig.savefig(path, dpi=dpi, bbox_inches='tight')
            plt.close(fig)
            saved['validation_map'] = path

        if class_ap_keys:
            fig, ax = plt.subplots(figsize=(9, 4))
            for key in class_ap_keys:
                xs, ys = _val_epoch_series(val_records, key)
                if xs:
                    ax.plot(xs, ys, label=key.split('/')[-1], marker='o',
                            linewidth=1.2, markersize=4)
            ax.set_title('Per-class AP (IoU=0.5)')
            ax.set_xlabel('epoch')
            ax.grid(True, alpha=0.3)
            ax.legend(loc='best', fontsize=7)
            fig.tight_layout()
            path = osp.join(out_dir, 'validation_class_ap.png')
            fig.savefig(path, dpi=dpi, bbox_inches='tight')
            plt.close(fig)
            saved['validation_class_ap'] = path

        if class_recall_keys:
            fig, ax = plt.subplots(figsize=(9, 4))
            for key in class_recall_keys:
                xs, ys = _val_epoch_series(val_records, key)
                if xs:
                    ax.plot(xs, ys, label=key.split('/')[-1], marker='o',
                            linewidth=1.2, markersize=4)
            ax.set_title('Per-class Recall (IoU=0.5)')
            ax.set_xlabel('epoch')
            ax.grid(True, alpha=0.3)
            ax.legend(loc='best', fontsize=7)
            fig.tight_layout()
            path = osp.join(out_dir, 'validation_class_recall.png')
            fig.savefig(path, dpi=dpi, bbox_inches='tight')
            plt.close(fig)
            saved['validation_class_recall'] = path

        fig, ax = plt.subplots(figsize=(8, 4))
        _plot_loss_vs_val_by_epoch(
            ax, train_records, val_records, overview_map_keys or map_keys)
        ax.set_title('Loss vs Val (by epoch)')
        fig.tight_layout()
        path = osp.join(out_dir, 'loss_vs_val_epoch.png')
        fig.savefig(path, dpi=dpi, bbox_inches='tight')
        plt.close(fig)
        saved['loss_vs_val_epoch'] = path

    return saved
