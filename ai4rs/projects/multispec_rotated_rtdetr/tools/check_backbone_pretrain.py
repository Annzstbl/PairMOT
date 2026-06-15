#!/usr/bin/env python3
# Copyright (c) AI4RS. All rights reserved.
"""Inspect or verify checkpoint backbone parameters.

Modes:
- ``--inspect``: list all parameter keys and tensor shapes (no pretrain compare)
- default: compare trained backbone against reference DOTA pretrained checkpoint
"""
import argparse
import os.path as osp
import sys
from collections import OrderedDict, defaultdict
from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F

_AI4RS_ROOT = osp.abspath(osp.join(osp.dirname(__file__), '../../..'))
if _AI4RS_ROOT not in sys.path:
    sys.path.insert(0, _AI4RS_ROOT)

from projects.multispec_rotated_rtdetr.configs.pretrain_paths import (
    O2_R18_DOTA_E72)
from projects.multispec_rotated_rtdetr.multispec_rotated_rtdetr.pretrain_utils import (
    convert_stem_conv2d_to_conv3d_weight, load_checkpoint_state_dict)
from projects.multispec_rotated_rtdetr.multispec_rotated_rtdetr.resnet import (
    _filter_backbone_state_dict)


def parse_args():
    parser = argparse.ArgumentParser(
        description='Verify backbone pretrained loading in a trained checkpoint.')
    parser.add_argument(
        'trained_ckpt',
        help='Path to trained checkpoint, e.g. epoch_72.pth')
    parser.add_argument(
        '--pretrain-ckpt',
        default=O2_R18_DOTA_E72,
        help='Reference DOTA pretrained checkpoint (default: R18 O2 DOTA e72)')
    parser.add_argument(
        '--cos-threshold',
        type=float,
        default=0.95,
        help='Cosine similarity threshold to treat as "loaded from pretrain"')
    parser.add_argument(
        '--max-diff-threshold',
        type=float,
        default=0.05,
        help='Max abs diff threshold (for weights that should match at init)')
    parser.add_argument(
        '--inspect',
        action='store_true',
        help='Only list parameter keys/shapes; skip pretrain comparison')
    parser.add_argument(
        '--module',
        choices=['all', 'backbone'],
        default='backbone',
        help='Which module to list in --inspect mode (default: backbone)')
    return parser.parse_args()


def inspect_checkpoint(state_dict: Dict[str, torch.Tensor], module: str) -> None:
    """Print parameter inventory and flag suspicious conv3d keys."""
    if module == 'backbone':
        items = sorted((k, v) for k, v in state_dict.items()
                       if k.startswith('backbone.'))
        title = 'backbone'
    else:
        items = sorted(state_dict.items())
        title = 'full model'

    print(f'=== {title}: {len(items)} tensors ===\n')

    groups: Dict[str, List[Tuple[str, Tuple[int, ...], torch.dtype]]] = (
        defaultdict(list))
    for key, value in items:
        top = key.split('.')[0] if module == 'all' else 'backbone'
        groups[top].append((key, tuple(value.shape), value.dtype))

    for group in sorted(groups):
        print(f'--- {group} ({len(groups[group])}) ---')
        for key, shape, dtype in groups[group]:
            note = ''
            if 'conv3d' in key:
                if key == 'backbone.stem.0.conv3d.weight':
                    note = '  [OK: expected 5D stem Conv3d]'
                else:
                    note = '  [BAD: unexpected conv3d key]'
            elif key.endswith('.conv1.weight') and '.conv3d.' in key:
                note = '  [BAD: layer conv1 wrongly remapped to conv3d]'
            print(f'  {key:<70} {str(shape):<28} {dtype}{note}')
        print()

    conv3d_keys = [k for k, _ in items if 'conv3d' in k]
    print('=== conv3d summary ===')
    for key in sorted(conv3d_keys):
        shape = tuple(state_dict[key].shape)
        print(f'  {key}: {shape}')
    print(f'Total conv3d keys: {len(conv3d_keys)}')

    bad = [
        k for k in conv3d_keys
        if k != 'backbone.stem.0.conv3d.weight'
    ]
    print()
    if bad:
        print(f'[FAIL] Unexpected conv3d keys ({len(bad)}):')
        for k in bad:
            print(f'  {k}: {tuple(state_dict[k].shape)}')
    else:
        print('[OK] Only backbone.stem.0.conv3d.weight is Conv3d; '
              'all layer*.conv1 remain Conv2d (N,C,H,W).')

    conv1_keys = [
        k for k, _ in items
        if k.endswith('.conv1.weight') and k.startswith('backbone.layer')
    ]
    print(f'\nlayer*.conv1.weight count: {len(conv1_keys)}')
    for k in conv1_keys:
        print(f'  {k}: {tuple(state_dict[k].shape)}')


def extract_backbone(state_dict: Dict[str, torch.Tensor],
                     prefix: str = 'backbone.') -> OrderedDict:
    out = OrderedDict()
    for key, value in state_dict.items():
        if key.startswith(prefix):
            out[key[len(prefix):]] = value
    return out


def cosine_similarity(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a.detach().flatten().float()
    b = b.detach().flatten().float()
    if a.numel() == 0 or b.numel() == 0:
        return float('nan')
    return F.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)).item()


def max_abs_diff(a: torch.Tensor, b: torch.Tensor) -> float:
    return (a.detach().float() - b.detach().float()).abs().max().item()


def build_correct_stem_adapted(pretrain_backbone: Dict[str, torch.Tensor]
                               ) -> Dict[str, torch.Tensor]:
    """Only remap ``stem.0.weight`` -> ``stem.0.conv3d.weight``."""
    adapted = dict(pretrain_backbone)
    stem_key = 'stem.0.weight'
    if stem_key not in adapted:
        raise KeyError(f'Missing {stem_key} in pretrained backbone.')
    weight = adapted.pop(stem_key)
    adapted['stem.0.conv3d.weight'] = convert_stem_conv2d_to_conv3d_weight(weight)
    return adapted


def build_buggy_stem_adapted(pretrain_backbone: Dict[str, torch.Tensor]
                             ) -> Dict[str, torch.Tensor]:
    """Legacy buggy logic: all keys ending with conv1.weight or stem.0.weight."""
    adapted = dict(pretrain_backbone)
    buggy_keys = [
        key for key in adapted
        if key.endswith('conv1.weight') or key.endswith('stem.0.weight')
    ]
    for key in buggy_keys:
        weight = adapted.pop(key)
        prefix = key.rsplit('.', 1)[0]
        adapted[f'{prefix}.conv3d.weight'] = (
            convert_stem_conv2d_to_conv3d_weight(weight))
    return adapted


def classify_key(key: str) -> str:
    if key.startswith('stem.0.conv3d'):
        return 'stem_3d_conv'
    if key.startswith('stem.0.se_'):
        return 'stem_se_random'
    if key.startswith('stem.'):
        return 'stem_other'
    if '.conv1.' in key or key.endswith('.conv1.weight'):
        return 'layer_conv1'
    if key.startswith('layer'):
        return 'layer_other'
    return 'other'


def compare_group(
        trained: Dict[str, torch.Tensor],
        reference: Dict[str, torch.Tensor],
        keys: List[str],
) -> List[Tuple[str, str, float, float, str]]:
    rows = []
    for key in keys:
        if key not in trained:
            rows.append((key, classify_key(key), float('nan'), float('nan'),
                         'missing_in_trained'))
            continue
        if key not in reference:
            rows.append((key, classify_key(key), float('nan'), float('nan'),
                         'no_pretrained_ref'))
            continue
        t, r = trained[key], reference[key]
        if t.shape != r.shape:
            rows.append((key, classify_key(key), float('nan'), float('nan'),
                         f'shape_mismatch {tuple(t.shape)} vs {tuple(r.shape)}'))
            continue
        cos = cosine_similarity(t, r)
        diff = max_abs_diff(t, r)
        rows.append((key, classify_key(key), cos, diff, 'ok'))
    return rows


def summarize_rows(rows: List[Tuple[str, str, float, float, str]],
                   cos_threshold: float) -> Dict[str, Dict[str, int]]:
    stats: Dict[str, Dict[str, int]] = {}
    for _, group, cos, _, status in rows:
        bucket = stats.setdefault(group, {
            'total': 0, 'high_sim': 0, 'low_sim': 0, 'missing': 0, 'other': 0
        })
        bucket['total'] += 1
        if status != 'ok':
            bucket['other'] += 1
            if 'missing' in status:
                bucket['missing'] += 1
            continue
        if cos >= cos_threshold:
            bucket['high_sim'] += 1
        else:
            bucket['low_sim'] += 1
    return stats


def print_table(rows: List[Tuple[str, str, float, float, str]],
                cos_threshold: float, title: str) -> None:
    print(f'\n{"=" * 80}')
    print(title)
    print(f'{"=" * 80}')
    print(f'{"key":<42} {"group":<14} {"cos":>8} {"max|d|":>10}  status')
    print('-' * 80)
    for key, group, cos, diff, status in rows:
        cos_s = f'{cos:.4f}' if cos == cos else '   n/a'
        diff_s = f'{diff:.4e}' if diff == diff else '       n/a'
        flag = ''
        if status == 'ok' and cos < cos_threshold:
            flag = '  <-- LOW'
        print(f'{key:<42} {group:<14} {cos_s:>8} {diff_s:>10}  {status}{flag}')


def main():
    args = parse_args()
    trained_sd = load_checkpoint_state_dict(args.trained_ckpt)

    if args.inspect:
        print(f'Checkpoint: {args.trained_ckpt}\n')
        inspect_checkpoint(trained_sd, args.module)
        return

    pretrain_sd = load_checkpoint_state_dict(args.pretrain_ckpt)

    trained_bb = extract_backbone(trained_sd)
    pretrain_bb = extract_backbone(pretrain_sd)

    print(f'Trained checkpoint : {args.trained_ckpt}')
    print(f'Pretrain reference : {args.pretrain_ckpt}')
    print(f'Backbone params    : trained={len(trained_bb)}, pretrain={len(pretrain_bb)}')

    invalid = [
        k for k in trained_bb
        if 'conv3d' in k and not k.startswith('stem.0.conv3d')
    ]
    if invalid:
        print('\n[FAIL] Invalid backbone keys (layer conv wrongly remapped):')
        for k in invalid[:10]:
            print(f'  - backbone.{k}')
    else:
        print('\n[OK] No invalid layer->conv3d keys in trained checkpoint.')

    correct_ref = build_correct_stem_adapted(pretrain_bb)
    buggy_ref = build_buggy_stem_adapted(pretrain_bb)

    # Keys only init_weights would touch (before load_from overwrites the rest).
    init_only_keys = sorted(set(correct_ref) - set(pretrain_bb))
    init_rows = compare_group(trained_bb, correct_ref, init_only_keys)
    print('\n--- Stage 1: stem remap (init_weights) ---')
    for key, group, cos, diff, status in init_rows:
        print(f'  backbone.{key}: cos={cos:.4f}, max|d|={diff:.4e}, {status}')

    # Full backbone vs pretrained (after load_from, layers should trace pretrain).
    shared_keys = sorted(set(trained_bb) & set(pretrain_bb))
    layer_rows = compare_group(trained_bb, pretrain_bb, shared_keys)
    print_table(
        layer_rows,
        args.cos_threshold,
        f'Stage 2: trained vs pretrained backbone ({len(shared_keys)} shared keys)')

    stats = summarize_rows(layer_rows, args.cos_threshold)
    print('\n--- Summary by group ---')
    for group, bucket in sorted(stats.items()):
        print(
            f'  {group:<14} total={bucket["total"]:3d}  '
            f'cos>={args.cos_threshold:.2f}: {bucket["high_sim"]:3d}  '
            f'cos<{args.cos_threshold:.2f}: {bucket["low_sim"]:3d}  '
            f'missing/other: {bucket["missing"] + bucket["other"]:3d}')

    # Simulate init_weights-only load (buggy) vs what load_from repairs.
    buggy_only = {
        k: v for k, v in buggy_ref.items()
        if k not in pretrain_bb and k.endswith('.conv3d.weight')
    }
    print('\n--- Buggy init_weights simulation ---')
    print(f'  Wrongly created conv3d keys: {len(buggy_only)}')
    if buggy_only:
        print('  Examples:')
        for k in list(buggy_only)[:3]:
            print(f'    {k}')

    low_sim = [
        (k, g, c, d) for k, g, c, d, s in layer_rows
        if s == 'ok' and c < args.cos_threshold
    ]
    print('\n--- Verdict ---')
    if invalid:
        print('  Structure: FAIL (corrupted key names)')
    else:
        print('  Structure: OK')

    stem3d = trained_bb.get('stem.0.conv3d.weight')
    stem3d_ref = correct_ref.get('stem.0.conv3d.weight')
    if stem3d is not None and stem3d_ref is not None:
        stem_cos = cosine_similarity(stem3d, stem3d_ref)
        print(f'  Stem 3D conv vs converted pretrain: cos={stem_cos:.4f} '
              f'({"OK" if stem_cos >= args.cos_threshold else "CHECK"})')

    conv1_rows = [r for r in layer_rows if r[1] == 'layer_conv1' and r[4] == 'ok']
    if conv1_rows:
        conv1_cos = [r[2] for r in conv1_rows]
        avg_conv1 = sum(conv1_cos) / len(conv1_cos)
        print(f'  Layer conv1 vs pretrain: avg_cos={avg_conv1:.4f} '
              f'({"OK" if avg_conv1 >= args.cos_threshold else "CHECK"})')
        print('  Note: after 72 epochs weights drift; high cosine means init likely '
              'loaded from pretrain (via load_from), not random init.')

    if low_sim:
        print(f'  Low-similarity keys ({len(low_sim)}):')
        for k, g, c, d in low_sim[:15]:
            print(f'    backbone.{k} ({g}) cos={c:.4f}')
        if len(low_sim) > 15:
            print(f'    ... and {len(low_sim) - 15} more')
    else:
        print('  All shared backbone keys are close to pretrained reference.')

    se_keys = [k for k in trained_bb if k.startswith('stem.0.se_')]
    print(f'  SE layers ({len(se_keys)} params): random init expected, no pretrain ref.')


if __name__ == '__main__':
    main()
