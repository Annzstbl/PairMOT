#!/usr/bin/env python3
"""Adapt single-frame O2-RTDETR checkpoint for PairRotatedRTDETR loading."""
from __future__ import annotations

import copy
import os.path as osp
from typing import Dict, Iterable, Tuple

import torch


def adapt_single_frame_ckpt_for_pair(
    state_dict: Dict[str, torch.Tensor],
    copy_cls_branches_curr: bool = False,
) -> Tuple[Dict[str, torch.Tensor], Dict[str, int]]:
    """Map single-frame RT-DETR weights onto Pair model keys.

    - ``decoder.layers.*.cross_attn.*`` -> ``cross_attn_prev`` + ``cross_attn_curr``
    - ``bbox_head.reg_branches.*`` -> ``reg_branches_curr`` (curr-frame decoder reg)
    - optionally ``bbox_head.cls_branches.*`` -> ``cls_branches_curr`` for
      dual-cls pair heads
    - Drop ``dn_query_generator.*`` (pair overfit uses ``dn_cfg=None``)
    """
    adapted: Dict[str, torch.Tensor] = {}
    stats = {
        'copied': 0,
        'cross_attn_expanded': 0,
        'reg_branches_curr_copied': 0,
        'cls_branches_curr_copied': 0,
        'dropped_dn': 0,
        'dropped_unmatched': 0,
    }

    for key, value in state_dict.items():
        if key.startswith('dn_query_generator.'):
            stats['dropped_dn'] += 1
            continue

        if key.startswith('bbox_head.reg_branches.'):
            curr_key = key.replace(
                'bbox_head.reg_branches.', 'bbox_head.reg_branches_curr.', 1)
            adapted[curr_key] = copy.deepcopy(value)
            stats['reg_branches_curr_copied'] += 1

        if copy_cls_branches_curr and key.startswith('bbox_head.cls_branches.'):
            curr_key = key.replace(
                'bbox_head.cls_branches.', 'bbox_head.cls_branches_curr.', 1)
            adapted[curr_key] = copy.deepcopy(value)
            stats['cls_branches_curr_copied'] += 1

        cross_marker = '.cross_attn.'
        if cross_marker in key and '.cross_attn_prev.' not in key:
            prefix, suffix = key.split(cross_marker, 1)
            prev_key = f'{prefix}.cross_attn_prev.{suffix}'
            curr_key = f'{prefix}.cross_attn_curr.{suffix}'
            adapted[prev_key] = value
            adapted[curr_key] = copy.deepcopy(value)
            stats['cross_attn_expanded'] += 1
            continue

        adapted[key] = value
        stats['copied'] += 1

    return adapted, stats


def ensure_pair_adapted_checkpoint(
    src_ckpt: str,
    cache_dir: str,
    force: bool = False,
    copy_cls_branches_curr: bool = False,
    output_name: str = 'pair_adapted_pretrain.pth',
) -> str:
    """Build (or reuse) a pair-adapted checkpoint under ``cache_dir``."""
    src_ckpt = osp.abspath(src_ckpt)
    cache_dir = osp.abspath(cache_dir)
    dst_ckpt = osp.join(cache_dir, output_name)
    if (not force and osp.isfile(dst_ckpt)
            and osp.getmtime(dst_ckpt) >= osp.getmtime(src_ckpt)):
        return dst_ckpt

    checkpoint = torch.load(src_ckpt, map_location='cpu')
    state_dict = checkpoint.get('state_dict', checkpoint)
    adapted_sd, stats = adapt_single_frame_ckpt_for_pair(
        state_dict, copy_cls_branches_curr=copy_cls_branches_curr)

    out = dict(checkpoint) if isinstance(checkpoint, dict) else {}
    out['state_dict'] = adapted_sd
    out['pair_pretrain_meta'] = {
        'source_checkpoint': src_ckpt,
        **stats,
    }

    import os
    os.makedirs(cache_dir, exist_ok=True)
    torch.save(out, dst_ckpt)
    print(
        f'Adapted pair pretrain checkpoint: {dst_ckpt}\n'
        f'  source: {src_ckpt}\n'
        f'  copied={stats["copied"]} cross_attn_expanded='
        f'{stats["cross_attn_expanded"]} reg_branches_curr_copied='
        f'{stats["reg_branches_curr_copied"]} cls_branches_curr_copied='
        f'{stats["cls_branches_curr_copied"]} dropped_dn='
        f'{stats["dropped_dn"]}')
    return dst_ckpt
