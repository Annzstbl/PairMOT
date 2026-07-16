#!/usr/bin/env python3
"""Adapt single-frame O2-RTDETR checkpoint for PairRotatedRTDETR loading."""
from __future__ import annotations

import copy
import json
import os
import os.path as osp
from collections import OrderedDict
from typing import Dict, Iterable, Optional, Tuple

import torch


def adapt_single_frame_ckpt_for_pair(
    state_dict: Dict[str, torch.Tensor],
    copy_cls_branches_curr: bool = False,
    copy_dn_query_generator_to_pair: bool = False,
) -> Tuple[Dict[str, torch.Tensor], Dict[str, int]]:
    """Map single-frame RT-DETR weights onto Pair model keys.

    - ``decoder.layers.*.cross_attn.*`` -> ``cross_attn_prev`` + ``cross_attn_curr``
    - ``bbox_head.reg_branches.*`` -> ``reg_branches_curr`` (curr-frame decoder reg)
    - optionally ``bbox_head.cls_branches.*`` -> ``cls_branches_curr`` for
      dual-cls pair heads
    - Drop ``dn_query_generator.*`` by default, or copy it to
      ``pair_dn_query_generator.*`` when PairDN is enabled.
    """
    adapted: Dict[str, torch.Tensor] = {}
    stats = {
        'copied': 0,
        'cross_attn_expanded': 0,
        'reg_branches_curr_copied': 0,
        'cls_branches_curr_copied': 0,
        'dropped_dn': 0,
        'pair_dn_copied': 0,
        'dropped_unmatched': 0,
    }

    for key, value in state_dict.items():
        if key.startswith('dn_query_generator.'):
            if copy_dn_query_generator_to_pair:
                pair_dn_key = key.replace(
                    'dn_query_generator.', 'pair_dn_query_generator.', 1)
                adapted[pair_dn_key] = copy.deepcopy(value)
                stats['pair_dn_copied'] += 1
                continue
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
    copy_dn_query_generator_to_pair: bool = False,
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
        state_dict,
        copy_cls_branches_curr=copy_cls_branches_curr,
        copy_dn_query_generator_to_pair=copy_dn_query_generator_to_pair)

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
        f'{stats["dropped_dn"]} pair_dn_copied='
        f'{stats["pair_dn_copied"]}')
    return dst_ckpt


def _load_pretrain_state_dict(src_ckpt: str) -> Tuple[Dict[str, torch.Tensor], str]:
    checkpoint = torch.load(src_ckpt, map_location='cpu')
    if isinstance(checkpoint, dict):
        if 'state_dict' in checkpoint:
            return checkpoint['state_dict'], 'state_dict'
        if 'ema' in checkpoint and isinstance(checkpoint['ema'], dict):
            ema = checkpoint['ema']
            if 'module' in ema:
                return ema['module'], 'ema.module'
        if 'model' in checkpoint:
            return checkpoint['model'], 'model'
    return checkpoint, 'checkpoint'


def _map_norm(src_key: str, target_prefix: str) -> str:
    suffix = src_key.rsplit('.norm.', 1)[1]
    return f'{target_prefix}.{suffix}'


def _map_bn(src_key: str, target_prefix: str) -> str:
    suffix = src_key.rsplit('.norm.', 1)[1]
    return f'{target_prefix}.bn.{suffix}'


def _convert_rtdetr_resnet_backbone(
        state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """Convert official RT-DETR Paddle-style ResNet keys to MMRotate.

    This handles both BasicBlock (R18/R34) and Bottleneck (R50+) variants.
    Blocks and stages are discovered from the checkpoint instead of being
    constrained to a specific backbone depth.
    """
    converted: Dict[str, torch.Tensor] = OrderedDict()
    stem_map = {
        'backbone.conv1.conv1_1': ('backbone.stem.0', 'backbone.stem.1'),
        'backbone.conv1.conv1_2': ('backbone.stem.3', 'backbone.stem.4'),
        'backbone.conv1.conv1_3': ('backbone.stem.6', 'backbone.stem.7'),
    }
    is_bottleneck = any('.branch2c.' in key for key in state_dict)
    for key, value in state_dict.items():
        for src_prefix, (conv_prefix, norm_prefix) in stem_map.items():
            if key == f'{src_prefix}.conv.weight':
                converted[f'{conv_prefix}.weight'] = value
            elif key.startswith(f'{src_prefix}.norm.'):
                converted[_map_norm(key, norm_prefix)] = value

    for key, value in state_dict.items():
        if not key.startswith('backbone.res_layers.'):
            continue
        parts = key.split('.')
        stage = int(parts[2])
        block = int(parts[4])
        layer_prefix = f'backbone.layer{stage + 1}.{block}'
        if '.branch2a.conv.weight' in key:
            converted[f'{layer_prefix}.conv1.weight'] = value
        elif '.branch2a.norm.' in key:
            converted[_map_norm(key, f'{layer_prefix}.bn1')] = value
        elif '.branch2b.conv.weight' in key:
            converted[f'{layer_prefix}.conv2.weight'] = value
        elif '.branch2b.norm.' in key:
            converted[_map_norm(key, f'{layer_prefix}.bn2')] = value
        elif '.branch2c.conv.weight' in key:
            converted[f'{layer_prefix}.conv3.weight'] = value
        elif '.branch2c.norm.' in key:
            converted[_map_norm(key, f'{layer_prefix}.bn3')] = value

        if is_bottleneck or stage > 0:
            down_conv_idx, down_norm_idx = 1, 2
        else:
            down_conv_idx, down_norm_idx = 0, 1
        if '.short.conv.weight' in key or '.short.conv.conv.weight' in key:
            converted[
                f'{layer_prefix}.downsample.{down_conv_idx}.weight'] = value
        elif '.short.norm.' in key or '.short.conv.norm.' in key:
            converted[_map_norm(
                key, f'{layer_prefix}.downsample.{down_norm_idx}')] = value
    return converted


def _convert_hybrid_encoder_keys(
        state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    converted: Dict[str, torch.Tensor] = OrderedDict()
    for key, value in state_dict.items():
        new_key: Optional[str] = None
        if key.startswith('encoder.input_proj.'):
            parts = key.split('.')
            idx = parts[2]
            if parts[3] == '0' and parts[4] == 'weight':
                new_key = f'neck.convs.{idx}.conv.weight'
            elif parts[3] == '1':
                suffix = '.'.join(parts[4:])
                new_key = f'neck.convs.{idx}.bn.{suffix}'
        elif key.startswith('encoder.encoder.0.layers.'):
            rest = key.replace('encoder.encoder.0.layers.',
                               'encoder.transformer_blocks.0.layers.', 1)
            rest = rest.replace('.self_attn.', '.self_attn.attn.')
            rest = rest.replace('.linear1.', '.ffn.layers.0.0.')
            rest = rest.replace('.linear2.', '.ffn.layers.1.')
            rest = rest.replace('.norm1.', '.norms.0.')
            rest = rest.replace('.norm2.', '.norms.1.')
            new_key = rest
        elif key.startswith('encoder.lateral_convs.'):
            parts = key.split('.')
            idx = parts[2]
            if '.conv.weight' in key:
                new_key = f'encoder.fpn.reduce_layers.{idx}.conv.weight'
            elif '.norm.' in key:
                new_key = _map_bn(key, f'encoder.fpn.reduce_layers.{idx}')
        elif key.startswith('encoder.downsample_convs.'):
            parts = key.split('.')
            idx = parts[2]
            if '.conv.weight' in key:
                new_key = f'encoder.fpn.downsamples.{idx}.conv.weight'
            elif '.norm.' in key:
                new_key = _map_bn(key, f'encoder.fpn.downsamples.{idx}')
        elif key.startswith('encoder.output_convs.'):
            parts = key.split('.')
            idx = parts[2]
            if '.conv.weight' in key:
                new_key = f'encoder.fpn.out_convs.{idx}.conv.weight'
            elif '.norm.' in key:
                new_key = _map_bn(key, f'encoder.fpn.out_convs.{idx}')
        elif (key.startswith('encoder.fpn_blocks.')
              or key.startswith('encoder.pan_blocks.')):
            src_top = 'encoder.fpn_blocks.'
            tgt_top = 'encoder.fpn.top_down_blocks.'
            if key.startswith('encoder.pan_blocks.'):
                src_top = 'encoder.pan_blocks.'
                tgt_top = 'encoder.fpn.bottom_up_blocks.'
            rest = key.replace(src_top, tgt_top, 1)
            rest = rest.replace('.bottlenecks.', '.blocks.')
            if '.blocks.' in rest:
                rest = rest.replace('.conv1.', '.rbr_dense.')
                rest = rest.replace('.conv2.', '.rbr_1x1.')
            else:
                rest = rest.replace('.conv1.', '.main_conv.')
                rest = rest.replace('.conv2.', '.short_conv.')
                rest = rest.replace('.conv3.', '.final_conv.')
            if '.conv.weight' in rest:
                new_key = rest
            elif '.norm.' in rest:
                new_key = rest.replace('.norm.', '.bn.')
        if new_key is not None:
            converted[new_key] = value
    return converted


def _convert_mlp_key(src_key: str, src_prefix: str,
                     target_prefix: str) -> Optional[str]:
    parts = src_key[len(src_prefix):].split('.')
    if len(parts) != 3 or parts[0] != 'layers':
        return None
    layer_map = {'0': '0', '1': '2', '2': '4'}
    target_layer = layer_map.get(parts[1])
    if target_layer is None:
        return None
    return f'{target_prefix}.{target_layer}.{parts[2]}'


def _convert_decoder_and_head_keys(
        state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    converted: Dict[str, torch.Tensor] = OrderedDict()
    decoder_layers = {
        int(key.split('.')[3])
        for key in state_dict
        if key.startswith('decoder.decoder.layers.')
    }
    encoder_branch_idx = max(decoder_layers) + 1 if decoder_layers else 0
    for key, value in state_dict.items():
        if key.startswith('decoder.decoder.layers.'):
            new_key = key.replace('decoder.decoder.layers.', 'decoder.layers.', 1)
            new_key = new_key.replace('.self_attn.', '.self_attn.attn.')
            new_key = new_key.replace('.linear1.', '.ffn.layers.0.0.')
            new_key = new_key.replace('.linear2.', '.ffn.layers.1.')
            new_key = new_key.replace('.norm1.', '.norms.0.')
            new_key = new_key.replace('.norm2.', '.norms.1.')
            new_key = new_key.replace('.norm3.', '.norms.2.')
            if '.cross_attn.' in new_key:
                prev_key = new_key.replace('.cross_attn.', '.cross_attn_prev.')
                curr_key = new_key.replace('.cross_attn.', '.cross_attn_curr.')
                converted[prev_key] = value
                converted[curr_key] = copy.deepcopy(value)
            else:
                converted[new_key] = value
        elif key.startswith('decoder.dec_bbox_head.'):
            parts = key.split('.')
            layer = parts[2]
            target_key = _convert_mlp_key(
                key, f'decoder.dec_bbox_head.{layer}.',
                f'bbox_head.reg_branches.{layer}')
            if target_key is None:
                continue
            converted[target_key] = value
            converted[target_key.replace(
                'reg_branches.', 'reg_branches_curr.')] = (
                copy.deepcopy(value))
        elif key.startswith('decoder.enc_bbox_head.'):
            target_key = _convert_mlp_key(
                key, 'decoder.enc_bbox_head.',
                f'bbox_head.reg_branches.{encoder_branch_idx}')
            if target_key is None:
                continue
            converted[target_key] = value
            converted[target_key.replace(
                'reg_branches.', 'reg_branches_curr.')] = copy.deepcopy(value)
        elif key.startswith('decoder.enc_output.0.'):
            converted[key.replace('decoder.enc_output.0.', 'memory_trans_fc.')] = value
        elif key.startswith('decoder.enc_output.1.'):
            converted[key.replace('decoder.enc_output.1.', 'memory_trans_norm.')] = value
        elif key.startswith('decoder.query_pos_head.'):
            converted[key.replace('decoder.query_pos_head.',
                                  'decoder.ref_point_head.')] = value
    return converted


def _adapt_2d_to_3d_weight(src: torch.Tensor,
                           target_shape: torch.Size) -> Optional[torch.Tensor]:
    if src.ndim != 4 or len(target_shape) != 5:
        return None
    out_c, in_c, depth, kh, kw = target_shape
    if src.shape[0] != out_c or src.shape[2:] != (kh, kw):
        return None
    # The project stem replaces Conv2d(3, C, 3) with Conv3d(1, C, 3, 3, 3).
    # Preserve each RGB filter along the spectral kernel axis exactly. This is
    # also the mapping used by ``convert_stem_conv2d_to_conv3d_weight``.
    if in_c == 1 and src.shape[1] == depth:
        return src.unsqueeze(1).contiguous()
    if src.shape[1] == in_c:
        return src.unsqueeze(2).repeat(1, 1, depth, 1, 1) / depth
    base = src.mean(dim=1, keepdim=True).repeat(1, in_c, 1, 1)
    scale = src.shape[1] / (in_c * depth)
    return base.unsqueeze(2).repeat(1, 1, depth, 1, 1) * scale


def _partial_copy(src: torch.Tensor, dst: torch.Tensor) -> Optional[torch.Tensor]:
    if src.ndim != dst.ndim:
        return None
    out = dst.clone()
    slices = tuple(slice(0, min(a, b)) for a, b in zip(src.shape, dst.shape))
    out[slices] = src[slices]
    return out


def build_pair_adapted_rtdetr_state_dict(
    src_ckpt: str,
    target_state_dict: Dict[str, torch.Tensor],
) -> Tuple[Dict[str, torch.Tensor], Dict[str, object]]:
    src_sd, source_key = _load_pretrain_state_dict(src_ckpt)
    candidates: Dict[str, torch.Tensor] = OrderedDict()
    candidates.update(_convert_rtdetr_resnet_backbone(src_sd))
    candidates.update(_convert_hybrid_encoder_keys(src_sd))
    candidates.update(_convert_decoder_and_head_keys(src_sd))

    adapted: Dict[str, torch.Tensor] = OrderedDict()
    stats = {
        'source_state_key': source_key,
        'source_tensors': len(src_sd),
        'candidate_tensors': len(candidates),
        'exact_copied': 0,
        'conv3d_stem_adapted': 0,
        'partial_copied': 0,
        'shape_mismatch_skipped': 0,
        'missing_in_target_skipped': 0,
        'adapted_conv3d_keys': [],
        'partial_keys': [],
        'shape_mismatch_keys': [],
        'missing_in_target_keys': [],
        'target_only_tensors': 0,
        'target_only_keys': [],
    }
    for key, value in candidates.items():
        target_key = key
        if key == 'backbone.stem.0.weight':
            target_key = 'backbone.stem.0.conv3d.weight'
        if target_key not in target_state_dict:
            stats['missing_in_target_skipped'] += 1
            stats['missing_in_target_keys'].append(target_key)
            continue
        target = target_state_dict[target_key]
        if tuple(value.shape) == tuple(target.shape):
            adapted[target_key] = value
            stats['exact_copied'] += 1
            continue
        conv3d = _adapt_2d_to_3d_weight(value, target.shape)
        if conv3d is not None:
            adapted[target_key] = conv3d
            stats['conv3d_stem_adapted'] += 1
            stats['adapted_conv3d_keys'].append(target_key)
            continue
        if (target_key.startswith('bbox_head.reg_branches')
                or target_key.startswith('bbox_head.reg_branches_curr')
                or target_key.startswith('decoder.ref_point_head')):
            partial = _partial_copy(value, target)
            if partial is not None:
                adapted[target_key] = partial
                stats['partial_copied'] += 1
                stats['partial_keys'].append(target_key)
                continue
        stats['shape_mismatch_skipped'] += 1
        stats['shape_mismatch_keys'].append({
            'key': target_key,
            'source_shape': list(value.shape),
            'target_shape': list(target.shape),
        })
    target_only = sorted(set(target_state_dict) - set(adapted))
    stats['target_only_tensors'] = len(target_only)
    stats['target_only_keys'] = target_only
    return adapted, stats


def build_pair_adapted_coco365_state_dict(
    src_ckpt: str,
    target_state_dict: Dict[str, torch.Tensor],
) -> Tuple[Dict[str, torch.Tensor], Dict[str, object]]:
    """Backward-compatible alias for the generalized RT-DETR adapter."""
    return build_pair_adapted_rtdetr_state_dict(src_ckpt, target_state_dict)


def ensure_rtdetr_pair_adapted_checkpoint(
    src_ckpt: str,
    target_config: str,
    cache_dir: str,
    force: bool = False,
    output_name: str = 'pair_coco_adapted_pretrain.pth',
) -> str:
    src_ckpt = osp.abspath(src_ckpt)
    target_config = osp.abspath(target_config)
    cache_dir = osp.abspath(cache_dir)
    dst_ckpt = osp.join(cache_dir, output_name)
    stats_path = dst_ckpt + '.json'
    if (not force and osp.isfile(dst_ckpt)
            and osp.getmtime(dst_ckpt) >= osp.getmtime(src_ckpt)
            and osp.getmtime(dst_ckpt) >= osp.getmtime(target_config)):
        return dst_ckpt

    from mmengine.config import Config
    from mmengine.registry import DefaultScope
    from mmengine.utils import import_modules_from_strings
    from mmdet.utils import register_all_modules as register_mmdet_modules
    from mmrotate.registry import MODELS
    from mmrotate.utils import register_all_modules as register_mmrotate_modules

    DefaultScope.get_instance('pair_rtdetr_adapt', scope_name='mmrotate')
    register_mmdet_modules(init_default_scope=False)
    register_mmrotate_modules(init_default_scope=False)
    cfg = Config.fromfile(target_config)
    if cfg.get('custom_imports', None):
        import_modules_from_strings(**cfg.custom_imports)
    target_sd = MODELS.build(cfg.model).state_dict()
    adapted_sd, stats = build_pair_adapted_rtdetr_state_dict(
        src_ckpt, target_sd)
    stats.update({
        'source_checkpoint': src_ckpt,
        'target_config': target_config,
        'target_tensors': len(target_sd),
        'output_checkpoint': dst_ckpt,
    })
    out = {
        'state_dict': adapted_sd,
        'meta': {
            'source_checkpoint': src_ckpt,
            'target_config': target_config,
            'note': (
                'Official RT-DETR checkpoint adapted to the requested PairMOT '
                'target architecture. Target-only extension parameters retain '
                'their model initialization. Class logits are skipped when '
                'class dimensions differ.'),
        },
        'pair_pretrain_meta': stats,
    }
    os.makedirs(cache_dir, exist_ok=True)
    torch.save(out, dst_ckpt)
    with open(stats_path, 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=2, sort_keys=True)
    print(
        f'Adapted RT-DETR pair pretrain checkpoint: {dst_ckpt}\n'
        f'  source: {src_ckpt}\n'
        f'  target: {target_config}\n'
        f'  exact={stats["exact_copied"]} conv3d='
        f'{stats["conv3d_stem_adapted"]} partial='
        f'{stats["partial_copied"]} missing='
        f'{stats["missing_in_target_skipped"]} shape_skip='
        f'{stats["shape_mismatch_skipped"]}\n'
        f'  stats: {stats_path}')
    return dst_ckpt


def ensure_coco365_pair_adapted_checkpoint(
    src_ckpt: str,
    target_config: str,
    cache_dir: str,
    force: bool = False,
    output_name: str = 'pair_coco365_full_adapted_pretrain.pth',
) -> str:
    """Backward-compatible wrapper for existing COCO+Objects365 configs."""
    return ensure_rtdetr_pair_adapted_checkpoint(
        src_ckpt=src_ckpt,
        target_config=target_config,
        cache_dir=cache_dir,
        force=force,
        output_name=output_name)
