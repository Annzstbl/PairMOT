#!/usr/bin/env python3
"""Visualize pair proposal initialization for HSMOT test first-frame pairs."""

from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import os.path as osp
import sys
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

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
from projects.multispec_pair_rotated_rtdetr.tools.run_pair_mot import (  # noqa: E402
    _frame_ids_from_images,
    _make_pair_info,
    _sequence_list,
)


DEFAULT_EXPERIMENTS = {
    'baseline': {
        'dir': '/data/users/litianhao01/PairMmot/workdir/'
        '0702_o2_pair_rtdetr_r18vd_2xb4_72e_hsmot_half_pairdn_gap1train_bothvis_dualcls_nopres',
        'config': '/data/users/litianhao01/PairMmot/ai4rs/projects/'
        'multispec_pair_rotated_rtdetr/configs/'
        'o2_pair_rtdetr_r18vd_2xb4_72e_hsmot_pairdn_gap1train_bothvis_dualcls_nopres.py',
        'checkpoint': 'latest',
    },
    'pairtopk_v2': {
        'dir': '/data/users/litianhao01/PairMmot/workdir/'
        '0702_o2_pair_rtdetr_r18vd_2xb4_72e_hsmot_half_pairdn_gap1train_bothvis_dualcls_nopres_pairtopk_v2',
        'config': '/data/users/litianhao01/PairMmot/ai4rs/projects/'
        'multispec_pair_rotated_rtdetr/configs/'
        'o2_pair_rtdetr_r18vd_2xb4_72e_hsmot_pairdn_gap1train_bothvis_dualcls_nopres_pairtopk_v2.py',
        'checkpoint': 'latest',
    },
    'pairtopk_v2_unique': {
        'dir': '/data/users/litianhao01/PairMmot/workdir/'
        '0702_o2_pair_rtdetr_r18vd_2xb4_72e_hsmot_half_pairdn_gap1train_bothvis_dualcls_nopres_pairtopk_v2_unique',
        'config': '/data/users/litianhao01/PairMmot/ai4rs/projects/'
        'multispec_pair_rotated_rtdetr/configs/'
        'o2_pair_rtdetr_r18vd_2xb4_72e_hsmot_pairdn_gap1train_bothvis_dualcls_nopres_pairtopk_v2_unique.py',
        'checkpoint': 'latest',
    },
}


@dataclass
class ProposalDiag:
    exp_name: str
    query_init: str
    seq_name: str
    prev_frame_id: int
    curr_frame_id: int
    prev_img_path: str
    curr_img_path: str
    candidate_prev: List[dict]
    candidate_curr: List[dict]
    candidate_pairs: List[dict]
    topk_pairs: List[dict]
    meta: dict


def _resolve_path(exp_cfg: dict, key: str) -> str:
    value = exp_cfg[key]
    if osp.isabs(value):
        return value
    return osp.join(exp_cfg['dir'], value)


def _latest_checkpoint(exp_dir: str) -> str:
    ckpts = glob.glob(osp.join(exp_dir, 'epoch_*.pth'))
    if not ckpts:
        raise FileNotFoundError(f'No epoch_*.pth checkpoints found in {exp_dir}')

    def epoch_num(path: str) -> int:
        name = osp.splitext(osp.basename(path))[0]
        try:
            return int(name.rsplit('_', 1)[1])
        except (IndexError, ValueError):
            return -1

    return max(ckpts, key=epoch_num)


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
    return data['inputs'].to(device), data['data_samples']


def _encoder_memory(model, batch_inputs: torch.Tensor, data_samples: list):
    img_feats = model.extract_feat(batch_inputs)
    encoder_inputs, decoder_inputs = model.pre_transformer(
        img_feats, data_samples)
    encoder_outputs = model.forward_encoder(**encoder_inputs)
    memory = encoder_outputs['memory']
    pair_batch, _ = model._split_pair_batch(img_feats[0].shape[0])
    return (
        memory[:pair_batch],
        memory[pair_batch:],
        encoder_outputs['memory_mask'],
        encoder_outputs['spatial_shapes'],
        decoder_inputs,
    )


def _factor(meta: dict, angle_factor: float, device, dtype) -> torch.Tensor:
    img_h, img_w = meta['img_shape']
    return torch.tensor(
        [img_w, img_h, img_w, img_h, angle_factor],
        device=device,
        dtype=dtype)


def _scale_factor(meta: dict) -> np.ndarray:
    sf = meta.get('scale_factor', (1.0, 1.0))
    if isinstance(sf, torch.Tensor):
        sf = sf.detach().cpu().numpy()
    sf = np.asarray(sf, dtype=np.float32).reshape(-1)
    if sf.size == 1:
        sf = np.repeat(sf, 2)
    if sf.size >= 4:
        return np.asarray([sf[0], sf[1], sf[2], sf[3], 1.0], dtype=np.float32)
    return np.asarray([sf[0], sf[1], sf[0], sf[1], 1.0], dtype=np.float32)


def _refs_to_original(refs: torch.Tensor, meta: dict,
                      angle_factor: float) -> np.ndarray:
    if refs.numel() == 0:
        return np.zeros((0, 5), dtype=np.float32)
    boxes = refs.detach() * _factor(meta, angle_factor, refs.device, refs.dtype)
    scale = torch.tensor(
        _scale_factor(meta), device=boxes.device, dtype=boxes.dtype).clamp_min(1e-6)
    boxes = boxes / scale
    return boxes.detach().cpu().float().numpy()


def _rows_from_refs(refs: torch.Tensor, scores: torch.Tensor, meta: dict,
                    angle_factor: float, indices: Optional[torch.Tensor] = None,
                    extra: Optional[Dict[str, Sequence]] = None) -> List[dict]:
    boxes = _refs_to_original(refs, meta, angle_factor)
    scores_np = scores.detach().cpu().float().numpy().reshape(-1)
    rows = []
    for rank, (box, score) in enumerate(zip(boxes, scores_np)):
        row = {
            'rank': rank,
            'score': float(score),
            'rbox': [float(x) for x in box.tolist()],
        }
        if indices is not None:
            row['proposal_index'] = int(indices.detach().cpu().reshape(-1)[rank])
        if extra:
            for key, values in extra.items():
                value = values[rank]
                if isinstance(value, torch.Tensor):
                    value = value.detach().cpu().item()
                if isinstance(value, np.generic):
                    value = value.item()
                row[key] = value
        rows.append(row)
    return rows


def _baseline_diag(model, memory_prev, memory_curr, memory_mask, spatial_shapes,
                   meta: dict, draw_candidates: int) -> Tuple[List[dict], List[dict], List[dict]]:
    num_layers = model.decoder.num_layers
    c = memory_prev.shape[-1]
    output_memory_p, output_proposals_p = model.gen_encoder_output_proposals(
        memory_prev, memory_mask, spatial_shapes)
    enc_cls_p = model.bbox_head.cls_branches[num_layers](output_memory_p)
    score_p = enc_cls_p.max(-1)[0].sigmoid()
    k = min(max(model.num_queries, draw_candidates), output_memory_p.size(1))
    top_idx = torch.topk(score_p, k=k, dim=1).indices
    query = torch.gather(output_memory_p, 1, top_idx.unsqueeze(-1).repeat(1, 1, c))
    props_p = torch.gather(output_proposals_p, 1, top_idx.unsqueeze(-1).repeat(1, 1, 5))
    ref_p = (model.bbox_head.reg_branches[num_layers](query) + props_p).sigmoid()

    output_memory_c, output_proposals_c = model.gen_encoder_output_proposals(
        memory_curr, memory_mask, spatial_shapes)
    props_c = torch.gather(output_proposals_c, 1, top_idx.unsqueeze(-1).repeat(1, 1, 5))
    ref_c = (model.bbox_head.reg_branches_curr[num_layers](query) + props_c).sigmoid()

    scores = torch.gather(score_p, 1, top_idx)
    angle_factor = model.decoder.angle_factor
    candidates_prev = _rows_from_refs(
        ref_p[0, :draw_candidates], scores[0, :draw_candidates], meta,
        angle_factor, top_idx[0, :draw_candidates])
    candidates_curr = _rows_from_refs(
        ref_c[0, :draw_candidates], scores[0, :draw_candidates], meta,
        angle_factor, top_idx[0, :draw_candidates])
    top_n = min(model.num_queries, ref_p.size(1))
    top_pairs = []
    prev_rows = _rows_from_refs(
        ref_p[0, :top_n], scores[0, :top_n], meta, angle_factor,
        top_idx[0, :top_n])
    curr_rows = _rows_from_refs(
        ref_c[0, :top_n], scores[0, :top_n], meta, angle_factor,
        top_idx[0, :top_n])
    for rank, (prev, curr) in enumerate(zip(prev_rows, curr_rows)):
        top_pairs.append({
            'rank': rank,
            'type': 'aligned_prev_topk',
            'score': prev['score'],
            'proposal_index': prev.get('proposal_index'),
            'prev': prev['rbox'],
            'curr': curr['rbox'],
        })
    return candidates_prev, candidates_curr, top_pairs


def _pairtopk_diag(model, memory_prev, memory_curr, memory_mask, spatial_shapes,
                   meta: dict, draw_candidates: int) -> Tuple[List[dict], List[dict], List[dict]]:
    cfg = model.pair_proposal_cfg
    c = memory_prev.shape[-1]
    pre_topk = int(cfg.get('pre_topk', model.num_queries * 2))
    num_layers = model.decoder.num_layers
    query_p, ref_p, score_p, idx_p = model._single_frame_topk_proposals(
        memory_prev, memory_mask, spatial_shapes,
        model.bbox_head.reg_branches[num_layers], pre_topk)
    query_c, ref_c, score_c, idx_c = model._single_frame_topk_proposals(
        memory_curr, memory_mask, spatial_shapes,
        model.bbox_head.reg_branches_curr[num_layers], pre_topk)
    angle_factor = model.decoder.angle_factor
    candidates_prev = _rows_from_refs(
        ref_p[0, :draw_candidates], score_p[0, :draw_candidates], meta,
        angle_factor, idx_p[0, :draw_candidates])
    candidates_curr = _rows_from_refs(
        ref_c[0, :draw_candidates], score_c[0, :draw_candidates], meta,
        angle_factor, idx_c[0, :draw_candidates])

    pair_scores = model._pair_match_score(
        query_p[0], query_c[0], ref_p[0], ref_c[0], score_p[0], score_c[0])
    best_scores, best_curr = pair_scores.max(dim=1)
    order = torch.argsort(best_scores, descending=True)
    match_score_thr = float(cfg.get('match_score_thr', 0.0))
    birth_score_thr = float(cfg.get('birth_score_thr', 0.35))
    death_score_thr = float(cfg.get('death_score_thr', 0.35))
    enable_birth = bool(cfg.get('enable_birth', True))
    enable_death = bool(cfg.get('enable_death', True))
    used_prev = torch.zeros(query_p.size(1), dtype=torch.bool, device=query_p.device)
    used_curr = torch.zeros(query_c.size(1), dtype=torch.bool, device=query_c.device)
    top_pairs = []

    def add_pair(kind: str, score, pi: Optional[int], ci: Optional[int],
                 prev_ref: torch.Tensor, curr_ref: torch.Tensor) -> None:
        prev_box = _refs_to_original(prev_ref[None], meta, angle_factor)[0]
        curr_box = _refs_to_original(curr_ref[None], meta, angle_factor)[0]
        top_pairs.append({
            'rank': len(top_pairs),
            'type': kind,
            'score': float(score.detach().cpu().item() if isinstance(score, torch.Tensor) else score),
            'prev_candidate_rank': pi,
            'curr_candidate_rank': ci,
            'prev_proposal_index': int(idx_p[0, pi].detach().cpu()) if pi is not None else None,
            'curr_proposal_index': int(idx_c[0, ci].detach().cpu()) if ci is not None else None,
            'prev': [float(x) for x in prev_box.tolist()],
            'curr': [float(x) for x in curr_box.tolist()],
        })

    for pi_t in order:
        pi = int(pi_t.detach().cpu())
        score = best_scores[pi_t]
        if score <= match_score_thr or score < -1e5:
            break
        ci_t = best_curr[pi_t]
        ci = int(ci_t.detach().cpu())
        if used_prev[pi_t] or used_curr[ci_t]:
            continue
        add_pair('matched', score, pi, ci, ref_p[0, pi_t], ref_c[0, ci_t])
        used_prev[pi_t] = True
        used_curr[ci_t] = True
        if len(top_pairs) >= model.num_queries:
            break

    if enable_birth and len(top_pairs) < model.num_queries:
        for ci_t in torch.argsort(score_c[0], descending=True):
            if used_curr[ci_t] or score_c[0, ci_t] < birth_score_thr:
                continue
            ci = int(ci_t.detach().cpu())
            ref_prev = model.decoder.ref_prev_embedding.weight.sigmoid()[len(top_pairs)].to(ref_c.device)
            add_pair('birth', score_c[0, ci_t], None, ci, ref_prev, ref_c[0, ci_t])
            used_curr[ci_t] = True
            if len(top_pairs) >= model.num_queries:
                break

    if enable_death and len(top_pairs) < model.num_queries:
        for pi_t in torch.argsort(score_p[0], descending=True):
            if used_prev[pi_t] or score_p[0, pi_t] < death_score_thr:
                continue
            pi = int(pi_t.detach().cpu())
            ref_curr = model.decoder.ref_curr_embedding.weight.sigmoid()[len(top_pairs)].to(ref_p.device)
            add_pair('death', score_p[0, pi_t], pi, None, ref_p[0, pi_t], ref_curr)
            used_prev[pi_t] = True
            if len(top_pairs) >= model.num_queries:
                break

    if len(top_pairs) < model.num_queries:
        learned_prev = model.decoder.ref_prev_embedding.weight.sigmoid()
        learned_curr = model.decoder.ref_curr_embedding.weight.sigmoid()
        while len(top_pairs) < model.num_queries:
            qi = len(top_pairs)
            add_pair('learned_pad', 0.0, None, None,
                     learned_prev[qi].to(memory_prev.device),
                     learned_curr[qi].to(memory_prev.device))
    return candidates_prev, candidates_curr, top_pairs


def _pairtopk_v2_diag(model, memory_prev, memory_curr, memory_mask,
                      spatial_shapes, data_samples: list, meta: dict,
                      draw_candidates: int) -> Tuple[List[dict], List[dict], List[dict]]:
    cfg = model.pair_proposal_cfg
    pre_topk = int(cfg.get('pre_topk', model.num_queries * 3))
    candidate_topk = int(cfg.get('candidate_topk', model.num_queries * 6))
    affinity_thr = float(cfg.get('affinity_thr', 0.25))
    proposal_quality_weight = float(cfg.get('proposal_quality_weight', 0.70))
    learned_quality_weight = float(cfg.get('learned_quality_weight', 0.20))
    affinity_rank_weight = float(cfg.get('affinity_rank_weight', 0.10))
    unique_pair_selection = bool(cfg.get('unique_pair_selection', False))
    pair_selection_mode = str(cfg.get('pair_selection_mode', 'rank'))
    num_layers = model.decoder.num_layers
    cls_prev_branch = model.bbox_head.cls_branches[num_layers]
    cls_curr_branch = getattr(model.bbox_head, 'cls_branches_curr',
                              model.bbox_head.cls_branches)[num_layers]
    query_p, ref_p, score_p, label_p, _, idx_p = (
        model._single_frame_topk_proposals_v2(
            memory_prev, memory_mask, spatial_shapes, cls_prev_branch,
            model.bbox_head.reg_branches[num_layers], pre_topk))
    query_c, ref_c, score_c, label_c, _, idx_c = (
        model._single_frame_topk_proposals_v2(
            memory_curr, memory_mask, spatial_shapes, cls_curr_branch,
            model.bbox_head.reg_branches_curr[num_layers], pre_topk))
    angle_factor = model.decoder.angle_factor
    candidates_prev = _rows_from_refs(
        ref_p[0, :draw_candidates], score_p[0, :draw_candidates], meta,
        angle_factor, idx_p[0, :draw_candidates], {
            'label': label_p[0, :draw_candidates],
        })
    candidates_curr = _rows_from_refs(
        ref_c[0, :draw_candidates], score_c[0, :draw_candidates], meta,
        angle_factor, idx_c[0, :draw_candidates], {
            'label': label_c[0, :draw_candidates],
        })

    img_shape = data_samples[0].metainfo.get('img_shape', (1, 1))
    gmc = model._as_gmc_tensor(
        data_samples, 0, memory_prev.device, memory_prev.dtype)
    affinity = model._pair_affinity_score_v2(
        query_p[0], query_c[0], ref_p[0], ref_c[0], score_p[0], score_c[0],
        label_p[0], label_c[0], gmc, img_shape)
    if pair_selection_mode == 'hungarian_affinity':
        pair_i, pair_j = model._hungarian_affinity_pairs(
            affinity, affinity_thr, candidate_topk)
    else:
        valid = affinity > affinity_thr
        pair_i, pair_j = torch.nonzero(valid, as_tuple=True)
        if pair_i.numel() == 0:
            flat_scores, flat_idx = torch.topk(
                affinity.reshape(-1), k=min(candidate_topk, affinity.numel()))
            keep = flat_scores > -1e5
            flat_idx = flat_idx[keep]
            pair_i = flat_idx // affinity.size(1)
            pair_j = flat_idx % affinity.size(1)
        if pair_i.numel() > candidate_topk:
            keep = torch.topk(affinity[pair_i, pair_j], k=candidate_topk).indices
            pair_i = pair_i[keep]
            pair_j = pair_j[keep]

    top_pairs = []
    if pair_i.numel() > 0:
        fused = model.pair_query_fusion(
            torch.cat([query_p[0, pair_i], query_c[0, pair_j]], dim=-1))
        prop_quality = torch.sqrt(
            score_p[0, pair_i].clamp(min=1e-6) *
            score_c[0, pair_j].clamp(min=1e-6))
        learned_quality = model.pair_quality_predictor(fused).squeeze(-1).sigmoid()
        aff = affinity[pair_i, pair_j].clamp(min=0.0)
        rank_score = (
            proposal_quality_weight * prop_quality +
            learned_quality_weight * learned_quality +
            affinity_rank_weight * aff)
        order = torch.argsort(rank_score, descending=True)
        if (pair_selection_mode != 'hungarian_affinity'
                and unique_pair_selection
                and order.numel() > model.num_queries):
            selected = []
            used_prev = set()
            used_curr = set()
            for idx in order.tolist():
                pi = int(pair_i[idx])
                pj = int(pair_j[idx])
                if pi in used_prev or pj in used_curr:
                    continue
                selected.append(idx)
                used_prev.add(pi)
                used_curr.add(pj)
                if len(selected) >= model.num_queries:
                    break
            if len(selected) < model.num_queries:
                selected_set = set(selected)
                for idx in order.tolist():
                    if idx not in selected_set:
                        selected.append(idx)
                        if len(selected) >= model.num_queries:
                            break
            order = order.new_tensor(selected)
        pair_i = pair_i[order]
        pair_j = pair_j[order]
        rank_score = rank_score[order]
        prop_quality = prop_quality[order]
        learned_quality = learned_quality[order]
        aff = aff[order]
        for rank in range(min(model.num_queries, pair_i.numel())):
            pi = int(pair_i[rank].detach().cpu())
            pj = int(pair_j[rank].detach().cpu())
            prev_box = _refs_to_original(ref_p[0, pi][None], meta, angle_factor)[0]
            curr_box = _refs_to_original(ref_c[0, pj][None], meta, angle_factor)[0]
            top_pairs.append({
                'rank': rank,
                'type': (
                    'v2_hungarian' if pair_selection_mode == 'hungarian_affinity'
                    else 'v2_unique' if unique_pair_selection else 'v2'),
                'score': float(rank_score[rank].detach().cpu().item()),
                'rank_score': float(rank_score[rank].detach().cpu().item()),
                'prop_quality': float(prop_quality[rank].detach().cpu().item()),
                'learned_quality': float(learned_quality[rank].detach().cpu().item()),
                'affinity': float(aff[rank].detach().cpu().item()),
                'prev_candidate_rank': pi,
                'curr_candidate_rank': pj,
                'prev_proposal_index': int(idx_p[0, pi].detach().cpu()),
                'curr_proposal_index': int(idx_c[0, pj].detach().cpu()),
                'prev_label': int(label_p[0, pi].detach().cpu()),
                'curr_label': int(label_c[0, pj].detach().cpu()),
                'prev_score': float(score_p[0, pi].detach().cpu()),
                'curr_score': float(score_c[0, pj].detach().cpu()),
                'prev': [float(x) for x in prev_box.tolist()],
                'curr': [float(x) for x in curr_box.tolist()],
            })

    learned_prev = model.decoder.ref_prev_embedding.weight.sigmoid()
    learned_curr = model.decoder.ref_curr_embedding.weight.sigmoid()
    while len(top_pairs) < model.num_queries:
        qi = len(top_pairs)
        prev_box = _refs_to_original(
            learned_prev[qi].to(memory_prev.device)[None], meta, angle_factor)[0]
        curr_box = _refs_to_original(
            learned_curr[qi].to(memory_prev.device)[None], meta, angle_factor)[0]
        top_pairs.append({
            'rank': qi,
            'type': 'learned_pad',
            'score': 0.0,
            'prev_candidate_rank': None,
            'curr_candidate_rank': None,
            'prev_proposal_index': None,
            'curr_proposal_index': None,
            'prev': [float(x) for x in prev_box.tolist()],
            'curr': [float(x) for x in curr_box.tolist()],
        })
    return candidates_prev, candidates_curr, top_pairs


def _sameidx_diag(model, memory_prev, memory_curr, memory_mask, spatial_shapes,
                  meta: dict, draw_candidates: int) -> Tuple[List[dict], List[dict], List[dict]]:
    cfg = model.pair_proposal_cfg
    num_layers = model.decoder.num_layers
    c = memory_prev.shape[-1]
    output_memory_p, output_proposals_p = model.gen_encoder_output_proposals(
        memory_prev, memory_mask, spatial_shapes)
    output_memory_c, output_proposals_c = model.gen_encoder_output_proposals(
        memory_curr, memory_mask, spatial_shapes)
    enc_cls_p = model.bbox_head.cls_branches[num_layers](output_memory_p)
    enc_cls_c = model.bbox_head.cls_branches[num_layers](output_memory_c)
    score_p = enc_cls_p.sigmoid().max(-1)[0]
    score_c = enc_cls_c.sigmoid().max(-1)[0]
    score_mode = str(cfg.get('sameidx_score_mode', 'sqrt'))
    if score_mode == 'prev':
        joint = score_p
    elif score_mode == 'mean':
        joint = 0.5 * (score_p + score_c)
    elif score_mode == 'min':
        joint = torch.minimum(score_p, score_c)
    else:
        joint = torch.sqrt(score_p.clamp(min=1e-6) * score_c.clamp(min=1e-6))
    k = min(max(model.num_queries, draw_candidates), output_memory_p.size(1))
    top_idx = torch.topk(joint, k=k, dim=1).indices
    query_p = torch.gather(output_memory_p, 1, top_idx.unsqueeze(-1).repeat(1, 1, c))
    query_c = torch.gather(output_memory_c, 1, top_idx.unsqueeze(-1).repeat(1, 1, c))
    props_p = torch.gather(output_proposals_p, 1, top_idx.unsqueeze(-1).repeat(1, 1, 5))
    props_c = torch.gather(output_proposals_c, 1, top_idx.unsqueeze(-1).repeat(1, 1, 5))
    query = model.pair_query_fusion(torch.cat([query_p, query_c], dim=-1))
    ref_source = str(cfg.get('sameidx_ref_source', 'frame'))
    if ref_source == 'fused':
        ref_p = (model.bbox_head.reg_branches[num_layers](query) + props_p).sigmoid()
        ref_c = (model.bbox_head.reg_branches_curr[num_layers](query) + props_c).sigmoid()
    else:
        ref_p = (model.bbox_head.reg_branches[num_layers](query_p) + props_p).sigmoid()
        ref_c = (model.bbox_head.reg_branches_curr[num_layers](query_c) + props_c).sigmoid()
    joint_scores = torch.gather(joint, 1, top_idx)
    angle_factor = model.decoder.angle_factor
    extra = {
        'prev_score': torch.gather(score_p, 1, top_idx)[0, :draw_candidates],
        'curr_score': torch.gather(score_c, 1, top_idx)[0, :draw_candidates],
    }
    candidates_prev = _rows_from_refs(
        ref_p[0, :draw_candidates], joint_scores[0, :draw_candidates], meta,
        angle_factor, top_idx[0, :draw_candidates], extra)
    candidates_curr = _rows_from_refs(
        ref_c[0, :draw_candidates], joint_scores[0, :draw_candidates], meta,
        angle_factor, top_idx[0, :draw_candidates], extra)
    top_pairs = []
    prev_rows = _rows_from_refs(
        ref_p[0, :model.num_queries], joint_scores[0, :model.num_queries],
        meta, angle_factor, top_idx[0, :model.num_queries], {
            'prev_score': torch.gather(score_p, 1, top_idx)[0, :model.num_queries],
            'curr_score': torch.gather(score_c, 1, top_idx)[0, :model.num_queries],
        })
    curr_rows = _rows_from_refs(
        ref_c[0, :model.num_queries], joint_scores[0, :model.num_queries],
        meta, angle_factor, top_idx[0, :model.num_queries])
    for rank, (prev, curr) in enumerate(zip(prev_rows, curr_rows)):
        top_pairs.append({
            'rank': rank,
            'type': f'sameidx_{score_mode}',
            'score': prev['score'],
            'prev_score': prev.get('prev_score'),
            'curr_score': prev.get('curr_score'),
            'proposal_index': prev.get('proposal_index'),
            'prev': prev['rbox'],
            'curr': curr['rbox'],
        })
    return candidates_prev, candidates_curr, top_pairs


def collect_diag(exp_name: str, model, preprocessor, pipeline, device,
                 pair_info: dict, draw_candidates: int) -> ProposalDiag:
    batch_inputs, data_samples = _prepare_pair(preprocessor, pipeline, pair_info, device)
    meta = data_samples[0].metainfo
    with torch.no_grad():
        memory_prev, memory_curr, memory_mask, spatial_shapes, _ = _encoder_memory(
            model, batch_inputs, data_samples)
        if model.query_init == 'dual_topk':
            cand_prev, cand_curr, top_pairs = _baseline_diag(
                model, memory_prev, memory_curr, memory_mask, spatial_shapes,
                meta, draw_candidates)
        elif model.query_init == 'pair_topk_v1':
            cand_prev, cand_curr, top_pairs = _pairtopk_diag(
                model, memory_prev, memory_curr, memory_mask, spatial_shapes,
                meta, draw_candidates)
        elif model.query_init == 'pair_topk_sameidx_v1':
            cand_prev, cand_curr, top_pairs = _sameidx_diag(
                model, memory_prev, memory_curr, memory_mask, spatial_shapes,
                meta, draw_candidates)
        elif model.query_init == 'pair_topk_v2':
            cand_prev, cand_curr, top_pairs = _pairtopk_v2_diag(
                model, memory_prev, memory_curr, memory_mask, spatial_shapes,
                data_samples, meta, draw_candidates)
        else:
            raise RuntimeError(f'Unsupported query_init for visualization: {model.query_init}')
    return ProposalDiag(
        exp_name=exp_name,
        query_init=model.query_init,
        seq_name=str(pair_info['seq_name']),
        prev_frame_id=int(pair_info['frame_id_prev']),
        curr_frame_id=int(pair_info['frame_id']),
        prev_img_path=str(pair_info['img_path_prev']),
        curr_img_path=str(pair_info['img_path']),
        candidate_prev=cand_prev,
        candidate_curr=cand_curr,
        candidate_pairs=top_pairs[:draw_candidates],
        topk_pairs=top_pairs,
        meta={
            'img_shape': list(meta.get('img_shape', [])),
            'ori_shape': list(meta.get('ori_shape', [])),
            'scale_factor': _scale_factor(meta).tolist(),
        },
    )


def _read_img(path: str) -> np.ndarray:
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(path)
    return img


def _rbox_to_poly(box: Sequence[float]) -> np.ndarray:
    tensor = torch.tensor(box, dtype=torch.float32).reshape(1, 5)
    return rbox2qbox(tensor).reshape(4, 2).cpu().numpy()


def _draw_box(img: np.ndarray, box: Sequence[float], color: Tuple[int, int, int],
              label: str, thickness: int = 1) -> Tuple[int, int]:
    poly = _rbox_to_poly(box)
    pts = np.round(poly).astype(np.int32).reshape(-1, 1, 2)
    cv2.polylines(img, [pts], True, color, thickness, cv2.LINE_AA)
    center = tuple(np.round(poly.mean(axis=0)).astype(np.int32).tolist())
    if label:
        x = int(np.clip(center[0], 0, img.shape[1] - 1))
        y = int(np.clip(center[1], 12, img.shape[0] - 1))
        cv2.putText(img, label, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.38,
                    color, 1, cv2.LINE_AA)
    return int(center[0]), int(center[1])


def _title_band(img: np.ndarray, title: str, lines: Sequence[str]) -> np.ndarray:
    h = 34 + 20 * len(lines)
    band = np.full((h, img.shape[1], 3), 245, dtype=np.uint8)
    cv2.putText(band, title, (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.62,
                (30, 30, 30), 2, cv2.LINE_AA)
    for i, line in enumerate(lines):
        cv2.putText(band, line, (10, 48 + i * 20), cv2.FONT_HERSHEY_SIMPLEX,
                    0.48, (60, 60, 60), 1, cv2.LINE_AA)
    return np.vstack([band, img])


def _side_by_side(prev: np.ndarray, curr: np.ndarray) -> np.ndarray:
    h = max(prev.shape[0], curr.shape[0])
    w = prev.shape[1] + curr.shape[1]
    canvas = np.full((h, w, 3), 255, dtype=np.uint8)
    canvas[:prev.shape[0], :prev.shape[1]] = prev
    canvas[:curr.shape[0], prev.shape[1]:prev.shape[1] + curr.shape[1]] = curr
    cv2.line(canvas, (prev.shape[1], 0), (prev.shape[1], h), (255, 255, 255), 3)
    return canvas


def draw_candidates(diag: ProposalDiag, out_path: str, limit: int) -> None:
    prev = _read_img(diag.prev_img_path)
    curr = _read_img(diag.curr_img_path)
    for row in diag.candidate_prev[:limit]:
        _draw_box(prev, row['rbox'], (160, 160, 160), '', 1)
    for row in diag.candidate_curr[:limit]:
        _draw_box(curr, row['rbox'], (160, 190, 255), '', 1)
    for row in diag.candidate_prev[:min(12, limit)]:
        _draw_box(prev, row['rbox'], (30, 220, 255),
                  f"p{row['rank']} {row['score']:.2f}", 2)
    for row in diag.candidate_curr[:min(12, limit)]:
        _draw_box(curr, row['rbox'], (255, 170, 40),
                  f"c{row['rank']} {row['score']:.2f}", 2)
    canvas = _side_by_side(prev, curr)
    canvas = _title_band(
        canvas,
        f"{diag.exp_name} candidates | {diag.seq_name} "
        f"{diag.prev_frame_id:06d}->{diag.curr_frame_id:06d}",
        [
            f"query_init={diag.query_init}  prev_candidates={len(diag.candidate_prev)} "
            f"curr_candidates={len(diag.candidate_curr)}  drawn={limit}",
            'gray: all drawn candidates; yellow/orange: highest ranked candidates',
        ])
    cv2.imwrite(out_path, canvas)


def draw_topk(diag: ProposalDiag, out_path: str, limit: int) -> None:
    prev = _read_img(diag.prev_img_path)
    curr = _read_img(diag.curr_img_path)
    canvas = _side_by_side(prev, curr)
    offset = prev.shape[1]
    palette = [
        (0, 255, 255), (0, 200, 0), (255, 128, 0), (255, 0, 255),
        (80, 180, 255), (180, 80, 255), (0, 120, 255), (120, 255, 0),
    ]
    type_counts: Dict[str, int] = {}
    for row in diag.topk_pairs[:limit]:
        color = palette[row['rank'] % len(palette)]
        label = f"{row['rank']} {row['score']:.2f}"
        if row.get('type') and row['type'] not in ('aligned_prev_topk',):
            label += f" {row['type'][:1]}"
        p_center = _draw_box(canvas, row['prev'], color, label, 2)
        curr_shift = list(row['curr'])
        curr_shift[0] += offset
        c_center = _draw_box(canvas, curr_shift, color, label, 2)
        cv2.line(canvas, p_center, c_center, color, 1, cv2.LINE_AA)
        type_counts[row.get('type', 'unknown')] = type_counts.get(row.get('type', 'unknown'), 0) + 1
    type_text = ', '.join(f'{k}:{v}' for k, v in sorted(type_counts.items()))
    canvas = _title_band(
        canvas,
        f"{diag.exp_name} final topk | {diag.seq_name} "
        f"{diag.prev_frame_id:06d}->{diag.curr_frame_id:06d}",
        [
            f"query_init={diag.query_init}  drawn_topk={min(limit, len(diag.topk_pairs))} "
            f"total_topk={len(diag.topk_pairs)}",
            f"pair types: {type_text}",
        ])
    cv2.imwrite(out_path, canvas)


def write_json(diag: ProposalDiag, out_path: str) -> None:
    payload = {
        'experiment': diag.exp_name,
        'query_init': diag.query_init,
        'seq_name': diag.seq_name,
        'prev_frame_id': diag.prev_frame_id,
        'curr_frame_id': diag.curr_frame_id,
        'prev_img_path': diag.prev_img_path,
        'curr_img_path': diag.curr_img_path,
        'meta': diag.meta,
        'candidate_prev': diag.candidate_prev,
        'candidate_curr': diag.candidate_curr,
        'topk_pairs': diag.topk_pairs,
    }
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(payload, f, indent=2)


def make_comparison(diags: Sequence[ProposalDiag], out_path: str,
                    topk_limit: int) -> None:
    rows = []
    tmp_paths = []
    tmp_dir = osp.join(osp.dirname(out_path), '_tmp')
    mkdir_or_exist(tmp_dir)
    for diag in diags:
        tmp_path = osp.join(tmp_dir, f'{diag.exp_name}_{diag.seq_name}.jpg')
        draw_topk(diag, tmp_path, topk_limit)
        img = cv2.imread(tmp_path, cv2.IMREAD_COLOR)
        if img is not None:
            rows.append(img)
            tmp_paths.append(tmp_path)
    if not rows:
        return
    width = max(img.shape[1] for img in rows)
    padded = []
    for img in rows:
        if img.shape[1] < width:
            pad = np.full((img.shape[0], width - img.shape[1], 3), 255, dtype=np.uint8)
            img = np.hstack([img, pad])
        padded.append(img)
    cv2.imwrite(out_path, np.vstack(padded))
    for path in tmp_paths:
        try:
            os.remove(path)
        except OSError:
            pass
    try:
        os.rmdir(tmp_dir)
    except OSError:
        pass


def _experiment_specs(args) -> Dict[str, dict]:
    specs = DEFAULT_EXPERIMENTS.copy()
    if args.experiment:
        selected = {}
        for item in args.experiment:
            name, exp_dir, config, checkpoint = item.split(',', 3)
            selected[name] = {'dir': exp_dir, 'config': config, 'checkpoint': checkpoint}
        specs = selected
    for name, spec in specs.items():
        spec['config_path'] = _resolve_path(spec, 'config')
        if spec['checkpoint'] == 'latest':
            spec['checkpoint_path'] = _latest_checkpoint(spec['dir'])
        else:
            spec['checkpoint_path'] = _resolve_path(spec, 'checkpoint')
        if not osp.isfile(spec['config_path']):
            raise FileNotFoundError(f'{name} config not found: {spec["config_path"]}')
        if not osp.isfile(spec['checkpoint_path']):
            raise FileNotFoundError(f'{name} checkpoint not found: {spec["checkpoint_path"]}')
    return specs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
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
        'proposal_vis_20260701')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--draw-candidates', type=int, default=80)
    parser.add_argument('--draw-topk', type=int, default=40)
    parser.add_argument('--max-seqs', type=int, default=0)
    parser.add_argument(
        '--experiment',
        action='append',
        help='Optional override: name,dir,config,checkpoint. May be repeated.')
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    specs = _experiment_specs(args)
    mkdir_or_exist(args.out_dir)
    comparison_dir = osp.join(args.out_dir, 'comparison')
    mkdir_or_exist(comparison_dir)
    seqs = _sequence_list(args.data_root, args.ann_file, args.ann_subdir)
    if args.max_seqs > 0:
        seqs = seqs[:args.max_seqs]
    img_root = osp.join(args.data_root, args.img_subdir)
    ann_dir = osp.join(args.data_root, args.ann_subdir)
    index_rows = []

    models = {}
    for name, spec in specs.items():
        print(f'[load] {name}: {spec["checkpoint_path"]}', flush=True)
        cfg, model, preprocessor, pipeline, device = _build_model_and_pipeline(
            spec['config_path'], spec['checkpoint_path'], args.device)
        models[name] = (cfg, model, preprocessor, pipeline, device, spec)

    for seq_idx, seq_name in enumerate(seqs, 1):
        ann_path = osp.join(ann_dir, f'{seq_name}.txt')
        frame_anns = load_hsmot_sequence_ann(ann_path)
        frame_ids = _frame_ids_from_images(osp.join(img_root, seq_name), args.img_format)
        if len(frame_ids) < 2:
            print(f'[skip] {seq_name}: less than 2 frames', flush=True)
            continue
        prev_id, curr_id = frame_ids[0], frame_ids[1]
        pair_info = _make_pair_info(
            seq_name, img_root, args.img_format, frame_anns, prev_id, curr_id)
        seq_diags = []
        print(f'[seq {seq_idx}/{len(seqs)}] {seq_name} {prev_id:06d}->{curr_id:06d}', flush=True)
        for name, (_, model, preprocessor, pipeline, device, spec) in models.items():
            diag = collect_diag(
                name, model, preprocessor, pipeline, device, pair_info,
                args.draw_candidates)
            exp_seq_dir = osp.join(args.out_dir, name, seq_name)
            mkdir_or_exist(exp_seq_dir)
            cand_path = osp.join(exp_seq_dir, f'{prev_id:06d}_{curr_id:06d}_01_candidates.jpg')
            topk_path = osp.join(exp_seq_dir, f'{prev_id:06d}_{curr_id:06d}_02_topk.jpg')
            json_path = osp.join(exp_seq_dir, f'{prev_id:06d}_{curr_id:06d}_summary.json')
            draw_candidates(diag, cand_path, args.draw_candidates)
            draw_topk(diag, topk_path, args.draw_topk)
            write_json(diag, json_path)
            seq_diags.append(diag)
            index_rows.append({
                'experiment': name,
                'seq_name': seq_name,
                'prev_frame_id': prev_id,
                'curr_frame_id': curr_id,
                'query_init': diag.query_init,
                'checkpoint': spec['checkpoint_path'],
                'candidates_image': cand_path,
                'topk_image': topk_path,
                'summary_json': json_path,
            })
        comp_path = osp.join(comparison_dir, f'{seq_name}_{prev_id:06d}_{curr_id:06d}_topk_grid.jpg')
        make_comparison(seq_diags, comp_path, args.draw_topk)

    index_path = osp.join(args.out_dir, 'index.csv')
    with open(index_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=[
            'experiment', 'seq_name', 'prev_frame_id', 'curr_frame_id',
            'query_init', 'checkpoint', 'candidates_image', 'topk_image',
            'summary_json'])
        writer.writeheader()
        writer.writerows(index_rows)

    readme_path = osp.join(args.out_dir, 'README.md')
    with open(readme_path, 'w', encoding='utf-8') as f:
        f.write('# Pair Proposal Visualization\n\n')
        f.write('Each test sequence is visualized on its first adjacent pair '
                '`000001 -> 000002`.\n\n')
        f.write('- `*/<seq>/000001_000002_01_candidates.jpg`: proposal candidates before final top-k.\n')
        f.write('- `*/<seq>/000001_000002_02_topk.jpg`: final selected top-k pair queries.\n')
        f.write('- `*/<seq>/000001_000002_summary.json`: numeric proposal/top-k records.\n')
        f.write('- `comparison/*_topk_grid.jpg`: stacked top-k comparison across experiments.\n')
        f.write('- `index.csv`: machine-readable file index.\n\n')
        for name, spec in specs.items():
            f.write(f'## {name}\n')
            f.write(f'- config: `{spec["config_path"]}`\n')
            f.write(f'- checkpoint: `{spec["checkpoint_path"]}`\n')
    print(f'[done] wrote {len(index_rows)} experiment-sequence records to {args.out_dir}', flush=True)


if __name__ == '__main__':
    main()
