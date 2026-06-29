# Copyright (c) AI4RS. All rights reserved.
"""Validation helpers for HSMOT image-pair samples."""
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from mmrotate.datasets.pair_gt import (
    INVALID_QBOX_PLACEHOLDER,
    build_pair_gt_from_instances,
)
from mmrotate.structures.bbox import rbox2qbox


def _angle_diff_deg(a: float, b: float) -> float:
    d = a - b
    while d > 180:
        d -= 360
    while d < -180:
        d += 360
    return d


@dataclass
class PairCheckResult:
    name: str
    ok: bool
    detail: str = ''


@dataclass
class PairValidationReport:
    checks: List[PairCheckResult] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return all(c.ok for c in self.checks)

    def summary(self) -> str:
        lines = []
        for c in self.checks:
            mark = 'OK' if c.ok else 'FAIL'
            lines.append(f'[{mark}] {c.name}: {c.detail}')
        for w in self.warnings:
            lines.append(f'[WARN] {w}')
        return '\n'.join(lines)


def _tensor_boxes(bboxes) -> torch.Tensor:
    if isinstance(bboxes, torch.Tensor):
        return bboxes
    if hasattr(bboxes, 'tensor'):
        return bboxes.tensor
    return torch.as_tensor(bboxes)


def _qbox_corners(qbox: np.ndarray) -> np.ndarray:
    return qbox.reshape(-1, 4, 2)


def validate_id_alignment(results: dict) -> PairCheckResult:
    """Track ids and valid masks match raw instance union."""
    expected = build_pair_gt_from_instances(
        results.get('instances_prev', []),
        results.get('instances_curr', []),
        video_id=results.get('video_id', ''),
        frame_id_prev=int(results.get('frame_id_prev', -1)),
        frame_id_curr=int(results.get('frame_id', -1)),
    )
    got_ids = results.get('pair_track_ids')
    if got_ids is None:
        return PairCheckResult('id_alignment', False, 'missing pair_track_ids')
    if isinstance(got_ids, torch.Tensor):
        got_ids = got_ids.cpu().numpy()
    got_vp = results['pair_valid_prev']
    got_vc = results['pair_valid_curr']
    if isinstance(got_vp, torch.Tensor):
        got_vp = got_vp.cpu().numpy()
    if isinstance(got_vc, torch.Tensor):
        got_vc = got_vc.cpu().numpy()

    if not np.array_equal(expected['track_ids'], got_ids):
        expected_set = set(expected['track_ids'].tolist())
        got_list = got_ids.tolist()
        if all(int(tid) in expected_set for tid in got_list) and got_list == sorted(got_list):
            exp_index = {
                int(tid): i
                for i, tid in enumerate(expected['track_ids'].tolist())
            }
            exp_vp = np.asarray(
                [expected['valid_prev'][exp_index[int(tid)]]
                 for tid in got_list],
                dtype=np.bool_)
            exp_vc = np.asarray(
                [expected['valid_curr'][exp_index[int(tid)]]
                 for tid in got_list],
                dtype=np.bool_)
            if np.any(got_vp & ~exp_vp):
                return PairCheckResult(
                    'id_alignment', False,
                    'valid_prev has true for raw-missing track')
            if np.any(got_vc & ~exp_vc):
                return PairCheckResult(
                    'id_alignment', False,
                    'valid_curr has true for raw-missing track')
            n_filtered = len(expected['track_ids']) - len(got_ids)
            n_new = int(np.sum(~got_vp & got_vc))
            n_dis = int(np.sum(got_vp & ~got_vc))
            n_persist = int(np.sum(got_vp & got_vc))
            return PairCheckResult(
                'id_alignment', True,
                f'persist={n_persist} new={n_new} disappear={n_dis} '
                f'total={len(got_ids)} geom_filtered={n_filtered}')
        return PairCheckResult(
            'id_alignment', False,
            f'track_ids mismatch exp={expected["track_ids"].tolist()} '
            f'got={got_ids.tolist()}')
    if np.any(got_vp & ~expected['valid_prev']):
        return PairCheckResult(
            'id_alignment', False,
            'valid_prev has true for raw-missing track')
    if np.any(got_vc & ~expected['valid_curr']):
        return PairCheckResult(
            'id_alignment', False,
            'valid_curr has true for raw-missing track')

    n_new = int(np.sum(~got_vp & got_vc))
    n_dis = int(np.sum(got_vp & ~got_vc))
    n_persist = int(np.sum(got_vp & got_vc))
    n_filtered = int(np.sum(expected['valid_prev'] & ~got_vp) +
                     np.sum(expected['valid_curr'] & ~got_vc))
    return PairCheckResult(
        'id_alignment', True,
        f'persist={n_persist} new={n_new} disappear={n_dis} '
        f'total={len(got_ids)} geom_filtered_sides={n_filtered}')


def validate_new_disappear_markers(results: dict) -> PairCheckResult:
    """Invalid-side placeholders stay zero after raw or geometric absence."""
    track_ids = results['pair_track_ids']
    vp = results['pair_valid_prev']
    vc = results['pair_valid_curr']
    if isinstance(track_ids, torch.Tensor):
        track_ids = track_ids.cpu().numpy()
    if isinstance(vp, torch.Tensor):
        vp = vp.cpu().numpy()
    if isinstance(vc, torch.Tensor):
        vc = vc.cpu().numpy()

    bprev = _tensor_boxes(results['gt_bboxes_prev']).cpu().numpy()
    bcurr = _tensor_boxes(results['gt_bboxes_curr']).cpu().numpy()
    n_prev, n_curr = len(bprev), len(bcurr)
    n_ids = len(track_ids)
    if n_prev != n_ids or n_curr != n_ids:
        return PairCheckResult(
            'new_disappear', False,
            f'bbox rows {n_prev}/{n_curr} != track_ids {n_ids}')

    for i, tid in enumerate(track_ids):
        tid = int(tid)
        if not vp[i] and vc[i]:
            if bprev.shape[-1] == 5:
                if np.any(bprev[i] != 0):
                    return PairCheckResult(
                        'new_disappear', False,
                        f'track {tid} new but prev bbox non-zero')
            elif not np.allclose(bprev[i], INVALID_QBOX_PLACEHOLDER):
                return PairCheckResult(
                    'new_disappear', False,
                    f'track {tid} new but prev qbox not placeholder')
        if vp[i] and not vc[i]:
            if bcurr.shape[-1] == 5:
                if np.any(bcurr[i] != 0):
                    return PairCheckResult(
                        'new_disappear', False,
                        f'track {tid} disappear but curr bbox non-zero')
            elif not np.allclose(bcurr[i], INVALID_QBOX_PLACEHOLDER):
                return PairCheckResult(
                    'new_disappear', False,
                    f'track {tid} disappear but curr qbox not placeholder')
    return PairCheckResult('new_disappear', True, 'placeholders and flags consistent')


def validate_angle_range(results: dict,
                         angle_min: float = -90.0,
                         angle_max: float = 90.0) -> PairCheckResult:
    """Valid rbox angles within le90 convention."""
    h, w = results['img_shape'][:2]
    issues = []
    for key, valid_key in (
            ('gt_bboxes_prev', 'pair_valid_prev'),
            ('gt_bboxes_curr', 'pair_valid_curr')):
        boxes = _tensor_boxes(results[key]).cpu().numpy()
        valid = results[valid_key]
        if isinstance(valid, torch.Tensor):
            valid = valid.cpu().numpy()
        for i, ok in enumerate(valid):
            if not ok or boxes.shape[0] <= i:
                continue
            if boxes.shape[-1] != 5:
                continue
            angle = float(boxes[i, 4])
            if angle < angle_min - 1e-3 or angle > angle_max + 1e-3:
                issues.append(f'{key}[{i}] angle={angle:.2f}')
    if issues:
        return PairCheckResult(
            'angle_range', False, '; '.join(issues[:5]))
    return PairCheckResult('angle_range', True, f'angles in [{angle_min}, {angle_max}]')


def validate_shared_rotation_delta(
        results: dict,
        raw_angle_prev: Dict[int, float],
        raw_angle_curr: Dict[int, float],
        tol_deg: float = 5.0) -> PairCheckResult:
    """Same image rotation applied to prev and curr (persistent tracks)."""
    track_ids = results['pair_track_ids']
    vp = results['pair_valid_prev']
    vc = results['pair_valid_curr']
    if isinstance(track_ids, torch.Tensor):
        track_ids = track_ids.cpu().numpy()
    if isinstance(vp, torch.Tensor):
        vp = vp.cpu().numpy()
    if isinstance(vc, torch.Tensor):
        vc = vc.cpu().numpy()

    bprev = _tensor_boxes(results['gt_bboxes_prev']).cpu().numpy()
    bcurr = _tensor_boxes(results['gt_bboxes_curr']).cpu().numpy()
    if bprev.shape[-1] != 5:
        return PairCheckResult('shared_rotation', True, 'skip (qbox mode)')

    deltas = []
    for i, tid in enumerate(track_ids):
        if not (vp[i] and vc[i]):
            continue
        tid = int(tid)
        if tid not in raw_angle_prev or tid not in raw_angle_curr:
            continue
        d_prev = _angle_diff_deg(bprev[i, 4], raw_angle_prev[tid])
        d_curr = _angle_diff_deg(bcurr[i, 4], raw_angle_curr[tid])
        if abs(_angle_diff_deg(d_prev, d_curr)) > tol_deg:
            return PairCheckResult(
                'shared_rotation', False,
                f'track {tid} rot_delta prev={d_prev:.2f} curr={d_curr:.2f}')
        deltas.append(d_prev)

    if not deltas:
        return PairCheckResult('shared_rotation', True, 'no persistent rboxes')
    return PairCheckResult(
        'shared_rotation', True,
        f'persistent n={len(deltas)} common_rot_delta~{np.mean(deltas):.2f}deg')


def validate_bbox_in_bounds(results: dict,
                            margin: float = 10.0) -> PairCheckResult:
    """Valid boxes: positive size and center inside image (corners may clip)."""
    h, w = results['img_shape'][:2]
    issues = []
    corner_issues = 0
    for key, valid_key in (
            ('gt_bboxes_prev', 'pair_valid_prev'),
            ('gt_bboxes_curr', 'pair_valid_curr')):
        boxes = results[key]
        valid = results[valid_key]
        if isinstance(valid, torch.Tensor):
            valid = valid.cpu().numpy()
        if len(valid) == 0:
            continue
        tensor = _tensor_boxes(boxes)
        if tensor.numel() == 0:
            continue
        polys = None
        if tensor.size(-1) == 5:
            qbox = rbox2qbox(tensor)
            polys = qbox.reshape(-1, 4, 2).cpu().numpy()
        for i, ok in enumerate(valid):
            if not ok:
                continue
            wh = tensor[i]
            if tensor.size(-1) == 5:
                if float(wh[2]) <= 0 or float(wh[3]) <= 0:
                    issues.append(f'{key}[{i}] non-positive wh')
                cx, cy = float(wh[0]), float(wh[1])
                if cx < -margin or cy < -margin or cx > w + margin or cy > h + margin:
                    issues.append(f'{key}[{i}] center OOB ({cx:.0f},{cy:.0f})')
            if polys is not None:
                poly = polys[i]
                xmin, ymin = poly[:, 0].min(), poly[:, 1].min()
                xmax, ymax = poly[:, 0].max(), poly[:, 1].max()
                if xmin < -margin or ymin < -margin:
                    corner_issues += 1
                if xmax > w + margin or ymax > h + margin:
                    corner_issues += 1
    if issues:
        return PairCheckResult('bbox_in_bounds', False, '; '.join(issues[:5]))
    detail = f'centers within [0,{w}]x[0,{h}]'
    if corner_issues:
        detail += f', corner_oob_warn={corner_issues}'
    return PairCheckResult('bbox_in_bounds', True, detail)


def validate_shared_geometry(results: dict) -> PairCheckResult:
    """Two frames share shape, scale_factor, flip decision."""
    img0, img1 = results['img'][0], results['img'][1]
    if img0.shape[:2] != img1.shape[:2]:
        return PairCheckResult(
            'shared_geometry', False,
            f'shape mismatch {img0.shape[:2]} vs {img1.shape[:2]}')

    details = [f'shape={img0.shape[:2]}']
    if 'scale_factor' in results:
        sf = results['scale_factor']
        if isinstance(sf, (list, tuple, np.ndarray)):
            sf_arr = np.asarray(sf, dtype=np.float64).ravel()
            if sf_arr.size >= 2:
                sf_w, sf_h = float(sf_arr[0]), float(sf_arr[1])
                if abs(sf_w - sf_h) > 1e-3:
                    return PairCheckResult(
                        'shared_geometry', False,
                        f'non-uniform scale_factor ({sf_w}, {sf_h})')
        details.append(f'scale_factor={sf}')
    if 'flip' in results:
        details.append(f'flip={results["flip"]} dir={results.get("flip_direction")}')

    # channel-wise mean should match when only shared geom (resize/flip/rotate)
    if img0.shape == img1.shape:
        diff = np.abs(img0.astype(np.float32) - img1.astype(np.float32))
        mean_diff = float(diff.mean())
        details.append(f'mean_pix_diff={mean_diff:.2f}')
    return PairCheckResult('shared_geometry', True, ', '.join(details))


def validate_pair_results(
        results: dict,
        raw_angle_prev: Optional[Dict[int, float]] = None,
        raw_angle_curr: Optional[Dict[int, float]] = None,
) -> PairValidationReport:
    """Run all pair validation checks on pipeline ``results`` dict."""
    report = PairValidationReport()
    report.checks.append(validate_id_alignment(results))
    report.checks.append(validate_new_disappear_markers(results))
    report.checks.append(validate_angle_range(results))
    report.checks.append(validate_bbox_in_bounds(results))
    report.checks.append(validate_shared_geometry(results))
    if raw_angle_prev is not None and raw_angle_curr is not None:
        report.checks.append(
            validate_shared_rotation_delta(
                results, raw_angle_prev, raw_angle_curr))
    return report


def compute_raw_rbox_angles(instances: Sequence[dict]) -> Dict[int, float]:
    from mmrotate.structures.bbox import RotatedBoxes, qbox2rbox

    out: Dict[int, float] = {}
    for inst in instances:
        tid = int(inst['track_id'])
        qbox = torch.from_numpy(
            np.asarray(inst['bbox'], dtype=np.float32).reshape(1, 8))
        rbox = RotatedBoxes(qbox2rbox(qbox))
        out[tid] = float(rbox.tensor[0, 4])
    return out
