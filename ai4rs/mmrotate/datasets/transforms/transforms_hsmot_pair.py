# Copyright (c) AI4RS. All rights reserved.
"""Shared geometric transforms for HSMOT image pairs."""
from typing import List, Optional, Union

import numpy as np
import torch
from mmcv.transforms import BaseTransform
from mmcv.transforms.utils import cache_randomness
from mmdet.datasets.transforms import RandomFlip, Resize

from mmrotate.registry import TRANSFORMS


def _restore_invalid_pair_boxes(results: dict) -> None:
    """Zero out bbox rows for invalid pair sides (new/disappear placeholders)."""
    from mmrotate.datasets.pair_gt import INVALID_QBOX_PLACEHOLDER

    pairs = (
        ('pair_valid_prev', 'gt_bboxes_prev'),
        ('pair_valid_curr', 'gt_bboxes_curr'),
    )
    for valid_key, bbox_key in pairs:
        if valid_key not in results or bbox_key not in results:
            continue
        valid = results[valid_key]
        boxes = results[bbox_key]
        if len(valid) == 0:
            continue
        if hasattr(valid, 'cpu'):
            valid_np = valid.cpu().numpy()
        else:
            valid_np = np.asarray(valid)
        tensor = boxes.tensor if hasattr(boxes, 'tensor') else boxes
        dim = tensor.size(-1)
        for i, ok in enumerate(valid_np):
            if ok:
                continue
            if dim == 5:
                tensor[i] = 0
            else:
                tensor[i] = torch.from_numpy(INVALID_QBOX_PLACEHOLDER.copy())
        if hasattr(boxes, 'tensor'):
            boxes.tensor = tensor
        else:
            results[bbox_key] = tensor


def _slice_pair_field(value, keep: np.ndarray):
    if isinstance(value, torch.Tensor):
        return value[torch.from_numpy(keep).to(device=value.device)]
    return np.asarray(value)[keep]


def _filter_fully_invalid_pair_rows(results: dict) -> None:
    if 'pair_valid_prev' not in results or 'pair_valid_curr' not in results:
        return
    valid_prev = results['pair_valid_prev']
    valid_curr = results['pair_valid_curr']
    if isinstance(valid_prev, torch.Tensor):
        vp = valid_prev.cpu().numpy().astype(bool)
    else:
        vp = np.asarray(valid_prev, dtype=bool)
    if isinstance(valid_curr, torch.Tensor):
        vc = valid_curr.cpu().numpy().astype(bool)
    else:
        vc = np.asarray(valid_curr, dtype=bool)
    keep = vp | vc
    if keep.all():
        return

    for key in ('pair_labels', 'pair_track_ids', 'pair_valid_prev',
                'pair_valid_curr'):
        if key in results:
            results[key] = _slice_pair_field(results[key], keep)
    for key in ('gt_bboxes_prev', 'gt_bboxes_curr'):
        if key in results:
            results[key] = results[key][keep]


def _filter_outside_pair_boxes(results: dict) -> None:
    """Mark side-specific boxes outside the image as invalid after geometry."""
    img_shape = results['img_shape'][:2]
    for valid_key, bbox_key in (
            ('pair_valid_prev', 'gt_bboxes_prev'),
            ('pair_valid_curr', 'gt_bboxes_curr')):
        if valid_key not in results or bbox_key not in results:
            continue
        boxes = results[bbox_key]
        if len(boxes) == 0:
            continue
        inside = boxes.is_inside(img_shape).cpu().numpy().astype(bool)
        valid = results[valid_key]
        if isinstance(valid, torch.Tensor):
            inside_t = torch.from_numpy(inside).to(
                device=valid.device, dtype=torch.bool)
            results[valid_key] = valid.bool() & inside_t
        else:
            results[valid_key] = np.asarray(valid, dtype=bool) & inside
    _restore_invalid_pair_boxes(results)
    _filter_fully_invalid_pair_rows(results)


def _subresult(results: dict, img_idx: int, bbox_key: str) -> dict:
    sub = {
        'img': results['img'][img_idx],
        'img_shape': results['img_shape'],
        'ori_shape': results.get('ori_shape', results['img_shape']),
    }
    if bbox_key in results:
        sub['gt_bboxes'] = results[bbox_key]
    return sub


def _write_subresult(results: dict, img_idx: int, bbox_key: str,
                     sub: dict) -> None:
    results['img'][img_idx] = sub['img']
    if bbox_key in results and 'gt_bboxes' in sub:
        results[bbox_key] = sub['gt_bboxes']


@TRANSFORMS.register_module()
class PairSharedResize(BaseTransform):
    """Resize both pair images with identical ``scale`` / ``scale_factor``."""

    def __init__(self,
                 scale: Optional[Union[int, tuple]] = None,
                 scale_factor: Optional[Union[float, tuple]] = None,
                 keep_ratio: bool = False,
                 clip_object_border: bool = True,
                 backend: str = 'cv2',
                 interpolation: str = 'bilinear') -> None:
        self.resize = Resize(
            scale=scale,
            scale_factor=scale_factor,
            keep_ratio=keep_ratio,
            clip_object_border=clip_object_border,
            backend=backend,
            interpolation=interpolation)

    def transform(self, results: dict) -> dict:
        bbox_keys = ('gt_bboxes_prev', 'gt_bboxes_curr')
        sub0 = _subresult(results, 0, bbox_keys[0])
        self.resize.transform(sub0)
        _write_subresult(results, 0, bbox_keys[0], sub0)

        for meta_key in ('scale', 'scale_factor', 'keep_ratio', 'img_shape'):
            if meta_key in sub0:
                results[meta_key] = sub0[meta_key]

        sub1 = _subresult(results, 1, bbox_keys[1])
        sub1['scale'] = sub0.get('scale')
        sub1['scale_factor'] = sub0.get('scale_factor')
        sub1['keep_ratio'] = sub0.get('keep_ratio', False)
        self.resize._resize_img(sub1)
        if sub1.get('gt_bboxes') is not None:
            self.resize._resize_bboxes(sub1)
        _write_subresult(results, 1, bbox_keys[1], sub1)
        _restore_invalid_pair_boxes(results)
        return results


@TRANSFORMS.register_module()
class PairSharedRandomFlip(BaseTransform):
    """Apply the same flip decision and direction to both pair frames."""

    def __init__(self,
                 prob: Optional[Union[float, List[float]]] = None,
                 direction: Union[str, List[str]] = 'horizontal') -> None:
        self.flip = RandomFlip(prob=prob, direction=direction)

    def transform(self, results: dict) -> dict:
        dummy = {
            'img': results['img'][0].copy(),
            'img_shape': results['img_shape'],
        }
        self.flip.transform(dummy)

        flip = dummy.get('flip', False)
        flip_direction = dummy.get('flip_direction', None)
        results['flip'] = flip
        results['flip_direction'] = flip_direction
        if 'homography_matrix' in dummy:
            results['homography_matrix'] = dummy['homography_matrix']

        for img_idx, bbox_key in enumerate(
                ('gt_bboxes_prev', 'gt_bboxes_curr')):
            sub = _subresult(results, img_idx, bbox_key)
            sub['flip'] = flip
            sub['flip_direction'] = flip_direction
            if flip:
                self.flip._flip(sub)
            results['img'][img_idx] = sub['img']
            if bbox_key in results and 'gt_bboxes' in sub:
                results[bbox_key] = sub['gt_bboxes']
        _restore_invalid_pair_boxes(results)
        return results


@TRANSFORMS.register_module()
class PairSharedRandomRotate(BaseTransform):
    """Rotate both pair frames with an identical random angle."""

    def __init__(self,
                 prob: float = 0.5,
                 angle_range: int = 180,
                 rect_obj_labels: Optional[List[int]] = None,
                 rotate_type: str = 'Rotate',
                 **rotate_kwargs) -> None:
        assert 0 < angle_range <= 180
        self.prob = prob
        self.angle_range = angle_range
        self.rect_obj_labels = rect_obj_labels
        self.rotate_type = rotate_type
        self.rotate_kwargs = rotate_kwargs
        self.horizontal_angles = [90, 180, -90, -180]

    @cache_randomness
    def _random_angle(self) -> float:
        return self.angle_range * (2 * np.random.rand() - 1)

    @cache_randomness
    def _random_horizontal_angle(self) -> int:
        return int(np.random.choice(self.horizontal_angles))

    @cache_randomness
    def _is_rotate(self) -> bool:
        return np.random.rand() < self.prob

    def _pick_angle(self, results: dict) -> float:
        if not self._is_rotate():
            return 0.0
        angle = self._random_angle()
        if self.rect_obj_labels is not None and len(results.get(
                'pair_labels', [])) > 0:
            labels = results['pair_labels']
            for label in self.rect_obj_labels:
                if np.any(labels == label):
                    return float(self._random_horizontal_angle())
        return angle

    def transform(self, results: dict) -> dict:
        angle = self._pick_angle(results)
        if angle == 0.0:
            return results

        rotate = TRANSFORMS.build({
            'type': self.rotate_type,
            'rotate_angle': angle,
            **self.rotate_kwargs,
        })

        for img_idx, bbox_key in enumerate(
                ('gt_bboxes_prev', 'gt_bboxes_curr')):
            sub = _subresult(results, img_idx, bbox_key)
            # Rotate without per-frame _filter_invalid: placeholders for
            # new/disappear tracks must stay row-aligned with pair_gt.
            rotate.homography_matrix = rotate._get_homography_matrix(sub)
            if results.get('homography_matrix', None) is None:
                results['homography_matrix'] = rotate.homography_matrix
            else:
                results['homography_matrix'] = (
                    rotate.homography_matrix @ results['homography_matrix'])
            rotate._transform_img(sub)
            if sub.get('gt_bboxes') is not None:
                rotate._transform_bboxes(sub)
            _write_subresult(results, img_idx, bbox_key, sub)
        results['img_shape'] = results['img'][0].shape[:2]
        _filter_outside_pair_boxes(results)
        return results
