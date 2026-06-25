# Copyright (c) AI4RS. All rights reserved.
"""Visualization helpers for HSMOT image pairs."""
from pathlib import Path
from typing import Optional, Sequence, Union

import cv2
import numpy as np
import torch

from mmrotate.structures.bbox import QuadriBoxes, rbox2qbox


def _to_numpy_image(img: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
    if isinstance(img, torch.Tensor):
        img = img.detach().cpu()
    if isinstance(img, torch.Tensor):
        img = img.numpy()
    if img.ndim == 3 and img.shape[0] in (1, 3, 8):
        img = np.transpose(img, (1, 2, 0))
    if img.dtype != np.uint8:
        if img.max() <= 1.0:
            img = img * 255.0
        img = np.clip(img, 0, 255).astype(np.uint8)
    return img


def _rgb_preview(img: np.ndarray) -> np.ndarray:
    """Use first 3 channels as BGR preview."""
    if img.shape[-1] >= 3:
        return cv2.cvtColor(img[..., :3], cv2.COLOR_RGB2BGR)
    return img


def _polygons_from_bboxes(bboxes) -> np.ndarray:
    if isinstance(bboxes, torch.Tensor):
        tensor = bboxes
    elif hasattr(bboxes, 'tensor'):
        tensor = bboxes.tensor
    else:
        tensor = torch.as_tensor(bboxes)
    if tensor.numel() == 0:
        return np.zeros((0, 4, 2), dtype=np.float32)
    if tensor.size(-1) == 5:
        qbox = rbox2qbox(tensor)
    elif tensor.size(-1) == 8:
        qbox = tensor
    else:
        raise ValueError(f'Unsupported bbox shape: {tensor.shape}')
    return qbox.reshape(-1, 4, 2).cpu().numpy()


def _draw_frame(img_rgb: np.ndarray,
                bboxes,
                track_ids: Optional[Sequence[int]],
                valid_mask: Optional[Sequence[bool]],
                title: str,
                angles: Optional[Sequence[float]] = None,
                out_of_bounds: Optional[Sequence[bool]] = None) -> np.ndarray:
    canvas = _rgb_preview(img_rgb).copy()
    polys = _polygons_from_bboxes(bboxes)
    if track_ids is None:
        track_ids = list(range(len(polys)))
    if valid_mask is None:
        valid_mask = [True] * len(polys)

    for idx, (poly, tid, valid) in enumerate(zip(polys, track_ids, valid_mask)):
        oob = out_of_bounds is not None and out_of_bounds[idx]
        if valid:
            color = (0, 0, 255) if oob else (0, 255, 0)
        else:
            color = (128, 128, 128)
        pts = poly.reshape(-1, 1, 2).astype(np.int32)
        cv2.polylines(canvas, [pts], isClosed=True, color=color, thickness=2)
        cx = int(np.clip(poly[:, 0].mean(), 0, canvas.shape[1] - 1))
        cy = int(np.clip(poly[:, 1].mean(), 0, canvas.shape[0] - 1))
        label = f'id={tid}'
        if angles is not None and idx < len(angles) and valid:
            label += f' a={angles[idx]:.1f}'
        if not valid:
            label += ' (invalid)'
        if oob:
            label += ' OOB'
        cv2.putText(canvas, label, (cx, cy), cv2.FONT_HERSHEY_SIMPLEX,
                    0.45, color, 1, cv2.LINE_AA)

    cv2.putText(canvas, title, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                (0, 255, 255), 2, cv2.LINE_AA)
    return canvas


def visualize_hsmot_pair(
    img_prev: Union[np.ndarray, torch.Tensor],
    img_curr: Union[np.ndarray, torch.Tensor],
    bboxes_prev,
    bboxes_curr,
    track_ids: Sequence[int],
    valid_prev: Optional[Sequence[bool]] = None,
    valid_curr: Optional[Sequence[bool]] = None,
    labels: Optional[Sequence[int]] = None,
    save_path: Optional[str] = None,
    show: bool = False,
    check_summary: Optional[str] = None,
    angles_prev: Optional[Sequence[float]] = None,
    angles_curr: Optional[Sequence[float]] = None,
    oob_prev: Optional[Sequence[bool]] = None,
    oob_curr: Optional[Sequence[bool]] = None,
    meta_line: Optional[str] = None,
) -> np.ndarray:
    """Draw previous/current frames with track ids and rotated boxes.

    Args:
        img_prev: Previous frame, (H,W,C) or (C,H,W).
        img_curr: Current frame.
        bboxes_prev: QuadriBoxes / RotatedBoxes / Tensor for previous frame.
        bboxes_curr: Same for current frame.
        track_ids: Aligned track id list.
        valid_prev: Valid mask for previous boxes.
        valid_curr: Valid mask for current boxes.
        labels: Optional class labels (unused in drawing, reserved).
        save_path: If set, write BGR visualization to this path.
        show: If True, display with OpenCV (requires GUI).

    Returns:
        np.ndarray: Side-by-side BGR visualization.
    """
    prev = _to_numpy_image(img_prev)
    curr = _to_numpy_image(img_curr)

    if valid_prev is None:
        valid_prev = [True] * len(track_ids)
    if valid_curr is None:
        valid_curr = [True] * len(track_ids)

    left = _draw_frame(
        prev, bboxes_prev, track_ids, valid_prev,
        'prev', angles=angles_prev, out_of_bounds=oob_prev)
    right = _draw_frame(
        curr, bboxes_curr, track_ids, valid_curr,
        'curr', angles=angles_curr, out_of_bounds=oob_curr)

    # tag NEW / DIS on frame titles
    n_new = sum(not vp and vc for vp, vc in zip(valid_prev, valid_curr))
    n_dis = sum(vp and not vc for vp, vc in zip(valid_prev, valid_curr))
    cv2.putText(left, f'new={n_new} dis={n_dis}', (8, 48),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1, cv2.LINE_AA)
    vis = np.concatenate([left, right], axis=1)

    banner_h = 56 if check_summary or meta_line else 0
    if banner_h:
        banner = np.zeros((banner_h, vis.shape[1], 3), dtype=np.uint8)
        y = 18
        if meta_line:
            cv2.putText(banner, meta_line, (8, y), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (200, 200, 200), 1, cv2.LINE_AA)
            y += 22
        if check_summary:
            color = (0, 255, 0) if check_summary.startswith('PASS') else (0, 0, 255)
            cv2.putText(banner, check_summary[:200], (8, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)
        vis = np.concatenate([vis, banner], axis=0)

    if save_path is not None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(save_path, vis)

    if show:
        cv2.imshow('hsmot_pair', vis)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return vis


def visualize_hsmot_pair_sample(data_sample, save_path: Optional[str] = None,
                                show: bool = False) -> np.ndarray:
    """Visualize from a packed ``DetDataSample`` with ``pair_gt_instances``."""
    if not hasattr(data_sample, 'pair_gt_instances'):
        raise AttributeError('data_sample has no pair_gt_instances')

    pair_gt = data_sample.pair_gt_instances
    inputs = data_sample.metainfo.get('inputs')
    if inputs is None:
        raise ValueError(
            'metainfo has no inputs; pass raw images or store inputs in meta.')

    img_prev = inputs[0]
    img_curr = inputs[1]
    return visualize_hsmot_pair(
        img_prev,
        img_curr,
        pair_gt.bboxes_prev,
        pair_gt.bboxes_curr,
        pair_gt.track_ids.tolist(),
        pair_gt.valid_prev.tolist(),
        pair_gt.valid_curr.tolist(),
        labels=pair_gt.labels.tolist(),
        save_path=save_path,
        show=show,
    )
