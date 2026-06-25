# Copyright (c) AI4RS. All rights reserved.
"""Utilities for constructing HSMOT image-pair ground truth."""
from typing import Dict, List, Sequence, Tuple

import numpy as np


class TrackIdClassMismatchError(ValueError):
    """Raised when the same track_id has different class labels across frames."""

    def __init__(self, track_id: int, label_prev: int, label_curr: int,
                 video_id: str = '', frame_id_prev: int = -1,
                 frame_id_curr: int = -1) -> None:
        self.track_id = track_id
        self.label_prev = label_prev
        self.label_curr = label_curr
        self.video_id = video_id
        self.frame_id_prev = frame_id_prev
        self.frame_id_curr = frame_id_curr
        super().__init__(
            f'Track id {track_id} has inconsistent labels across pair frames: '
            f'prev={label_prev}, curr={label_curr} '
            f'(video_id={video_id!r}, prev_frame={frame_id_prev}, '
            f'curr_frame={frame_id_curr}).')


INVALID_QBOX_PLACEHOLDER = np.zeros(8, dtype=np.float32)


def build_pair_gt_from_instances(
    instances_prev: Sequence[dict],
    instances_curr: Sequence[dict],
    video_id: str = '',
    frame_id_prev: int = -1,
    frame_id_curr: int = -1,
) -> Dict[str, np.ndarray]:
    """Build aligned pair GT from per-frame instance lists.

    Each instance dict must contain ``track_id``, ``bbox`` (8-d qbox), and
    ``bbox_label``. The union of track ids defines the output rows. Missing
    boxes are filled with ``INVALID_QBOX_PLACEHOLDER`` and ``valid_*`` is 0.

    Args:
        instances_prev: Instances on the previous frame.
        instances_curr: Instances on the current frame.
        video_id: Sequence / video identifier for error messages.
        frame_id_prev: Previous frame index (1-based MOT convention).
        frame_id_curr: Current frame index.

    Returns:
        dict with keys ``labels``, ``track_ids``, ``bboxes_prev``,
        ``bboxes_curr``, ``valid_prev``, ``valid_curr``.
    """
    prev_map = {
        int(inst['track_id']): inst for inst in instances_prev
    }
    curr_map = {
        int(inst['track_id']): inst for inst in instances_curr
    }
    all_track_ids = sorted(set(prev_map.keys()) | set(curr_map.keys()))

    if len(all_track_ids) == 0:
        return {
            'labels': np.zeros((0,), dtype=np.int64),
            'track_ids': np.zeros((0,), dtype=np.int64),
            'bboxes_prev': np.zeros((0, 8), dtype=np.float32),
            'bboxes_curr': np.zeros((0, 8), dtype=np.float32),
            'valid_prev': np.zeros((0,), dtype=np.bool_),
            'valid_curr': np.zeros((0,), dtype=np.bool_),
        }

    labels: List[int] = []
    track_ids: List[int] = []
    bboxes_prev: List[np.ndarray] = []
    bboxes_curr: List[np.ndarray] = []
    valid_prev: List[bool] = []
    valid_curr: List[bool] = []

    for track_id in all_track_ids:
        in_prev = track_id in prev_map
        in_curr = track_id in curr_map

        if in_prev and in_curr:
            label_prev = int(prev_map[track_id]['bbox_label'])
            label_curr = int(curr_map[track_id]['bbox_label'])
            if label_prev != label_curr:
                raise TrackIdClassMismatchError(
                    track_id=track_id,
                    label_prev=label_prev,
                    label_curr=label_curr,
                    video_id=video_id,
                    frame_id_prev=frame_id_prev,
                    frame_id_curr=frame_id_curr,
                )
            label = label_prev
        elif in_prev:
            label = int(prev_map[track_id]['bbox_label'])
        else:
            label = int(curr_map[track_id]['bbox_label'])

        track_ids.append(track_id)
        labels.append(label)
        valid_prev.append(in_prev)
        valid_curr.append(in_curr)
        bboxes_prev.append(
            prev_map[track_id]['bbox'].astype(np.float32, copy=False)
            if in_prev else INVALID_QBOX_PLACEHOLDER.copy())
        bboxes_curr.append(
            curr_map[track_id]['bbox'].astype(np.float32, copy=False)
            if in_curr else INVALID_QBOX_PLACEHOLDER.copy())

    return {
        'labels': np.asarray(labels, dtype=np.int64),
        'track_ids': np.asarray(track_ids, dtype=np.int64),
        'bboxes_prev': np.stack(bboxes_prev, axis=0),
        'bboxes_curr': np.stack(bboxes_curr, axis=0),
        'valid_prev': np.asarray(valid_prev, dtype=np.bool_),
        'valid_curr': np.asarray(valid_curr, dtype=np.bool_),
    }
