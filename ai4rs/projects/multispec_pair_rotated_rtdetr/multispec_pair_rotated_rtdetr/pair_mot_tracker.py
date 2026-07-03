"""Pair-detection based online MOT utilities for HSMOT.

The tracker consumes cached pair detections.  Model inference and tracking are
kept separate so threshold and lifecycle parameters can be swept without
running the pair detector again.
"""
from __future__ import annotations

import math
import os
import os.path as osp
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment

from mmrotate.structures.bbox import qbox2rbox, rbbox_overlaps, rbox2qbox


def wrap_angle(angle: float) -> float:
    """Wrap an angle to [-pi, pi)."""
    return (float(angle) + math.pi) % (2 * math.pi) - math.pi


def rbox_to_qbox_list(rbox: Sequence[float]) -> List[float]:
    tensor = torch.as_tensor(rbox, dtype=torch.float32).reshape(1, 5)
    qbox = rbox2qbox(tensor).reshape(-1).cpu().numpy()
    return [float(x) for x in qbox.tolist()]


def rotated_iou_matrix(
    boxes1: Sequence[Sequence[float]],
    boxes2: Sequence[Sequence[float]],
) -> np.ndarray:
    if len(boxes1) == 0 or len(boxes2) == 0:
        return np.zeros((len(boxes1), len(boxes2)), dtype=np.float32)
    b1 = torch.as_tensor(boxes1, dtype=torch.float32)
    b2 = torch.as_tensor(boxes2, dtype=torch.float32)
    return rbbox_overlaps(b1, b2).detach().cpu().numpy().astype(np.float32)


@dataclass
class PairDetection:
    index: int
    prev_bbox: List[float]
    curr_bbox: List[float]
    score: float
    cls_score: float
    label: int
    presence_prev: Optional[float] = None
    presence_curr: Optional[float] = None
    score_prev: Optional[float] = None
    score_curr: Optional[float] = None
    label_prev: Optional[int] = None
    label_curr: Optional[int] = None

    def prev_side_score(self) -> float:
        if self.score_prev is not None:
            return float(self.score_prev)
        if self.presence_prev is not None:
            return float(self.cls_score) * float(self.presence_prev)
        return float(self.score)

    def curr_side_score(self) -> float:
        if self.score_curr is not None:
            return float(self.score_curr)
        if self.presence_curr is not None:
            return float(self.cls_score) * float(self.presence_curr)
        return float(self.score)

    def pair_score(self) -> float:
        return math.sqrt(
            max(self.prev_side_score(), 1e-6) *
            max(self.curr_side_score(), 1e-6))

    def birth_score(self) -> float:
        return self.curr_side_score()


@dataclass
class PairFrameRecord:
    seq_name: str
    prev_frame_id: int
    curr_frame_id: int
    frame_gap: int
    prev_img_path: str
    curr_img_path: str
    img_shape: List[int]
    ori_shape: List[int]
    scale_factor: List[float]
    detections: List[PairDetection]
    is_first_pair: bool = False


PAIR_DET_TXT_HEADER = (
    '# curr_frame,prev_frame,det_index,'
    'prev_x1,prev_y1,prev_x2,prev_y2,prev_x3,prev_y3,prev_x4,prev_y4,'
    'prev_cls,prev_score,'
    'curr_x1,curr_y1,curr_x2,curr_y2,curr_x3,curr_y3,curr_x4,curr_y4,'
    'curr_cls,curr_score,pair_cls,pair_score,cls_score,'
    'presence_prev,presence_curr\n')


def write_pair_det_txt(path: str, records: Iterable[PairFrameRecord]) -> None:
    """Write pair detections as the canonical text cache format.

    The text cache intentionally starts from real temporal pairs such as
    01-02. Same-frame bootstrap records are skipped.
    """
    records = sorted(
        records, key=lambda item: (item.prev_frame_id, item.curr_frame_id))
    os.makedirs(osp.dirname(path), exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        f.write(PAIR_DET_TXT_HEADER)
        for rec in records:
            if rec.prev_frame_id == rec.curr_frame_id:
                continue
            dets = sorted(rec.detections, key=lambda det: det.index)
            if dets:
                prev_q = rbox2qbox(torch.as_tensor(
                    [det.prev_bbox for det in dets], dtype=torch.float32))
                curr_q = rbox2qbox(torch.as_tensor(
                    [det.curr_bbox for det in dets], dtype=torch.float32))
                prev_q_np = prev_q.detach().cpu().numpy()
                curr_q_np = curr_q.detach().cpu().numpy()
            else:
                prev_q_np = np.zeros((0, 8), dtype=np.float32)
                curr_q_np = np.zeros((0, 8), dtype=np.float32)
            for row_idx, det in enumerate(dets):
                prev_cls = det.label_prev if det.label_prev is not None else det.label
                curr_cls = det.label_curr if det.label_curr is not None else det.label
                presence_prev = 1.0 if det.presence_prev is None else det.presence_prev
                presence_curr = 1.0 if det.presence_curr is None else det.presence_curr
                vals = [
                    int(rec.curr_frame_id),
                    int(rec.prev_frame_id),
                    int(det.index),
                ]
                vals += [f'{float(v):.2f}' for v in prev_q_np[row_idx].tolist()]
                vals += [int(prev_cls), f'{det.prev_side_score():.6f}']
                vals += [f'{float(v):.2f}' for v in curr_q_np[row_idx].tolist()]
                vals += [int(curr_cls), f'{det.curr_side_score():.6f}']
                vals += [
                    int(det.label),
                    f'{det.pair_score():.6f}',
                    f'{float(det.cls_score):.6f}',
                    f'{float(presence_prev):.6f}',
                    f'{float(presence_curr):.6f}',
                ]
                f.write(','.join(map(str, vals)) + '\n')


def read_pair_det_txt(path: str, seq_name: Optional[str] = None
                      ) -> List[PairFrameRecord]:
    """Read canonical pair detection txt into PairFrameRecord objects."""
    if seq_name is None:
        seq_name = osp.splitext(osp.basename(path))[0]
    by_pair: Dict[Tuple[int, int], List[PairDetection]] = {}
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split(',')
            if len(parts) < 28:
                raise ValueError(f'Invalid pair det row in {path}: {line}')
            curr_frame = int(float(parts[0]))
            prev_frame = int(float(parts[1]))
            det_index = int(float(parts[2]))
            prev_q = torch.as_tensor(
                [float(v) for v in parts[3:11]], dtype=torch.float32).view(1, 8)
            prev_bbox = qbox2rbox(prev_q).reshape(-1).tolist()
            prev_cls = int(float(parts[11]))
            prev_score = float(parts[12])
            curr_q = torch.as_tensor(
                [float(v) for v in parts[13:21]], dtype=torch.float32).view(1, 8)
            curr_bbox = qbox2rbox(curr_q).reshape(-1).tolist()
            curr_cls = int(float(parts[21]))
            curr_score = float(parts[22])
            pair_cls = int(float(parts[23]))
            pair_score = float(parts[24])
            cls_score = float(parts[25])
            presence_prev = float(parts[26]) if len(parts) > 26 else None
            presence_curr = float(parts[27]) if len(parts) > 27 else None
            by_pair.setdefault((prev_frame, curr_frame), []).append(
                PairDetection(
                    index=det_index,
                    prev_bbox=[float(x) for x in prev_bbox],
                    curr_bbox=[float(x) for x in curr_bbox],
                    score=pair_score,
                    cls_score=cls_score,
                    label=pair_cls,
                    presence_prev=presence_prev,
                    presence_curr=presence_curr,
                    score_prev=prev_score,
                    score_curr=curr_score,
                    label_prev=prev_cls,
                    label_curr=curr_cls,
                ))
    records = []
    for (prev_frame, curr_frame), detections in sorted(by_pair.items()):
        records.append(PairFrameRecord(
            seq_name=str(seq_name),
            prev_frame_id=int(prev_frame),
            curr_frame_id=int(curr_frame),
            frame_gap=int(curr_frame) - int(prev_frame),
            prev_img_path='',
            curr_img_path='',
            img_shape=[],
            ori_shape=[],
            scale_factor=[],
            detections=sorted(detections, key=lambda det: det.index),
            is_first_pair=False,
        ))
    return records


def bootstrap_first_record_from_pair(record: PairFrameRecord) -> PairFrameRecord:
    """Build frame-1 initialization from the previous side of a 01-02 pair."""
    detections = []
    for det in record.detections:
        prev_score = det.prev_side_score()
        detections.append(PairDetection(
            index=det.index,
            prev_bbox=list(det.prev_bbox),
            curr_bbox=list(det.prev_bbox),
            score=prev_score,
            cls_score=det.cls_score,
            label=det.label_prev if det.label_prev is not None else det.label,
            presence_prev=det.presence_prev,
            presence_curr=det.presence_prev,
            score_prev=det.score_prev,
            score_curr=det.score_prev,
            label_prev=det.label_prev,
            label_curr=det.label_prev,
        ))
    return PairFrameRecord(
        seq_name=record.seq_name,
        prev_frame_id=record.prev_frame_id,
        curr_frame_id=record.prev_frame_id,
        frame_gap=0,
        prev_img_path=record.prev_img_path,
        curr_img_path=record.prev_img_path,
        img_shape=list(record.img_shape),
        ori_shape=list(record.ori_shape),
        scale_factor=list(record.scale_factor),
        detections=detections,
        is_first_pair=True,
    )


def detection_score(
    cls_score: float,
    presence_prev: Optional[float],
    presence_curr: Optional[float],
    mode: str,
) -> float:
    cls_score = float(cls_score)
    if mode == 'cls':
        return cls_score
    if presence_prev is None or presence_curr is None:
        return cls_score
    pres_min = min(float(presence_prev), float(presence_curr))
    if mode == 'cls_min_presence':
        return cls_score * pres_min
    if mode == 'auto':
        # Presence can be badly calibrated; only down-weight when it is clearly
        # confident.  Otherwise fall back to class score.
        if max(float(presence_prev), float(presence_curr)) >= 0.5:
            return cls_score * pres_min
        return cls_score
    raise ValueError(f'Unsupported score mode: {mode}')


class RotatedKalman:
    """Small constant-velocity Kalman filter for cx, cy, w, h, angle."""

    ndim = 5
    dt = 1.0

    def initiate(self, measurement: Sequence[float]) -> Tuple[np.ndarray, np.ndarray]:
        mean = np.zeros((2 * self.ndim,), dtype=np.float32)
        mean[:self.ndim] = np.asarray(measurement, dtype=np.float32)
        mean[4] = wrap_angle(float(mean[4]))
        covariance = np.eye(2 * self.ndim, dtype=np.float32)
        covariance[:self.ndim, :self.ndim] *= 10.0
        covariance[self.ndim:, self.ndim:] *= 100.0
        return mean, covariance

    def predict(self, mean: np.ndarray,
                covariance: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        motion = np.eye(2 * self.ndim, dtype=np.float32)
        for i in range(self.ndim):
            motion[i, self.ndim + i] = self.dt
        std_pos = np.array([2.0, 2.0, 1.0, 1.0, 0.05], dtype=np.float32)
        std_vel = np.array([1.0, 1.0, 0.5, 0.5, 0.02], dtype=np.float32)
        motion_cov = np.diag(np.r_[std_pos, std_vel] ** 2).astype(np.float32)
        mean = motion @ mean
        mean[4] = wrap_angle(float(mean[4]))
        covariance = motion @ covariance @ motion.T + motion_cov
        return mean, covariance

    def update(self, mean: np.ndarray, covariance: np.ndarray,
               measurement: Sequence[float]) -> Tuple[np.ndarray, np.ndarray]:
        measurement = np.asarray(measurement, dtype=np.float32).copy()
        measurement[4] = mean[4] + wrap_angle(float(measurement[4] - mean[4]))
        proj = np.eye(self.ndim, 2 * self.ndim, dtype=np.float32)
        innovation_cov = np.diag(
            np.array([1.0, 1.0, 0.5, 0.5, 0.03], dtype=np.float32) ** 2)
        projected_mean = proj @ mean
        projected_cov = proj @ covariance @ proj.T + innovation_cov
        kalman_gain = covariance @ proj.T @ np.linalg.inv(projected_cov)
        innovation = measurement - projected_mean
        innovation[4] = wrap_angle(float(innovation[4]))
        mean = mean + kalman_gain @ innovation
        mean[4] = wrap_angle(float(mean[4]))
        covariance = covariance - kalman_gain @ projected_cov @ kalman_gain.T
        return mean.astype(np.float32), covariance.astype(np.float32)


@dataclass
class Track:
    track_id: int
    label: int
    score: float
    bbox: List[float]
    mean: np.ndarray
    covariance: np.ndarray
    state: str = 'Tracked'
    age: int = 1
    hits: int = 1
    time_since_update: int = 0
    last_frame_id: int = 0
    history: List[Tuple[int, List[float], float, int]] = field(default_factory=list)

    def predicted_bbox(self) -> List[float]:
        box = self.mean[:5].astype(float).tolist()
        box[2] = max(box[2], 1.0)
        box[3] = max(box[3], 1.0)
        box[4] = wrap_angle(box[4])
        return box


class PairMOTTracker:
    """Online tracker using pair detections as the association signal."""

    def __init__(self,
                 *,
                 new_born_th: float = 0.5,
                 track_th: float = 0.2,
                 match_iou_th: float = 0.3,
                 new_birth_iou_th: float = 0.6,
                 max_age: int = 30,
                 init_same_iou_th: float = 0.3,
                 class_aware: bool = False) -> None:
        self.new_born_th = float(new_born_th)
        self.track_th = float(track_th)
        self.match_iou_th = float(match_iou_th)
        self.new_birth_iou_th = float(new_birth_iou_th)
        self.max_age = int(max_age)
        self.init_same_iou_th = float(init_same_iou_th)
        self.class_aware = bool(class_aware)
        self.kalman = RotatedKalman()
        self.tracks: List[Track] = []
        self.finished_tracks: List[Track] = []
        self.last_events: List[dict] = []
        self.next_id = 1

    def reset(self) -> None:
        self.tracks = []
        self.finished_tracks = []
        self.last_events = []
        self.next_id = 1

    def predict(self) -> None:
        for track in self.tracks:
            self._advance_track_to_frame(track, track.last_frame_id + 1)

    def _advance_track_to_frame(self, track: Track, frame_id: int) -> None:
        frame_id = int(frame_id)
        if track.last_frame_id > frame_id:
            raise ValueError(
                f'Track {track.track_id} is at frame {track.last_frame_id}, '
                f'cannot move back to frame {frame_id}')
        while track.last_frame_id < frame_id:
            track.mean, track.covariance = self.kalman.predict(
                track.mean, track.covariance)
            track.bbox = track.predicted_bbox()
            track.age += 1
            track.time_since_update += 1
            track.last_frame_id += 1

    def _new_track(self, bbox: Sequence[float], score: float, label: int,
                   frame_id: int) -> Track:
        mean, covariance = self.kalman.initiate(bbox)
        track = Track(
            track_id=self.next_id,
            label=int(label),
            score=float(score),
            bbox=[float(x) for x in bbox],
            mean=mean,
            covariance=covariance,
            last_frame_id=int(frame_id),
        )
        track.history.append((int(frame_id), track.bbox, float(score), int(label)))
        self.tracks.append(track)
        self.next_id += 1
        return track

    def _duplicate_iou_with_tracks(self, bbox: Sequence[float], label: int,
                                   tracks: Sequence[Track]) -> float:
        candidates = [
            tr.bbox for tr in tracks
            if tr.state == 'Tracked'
            and tr.time_since_update == 0
            and ((not self.class_aware) or tr.label == int(label))
        ]
        if not candidates:
            return 0.0
        ious = rotated_iou_matrix([bbox], candidates)
        return float(ious.max()) if ious.size else 0.0

    def _update_track(self, track: Track, bbox: Sequence[float], score: float,
                      label: int, frame_id: int) -> None:
        if track.last_frame_id < int(frame_id):
            self._advance_track_to_frame(track, int(frame_id))
        track.mean, track.covariance = self.kalman.update(
            track.mean, track.covariance, bbox)
        track.bbox = [float(x) for x in bbox]
        track.score = float(score)
        track.label = int(label)
        track.state = 'Tracked'
        track.time_since_update = 0
        track.hits += 1
        track.last_frame_id = int(frame_id)
        track.history.append((int(frame_id), track.bbox, float(score), int(label)))

    def _mark_lost_and_prune(self) -> None:
        kept = []
        for track in self.tracks:
            if track.time_since_update > 0:
                track.state = 'Lost'
            if track.time_since_update <= self.max_age:
                kept.append(track)
            else:
                track.state = 'Removed'
                self.finished_tracks.append(track)
        self.tracks = kept

    def init_first_frame(self, record: PairFrameRecord) -> List[Track]:
        created = []
        self.last_events = []
        for det in record.detections:
            if det.birth_score() < self.new_born_th:
                continue
            iou = rotated_iou_matrix([det.prev_bbox], [det.curr_bbox])[0, 0]
            if iou >= self.init_same_iou_th:
                bbox = [
                    float((a + b) * 0.5)
                    for a, b in zip(det.prev_bbox, det.curr_bbox)
                ]
                bbox[4] = wrap_angle(bbox[4])
            else:
                bbox = det.curr_bbox
            duplicate_iou = self._duplicate_iou_with_tracks(
                bbox, det.label, created)
            if duplicate_iou >= self.new_birth_iou_th:
                self.last_events.append({
                    'event': 'birth_suppressed',
                    'frame_id': record.curr_frame_id,
                    'det_index': det.index,
                    'score': det.birth_score(),
                    'pair_score': det.pair_score(),
                    'prev_score': det.prev_side_score(),
                    'curr_score': det.curr_side_score(),
                    'label': det.label,
                    'duplicate_iou': duplicate_iou,
                    'new_birth_iou_th': self.new_birth_iou_th,
                })
                continue
            track = self._new_track(
                bbox, det.birth_score(), det.label, record.curr_frame_id)
            created.append(track)
            self.last_events.append({
                'event': 'birth',
                'frame_id': record.curr_frame_id,
                'track_id': track.track_id,
                'det_index': det.index,
                'score': det.birth_score(),
                'pair_score': det.pair_score(),
                'prev_score': det.prev_side_score(),
                'curr_score': det.curr_side_score(),
                'label': det.label,
                'same_frame_iou': float(iou),
            })
        return created

    def update_pair(self, record: PairFrameRecord) -> List[Track]:
        self.last_events = []
        for track in self.tracks:
            self._advance_track_to_frame(track, record.prev_frame_id)
        # Match to previous-frame tracks only when the previous side is
        # credible. Current-side confidence is checked after a match and is
        # also used independently for new births.
        detections = [
            det for det in record.detections
            if det.prev_side_score() >= self.track_th
        ]
        candidate_tracks = [
            tr for tr in self.tracks
            if tr.state in ('Tracked', 'Lost') and tr.time_since_update <= self.max_age
            and tr.last_frame_id == record.prev_frame_id
        ]

        matches: List[Tuple[int, int, float]] = []
        unmatched_track_idx = set(range(len(candidate_tracks)))
        unmatched_det_idx = set(range(len(detections)))
        diag_by_track: Dict[int, dict] = {}
        consumed_det_indices = set()
        curr_low_track_idx = set()
        if candidate_tracks and detections:
            track_boxes = [tr.bbox for tr in candidate_tracks]
            det_prev_boxes = [det.prev_bbox for det in detections]
            ious = rotated_iou_matrix(track_boxes, det_prev_boxes)
            raw_ious = ious.copy()
            if self.class_aware:
                for ti, tr in enumerate(candidate_tracks):
                    for di, det in enumerate(detections):
                        if tr.label != det.label:
                            ious[ti, di] = -1.0
            for ti, tr in enumerate(candidate_tracks):
                if raw_ious.shape[1] == 0:
                    continue
                best_di = int(np.argmax(raw_ious[ti]))
                best_det = detections[best_di]
                best_iou = float(raw_ious[ti, best_di])
                class_ok = (not self.class_aware) or tr.label == best_det.label
                diag_by_track[ti] = {
                    'event': 'match_diag',
                    'frame_id': record.curr_frame_id,
                    'prev_frame_id': record.prev_frame_id,
                    'track_id': tr.track_id,
                    'track_state': tr.state,
                    'track_label': tr.label,
                    'track_score': tr.score,
                    'track_time_since_update': tr.time_since_update,
                    'track_bbox': tr.bbox,
                    'best_det_index': best_det.index,
                    'best_det_label': best_det.label,
                    'best_det_score': best_det.score,
                    'best_det_pair_score': best_det.pair_score(),
                    'best_det_prev_score': best_det.prev_side_score(),
                    'best_det_curr_score': best_det.curr_side_score(),
                    'best_iou': best_iou,
                    'match_iou_th': self.match_iou_th,
                    'class_ok': class_ok,
                    'would_match_by_iou': best_iou >= self.match_iou_th,
                    'matched': False,
                }
            rows, cols = linear_sum_assignment(-ious)
            for row, col in zip(rows.tolist(), cols.tolist()):
                if ious[row, col] < self.match_iou_th:
                    continue
                matches.append((row, col, float(ious[row, col])))

        for track_idx, det_idx, match_iou in matches:
            track = candidate_tracks[track_idx]
            det = detections[det_idx]
            consumed_det_indices.add(det.index)
            unmatched_det_idx.discard(det_idx)
            if det.curr_side_score() >= self.track_th:
                unmatched_track_idx.discard(track_idx)
                self._update_track(
                    track, det.curr_bbox, det.curr_side_score(), det.label,
                    record.curr_frame_id)
                self.last_events.append({
                    'event': 'match',
                    'frame_id': record.curr_frame_id,
                    'track_id': track.track_id,
                    'det_index': det.index,
                    'score': det.curr_side_score(),
                    'pair_score': det.pair_score(),
                    'prev_score': det.prev_side_score(),
                    'curr_score': det.curr_side_score(),
                    'label': det.label,
                    'match_iou': match_iou,
                })
            else:
                curr_low_track_idx.add(track_idx)
                self.last_events.append({
                    'event': 'matched_prev_curr_low',
                    'frame_id': record.curr_frame_id,
                    'track_id': track.track_id,
                    'det_index': det.index,
                    'score': det.curr_side_score(),
                    'pair_score': det.pair_score(),
                    'prev_score': det.prev_side_score(),
                    'curr_score': det.curr_side_score(),
                    'label': det.label,
                    'match_iou': match_iou,
                    'track_th': self.track_th,
                })

        matched_track_idx = {track_idx for track_idx, _, _ in matches}
        matched_det_idx = {det_idx for _, det_idx, _ in matches}
        for track_idx in sorted(unmatched_track_idx):
            diag = diag_by_track.get(track_idx)
            if diag is None:
                continue
            best_det_idx = next(
                (idx for idx, det in enumerate(detections)
                 if det.index == diag['best_det_index']),
                None)
            if not diag['class_ok']:
                diag['reason'] = 'class_constraint'
            elif diag['best_iou'] < self.match_iou_th:
                diag['reason'] = 'iou_below_threshold'
            elif track_idx in curr_low_track_idx:
                diag['reason'] = 'curr_score_below_track_threshold'
            elif best_det_idx in matched_det_idx:
                diag['reason'] = 'best_det_taken_by_one_to_one_assignment'
            else:
                diag['reason'] = 'not_selected_by_hungarian'
            self.last_events.append(diag)

        for track_idx in sorted(unmatched_track_idx):
            track = candidate_tracks[track_idx]
            self._advance_track_to_frame(track, record.curr_frame_id)

        self._mark_lost_and_prune()

        # Birth is a current-frame decision. It must consider all detections,
        # including those whose previous side was too weak for track matching.
        for det in sorted(record.detections, key=lambda item: item.index):
            if det.index in consumed_det_indices:
                continue
            if det.birth_score() >= self.new_born_th:
                duplicate_iou = self._duplicate_iou_with_tracks(
                    det.curr_bbox, det.label, self.tracks)
                if duplicate_iou >= self.new_birth_iou_th:
                    self.last_events.append({
                        'event': 'birth_suppressed',
                        'frame_id': record.curr_frame_id,
                        'det_index': det.index,
                        'score': det.birth_score(),
                        'pair_score': det.pair_score(),
                        'prev_score': det.prev_side_score(),
                        'curr_score': det.curr_side_score(),
                        'label': det.label,
                        'duplicate_iou': duplicate_iou,
                        'new_birth_iou_th': self.new_birth_iou_th,
                    })
                    continue
                track = self._new_track(
                    det.curr_bbox, det.birth_score(), det.label,
                    record.curr_frame_id)
                self.last_events.append({
                    'event': 'birth',
                    'frame_id': record.curr_frame_id,
                    'track_id': track.track_id,
                    'det_index': det.index,
                    'score': det.birth_score(),
                    'pair_score': det.pair_score(),
                    'prev_score': det.prev_side_score(),
                    'curr_score': det.curr_side_score(),
                    'label': det.label,
                })

        return [tr for tr in self.tracks if tr.state == 'Tracked']

    def all_history(self) -> List[Tuple[int, int, List[float], float, int]]:
        rows = []
        for track in self.finished_tracks + self.tracks:
            for frame_id, bbox, score, label in track.history:
                rows.append((frame_id, track.track_id, bbox, score, label))
        rows.sort(key=lambda item: (item[0], item[1]))
        return rows


def write_trackeval_txt(path: str,
                        rows: Iterable[Tuple[int, int, Sequence[float], float, int]]
                        ) -> None:
    rows = list(rows)
    os.makedirs(osp.dirname(path), exist_ok=True)
    if not rows:
        open(path, 'w', encoding='utf-8').close()
        return
    rboxes = torch.as_tensor([row[2] for row in rows], dtype=torch.float32)
    qboxes = rbox2qbox(rboxes).detach().cpu().numpy()
    with open(path, 'w', encoding='utf-8') as f:
        for (frame_id, track_id, _, score, label), qbox in zip(rows, qboxes):
            vals = [int(frame_id), int(track_id)]
            vals += [f'{float(v):.2f}' for v in qbox.tolist()]
            vals += [f'{float(score):.6f}', int(label), 0]
            f.write(','.join(map(str, vals)) + '\n')
