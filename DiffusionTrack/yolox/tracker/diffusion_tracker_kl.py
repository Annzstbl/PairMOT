"""Kalman DiffusionTracker adapted to multispectral rotated HSMOT.

The tracking state machine is the original paper/default KL implementation:
active tracks are propagated by Pair Diffusion, unmatched lost tracks are
predicted by Kalman filtering and can be recovered by current-frame detections.
Only image/box/class representations and their geometry are changed here.
"""

import math
import time

import numpy as np
import torch

from yolox.utils.rotated_boxes import (
    batched_rotated_nms, pair_cluster_nms_rotated, rotated_iou)

from .basetrack import BaseTrack, TrackState
from .kalman_filter import KalmanFilter
from . import matching


def _as_rboxes(items):
    if len(items) == 0:
        return np.empty((0, 5), dtype=np.float32)
    first = items[0]
    if isinstance(first, STrack):
        return np.asarray([item.rbox for item in items], dtype=np.float32)
    array = np.asarray(items, dtype=np.float32)
    return array[:, :5] if array.ndim == 2 else array.reshape(-1, 5)


def rotated_distance(items_a, items_b, class_aware=False):
    """Return the original association cost convention, ``1 - rotated IoU``."""
    boxes_a, boxes_b = _as_rboxes(items_a), _as_rboxes(items_b)
    if len(boxes_a) == 0 or len(boxes_b) == 0:
        return np.ones((len(boxes_a), len(boxes_b)), dtype=np.float32)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    distance = 1 - rotated_iou(
        torch.from_numpy(boxes_a).to(device),
        torch.from_numpy(boxes_b).to(device)).cpu().numpy()
    if class_aware:
        classes_a = np.asarray([
            item.class_id if isinstance(item, STrack) else item[7]
            for item in items_a], dtype=np.int64)
        classes_b = np.asarray([
            item.class_id if isinstance(item, STrack) else item[7]
            for item in items_b], dtype=np.int64)
        distance[classes_a[:, None] != classes_b[None, :]] = 1.0
    return distance


class STrack(BaseTrack):
    """Rotated track whose cxcywh motion follows the original xyah Kalman filter."""

    shared_kalman = KalmanFilter()

    def __init__(self, rbox, score, class_id=0):
        self._rbox = np.asarray(rbox, dtype=np.float32)
        self.kalman_filter = None
        self.mean, self.covariance = None, None
        self.is_activated = False
        self.score = float(score)
        self.class_id = int(class_id)
        self.tracklet_len = 0

    @staticmethod
    def rbox_to_xyah(rbox):
        ret = np.asarray(rbox[:4], dtype=np.float32).copy()
        ret[2] /= max(ret[3], 1e-6)
        return ret

    @property
    def rbox(self):
        if self.mean is None:
            return self._rbox.copy()
        ret = self._rbox.copy()
        ret[:4] = self.mean[:4]
        ret[2] *= ret[3]
        return ret

    @property
    def tlwh(self):
        ret = self.rbox[:4].copy()
        ret[:2] -= ret[2:] / 2
        return ret

    @property
    def tlbr(self):
        ret = self.tlwh
        ret[2:] += ret[:2]
        return ret

    def predict(self):
        mean_state = self.mean.copy()
        if self.state != TrackState.Tracked:
            mean_state[7] = 0
        self.mean, self.covariance = self.kalman_filter.predict(
            mean_state, self.covariance)

    @staticmethod
    def multi_predict(stracks):
        if not stracks:
            return
        multi_mean = np.asarray([st.mean.copy() for st in stracks])
        multi_covariance = np.asarray([st.covariance for st in stracks])
        for index, track in enumerate(stracks):
            if track.state != TrackState.Tracked:
                multi_mean[index][7] = 0
        multi_mean, multi_covariance = STrack.shared_kalman.multi_predict(
            multi_mean, multi_covariance)
        for track, mean, covariance in zip(stracks, multi_mean, multi_covariance):
            track.mean, track.covariance = mean, covariance

    def activate(self, kalman_filter, frame_id):
        self.kalman_filter = kalman_filter
        self.track_id = self.next_id()
        self.mean, self.covariance = self.kalman_filter.initiate(
            self.rbox_to_xyah(self._rbox))
        self.tracklet_len = 0
        self.state = TrackState.Tracked
        if frame_id == 1:
            self.is_activated = True
        self.frame_id = frame_id
        self.start_frame = frame_id

    def re_activate(self, new_track, frame_id, new_id=False):
        self.mean, self.covariance = self.kalman_filter.update(
            self.mean, self.covariance, self.rbox_to_xyah(new_track.rbox))
        self._rbox = new_track.rbox.copy()
        self.tracklet_len = 0
        self.state = TrackState.Tracked
        self.is_activated = True
        self.frame_id = frame_id
        if new_id:
            self.track_id = self.next_id()
        self.score = new_track.score
        self.class_id = new_track.class_id

    def update(self, new_track, frame_id):
        self.frame_id = frame_id
        self.tracklet_len += 1
        new_rbox = new_track.rbox
        self.mean, self.covariance = self.kalman_filter.update(
            self.mean, self.covariance, self.rbox_to_xyah(new_rbox))
        self._rbox = new_rbox.copy()
        self.state = TrackState.Tracked
        self.is_activated = True
        self.score = new_track.score
        self.class_id = new_track.class_id

    @staticmethod
    def tlbr_to_tlwh(tlbr):
        ret = np.asarray(tlbr).copy()
        ret[2:] -= ret[:2]
        return ret

    @staticmethod
    def tlwh_to_tlbr(tlwh):
        ret = np.asarray(tlwh).copy()
        ret[2:] += ret[:2]
        return ret

    def __repr__(self):
        return 'OT_{}_({}-{})'.format(
            self.track_id, self.start_frame, self.end_frame)


class DiffusionTracker:
    def __init__(self, model, tensor_type, conf_thresh=0.7, det_thresh=0.6,
                 nms_thresh_3d=0.7, nms_thresh_2d=0.75, interval=5,
                 detections=None):
        self.frame_id = 0
        self.backbone = model.backbone
        self.feature_projs = model.projs
        self.diffusion_model = model.head
        self.feature_extractor = self.diffusion_model.head.box_pooler
        self.det_thresh = det_thresh
        self.association_thresh = conf_thresh
        self.nms_thresh_2d = nms_thresh_2d
        self.nms_thresh_3d = nms_thresh_3d
        self.same_thresh = 0.9
        self.pre_features = None
        self.data_type = tensor_type
        self.detections = detections

        self.tracked_stracks = []
        self.lost_stracks = []
        self.removed_stracks = []
        self.max_time_lost = 30
        self.kalman_filter = KalmanFilter()

        self.repeat_times = 0
        self.dynamic_time = True
        self.sampling_steps = 1
        self.num_boxes = 500
        self.track_t = 400
        self.mot17 = False
        self.last_pair_detections = None

    def _new_track(self, detection):
        return STrack(detection[:5], detection[6], detection[7])

    def update(self, cur_image):
        self.frame_id += 1
        activated_stracks, refind_stracks = [], []
        lost_stracks, removed_stracks = [], []
        cur_features, mate_info = self.extract_feature(cur_image)
        mate_shape, mate_device, mate_dtype = mate_info
        self.diffusion_model.device = mate_device
        self.diffusion_model.dtype = mate_dtype
        batch, _, height, width = mate_shape
        images_whwh = torch.tensor(
            [width, height, width, height], dtype=mate_dtype,
            device=mate_device)[None].expand(4 * batch, 4)

        if self.frame_id == 1:
            self.last_pair_detections = None
            if self.pre_features is None:
                self.pre_features = cur_features
            inps = self.prepare_input(self.pre_features, cur_features)
            outputs, conf_scores, association_time = \
                self.diffusion_model.new_ddim_sample(
                    inps, images_whwh, num_timesteps=self.sampling_steps,
                    num_proposals=self.num_boxes, dynamic_time=self.dynamic_time,
                    track_candidate=self.repeat_times)
            _, _, detections = self.diffusion_postprocess(
                outputs, conf_scores, self.nms_thresh_3d,
                self.association_thresh)
            detections = self.diffusion_det_filt(
                detections, self.det_thresh, self.nms_thresh_2d)
            for detection in detections:
                track = self._new_track(detection)
                track.activate(self.kalman_filter, self.frame_id)
                self.tracked_stracks.append(track)
            return [t for t in self.tracked_stracks if t.is_activated], association_time

        ref_rboxes = np.asarray(
            [track.rbox for track in self.tracked_stracks], dtype=np.float32)
        inps = self.prepare_input(self.pre_features, cur_features)
        if len(ref_rboxes):
            ref_targets = torch.as_tensor(
                ref_rboxes, device=mate_device, dtype=mate_dtype
            ).reshape(1, -1, 5).repeat(2, 1, 1)
        else:
            ref_targets = None
        outputs, conf_scores, association_time = \
            self.diffusion_model.new_ddim_sample(
                inps, images_whwh, num_timesteps=self.sampling_steps,
                num_proposals=self.num_boxes, ref_targets=ref_targets,
                dynamic_time=self.dynamic_time,
                track_candidate=self.repeat_times, diffusion_t=self.track_t)
        ref_dets, track_dets, detections = self.diffusion_postprocess(
            outputs, conf_scores, self.nms_thresh_3d,
            self.association_thresh)
        detections = self.diffusion_det_filt(
            detections, self.det_thresh, self.nms_thresh_2d)
        ref_dets, track_dets = self.diffusion_track_filt(
            ref_dets, track_dets, self.det_thresh, self.nms_thresh_2d)
        self.last_pair_detections = (ref_dets.copy(), track_dets.copy())

        start_time = time.time()
        STrack.multi_predict(self.tracked_stracks)
        dists = rotated_distance(self.tracked_stracks, ref_dets, class_aware=True)
        matches, unmatched_tracks, _ = matching.linear_assignment(
            dists, thresh=self.same_thresh)

        unmatched_detections = np.arange(len(detections))
        if len(matches):
            paired_current = track_dets[matches[:, 1]]
            dists_fix = rotated_distance(
                paired_current, detections, class_aware=True)
            matches_fix, _, unmatched_detections = matching.linear_assignment(
                dists_fix, thresh=self.same_thresh)
            for pair_index, detection_index in matches_fix:
                paired_current[pair_index, :8] = detections[detection_index, :8]
            track_dets[matches[:, 1]] = paired_current

        remaining_detections = detections[unmatched_detections]
        ref_box_t, track_box_t = [], []
        for track_index, detection_index in matches:
            track = self.tracked_stracks[track_index]
            detection = track_dets[detection_index]
            ref_box_t.append(track.rbox[:4])
            track_box_t.append(detection[:4])
            new_track = self._new_track(detection)
            if track.state == TrackState.Tracked:
                track.update(new_track, self.frame_id)
                activated_stracks.append(track)
            else:
                track.re_activate(new_track, self.frame_id, new_id=False)
                refind_stracks.append(track)
        if ref_box_t:
            self.track_t = self.extract_mean_track_t(
                np.asarray(ref_box_t), np.asarray(track_box_t))
        for track_index in unmatched_tracks:
            track = self.tracked_stracks[track_index]
            if track.state != TrackState.Lost:
                track.mark_lost()
                lost_stracks.append(track)

        # This is the original paper's lost-target mechanism: Kalman predicts
        # old tracks, then current-frame detections can recover their IDs.
        STrack.multi_predict(self.lost_stracks)
        dists_lost = rotated_distance(
            self.lost_stracks, remaining_detections, class_aware=True)
        matches_lost, _, unmatched_detection_lost = matching.linear_assignment(
            dists_lost, thresh=self.same_thresh)
        for track_index, detection_index in matches_lost:
            track = self.lost_stracks[track_index]
            new_track = self._new_track(remaining_detections[detection_index])
            track.re_activate(new_track, self.frame_id, new_id=False)
            refind_stracks.append(track)

        for detection_index in unmatched_detection_lost:
            track = self._new_track(remaining_detections[detection_index])
            track.activate(self.kalman_filter, self.frame_id)
            activated_stracks.append(track)

        for track in self.lost_stracks:
            if self.frame_id - track.end_frame > self.max_time_lost:
                track.mark_removed()
                removed_stracks.append(track)

        self.tracked_stracks = [
            track for track in self.tracked_stracks
            if track.state == TrackState.Tracked]
        self.tracked_stracks = joint_stracks(
            self.tracked_stracks, activated_stracks)
        self.tracked_stracks = joint_stracks(
            self.tracked_stracks, refind_stracks)
        self.lost_stracks = sub_stracks(
            self.lost_stracks, self.tracked_stracks)
        self.lost_stracks.extend(lost_stracks)
        self.lost_stracks = sub_stracks(
            self.lost_stracks, self.removed_stracks)
        self.removed_stracks.extend(removed_stracks)
        self.tracked_stracks, self.lost_stracks = remove_duplicate_stracks(
            self.tracked_stracks, self.lost_stracks)

        self.pre_features = cur_features
        return self.tracked_stracks, association_time + time.time() - start_time

    def extract_feature(self, cur_image):
        fpn_outs = self.backbone(cur_image)
        cur_features = [
            projection(feature)
            for projection, feature in zip(self.feature_projs, fpn_outs)]
        mate_info = (cur_image.shape, cur_image.device, cur_image.dtype)
        return cur_features, mate_info

    def extract_mean_track_t(self, pre_box, cur_box):
        relative = np.abs(pre_box - cur_box) / (np.abs(pre_box) + 1e-5)
        return min(max(int(np.mean(relative.sum(axis=1) / 4) * 1000), 1), 999)

    def diffusion_postprocess(self, diffusion_outputs, conf_scores,
                              nms_thre=0.7, conf_thre=0.6):
        pre_predictions, cur_predictions = diffusion_outputs.split(
            len(diffusion_outputs) // 2, dim=0)
        output = []
        for pre_pred, cur_pred, association_score in zip(
                pre_predictions, cur_predictions, conf_scores):
            association_score = association_score.flatten()
            class_conf_pre, class_pre = pre_pred[:, 5:].sigmoid().max(dim=1)
            class_conf_cur, class_cur = cur_pred[:, 5:].sigmoid().max(dim=1)
            pair_class_conf = torch.sqrt(class_conf_pre * class_conf_cur)
            pair_class = torch.where(
                class_conf_pre >= class_conf_cur, class_pre, class_cur)
            detections = pre_pred.new_zeros((2, len(cur_pred), 8))
            detections[0, :, :5] = pre_pred[:, :5]
            detections[1, :, :5] = cur_pred[:, :5]
            detections[:, :, 5] = association_score[None]
            detections[:, :, 6] = torch.sqrt(
                pair_class_conf * association_score)[None]
            detections[:, :, 7] = pair_class.to(detections.dtype)[None]
            keep = association_score > conf_thre
            detections = detections[:, keep]
            if detections.size(1):
                kept_by_class = []
                for class_id in detections[0, :, 7].unique(sorted=True):
                    indices = torch.where(detections[0, :, 7] == class_id)[0]
                    local = pair_cluster_nms_rotated(
                        detections[0, indices, :5],
                        detections[1, indices, :5],
                        detections[0, indices, 5], nms_thre)
                    kept_by_class.append(indices[local])
                keep = torch.cat(kept_by_class)
                keep = keep[torch.argsort(
                    detections[0, keep, 5], descending=True)]
                detections = detections[:, keep]
            output.append(detections)

        empty = diffusion_outputs.new_zeros((0, 8))
        ref = output[0][0] if output else empty
        track = output[0][1] if output else empty
        current = (torch.cat([output[1][0], output[1][1]], dim=0)
                   if len(output) >= 2 else empty)
        return ref, track, current

    def diffusion_track_filt(self, ref_detections, track_detections,
                             conf_thre=0.6, nms_thre=0.7):
        if not ref_detections.size(0):
            return (ref_detections.cpu().numpy(),
                    track_detections.cpu().numpy())
        keep = ref_detections[:, 6] > conf_thre
        ref_detections, track_detections = (
            ref_detections[keep], track_detections[keep])
        keep = batched_rotated_nms(
            ref_detections[:, :5], ref_detections[:, 6],
            ref_detections[:, 7].long(), nms_thre)
        return (ref_detections[keep].cpu().numpy(),
                track_detections[keep].cpu().numpy())

    def diffusion_det_filt(self, detections, conf_thre=0.6, nms_thre=0.7):
        if not detections.size(0):
            return detections.cpu().numpy()
        detections = detections[detections[:, 6] > conf_thre]
        if not detections.size(0):
            return detections.cpu().numpy()
        keep = batched_rotated_nms(
            detections[:, :5], detections[:, 6],
            detections[:, 7].long(), nms_thre)
        return detections[keep].cpu().numpy()

    def proposal_schedule(self, num_ref_bboxes):
        return 16 * num_ref_bboxes

    def sampling_steps_schedule(self, num_ref_bboxes):
        value = (num_ref_bboxes - 10) * 3 / 90 + 1
        return min(max(int(value), 1), 4)

    def prepare_input(self, pre_features, cur_features):
        inputs_pre, inputs_cur = [], []
        for pre_feature, cur_feature in zip(pre_features, cur_features):
            inputs_pre.append(torch.cat(
                [pre_feature.clone(), cur_feature.clone()], dim=0))
            inputs_cur.append(torch.cat(
                [cur_feature.clone(), cur_feature.clone()], dim=0))
        return inputs_pre, inputs_cur


def joint_stracks(tlista, tlistb):
    exists, result = {}, []
    for track in tlista:
        exists[track.track_id] = True
        result.append(track)
    for track in tlistb:
        if track.track_id not in exists:
            exists[track.track_id] = True
            result.append(track)
    return result


def sub_stracks(tlista, tlistb):
    tracks = {track.track_id: track for track in tlista}
    for track in tlistb:
        tracks.pop(track.track_id, None)
    return list(tracks.values())


def remove_duplicate_stracks(stracksa, stracksb):
    distances = rotated_distance(stracksa, stracksb, class_aware=True)
    pairs = np.where(distances < 0.15)
    duplicate_a, duplicate_b = [], []
    for index_a, index_b in zip(*pairs):
        duration_a = stracksa[index_a].frame_id - stracksa[index_a].start_frame
        duration_b = stracksb[index_b].frame_id - stracksb[index_b].start_frame
        if duration_a > duration_b:
            duplicate_b.append(index_b)
        else:
            duplicate_a.append(index_a)
    return ([track for index, track in enumerate(stracksa)
             if index not in duplicate_a],
            [track for index, track in enumerate(stracksb)
             if index not in duplicate_b])
