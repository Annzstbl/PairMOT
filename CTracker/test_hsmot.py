"""Run rotated CTracker on HSMOT 8-channel sequences."""

import argparse
import csv
import glob
import json
import os
import os.path as osp
import subprocess
import sys

import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment

import model  # noqa: F401 - required by full-model torch checkpoints
from hsmot_adapter import HSMOT_CLASSES
from mmrotate.datasets.hsmot import HSMOT_MEAN, HSMOT_STD
from mmrotate.datasets.transforms.loading_hsmot_pair import (
    _load_multichannel_image)
from mmcv.ops import box_iou_rotated
from rotated_ops import rboxes_to_qboxes
from projects.multispec_pair_rotated_rtdetr.multispec_pair_rotated_rtdetr.pair_mot_tracker import (  # noqa: E501
    PairDetection, PairFrameRecord, write_pair_det_txt)


class Detection:
    def __init__(self, frame_id, prev_box, curr_box, score, label):
        self.frame_id = frame_id
        self.prev_box = prev_box
        self.curr_box = curr_box
        self.score = float(score)
        self.label = int(label)
        self.track_id = -1

    @property
    def position(self):
        return self.prev_box[:2]

    @property
    def size(self):
        return self.prev_box[2:4]


class Track:
    def __init__(self, detection):
        self.track_id = detection.track_id
        self.label = detection.label
        self.detections = [detection]
        self.num_detections = 1
        self.last = detection
        self.last_frame = detection.frame_id
        self.no_match_frame = 0

    def update(self, detection):
        self.detections.append(detection)
        self.num_detections += 1
        self.last = detection
        self.last_frame = detection.frame_id

    @property
    def velocity(self):
        if self.num_detections < 2:
            return self.last.position.new_zeros(2)
        if self.num_detections < 6:
            last = self.detections[-1]
            previous = self.detections[-2]
            return ((last.position - previous.position) /
                    (last.frame_id - previous.frame_id))
        first = ((self.detections[-1].position -
                  self.detections[-4].position) /
                 (self.detections[-1].frame_id -
                  self.detections[-4].frame_id))
        second = ((self.detections[-2].position -
                   self.detections[-5].position) /
                  (self.detections[-2].frame_id -
                   self.detections[-5].frame_id))
        third = ((self.detections[-3].position -
                  self.detections[-6].position) /
                 (self.detections[-3].frame_id -
                  self.detections[-6].frame_id))
        return (first + second + third) / 3


def rotated_iou(box1, box2):
    return float(box_iou_rotated(
        box1[None].float(), box2[None].float())[0, 0])


def track_detection_similarity(track, detection):
    """Original CTracker association, adapted only from HBB to RBB IoU."""
    if detection.frame_id <= track.last_frame:
        return 0.0
    if detection.label != track.label:
        return 0.0
    frame_delta = detection.frame_id - track.last_frame
    if frame_delta == 1:
        return rotated_iou(track.last.curr_box, detection.prev_box)
    predicted = track.last.prev_box.clone()
    predicted[:2] += track.velocity * frame_delta
    return rotated_iou(predicted, detection.prev_box)


def match_tracks(tracks, detections, iou_threshold):
    if not tracks or not detections:
        return [], list(range(len(tracks))), list(range(len(detections)))
    costs = torch.zeros(len(tracks), len(detections))
    for track_index, track in enumerate(tracks):
        for detection_index, detection in enumerate(detections):
            costs[track_index, detection_index] = -track_detection_similarity(
                track, detection)
    rows, cols = linear_sum_assignment(costs.numpy())
    matches = []
    unmatched_tracks = set(range(len(tracks)))
    unmatched_detections = set(range(len(detections)))
    for row, col in zip(rows, cols):
        if float(costs[row, col]) <= -iou_threshold:
            matches.append((int(row), int(col)))
            unmatched_tracks.discard(int(row))
            unmatched_detections.discard(int(col))
    return matches, sorted(unmatched_tracks), sorted(unmatched_detections)


def preprocess(path, image_scale, device):
    image = _load_multichannel_image(path, to_float32=False)
    height, width = image.shape[:2]
    # Match the HSMOT pair pipeline's scale=(height, width) convention.
    scale = min(image_scale[0] / height, image_scale[1] / width)
    resized_h = max(1, round(height * scale))
    resized_w = max(1, round(width * scale))
    tensor = torch.from_numpy(image).permute(2, 0, 1).float()[None]
    tensor = F.interpolate(
        tensor, size=(resized_h, resized_w), mode='bilinear',
        align_corners=False)
    mean = tensor.new_tensor(HSMOT_MEAN).view(1, 8, 1, 1)
    std = tensor.new_tensor(HSMOT_STD).view(1, 8, 1, 1)
    tensor = (tensor / 255.0 - mean) / std
    pad_h = (32 - resized_h % 32) % 32
    pad_w = (32 - resized_w % 32) % 32
    tensor = F.pad(tensor, (0, pad_w, 0, pad_h))
    return tensor.to(device), scale, (height, width)


def detections_from_output(output, frame_id, score_threshold):
    keep = output['scores'] >= score_threshold
    boxes = output['paired_boxes'][keep]
    scores = output['scores'][keep]
    labels = output['labels'][keep]
    return [
        Detection(frame_id, box[:5], box[5:], score, label)
        for box, score, label in zip(boxes, scores, labels)
    ]


def write_results(path, frame_detections, scale):
    with open(path, 'w', encoding='utf-8') as output_file:
        for frame_id in sorted(frame_detections):
            detections = frame_detections[frame_id]
            if not detections:
                continue
            boxes = torch.stack([det.prev_box for det in detections]).clone()
            boxes[:, :4] /= scale
            qboxes = rboxes_to_qboxes(boxes).detach().cpu().numpy()
            for detection, qbox in zip(detections, qboxes):
                qbox_text = ','.join(f'{value:.2f}' for value in qbox)
                output_file.write(
                    f'{frame_id},{detection.track_id},{qbox_text},'
                    f'{detection.score:.6f},{detection.label},0\n')


def _pair_record(output, seq_name, prev_path, curr_path, prev_frame_id,
                 curr_frame_id, scale, original_shape):
    paired_boxes = output['paired_boxes'].detach().cpu().clone()
    paired_boxes[:, :4] /= scale
    paired_boxes[:, 5:9] /= scale
    scores = output['scores'].detach().cpu()
    labels = output['labels'].detach().cpu()
    detections = []
    for index, (boxes, score, label) in enumerate(
            zip(paired_boxes, scores, labels)):
        value = float(score)
        detections.append(PairDetection(
            index=index,
            prev_bbox=[float(item) for item in boxes[:5]],
            curr_bbox=[float(item) for item in boxes[5:]],
            score=value,
            cls_score=value,
            label=int(label),
            score_prev=value,
            score_curr=value,
            label_prev=int(label),
            label_curr=int(label),
        ))
    height, width = original_shape
    return PairFrameRecord(
        seq_name=seq_name,
        prev_frame_id=prev_frame_id,
        curr_frame_id=curr_frame_id,
        frame_gap=curr_frame_id - prev_frame_id,
        prev_img_path=prev_path,
        curr_img_path=curr_path,
        img_shape=[round(height * scale), round(width * scale)],
        ori_shape=[height, width],
        scale_factor=[scale, scale],
        detections=detections,
        is_first_pair=False,
    )


def run_sequence(network, sequence_dir, output_path, pair_output_path, device,
                 image_scale, score_threshold=0.4, iou_threshold=0.5,
                 retention=10):
    paths = sorted(glob.glob(osp.join(sequence_dir, '*_p1.jpg')))
    if not paths:
        raise FileNotFoundError(f'No HSMOT frames found in {sequence_dir}')
    tracks = []
    next_track_id = 1
    frame_detections = {}
    previous_features = None
    last_scale = 1.0
    pair_records = []
    sequence_name = osp.basename(sequence_dir)

    # Duplicate the final frame to flush CTracker's one-frame delayed output.
    for index, path in enumerate(paths + [paths[-1]]):
        tensor, scale, original_shape = preprocess(path, image_scale, device)
        last_scale = scale
        with torch.no_grad():
            output, features = network(tensor, last_feat=previous_features)
        previous_features = features
        if index == 0:
            continue
        emitted_path = paths[index - 1]
        frame_id = int(osp.basename(emitted_path).split('_', 1)[0])
        if index < len(paths):
            current_path = paths[index]
            current_frame_id = int(
                osp.basename(current_path).split('_', 1)[0])
            pair_records.append(_pair_record(
                output, sequence_name, emitted_path, current_path, frame_id,
                current_frame_id, scale, original_shape))
        detections = detections_from_output(
            output, frame_id, score_threshold)
        matches, unmatched_tracks, unmatched_detections = match_tracks(
            tracks, detections, iou_threshold)
        for track_index, detection_index in matches:
            detection = detections[detection_index]
            detection.track_id = tracks[track_index].track_id
            tracks[track_index].update(detection)
        for track_index in unmatched_tracks:
            tracks[track_index].no_match_frame += 1
        tracks = [
            track for track in tracks
            if track.no_match_frame < retention
        ]
        for detection_index in unmatched_detections:
            detection = detections[detection_index]
            detection.track_id = next_track_id
            next_track_id += 1
            tracks.append(Track(detection))
        frame_detections[frame_id] = detections

    write_results(output_path, frame_detections, last_scale)
    write_pair_det_txt(pair_output_path, pair_records)
    return sum(len(value) for value in frame_detections.values())


def load_network(path, device):
    payload = torch.load(path, map_location='cpu', weights_only=False)
    if isinstance(payload, torch.nn.DataParallel):
        payload = payload.module
    if isinstance(payload, torch.nn.Module):
        return payload.to(device).eval(), None
    if not isinstance(payload, dict) or 'model' not in payload:
        raise ValueError(f'Unsupported model/checkpoint format: {path}')
    args = payload.get('args', {})
    depth = int(args.get('depth', 50))
    factories = {
        18: model.resnet18,
        34: model.resnet34,
        50: model.resnet50,
        101: model.resnet101,
        152: model.resnet152,
    }
    if depth not in factories:
        raise ValueError(f'Unsupported checkpoint depth: {depth}')
    network = factories[depth](
        num_classes=len(HSMOT_CLASSES), pretrained=False,
        num_spectral=8, use_3d_se_stem=True, rotated=True)
    network.load_state_dict(payload['model'], strict=True)
    checkpoint_info = dict(
        epoch=int(payload.get('epoch', -1)) + 1,
        total_iter=int(payload.get('total_iter', 0)),
    )
    return network.to(device).eval(), checkpoint_info


def run_trackeval(output_dir, tracker_name, tracker_sub_folder, gt_dir,
                  img_dir, trackeval_root):
    cmd = [
        sys.executable,
        osp.join(osp.abspath(trackeval_root), 'scripts/run_hsmot_8ch.py'),
        '--USE_PARALLEL', 'False',
        '--METRICS', 'HOTA', 'CLEAR', 'Identity',
        '--TRACKERS_TO_EVAL', tracker_name,
        '--TRACKER_SUB_FOLDER', tracker_sub_folder,
        '--GT_FOLDER', osp.abspath(gt_dir),
        '--IMG_FOLDER', osp.abspath(img_dir),
        '--TRACKERS_FOLDER', osp.abspath(osp.join(output_dir, 'trackers')),
        '--OUTPUT_FOLDER', osp.abspath(osp.join(output_dir, 'trackers')),
    ]
    pairmot_root = osp.abspath(osp.join(osp.dirname(__file__), '..'))
    ai4rs_root = osp.join(pairmot_root, 'ai4rs')
    hsmot_root = osp.join(pairmot_root, 'hsmot')
    env = os.environ.copy()
    env['PYTHONPATH'] = os.pathsep.join(
        [pairmot_root, ai4rs_root, hsmot_root] +
        ([env['PYTHONPATH']] if env.get('PYTHONPATH') else []))
    completed = subprocess.run(
        cmd, cwd=osp.abspath(trackeval_root), check=False, env=env,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    stdout_path = osp.join(output_dir, 'trackeval_stdout.log')
    with open(stdout_path, 'w', encoding='utf-8') as output_file:
        output_file.write(completed.stdout)
    if completed.returncode:
        raise RuntimeError(
            f'TrackEval failed with exit code {completed.returncode}; '
            f'see {stdout_path}')

    eval_dir = osp.join(
        output_dir, 'trackers', tracker_name, 'eval')
    metrics = {}
    for path in sorted(glob.glob(osp.join(eval_dir, '*_summary.csv'))):
        with open(path, encoding='utf-8') as input_file:
            row = next(csv.DictReader(input_file), None)
        if row is not None:
            metrics[osp.basename(path)[:-len('_summary.csv')]] = row
    metrics_path = osp.join(output_dir, 'metrics.json')
    with open(metrics_path, 'w', encoding='utf-8') as output_file:
        json.dump(metrics, output_file, indent=2, ensure_ascii=False)
    return metrics_path, stdout_path


def main(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', required=True)
    parser.add_argument(
        '--model', required=True,
        help='Complete model_final.pt or training checkpoint_epoch_*.pt.')
    parser.add_argument('--output_dir', required=True)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available()
                        else 'cpu')
    parser.add_argument('--image_scale', nargs=2, type=int,
                        default=(900, 1200), metavar=('H', 'W'))
    parser.add_argument('--score_threshold', type=float, default=0.4)
    parser.add_argument('--iou_threshold', type=float, default=0.5)
    parser.add_argument('--tracker_name', default='ctracker')
    parser.add_argument('--tracker_sub_folder', default='preds')
    parser.add_argument('--evaluate', action='store_true')
    parser.add_argument('--gt_dir', default='')
    parser.add_argument('--trackeval_root', default='../TrackEval')
    parsed = parser.parse_args(args)

    device = torch.device(parsed.device)
    network, checkpoint_info = load_network(parsed.model, device)
    if not getattr(network, 'rotated', False):
        raise ValueError('The checkpoint is not a rotated CTracker model')
    if checkpoint_info is not None:
        print(f'Loaded checkpoint: {checkpoint_info}')
    os.makedirs(parsed.output_dir, exist_ok=True)
    pair_output_dir = osp.join(parsed.output_dir, 'val_det')
    track_output_dir = osp.join(
        parsed.output_dir, 'trackers', parsed.tracker_name,
        parsed.tracker_sub_folder)
    os.makedirs(pair_output_dir, exist_ok=True)
    os.makedirs(track_output_dir, exist_ok=True)
    sequence_dirs = sorted(
        path for path in glob.glob(osp.join(parsed.data_root, '*'))
        if osp.isdir(path))
    total = 0
    for sequence_dir in sequence_dirs:
        sequence = osp.basename(sequence_dir)
        output_path = osp.join(track_output_dir, f'{sequence}.txt')
        pair_output_path = osp.join(pair_output_dir, f'{sequence}.txt')
        count = run_sequence(
            network, sequence_dir, output_path, pair_output_path, device,
            tuple(parsed.image_scale), parsed.score_threshold,
            parsed.iou_threshold)
        total += count
        print(f'{sequence}: {count} detections -> {output_path}')
    print(f'Completed {len(sequence_dirs)} sequences, {total} detections; '
          f'classes={HSMOT_CLASSES}')
    if parsed.evaluate:
        gt_dir = parsed.gt_dir or osp.join(
            osp.dirname(parsed.data_root), 'mot')
        metrics_path, stdout_path = run_trackeval(
            parsed.output_dir, parsed.tracker_name,
            parsed.tracker_sub_folder, gt_dir, parsed.data_root,
            parsed.trackeval_root)
        print(f'TrackEval metrics -> {metrics_path}')
        print(f'TrackEval stdout -> {stdout_path}')


if __name__ == '__main__':
    main()
