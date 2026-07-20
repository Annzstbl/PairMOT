"""Portable HSMOT pair-detection cache shared by detection and tracking."""

import json
import os
from collections import defaultdict

import numpy as np
import torch

from yolox.utils.rotated_boxes import qbox_to_rbox, rbox_to_qbox


PAIR_HEADER = (
    '# curr_frame,prev_frame,det_index,'
    'prev_x1,prev_y1,prev_x2,prev_y2,prev_x3,prev_y3,prev_x4,prev_y4,'
    'prev_cls,prev_score,'
    'curr_x1,curr_y1,curr_x2,curr_y2,curr_x3,curr_y3,curr_x4,curr_y4,'
    'curr_cls,curr_score,pair_cls,pair_score,cls_score,'
    'presence_prev,presence_curr\n')
FRAME_HEADER = (
    '# frame,det_index,x1,y1,x2,y2,x3,y3,x4,y4,class,score\n')


def _original_qboxes(detections, scale):
    boxes = torch.as_tensor(detections[:, :5], dtype=torch.float32).clone()
    boxes[:, :4] /= float(scale)
    return rbox_to_qbox(boxes).cpu().numpy()


def write_detection_cache(root, records, metadata=None):
    """Write canonical pair rows plus independent current-frame detections."""
    pair_dir = os.path.join(root, 'pair_detections')
    frame_dir = os.path.join(root, 'frame_detections')
    os.makedirs(pair_dir, exist_ok=True)
    os.makedirs(frame_dir, exist_ok=True)
    by_sequence = defaultdict(list)
    for record in records:
        by_sequence[record['sequence']].append(record)

    manifest = dict(version=1, coordinate_space='original', sequences={},
                    metadata=metadata or {})
    for sequence, sequence_records in sorted(by_sequence.items()):
        sequence_records.sort(key=lambda item: item['frame_id'])
        manifest['sequences'][sequence] = [
            dict(frame_id=int(item['frame_id']),
                 prev_frame_id=int(item['prev_frame_id']),
                 scale=float(item['scale']))
            for item in sequence_records]
        with open(os.path.join(pair_dir, sequence + '.txt'), 'w',
                  encoding='utf-8') as stream:
            stream.write(PAIR_HEADER)
            for item in sequence_records:
                ref_dets, cur_dets = item['ref_dets'], item['cur_dets']
                if not len(ref_dets):
                    continue
                ref_qboxes = _original_qboxes(ref_dets, item['scale'])
                cur_qboxes = _original_qboxes(cur_dets, item['scale'])
                for index, (ref, cur, ref_qbox, cur_qbox) in enumerate(zip(
                        ref_dets, cur_dets, ref_qboxes, cur_qboxes)):
                    pair_class = int(ref[7])
                    values = [int(item['frame_id']),
                              int(item['prev_frame_id']), index]
                    values += [f'{value:.4f}' for value in ref_qbox]
                    values += [int(ref[7]), f'{ref[6]:.7f}']
                    values += [f'{value:.4f}' for value in cur_qbox]
                    values += [int(cur[7]), f'{cur[6]:.7f}', pair_class,
                               f'{ref[5]:.7f}',
                               f'{np.sqrt(max(ref[6], 0) * max(cur[6], 0)):.7f}',
                               '1.0000000', '1.0000000']
                    stream.write(','.join(map(str, values)) + '\n')
        with open(os.path.join(frame_dir, sequence + '.txt'), 'w',
                  encoding='utf-8') as stream:
            stream.write(FRAME_HEADER)
            for item in sequence_records:
                detections = item['detections']
                if not len(detections):
                    continue
                qboxes = _original_qboxes(detections, item['scale'])
                for index, (detection, qbox) in enumerate(zip(
                        detections, qboxes)):
                    values = [int(item['frame_id']), index]
                    values += [f'{value:.4f}' for value in qbox]
                    values += [int(detection[7]), f'{detection[6]:.7f}']
                    stream.write(','.join(map(str, values)) + '\n')
    with open(os.path.join(root, 'manifest.json'), 'w',
              encoding='utf-8') as stream:
        json.dump(manifest, stream, indent=2, sort_keys=True)
    return root


def _rbox_from_fields(fields):
    qbox = torch.as_tensor(
        [float(value) for value in fields], dtype=torch.float32).reshape(1, 8)
    return qbox_to_rbox(qbox)[0].cpu().numpy()


def load_detection_cache(root):
    """Load cache as ``sequence -> ordered frame records``."""
    with open(os.path.join(root, 'manifest.json'), 'r',
              encoding='utf-8') as stream:
        manifest = json.load(stream)
    result = {}
    for sequence, frame_meta in manifest['sequences'].items():
        frames = {
            int(item['frame_id']): dict(
                sequence=sequence, frame_id=int(item['frame_id']),
                prev_frame_id=int(item['prev_frame_id']), scale=1.0,
                ref_dets=[], cur_dets=[], detections=[])
            for item in frame_meta}
        pair_path = os.path.join(root, 'pair_detections', sequence + '.txt')
        with open(pair_path, 'r', encoding='utf-8') as stream:
            for line in stream:
                if not line.strip() or line.startswith('#'):
                    continue
                fields = line.strip().split(',')
                frame_id = int(float(fields[0]))
                ref = np.zeros(8, dtype=np.float32)
                cur = np.zeros(8, dtype=np.float32)
                ref[:5] = _rbox_from_fields(fields[3:11])
                cur[:5] = _rbox_from_fields(fields[13:21])
                ref[5] = cur[5] = float(fields[24])
                ref[6], cur[6] = float(fields[12]), float(fields[22])
                ref[7], cur[7] = int(float(fields[11])), int(float(fields[21]))
                frames[frame_id]['ref_dets'].append(ref)
                frames[frame_id]['cur_dets'].append(cur)
        frame_path = os.path.join(
            root, 'frame_detections', sequence + '.txt')
        with open(frame_path, 'r', encoding='utf-8') as stream:
            for line in stream:
                if not line.strip() or line.startswith('#'):
                    continue
                fields = line.strip().split(',')
                frame_id = int(float(fields[0]))
                detection = np.zeros(8, dtype=np.float32)
                detection[:5] = _rbox_from_fields(fields[2:10])
                detection[5] = detection[6] = float(fields[11])
                detection[7] = int(float(fields[10]))
                frames[frame_id]['detections'].append(detection)
        for item in frames.values():
            for key in ('ref_dets', 'cur_dets', 'detections'):
                item[key] = np.asarray(item[key], dtype=np.float32).reshape(-1, 8)
        result[sequence] = [frames[key] for key in sorted(frames)]
    return result
