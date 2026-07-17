"""HSMOT evaluator/output writer for the original Kalman DiffusionTracker."""

from collections import defaultdict
import os
import time

from loguru import logger
import numpy as np
import torch
from tqdm import tqdm

from yolox.tracker.diffusion_tracker_kl import DiffusionTracker
from yolox.utils import is_main_process, synchronize
from yolox.utils.rotated_boxes import rbox_to_qbox


PAIR_HEADER = (
    '# curr_frame,prev_frame,det_index,'
    'prev_x1,prev_y1,prev_x2,prev_y2,prev_x3,prev_y3,prev_x4,prev_y4,'
    'prev_cls,prev_score,'
    'curr_x1,curr_y1,curr_x2,curr_y2,curr_x3,curr_y3,curr_x4,curr_y4,'
    'curr_cls,curr_score,pair_cls,pair_score,cls_score,'
    'presence_prev,presence_curr\n')


def write_results(filename, results):
    """Write TrackEval HSMOT rows: frame,id,qbox8,score,class,truncation."""
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, 'w', encoding='utf-8') as stream:
        for frame_id, rboxes, track_ids, scores, classes in results:
            if not rboxes:
                continue
            qboxes = rbox_to_qbox(
                torch.as_tensor(np.asarray(rboxes), dtype=torch.float32)).numpy()
            for qbox, track_id, score, class_id in zip(
                    qboxes, track_ids, scores, classes):
                if track_id < 0:
                    continue
                polygon = ','.join('{:.2f}'.format(value) for value in qbox)
                stream.write(
                    f'{frame_id},{track_id},{polygon},{score:.6f},'
                    f'{class_id},0\n')
    logger.info('saved tracking results to {}', filename)


def write_pair_results(filename, records):
    """Write canonical PairMOT pair-detection cache used by project tools."""
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, 'w', encoding='utf-8') as stream:
        stream.write(PAIR_HEADER)
        for curr_frame, prev_frame, ref_dets, cur_dets, scale in records:
            if not len(ref_dets):
                continue
            ref_boxes = torch.as_tensor(ref_dets[:, :5]).clone()
            cur_boxes = torch.as_tensor(cur_dets[:, :5]).clone()
            ref_boxes[:, :4] /= scale
            cur_boxes[:, :4] /= scale
            ref_qboxes = rbox_to_qbox(ref_boxes).numpy()
            cur_qboxes = rbox_to_qbox(cur_boxes).numpy()
            for index, (ref, cur, ref_qbox, cur_qbox) in enumerate(zip(
                    ref_dets, cur_dets, ref_qboxes, cur_qboxes)):
                ref_text = [f'{value:.2f}' for value in ref_qbox]
                cur_text = [f'{value:.2f}' for value in cur_qbox]
                class_id = int(ref[7])
                pair_score = float(np.sqrt(max(ref[6], 1e-6) *
                                           max(cur[6], 1e-6)))
                values = [curr_frame, prev_frame, index]
                values += ref_text + [class_id, f'{ref[6]:.6f}']
                values += cur_text + [int(cur[7]), f'{cur[6]:.6f}']
                values += ['nan', f'{pair_score:.6f}', f'{pair_score:.6f}',
                           '1.000000', '1.000000']
                stream.write(','.join(map(str, values)) + '\n')
    logger.info('saved pair detections to {}', filename)


class DiffusionMOTEvaluatorKL:
    def __init__(self, args, dataloader, img_size, confthre, nmsthre3d,
                 detthre, nmsthre2d, interval, num_classes):
        self.args = args
        self.dataloader = dataloader
        self.img_size = img_size
        self.confthre = confthre
        self.nmsthre3d = nmsthre3d
        self.detthre = detthre
        self.nmsthre2d = nmsthre2d
        self.num_classes = num_classes
        self.association_interval = interval

    def evaluate(self, model, distributed=False, half=False, trt_file=None,
                 decoder=None, test_size=None, result_folder=None):
        if distributed:
            raise ValueError('stateful HSMOT tracking evaluation must use one GPU')
        if trt_file is not None:
            raise ValueError('YOLO11 Conv3D-SE DiffusionTrack does not support TRT')
        tensor_type = torch.cuda.HalfTensor if half else torch.cuda.FloatTensor
        model = model.eval().half() if half else model.eval()
        progress = tqdm if is_main_process() else iter
        result_folder = result_folder or 'track_results'
        pair_folder = os.path.join(os.path.dirname(result_folder), 'pair_detections')

        tracker = None
        current_video = None
        track_records, pair_records = [], []
        track_time, sample_count = 0.0, 0

        def flush(video_name):
            if video_name is None:
                return
            write_results(
                os.path.join(result_folder, video_name + '.txt'),
                track_records)
            write_pair_results(
                os.path.join(pair_folder, video_name + '.txt'), pair_records)

        for imgs, _, info_imgs, _ in progress(self.dataloader):
            frame_id = int(info_imgs[2].item())
            image_name = info_imgs[4][0]
            video_name = image_name.split(os.sep)[0]
            if video_name != current_video:
                flush(current_video)
                track_records.clear()
                pair_records.clear()
                current_video = video_name
                tracker = DiffusionTracker(
                    model, tensor_type, self.confthre, self.detthre,
                    self.nmsthre3d, self.nmsthre2d,
                    self.association_interval)

            original_h = float(info_imgs[0].item())
            original_w = float(info_imgs[1].item())
            scale = min(self.img_size[0] / original_h,
                        self.img_size[1] / original_w)
            with torch.no_grad():
                output, association_time = tracker.update(imgs.type(tensor_type))
            track_time += association_time
            sample_count += 1

            rboxes, track_ids, scores, classes = [], [], [], []
            for track in output:
                rbox = track.rbox.copy()
                rbox[:4] /= scale
                if rbox[2] * rbox[3] <= self.args.min_box_area:
                    continue
                rboxes.append(rbox)
                track_ids.append(track.track_id)
                scores.append(track.score)
                classes.append(track.class_id)
            track_records.append(
                (frame_id, rboxes, track_ids, scores, classes))
            if tracker.last_pair_detections is not None:
                ref_dets, cur_dets = tracker.last_pair_detections
                pair_records.append(
                    (frame_id, frame_id - 1, ref_dets, cur_dets, scale))

        flush(current_video)
        fps = sample_count / max(track_time, 1e-9)
        info = (f'HSMOT sequences written to {result_folder}; '
                f'pair detections written to {pair_folder}; '
                f'tracking FPS={fps:.3f}')
        logger.info(info)
        synchronize()
        return 0.0, 0.0, info
