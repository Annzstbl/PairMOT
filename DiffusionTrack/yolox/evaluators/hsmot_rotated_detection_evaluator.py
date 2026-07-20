"""Class-aware rotated detection AP for HSMOT Stage-1 validation."""

import os
import time

import numpy as np
import torch
import torch.distributed as dist
from loguru import logger
from tqdm import tqdm

from yolox.tracker.diffusion_tracker_kl import DiffusionTracker
from yolox.utils import is_main_process
from yolox.utils.rotated_boxes import qbox_to_rbox, rotated_iou
from .hsmot_detection_cache import write_detection_cache


class HSMOTRotatedDetectionEvaluator:
    """Evaluate rotated mAP@[.50:.95] and mAP@.50 on HSMOT.

    Stage 1 uses the detector's original duplicated-pair inference path for
    each validation image, then performs class-aware rotated NMS. AP uses the
    COCO 101-point interpolation rule and rotated IoU thresholds 0.50:0.05:0.95.
    """

    def __init__(self, dataloader, num_classes, confthre=0.001,
                 detthre=0.001, nmsthre3d=0.7, nmsthre2d=0.75, amp=True,
                 cache_root=None, max_dets=100):
        self.dataloader = dataloader
        self.num_classes = num_classes
        self.confthre = confthre
        self.detthre = detthre
        self.nmsthre3d = nmsthre3d
        self.nmsthre2d = nmsthre2d
        self.amp = amp
        self.amp_dtype = ({"bf16": torch.bfloat16, "fp16": torch.float16}
                          .get(amp, torch.float16))
        self.iou_thresholds = np.arange(0.50, 0.96, 0.05)
        self.cache_root = cache_root
        self.validation_name = None
        self.max_dets = int(max_dets)

    @staticmethod
    def _interpolated_ap(tp, fp, num_gt):
        if num_gt == 0:
            return np.nan
        tp = np.cumsum(tp, dtype=np.float64)
        fp = np.cumsum(fp, dtype=np.float64)
        recall = tp / num_gt
        precision = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
        precision = np.maximum.accumulate(precision[::-1])[::-1]
        recall_points = np.linspace(0.0, 1.0, 101)
        indices = np.searchsorted(recall, recall_points, side="left")
        sampled = np.zeros_like(recall_points)
        valid = indices < len(precision)
        sampled[valid] = precision[indices[valid]]
        return float(sampled.mean())

    def _summarize(self, records, gt_counts):
        per_class = []
        for class_id in range(self.num_classes):
            class_records = sorted(
                records[class_id], key=lambda item: item[0], reverse=True)
            class_aps = []
            for threshold in self.iou_thresholds:
                matched = {}
                tp = np.zeros(len(class_records), dtype=np.float32)
                fp = np.zeros(len(class_records), dtype=np.float32)
                for index, (_, image_id, overlaps) in enumerate(class_records):
                    used = matched.setdefault(
                        image_id, np.zeros(len(overlaps), dtype=bool))
                    candidates = np.where(~used)[0]
                    if len(candidates):
                        best = candidates[np.argmax(overlaps[candidates])]
                        if overlaps[best] >= threshold:
                            used[best] = True
                            tp[index] = 1
                            continue
                    fp[index] = 1
                class_aps.append(self._interpolated_ap(
                    tp, fp, gt_counts[class_id]))
            per_class.append(class_aps)

        values = np.asarray(per_class, dtype=np.float64)
        valid_classes = np.asarray(gt_counts) > 0
        if not valid_classes.any():
            return 0.0, 0.0, values
        per_threshold = np.nanmean(values[valid_classes], axis=0)
        return float(per_threshold.mean()), float(per_threshold[0]), values

    @torch.no_grad()
    def evaluate(self, model, distributed=False, half=False, **kwargs):
        if hasattr(model, "module"):
            model = model.module
        model = model.eval()
        # Keep EMA weights in FP32 across repeated validation runs. Autocast
        # reduces activation memory without destructively converting them.
        tensor_type = torch.cuda.FloatTensor
        records = [[] for _ in range(self.num_classes)]
        gt_counts = np.zeros(self.num_classes, dtype=np.int64)
        cache_records = []
        start = time.time()
        tracker = DiffusionTracker(
            model, tensor_type, conf_thresh=self.confthre,
            det_thresh=self.detthre, nms_thresh_3d=self.nmsthre3d,
            nms_thresh_2d=self.nmsthre2d)

        iterator = tqdm(self.dataloader, desc="HSMOT rotated val",
                        disable=not is_main_process())
        for prev_images, curr_images, targets, meta in iterator:
            prev_images = prev_images.type(tensor_type, non_blocking=True)
            curr_images = curr_images.type(tensor_type, non_blocking=True)
            with torch.cuda.amp.autocast(
                    enabled=bool(self.amp) or half, dtype=self.amp_dtype):
                batch_results, _ = tracker.detect_batch(
                    prev_images, curr_images)
            for batch_index, (ref_dets, cur_dets, detections) in enumerate(
                    batch_results):
                if len(ref_dets) > self.max_dets:
                    keep = np.argsort(-ref_dets[:, 6])[:self.max_dets]
                    ref_dets, cur_dets = ref_dets[keep], cur_dets[keep]
                if len(detections) > self.max_dets:
                    keep = np.argsort(
                        -detections[:, 6])[:self.max_dets]
                    detections = detections[keep]
                target = targets[batch_index]
                valid = target[:, 1:9].abs().sum(dim=1) > 0
                target = target[valid]
                gt_by_class = []
                image_id = int(meta['image_id'][batch_index])
                for class_id in range(self.num_classes):
                    class_target = target[target[:, 0].long() == class_id]
                    boxes = qbox_to_rbox(
                        class_target[:, 1:9]).to(curr_images.device)
                    gt_by_class.append(boxes)
                    gt_counts[class_id] += len(boxes)
                for class_id in range(self.num_classes):
                    class_detections = detections[
                        detections[:, 7].astype(np.int64) == class_id]
                    if not len(class_detections):
                        continue
                    overlaps = rotated_iou(
                        torch.as_tensor(
                            class_detections[:, :5],
                            device=curr_images.device,
                            dtype=torch.float32),
                        gt_by_class[class_id]).cpu().numpy()
                    records[class_id].extend(
                        (float(detection[6]), image_id, overlap)
                        for detection, overlap in zip(
                            class_detections, overlaps))
                original_h = float(meta['original_height'][batch_index])
                original_w = float(meta['original_width'][batch_index])
                scale = min(curr_images.shape[-2] / original_h,
                            curr_images.shape[-1] / original_w)
                cache_records.append(dict(
                    image_id=image_id,
                    sequence=meta['sequence'][batch_index],
                    frame_id=int(meta['frame_id'][batch_index]),
                    prev_frame_id=int(meta['prev_frame_id'][batch_index]),
                    scale=scale, ref_dets=ref_dets, cur_dets=cur_dets,
                    detections=detections))

        if distributed:
            # Each rank performs pair detection on its disjoint sampler shard.
            # Gather compact CPU/Numpy evaluation records only after inference;
            # no detector output is communicated on the hot path.
            torch.cuda.empty_cache()
            gathered = ([None] * dist.get_world_size()
                        if is_main_process() else None)
            dist.gather_object(
                (records, gt_counts, cache_records), gathered, dst=0)
            if is_main_process():
                records = [[] for _ in range(self.num_classes)]
                gt_counts = np.zeros(self.num_classes, dtype=np.int64)
                cache_records = []
                for rank_records, rank_gt_counts, rank_cache in gathered:
                    for class_id in range(self.num_classes):
                        records[class_id].extend(rank_records[class_id])
                    gt_counts += rank_gt_counts
                    cache_records.extend(rank_cache)
                cache_records.sort(
                    key=lambda item: int(item.get('image_id', 0)))

        if distributed and not is_main_process():
            result = [None]
            dist.broadcast_object_list(result, src=0)
            return result[0]

        cache_path = None
        if self.cache_root:
            cache_path = os.path.join(
                self.cache_root, self.validation_name or 'latest')
            write_detection_cache(
                cache_path, cache_records,
                metadata=dict(conf_threshold=self.confthre,
                              detection_threshold=self.detthre,
                              pair_nms_threshold=self.nmsthre3d,
                              frame_nms_threshold=self.nmsthre2d))
            logger.info('saved batched pair-detection cache to {}', cache_path)

        map50_95, map50, per_class = self._summarize(records, gt_counts)
        class_lines = []
        class_names = getattr(self.dataloader.dataset, "_classes", None)
        for class_id in range(self.num_classes):
            if gt_counts[class_id] == 0:
                continue
            name = (class_names[class_id] if class_names is not None
                    else str(class_id))
            class_lines.append(
                "{}: AP50={:.4f}, AP50:95={:.4f}, GT={}".format(
                    name, per_class[class_id, 0],
                    np.nanmean(per_class[class_id]), gt_counts[class_id]))
        summary = (
            "HSMOT rotated detection validation\n"
            "mAP50:95={:.4f}, mAP50={:.4f}, images={}, time={:.1f}s, "
            "pair_cache={}\n{}"
        ).format(map50_95, map50, len(self.dataloader.dataset),
                 time.time() - start, cache_path, "\n".join(class_lines))
        logger.info(summary)
        if distributed:
            result = [(map50_95, map50, summary)]
            dist.broadcast_object_list(result, src=0)
            return result[0]
        return map50_95, map50, summary
