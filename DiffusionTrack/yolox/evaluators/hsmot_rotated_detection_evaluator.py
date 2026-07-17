"""Class-aware rotated detection AP for HSMOT Stage-1 validation."""

import time

import numpy as np
import torch
from loguru import logger
from tqdm import tqdm

from yolox.tracker.diffusion_tracker_kl import DiffusionTracker
from yolox.utils import get_rank, is_main_process, synchronize
from yolox.utils.rotated_boxes import qbox_to_rbox, rotated_iou


class HSMOTRotatedDetectionEvaluator:
    """Evaluate rotated mAP@[.50:.95] and mAP@.50 on HSMOT.

    Stage 1 uses the detector's original duplicated-pair inference path for
    each validation image, then performs class-aware rotated NMS. AP uses the
    COCO 101-point interpolation rule and rotated IoU thresholds 0.50:0.05:0.95.
    """

    def __init__(self, dataloader, num_classes, confthre=0.001,
                 detthre=0.001, nmsthre3d=0.7, nmsthre2d=0.75, amp=True):
        self.dataloader = dataloader
        self.num_classes = num_classes
        self.confthre = confthre
        self.detthre = detthre
        self.nmsthre3d = nmsthre3d
        self.nmsthre2d = nmsthre2d
        self.amp = amp
        self.iou_thresholds = np.arange(0.50, 0.96, 0.05)

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
        # Stateful duplicated-pair inference is deliberately run only on rank
        # 0. Other ranks wait here so DDP training resumes in lockstep.
        if distributed and not is_main_process():
            synchronize()
            return 0.0, 0.0, "validation executed on rank 0"

        if hasattr(model, "module"):
            model = model.module
        model = model.eval()
        # Keep EMA weights in FP32 across repeated validation runs. Autocast
        # reduces activation memory without destructively converting them.
        tensor_type = torch.cuda.FloatTensor
        records = [[] for _ in range(self.num_classes)]
        gt_counts = np.zeros(self.num_classes, dtype=np.int64)
        start = time.time()

        iterator = tqdm(self.dataloader, desc="HSMOT rotated val",
                        disable=not is_main_process())
        for image_id, (images, targets, _, _) in enumerate(iterator):
            images = images.type(tensor_type, non_blocking=True)
            target = targets[0]
            valid = target[:, 1:9].abs().sum(dim=1) > 0
            target = target[valid]
            gt_by_class = []
            for class_id in range(self.num_classes):
                class_target = target[target[:, 0].long() == class_id]
                boxes = qbox_to_rbox(class_target[:, 1:9]).cuda()
                gt_by_class.append(boxes)
                gt_counts[class_id] += len(boxes)

            # Recreate the first-frame inference state for every image so this
            # metric measures detection rather than temporal tracking quality.
            tracker = DiffusionTracker(
                model, tensor_type, conf_thresh=self.confthre,
                det_thresh=self.detthre, nms_thresh_3d=self.nmsthre3d,
                nms_thresh_2d=self.nmsthre2d)
            with torch.cuda.amp.autocast(enabled=self.amp or half):
                detections, _ = tracker.update(images)
            for detection in detections:
                class_id = int(detection.class_id)
                if not 0 <= class_id < self.num_classes:
                    continue
                box = torch.as_tensor(
                    detection.rbox, device=images.device,
                    dtype=torch.float32).reshape(1, 5)
                overlaps = rotated_iou(
                    box, gt_by_class[class_id]).squeeze(0).cpu().numpy()
                records[class_id].append(
                    (float(detection.score), image_id, overlaps))

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
            "mAP50:95={:.4f}, mAP50={:.4f}, images={}, time={:.1f}s\n{}"
        ).format(map50_95, map50, len(self.dataloader.dataset),
                 time.time() - start, "\n".join(class_lines))
        logger.info(summary)
        if distributed:
            synchronize()
        return map50_95, map50, summary
