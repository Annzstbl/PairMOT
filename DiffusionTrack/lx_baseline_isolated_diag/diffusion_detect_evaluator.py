import itertools
import json
import os

import numpy as np
import torch
from loguru import logger
from tqdm import tqdm
from torchvision.ops import batched_nms, box_iou

from yolox.utils import gather, is_main_process, synchronize, time_synchronized
from utils.box_ops import box_cxcywhtheta_to_xyxyxyxy, box_xyxyxyxy_to_cxcywhtheta


class DiffusionDetectEvaluator:
    """
    Detection-only evaluator for diffusion detector models.
    It runs single-frame inference and computes rotated mAP
    at IoU=0.50 and IoU=0.50:0.95.
    """

    def __init__(
        self,
        dataloader,
        img_size,
        confthre,
        nmsthre3d,
        detthre,
        nmsthre2d,
        num_classes,
        testdev=False,
    ):
        self.dataloader = dataloader
        self.img_size = img_size
        self.confthre = confthre
        self.nmsthre3d = nmsthre3d
        self.detthre = detthre
        self.nmsthre2d = nmsthre2d
        self.num_classes = num_classes
        self.testdev = testdev
        self.cache_root = None
        self.validation_name = None

    def _dataset_attr(self, name, default=None):
        ds = self.dataloader.dataset
        if hasattr(ds, name):
            return getattr(ds, name)
        base = getattr(ds, "dataset", None)
        if base is not None and hasattr(base, name):
            return getattr(base, name)
        return default

    @torch.no_grad()
    def evaluate(
        self,
        model,
        distributed=False,
        half=False,
        trt_file=None,
        decoder=None,
        test_size=None,
    ):
        tensor_type = torch.cuda.HalfTensor if half else torch.cuda.FloatTensor
        model = model.eval()
        if half:
            model = model.half()

        data_list = []
        inference_time = 0.0
        n_samples = len(self.dataloader) - 1
        progress_bar = tqdm if is_main_process() else iter

        for cur_iter, (imgs, _, info_imgs, ids) in enumerate(progress_bar(self.dataloader)):
            with torch.no_grad():
                imgs = imgs.type(tensor_type)
                infer_start = time_synchronized()
                outputs = model((imgs, None), (None, None))
                inference_time += time_synchronized() - infer_start

                output_results = self.convert_to_detection_format(outputs, info_imgs, ids)
                data_list.extend(output_results)

        statistics = torch.cuda.FloatTensor([inference_time, n_samples])
        if distributed:
            data_list = gather(data_list, dst=0)
            data_list = list(itertools.chain(*data_list))
            torch.distributed.reduce(statistics, dst=0)

        eval_results = self.evaluate_prediction(data_list, statistics)
        synchronize()
        return eval_results

    def _postprocess_single(self, pred, assoc):
        """
        pred: [N, 5 + C] = (cx, cy, w, h, theta_deg, cls_logits...)
        assoc: [N] (0~1)
        returns: np.ndarray [M, 7] = (cx, cy, w, h, theta_deg, score, cls_id)
        """
        if pred is None or pred.numel() == 0:
            return np.zeros((0, 7), dtype=np.float32)

        assoc = assoc.flatten()
        cls_prob = torch.sigmoid(pred[:, 5:])
        cls_conf, cls_id = cls_prob.max(dim=-1)
        det_score = torch.sqrt(cls_conf * assoc)

        keep = det_score > float(self.detthre)
        if keep.sum() == 0:
            return np.zeros((0, 7), dtype=np.float32)

        boxes = pred[keep, :5]
        scores = det_score[keep]
        cls_ids = cls_id[keep].long()

        try:
            from detectron2.layers import batched_nms_rotated

            nms_keep = batched_nms_rotated(boxes, scores, cls_ids, float(self.nmsthre2d))
        except Exception:
            x1 = boxes[:, 0] - boxes[:, 2] * 0.5
            y1 = boxes[:, 1] - boxes[:, 3] * 0.5
            x2 = boxes[:, 0] + boxes[:, 2] * 0.5
            y2 = boxes[:, 1] + boxes[:, 3] * 0.5
            boxes_xyxy = torch.stack([x1, y1, x2, y2], dim=-1)
            nms_keep = batched_nms(boxes_xyxy, scores, cls_ids, float(self.nmsthre2d))
        boxes = boxes[nms_keep]
        scores = scores[nms_keep]
        cls_ids = cls_ids[nms_keep]

        out = torch.zeros((boxes.shape[0], 7), dtype=boxes.dtype, device=boxes.device)
        out[:, :5] = boxes
        out[:, 5] = scores
        out[:, 6] = cls_ids.to(out.dtype)
        return out.cpu().numpy().astype(np.float32)

    def convert_to_detection_format(self, outputs, info_imgs, ids):
        data_list = []
        if outputs is None:
            return data_list

        diffusion_outputs, conf_scores, _ = outputs
        if diffusion_outputs is None or conf_scores is None:
            return data_list

        bs = diffusion_outputs.shape[0] // 2
        cur_prediction = diffusion_outputs[bs:]

        for bi in range(bs):
            # info_imgs is tuple-like batch fields from dataloader collate
            img_h = float(info_imgs[0][bi]) if hasattr(info_imgs[0], "__len__") else float(info_imgs[0])
            img_w = float(info_imgs[1][bi]) if hasattr(info_imgs[1], "__len__") else float(info_imgs[1])
            scale = min(self.img_size[0] / img_h, self.img_size[1] / img_w)

            preds = self._postprocess_single(cur_prediction[bi], conf_scores[bi])
            if preds.shape[0] == 0:
                continue

            # map predictions back to original image size
            preds[:, 0] /= scale
            preds[:, 1] /= scale
            preds[:, 2] /= scale
            preds[:, 3] /= scale

            image_id = int(ids[bi]) if hasattr(ids, "__len__") else int(ids)
            for det in preds:
                cls_idx = int(det[6])
                class_ids = self._dataset_attr("class_ids", [])
                if cls_idx < 0 or cls_idx >= len(class_ids):
                    continue
                category_id = class_ids[cls_idx]
                pred_data = {
                    "image_id": image_id,
                    "category_id": int(category_id),
                    "rbox": [
                        float(det[0]),
                        float(det[1]),
                        float(det[2]),
                        float(det[3]),
                        float(det[4]),
                    ],
                    "score": float(det[5]),
                }
                data_list.append(pred_data)
        return data_list

    @staticmethod
    def _ann_to_rbox(ann):
        bbox = ann.get("bbox", [])
        if isinstance(bbox, list) and len(bbox) == 8:
            box_t = torch.tensor([bbox], dtype=torch.float32)
            rbox = box_xyxyxyxy_to_cxcywhtheta(box_t).cpu().numpy().reshape(-1).tolist()
            return [float(rbox[0]), float(rbox[1]), float(rbox[2]), float(rbox[3]), float(rbox[4])]
        if isinstance(bbox, list) and len(bbox) == 5:
            return [float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3]), float(bbox[4])]
        if isinstance(bbox, list) and len(bbox) == 4:
            x, y, w, h = bbox
            return [float(x + w * 0.5), float(y + h * 0.5), float(w), float(h), 0.0]
        return None

    @staticmethod
    def _compute_iou_matrix(gt_rboxes, dt_rboxes):
        if len(gt_rboxes) == 0 or len(dt_rboxes) == 0:
            return np.zeros((len(gt_rboxes), len(dt_rboxes)), dtype=np.float32)

        gt_t = torch.tensor(gt_rboxes, dtype=torch.float32)
        dt_t = torch.tensor(dt_rboxes, dtype=torch.float32)
        try:
            from detectron2.layers.rotated_boxes import pairwise_iou_rotated

            ious = pairwise_iou_rotated(gt_t, dt_t)
            return ious.cpu().numpy().astype(np.float32)
        except Exception:
            gt_poly = box_cxcywhtheta_to_xyxyxyxy(gt_t).view(-1, 4, 2)
            dt_poly = box_cxcywhtheta_to_xyxyxyxy(dt_t).view(-1, 4, 2)
            gt_xyxy = torch.stack(
                [gt_poly[:, :, 0].min(dim=1).values, gt_poly[:, :, 1].min(dim=1).values,
                 gt_poly[:, :, 0].max(dim=1).values, gt_poly[:, :, 1].max(dim=1).values],
                dim=1,
            )
            dt_xyxy = torch.stack(
                [dt_poly[:, :, 0].min(dim=1).values, dt_poly[:, :, 1].min(dim=1).values,
                 dt_poly[:, :, 0].max(dim=1).values, dt_poly[:, :, 1].max(dim=1).values],
                dim=1,
            )
            return box_iou(gt_xyxy, dt_xyxy).cpu().numpy().astype(np.float32)

    @staticmethod
    def _compute_ap_from_pr(recalls, precisions):
        if recalls.size == 0:
            return 0.0
        mrec = np.concatenate(([0.0], recalls, [1.0]))
        mpre = np.concatenate(([0.0], precisions, [0.0]))
        for i in range(mpre.size - 1, 0, -1):
            mpre[i - 1] = max(mpre[i - 1], mpre[i])
        recall_points = np.linspace(0.0, 1.0, 101)
        ap = 0.0
        for r in recall_points:
            inds = np.where(mrec >= r)[0]
            p = np.max(mpre[inds]) if inds.size > 0 else 0.0
            ap += p
        return float(ap / 101.0)

    def evaluate_prediction(self, data_dict, statistics):
        if not is_main_process():
            return 0, 0, None

        logger.info("Evaluate rotated detection mAP in main process...")
        inference_time = statistics[0].item()
        n_samples = max(statistics[1].item(), 1)
        a_infer_time = 1000 * inference_time / (n_samples * self.dataloader.batch_size)
        info = "Average forward time: {:.2f} ms\n".format(a_infer_time)

        if len(data_dict) == 0:
            return 0, 0, info

        coco_src = self._dataset_attr("coco", None)
        if coco_src is None:
            return 0, 0, info + "Evaluator could not find dataset.coco.\n"

        gt_img_cat = {}
        gt_count_per_cat = {}
        for ann in coco_src.dataset.get("annotations", []):
            cat = int(ann.get("category_id", -1))
            img = int(ann.get("image_id", -1))
            rbox = self._ann_to_rbox(ann)
            if cat < 0 or img < 0 or rbox is None:
                continue
            gt_img_cat.setdefault((img, cat), []).append(rbox)
            gt_count_per_cat[cat] = gt_count_per_cat.get(cat, 0) + 1

        dt_per_cat = {}
        for det in data_dict:
            cat = int(det["category_id"])
            dt_per_cat.setdefault(cat, []).append(det)
        for cat in dt_per_cat.keys():
            dt_per_cat[cat].sort(key=lambda x: float(x["score"]), reverse=True)

        iou_thrs = np.arange(0.50, 0.96, 0.05)
        cat_ids = sorted(set(list(gt_count_per_cat.keys()) + list(dt_per_cat.keys())))
        aps_by_thr = {float(t): [] for t in iou_thrs}

        for cat in cat_ids:
            num_gt = int(gt_count_per_cat.get(cat, 0))
            if num_gt <= 0:
                continue

            detections = dt_per_cat.get(cat, [])
            for thr in iou_thrs:
                matched = {
                    key: np.zeros(len(boxes), dtype=bool)
                    for key, boxes in gt_img_cat.items()
                    if key[1] == cat
                }
                tps = []
                fps = []
                for det in detections:
                    img_id = int(det["image_id"])
                    key = (img_id, cat)
                    gt_boxes = gt_img_cat.get(key, [])
                    if len(gt_boxes) == 0:
                        tps.append(0.0)
                        fps.append(1.0)
                        continue
                    dt_box = [det["rbox"]]
                    ious = self._compute_iou_matrix(gt_boxes, dt_box).reshape(-1)
                    if ious.size == 0:
                        tps.append(0.0)
                        fps.append(1.0)
                        continue
                    max_idx = int(np.argmax(ious))
                    max_iou = float(ious[max_idx])
                    if max_iou >= float(thr) and not matched[key][max_idx]:
                        matched[key][max_idx] = True
                        tps.append(1.0)
                        fps.append(0.0)
                    else:
                        tps.append(0.0)
                        fps.append(1.0)

                if len(tps) == 0:
                    aps_by_thr[float(thr)].append(0.0)
                    continue
                tps = np.array(tps, dtype=np.float32)
                fps = np.array(fps, dtype=np.float32)
                cum_tp = np.cumsum(tps)
                cum_fp = np.cumsum(fps)
                recalls = cum_tp / max(float(num_gt), 1e-12)
                precisions = cum_tp / np.maximum(cum_tp + cum_fp, 1e-12)
                ap = self._compute_ap_from_pr(recalls, precisions)
                aps_by_thr[float(thr)].append(ap)

        ap50 = float(np.mean(aps_by_thr[0.50])) if len(aps_by_thr[0.50]) > 0 else 0.0
        mean_aps = []
        for thr in iou_thrs:
            vals = aps_by_thr[float(thr)]
            mean_aps.append(float(np.mean(vals)) if len(vals) > 0 else 0.0)
        ap50_95 = float(np.mean(mean_aps)) if len(mean_aps) > 0 else 0.0

        info += "Rotated mAP summary (IoU=0.50:0.95):\n"
        info += "  AP50: {:.4f}\n".format(ap50)
        info += "  AP50_95: {:.4f}\n".format(ap50_95)
        if self.cache_root:
            output_dir = os.path.join(
                self.cache_root, self.validation_name or "latest")
            os.makedirs(output_dir, exist_ok=True)
            with open(
                    os.path.join(output_dir, "detections.json"),
                    "w", encoding="utf-8") as handle:
                json.dump(data_dict, handle)
            with open(
                    os.path.join(output_dir, "metrics.json"),
                    "w", encoding="utf-8") as handle:
                json.dump({
                    "conf_threshold": float(self.confthre),
                    "detection_threshold": float(self.detthre),
                    "mAP50": ap50,
                    "mAP50_95": ap50_95,
                    "num_detections": len(data_dict),
                }, handle, indent=2)
            info += "  val_det: {}\n".format(output_dir)
        return ap50_95, ap50, info
