"""VT-Tiny-MOT 检测评估：从 COCO-MOT JSON 读取 GT，与 submit 阶段 det/*.txt 对齐。

JSON 解析逻辑与 ``MeMOTR/data/vt_tiny_mot.py`` 的 ``load_coco_annotations`` 保持一致。
"""

from __future__ import annotations

import json
import os
import os.path as osp
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torchvision.ops import box_iou
from tqdm import tqdm

from hsmot.eval.validator import PredictValidator

ANN_FILE_TEMPLATES = {
    "00": "instances_00_{split}2017.json",
    "01": "instances_01_{split}2017.json",
    "plain": "instances_{split}2017.json",
}

VT_TINY_CLASS_NAMES = [
    "ship",
    "car",
    "cyclist",
    "pedestrian",
    "bus",
    "drone",
    "plane",
]

# 训练/推理 submit 仍按 00/ 帧序；评测时 00、01 各用对应 JSON 各评一次
VT_TINY_EVAL_CHANNELS = ("00", "01")
SUBMIT_FRAME_IMAGE_SUBDIR = "00"


def parse_scene_from_file_name(file_name: str) -> str:
    return file_name.split("/")[0]


def parse_frame_id(img: dict) -> int:
    if "frame_id" in img:
        return int(img["frame_id"])
    if "mot_frame_id" in img:
        return int(img["mot_frame_id"])
    return int(osp.splitext(img["file_name"].split("/")[-1])[0])


def parse_file_frame_id(img: dict) -> int:
    """从 file_name 解析磁盘上的帧编号（如 00445.jpg -> 445）。"""
    return int(osp.splitext(img["file_name"].split("/")[-1])[0])


def load_vt_tiny_coco_gt(
    ann_path: str,
    ann_mode: str = "plain",
    ir_ann_path: Optional[str] = None,
) -> Tuple[Dict[str, Dict[int, List[np.ndarray]]], Dict[str, Dict[int, int]]]:
    """解析 COCO-MOT JSON（与 MeMOTR/data/vt_tiny_mot.py 一致）。

    Returns:
        labels_full[scene][mot_frame_id]: 每帧 GT，shape (N, 5) -> x,y,w,h,cls
        frame_file_ids[scene][mot_frame_id]: MOT 帧 id -> 磁盘帧编号
    """
    ann_mode = ann_mode.lower()
    if ann_mode not in ANN_FILE_TEMPLATES:
        raise ValueError(f"Unsupported ann_mode={ann_mode}, choose from {list(ANN_FILE_TEMPLATES)}")

    with open(ann_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    images_by_id = {img["id"]: img for img in data["images"]}
    ir_images_by_id: Dict[int, dict] = {}
    if ann_mode == "plain":
        if ir_ann_path is None:
            raise ValueError("plain 模式需要提供 ir_ann_path (instances_01_*.json)")
        with open(ir_ann_path, "r", encoding="utf-8") as f:
            ir_data = json.load(f)
        ir_images_by_id = {img["id"]: img for img in ir_data["images"]}

    labels_full: Dict[str, Dict[int, List[np.ndarray]]] = defaultdict(lambda: defaultdict(list))
    frame_file_ids: Dict[str, Dict[int, int]] = defaultdict(dict)
    skipped = 0

    if ann_mode in VT_TINY_EVAL_CHANNELS:
        channel_marker = f"/{ann_mode}/"
    else:
        channel_marker = f"/{SUBMIT_FRAME_IMAGE_SUBDIR}/"

    for img in data["images"]:
        if channel_marker not in img["file_name"]:
            continue
        scene = parse_scene_from_file_name(img["file_name"])
        mot_frame_id = parse_frame_id(img)
        frame_file_ids[scene][mot_frame_id] = parse_file_frame_id(img)

    for ann in data["annotations"]:
        img = images_by_id.get(ann["image_id"])
        if img is None:
            if ann_mode == "plain" and ann.get("type") == 2:
                img = ir_images_by_id.get(ann["image_id"])
            if img is None:
                skipped += 1
                continue

        if ann_mode in VT_TINY_EVAL_CHANNELS and channel_marker not in img["file_name"]:
            continue

        scene = parse_scene_from_file_name(img["file_name"])
        frame_id = parse_frame_id(img)
        x, y, w, h = ann["bbox"]
        labels_full[scene][frame_id].append(
            np.array([x, y, w, h, ann["category_id"]], dtype=np.float32)
        )

    if skipped:
        print(f"[VT-Tiny det eval] skipped {skipped} annotations with unresolved image_id in {ann_path}")

    return labels_full, frame_file_ids


def build_submit_frame_to_mot_id(
    scene: str,
    scene_dir: str,
    frame_file_ids: Dict[str, Dict[int, int]],
    image_subdir: str = SUBMIT_FRAME_IMAGE_SUBDIR,
) -> Dict[int, int]:
    """将 submit det txt 的帧号（1-based dataloader 序号）映射到 COCO mot_frame_id。

    预测始终按 ``00/`` 排序帧写出；``frame_file_ids`` 来自当前评测通道（00 或 01）的 GT JSON。
    """
    rgb_dir = osp.join(scene_dir, image_subdir)
    if not osp.isdir(rgb_dir):
        raise FileNotFoundError(f"VT-Tiny RGB dir not found: {rgb_dir}")

    image_paths = sorted(
        osp.join(rgb_dir, name)
        for name in os.listdir(rgb_dir)
        if name.endswith(".jpg")
    )
    file_id_to_mot_id = {fid: mot_id for mot_id, fid in frame_file_ids.get(scene, {}).items()}
    mapping: Dict[int, int] = {}
    for idx, img_path in enumerate(image_paths):
        submit_frame = idx + 1
        file_id = int(osp.splitext(osp.basename(img_path))[0])
        mot_id = file_id_to_mot_id.get(file_id)
        if mot_id is None:
            mot_id = submit_frame - 1
        mapping[submit_frame] = mot_id
    return mapping


def read_rect_det_txt(txt_path: str) -> Dict[int, List[List[float]]]:
    """读取 ONLY_TRAIN_DETR 写出的 det txt：frame,x,y,w,h,score,cls,-1,-1。"""
    results: Dict[int, List[List[float]]] = {}
    if not osp.isfile(txt_path):
        return results
    with open(txt_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(",")
            if len(parts) < 7:
                continue
            frame_id = int(parts[0])
            x, y, w, h = map(float, parts[1:5])
            score = float(parts[5])
            cls = float(parts[6])
            results.setdefault(frame_id, []).append([x, y, w, h, cls, score])
    return results


def _xywh_to_xyxy(xywh: torch.Tensor) -> torch.Tensor:
    x, y, w, h = xywh.unbind(-1)
    return torch.stack((x, y, x + w, y + h), dim=-1)


def box_iou_xywh(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    """axis-aligned IoU，输入 xywh (top-left x,y,w,h)。"""
    if boxes1.numel() == 0 or boxes2.numel() == 0:
        return torch.zeros((boxes1.shape[0], boxes2.shape[0]), dtype=torch.float32)
    return box_iou(_xywh_to_xyxy(boxes1), _xywh_to_xyxy(boxes2))


class RectPredictValidator(PredictValidator):
    """正框检测验证：GT/Pred 均为 xywh + cls（Pred 另含 score）。"""

    def update_metrics(self, preds: torch.Tensor, gts: torch.Tensor):
        """
        preds: [m, 6]  x, y, w, h, cls, score
        gts:   [n, 5]  x, y, w, h, cls
        """
        self.seen += 1
        device = preds.device if preds.numel() else gts.device if gts.numel() else torch.device("cpu")
        if self.device is None:
            self.device = device

        pred_cls = preds[:, 4].view(-1) if preds.numel() else torch.zeros(0, device=device)
        gt_cls = gts[:, 4].view(-1) if gts.numel() else torch.zeros(0, device=device)
        npr = len(preds)
        nl = len(gts)

        stat = dict(
            conf=torch.zeros(0, device=device),
            pred_cls=torch.zeros(0, device=device),
            tp=torch.zeros(npr, self.niou, dtype=torch.bool, device=device),
        )
        stat["target_cls"] = gt_cls
        stat["target_img"] = gt_cls.unique()

        if npr == 0:
            if nl:
                for k in self.stats.keys():
                    self.stats[k].append(stat[k])
            return

        stat["conf"] = preds[:, 5]
        stat["pred_cls"] = pred_cls
        if nl:
            iou = box_iou_xywh(gts[:, :4], preds[:, :4])
            stat["tp"] = self.match_predictions(pred_cls, gt_cls, iou, use_scipy=True)

        for k in self.stats.keys():
            self.stats[k].append(stat[k])


def resolve_vt_tiny_ann_paths(
    dataset_root: str,
    dataset_split: str,
    ann_mode: str = "plain",
) -> Tuple[str, Optional[str]]:
    ann_dir = osp.join(dataset_root, "annotations")
    ann_mode = ann_mode.lower()
    ann_path = osp.join(ann_dir, ANN_FILE_TEMPLATES[ann_mode].format(split=dataset_split))
    ir_ann_path = None
    if ann_mode == "plain":
        ir_ann_path = osp.join(ann_dir, ANN_FILE_TEMPLATES["01"].format(split=dataset_split))
    return ann_path, ir_ann_path


def list_vt_tiny_channel_eval_runs(
    dataset_root: str,
    dataset_split: str,
    mot_stage: bool = False,
) -> List[Dict[str, str]]:
    """00 / 01 各评一次的配置列表。

    mot_stage=True 时 output_sub_folder 为 eval_00 / eval_01（TrackEval）；
    检测阶段仅区分 ann_mode，结果写入 log。
    """
    runs = []
    for ch in VT_TINY_EVAL_CHANNELS:
        gt_path, _ = resolve_vt_tiny_ann_paths(dataset_root, dataset_split, ann_mode=ch)
        entry = {"channel": ch, "ann_mode": ch, "gt_coco_ann": gt_path}
        if mot_stage:
            entry["output_sub_folder"] = f"eval_{ch}"
        runs.append(entry)
    return runs


def val_vt_tiny_coco_det(
    gt_coco_ann: str,
    pred_det_folder: str,
    data_split_dir: str,
    ann_mode: str = "plain",
    ir_ann_path: Optional[str] = None,
    nc: int = 7,
    names: Optional[List[str]] = None,
) -> List[str]:
    """对 VT-Tiny submit ``det/`` 目录做检测 mAP 评估。

    Args:
        gt_coco_ann: 如 ``instances_test2017.json``（RGB / plain 主标注）
        pred_det_folder: ``{epoch_dir}/{split}/det/``，内含 ``{scene}_det.txt``
        data_split_dir: 如 ``.../test2017``，用于定位各 scene 的 ``00/`` 图像序列
    """
    names = names or VT_TINY_CLASS_NAMES
    print(
        f"Start VT-Tiny COCO det eval: gt={gt_coco_ann}, pred_det={pred_det_folder}, "
        f"split_dir={data_split_dir}, ann_mode={ann_mode}"
    )

    labels_full, frame_file_ids = load_vt_tiny_coco_gt(
        gt_coco_ann, ann_mode=ann_mode, ir_ann_path=ir_ann_path
    )
    validator = RectPredictValidator(nc=nc, names=names)

    scenes = sorted(labels_full.keys())
    for scene in tqdm(scenes, desc=f"VT-Tiny det eval ({ann_mode})"):
        scene_dir = osp.join(data_split_dir, scene)
        pred_path = osp.join(pred_det_folder, f"{scene}_det.txt")
        pred_by_submit_frame = read_rect_det_txt(pred_path)
        frame_map = build_submit_frame_to_mot_id(scene, scene_dir, frame_file_ids)

        for submit_frame, mot_frame_id in sorted(frame_map.items()):
            gts_list = labels_full[scene].get(mot_frame_id, [])
            preds_list = pred_by_submit_frame.get(submit_frame, [])

            gts = (
                torch.tensor(gts_list, dtype=torch.float32)
                if gts_list
                else torch.zeros((0, 5), dtype=torch.float32)
            )
            preds = (
                torch.tensor(preds_list, dtype=torch.float32)
                if preds_list
                else torch.zeros((0, 6), dtype=torch.float32)
            )
            validator.update_metrics(preds, gts)

    return validator.final()


__all__ = [
    "ANN_FILE_TEMPLATES",
    "VT_TINY_CLASS_NAMES",
    "VT_TINY_EVAL_CHANNELS",
    "SUBMIT_FRAME_IMAGE_SUBDIR",
    "load_vt_tiny_coco_gt",
    "build_submit_frame_to_mot_id",
    "read_rect_det_txt",
    "RectPredictValidator",
    "resolve_vt_tiny_ann_paths",
    "list_vt_tiny_channel_eval_runs",
    "val_vt_tiny_coco_det",
]
