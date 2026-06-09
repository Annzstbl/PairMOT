"""几何旋转框 IoU（评测 / 数据管线），替代 mmcv.ops.box_iou_rotated。

训练时可微 IoU 请使用 ``hsmot.loss.prob_iou_loss`` 中的 ``probiou`` / ``batch_probiou``。
"""

from typing import Union

import cv2
import numpy as np
import torch

__all__ = ["box_iou_rotated"]


def _to_numpy(boxes: Union[torch.Tensor, np.ndarray]) -> np.ndarray:
    if isinstance(boxes, torch.Tensor):
        return boxes.detach().cpu().numpy().astype(np.float64)
    return np.asarray(boxes, dtype=np.float64)


def _box_to_corners(boxes: np.ndarray, clockwise: bool = True) -> np.ndarray:
    """将 (N, 5) xywhr 转为 (N, 4, 2) 顶点坐标。

    角度约定与 mmrotate / mmcv ``box_iou_rotated(clockwise=True)`` 一致：
    弧度制，绕框中心逆时针旋转矩阵（与 le135 编码配套）。
    ``clockwise=False`` 时对角度取反，与 mmcv 参数语义一致。
    """
    boxes = boxes.copy()
    if not clockwise:
        boxes[:, 4] = -boxes[:, 4]

    x = boxes[:, 0]
    y = boxes[:, 1]
    w = boxes[:, 2]
    h = boxes[:, 3]
    alpha = boxes[:, 4]

    x4 = np.array([0.5, -0.5, -0.5, 0.5], dtype=np.float64) * w[:, None]
    y4 = np.array([0.5, 0.5, -0.5, -0.5], dtype=np.float64) * h[:, None]

    cos_a = np.cos(alpha)
    sin_a = np.sin(alpha)
    x_rot = x4 * cos_a[:, None] - y4 * sin_a[:, None] + x[:, None]
    y_rot = x4 * sin_a[:, None] + y4 * cos_a[:, None] + y[:, None]
    return np.stack([x_rot, y_rot], axis=-1).astype(np.float32)


def _convex_intersection_area(pts1: np.ndarray, pts2: np.ndarray) -> float:
    """计算两个凸四边形交集面积（OpenCV，非可微）。"""
    ret, intersect_pts = cv2.intersectConvexConvex(
        pts1.astype(np.float32),
        pts2.astype(np.float32),
        handleNested=True,
    )
    if ret <= 0 or intersect_pts is None or len(intersect_pts) < 3:
        return 0.0
    return float(cv2.contourArea(intersect_pts))


def _pair_iou(
    box1: np.ndarray,
    corners1: np.ndarray,
    area1: float,
    box2: np.ndarray,
    corners2: np.ndarray,
    area2: float,
    mode: str,
) -> float:
    inter = _convex_intersection_area(corners1, corners2)
    if mode == "iof":
        return inter / (area1 + 1e-8)
    union = area1 + area2 - inter
    return inter / (union + 1e-8)


def _compute_iou_numpy(
    bboxes1: np.ndarray,
    bboxes2: np.ndarray,
    mode: str = "iou",
    aligned: bool = False,
    clockwise: bool = True,
) -> np.ndarray:
    corners1 = _box_to_corners(bboxes1, clockwise)
    corners2 = _box_to_corners(bboxes2, clockwise)
    areas1 = bboxes1[:, 2] * bboxes1[:, 3]
    areas2 = bboxes2[:, 2] * bboxes2[:, 3]

    if aligned:
        assert bboxes1.shape[0] == bboxes2.shape[0]
        n = bboxes1.shape[0]
        out = np.zeros(n, dtype=np.float32)
        for i in range(n):
            out[i] = _pair_iou(
                bboxes1[i],
                corners1[i],
                areas1[i],
                bboxes2[i],
                corners2[i],
                areas2[i],
                mode,
            )
        return out

    n, m = bboxes1.shape[0], bboxes2.shape[0]
    out = np.zeros((n, m), dtype=np.float32)
    for i in range(n):
        for j in range(m):
            out[i, j] = _pair_iou(
                bboxes1[i],
                corners1[i],
                areas1[i],
                bboxes2[j],
                corners2[j],
                areas2[j],
                mode,
            )
    return out


def box_iou_rotated(
    bboxes1: Union[torch.Tensor, np.ndarray],
    bboxes2: Union[torch.Tensor, np.ndarray],
    mode: str = "iou",
    aligned: bool = False,
    clockwise: bool = True,
) -> torch.Tensor:
    """计算旋转框几何 IoU / IoF（非可微，用于评测与数据过滤）。

    与 ``mmcv.ops.box_iou_rotated`` API 兼容，基于 OpenCV 凸多边形求交实现，
    不依赖 mmcv。训练 loss 请使用 ``probiou``（可微 ProbIoU）。

    Args:
        bboxes1: (N, 5) xywhr，[cx, cy, w, h, angle_rad]。
        bboxes2: (M, 5) 或 aligned=True 时 (N, 5)。
        mode: ``iou`` 或 ``iof``（intersection over area of bboxes1）。
        aligned: True 时逐对计算，返回 (N,)；否则返回 (N, M) 矩阵。
        clockwise: 角度方向，与 mmcv 一致，默认 True。

    Returns:
        torch.Tensor，float32，值域 [0, 1]。
    """
    if mode not in ("iou", "iof"):
        raise ValueError(f"Unsupported mode: {mode!r}")

    ref = bboxes1 if isinstance(bboxes1, torch.Tensor) else bboxes2
    device = ref.device if isinstance(ref, torch.Tensor) else torch.device("cpu")

    b1 = _to_numpy(bboxes1)
    b2 = _to_numpy(bboxes2)
    result = _compute_iou_numpy(b1, b2, mode=mode, aligned=aligned, clockwise=clockwise)
    return torch.from_numpy(result).to(device=device, dtype=torch.float32)
