"""水平正框 (xyxy) MOT 相关工具。"""

from __future__ import annotations

import torch


def xyxy_to_cxcywh(boxes: torch.Tensor) -> torch.Tensor:
    x1, y1, x2, y2 = boxes.unbind(-1)
    return torch.stack([(x1 + x2) * 0.5, (y1 + y2) * 0.5, x2 - x1, y2 - y1], dim=-1)


def normalize_cxcywh(boxes: torch.Tensor, img_shape: tuple[int, int]) -> torch.Tensor:
    h, w = img_shape
    return boxes / boxes.new_tensor([w, h, w, h])


def denormalize_cxcywh(norm_boxes: torch.Tensor, img_shape: tuple[int, int]) -> torch.Tensor:
    h, w = img_shape
    return norm_boxes * norm_boxes.new_tensor([w, h, w, h])
