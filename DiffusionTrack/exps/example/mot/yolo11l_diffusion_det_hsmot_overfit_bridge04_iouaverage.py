"""Controlled bridge 04: average the two frame-wise rotated-IoU losses."""

import os

from yolo11l_diffusion_det_hsmot_overfit_bridge03r_lxroialignangle import (
    Exp as LxRoiAlignExp,
)


class Exp(LxRoiAlignExp):
    """Change only criterion IoU normalization from pair sum to frame mean."""

    def __init__(self):
        super().__init__()
        self.exp_name = os.path.splitext(os.path.basename(__file__))[0]

    def get_model(self):
        model = super().get_model()
        criterion = model.head.criterion
        if criterion.average_pair_iou_loss:
            raise ValueError("bridge04 requires bridge03 pair-sum IoU loss")
        criterion.average_pair_iou_loss = True
        return model
