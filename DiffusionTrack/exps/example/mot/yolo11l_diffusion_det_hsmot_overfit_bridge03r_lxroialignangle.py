"""Controlled bridge 03R: align rotated ROIAlign's angle direction to LX."""

import os

from yolo11l_diffusion_det_hsmot_overfit_bridge03_qbox8match import (
    Exp as Qbox8MatchExp,
)


class Exp(Qbox8MatchExp):
    """Change only MMCV ROIAlignRotated from clockwise to counter-clockwise."""

    def __init__(self):
        super().__init__()
        self.exp_name = os.path.splitext(os.path.basename(__file__))[0]

    def get_model(self):
        model = super().get_model()
        pooler = model.head.head.box_pooler
        if not pooler.clockwise:
            raise RuntimeError(
                "Bridge03R expected clockwise ROIAlign input state")
        pooler.clockwise = False
        return model
