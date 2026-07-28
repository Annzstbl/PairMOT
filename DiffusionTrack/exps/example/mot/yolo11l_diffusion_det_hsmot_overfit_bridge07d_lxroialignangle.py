"""Bridge 07D: align MMCV RotatedROIAlign's angle direction with Detectron2."""

import os

from yolo11l_diffusion_det_hsmot_overfit_bridge07c_minrefined2 import (
    Exp as MinRefinedSideExp,
)


class Exp(MinRefinedSideExp):
    """Change only ROIAlign angle direction from clockwise to counterclockwise."""

    def __init__(self):
        super().__init__()
        self.exp_name = os.path.splitext(os.path.basename(__file__))[0]

    def get_model(self):
        model = super().get_model()
        pooler = model.head.head.box_pooler
        if not pooler.clockwise:
            raise ValueError("bridge07D requires historical clockwise=True")
        pooler.clockwise = False
        return model
