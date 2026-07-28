"""Bridge 08B: use LX's raw differentiable rotated-IoU loss kernel."""

import os

from yolo11l_diffusion_det_hsmot_overfit_bridge08_fp32 import (
    Exp as Fp32Exp,
)


class Exp(Fp32Exp):
    """Change only criterion IoU numerical guards; ordinary IoU is retained."""

    def __init__(self):
        super().__init__()
        self.exp_name = os.path.splitext(os.path.basename(__file__))[0]

    def get_model(self):
        model = super().get_model()
        criterion = model.head.criterion
        if criterion.iou_numerical_mode != "guarded":
            raise ValueError("bridge08B requires guarded ordinary rotated IoU")
        criterion.iou_numerical_mode = "lx_raw"
        return model
