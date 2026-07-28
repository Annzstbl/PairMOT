"""Bridge 08C: reproduce LX's unclamped score/logit fusion."""

import os

from yolo11l_diffusion_det_hsmot_overfit_bridge08b_lxrawiou import (
    Exp as LxRawIouExp,
)


class Exp(LxRawIouExp):
    """Change only criterion classification fusion arithmetic."""

    def __init__(self):
        super().__init__()
        self.exp_name = os.path.splitext(os.path.basename(__file__))[0]

    def get_model(self):
        model = super().get_model()
        criterion = model.head.criterion
        if criterion.class_fusion_mode != "stable":
            raise ValueError("bridge08C requires stable class fusion")
        criterion.class_fusion_mode = "lx_raw"
        return model
