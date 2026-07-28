"""Bridge 08D: reproduce LX's raw matcher IoU/nonfinite arithmetic."""

import os

from yolo11l_diffusion_det_hsmot_overfit_bridge08c_lxclassfusion import (
    Exp as LxClassFusionExp,
)


class Exp(LxClassFusionExp):
    """Change only matcher numerical guards; costs/coverage stay unchanged."""

    def __init__(self):
        super().__init__()
        self.exp_name = os.path.splitext(os.path.basename(__file__))[0]

    def get_model(self):
        model = super().get_model()
        matcher = model.head.criterion.matcher
        if matcher.numerical_mode != "guarded":
            raise ValueError("bridge08D requires guarded matcher arithmetic")
        matcher.numerical_mode = "lx_raw"
        return model
