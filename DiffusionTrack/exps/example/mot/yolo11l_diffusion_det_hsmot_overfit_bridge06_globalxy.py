"""Controlled bridge 06: reproduce LX global-axis center refinement."""

import os

from yolo11l_diffusion_det_hsmot_overfit_bridge05d_lxheadrng import (
    Exp as LxHeadRngExp,
)


class Exp(LxHeadRngExp):
    """Change only bbox-delta center projection from local to global axes."""

    def __init__(self):
        super().__init__()
        self.exp_name = os.path.splitext(os.path.basename(__file__))[0]

    def get_model(self):
        model = super().get_model()
        for head in model.head.head.head_series:
            if not head.proj_xy:
                raise ValueError("bridge06 requires bridge05 proj_xy=True")
            head.proj_xy = False
        return model
