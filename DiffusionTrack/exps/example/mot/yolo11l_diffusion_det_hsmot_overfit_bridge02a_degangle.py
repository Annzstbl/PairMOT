"""Controlled bridge 02A: reproduce LX degree-sized angle refinement."""

import math
import os

from yolo11l_diffusion_det_hsmot_overfit_bridge02_angleweight1 import (
    Exp as AngleWeightExp,
)


class Exp(AngleWeightExp):
    """Change only raw dtheta physical scale from one radian to one degree."""

    def __init__(self):
        super().__init__()
        self.exp_name = os.path.splitext(os.path.basename(__file__))[0]

    def get_model(self):
        model = super().get_model()
        for head in model.head.head.head_series:
            if head.angle_delta_scale != 1.0:
                raise ValueError(
                    "bridge02A requires one-radian raw angle deltas")
            head.angle_delta_scale = math.pi / 180.0
        return model
