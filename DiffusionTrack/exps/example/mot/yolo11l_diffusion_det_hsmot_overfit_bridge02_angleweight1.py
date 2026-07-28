"""Controlled bridge 02: full normalized angle weight in box loss only."""

import os

from yolo11l_diffusion_det_hsmot_overfit_bridge01_centeractive import (
    Exp as CenterActiveExp,
)


class Exp(CenterActiveExp):
    """Change only regression-loss angle L1 weight from 0.05 to 1.0.

    Matching remains exactly Bridge 01. This separates dense regression
    gradient effects from assignment effects.
    """

    def __init__(self):
        super().__init__()
        self.exp_name = os.path.splitext(os.path.basename(__file__))[0]

    def get_model(self):
        model = super().get_model()
        head = model.head
        criterion = head.criterion
        matcher = criterion.matcher
        if not (
                head.bbox_angle_l1_weight
                == criterion.bbox_angle_l1_weight
                == matcher.bbox_angle_l1_weight
                == 0.05):
            raise ValueError("bridge02 requires the 0.05 bridge01 baseline")
        criterion.bbox_angle_l1_weight = 1.0
        return model
