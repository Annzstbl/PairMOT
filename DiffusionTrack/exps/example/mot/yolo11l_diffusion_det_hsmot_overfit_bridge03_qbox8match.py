"""Controlled bridge 03: reproduce LX qbox8 matching L1 only."""

import os

from yolo11l_diffusion_det_hsmot_overfit_bridge02a_degangle import (
    Exp as DegreeAngleExp,
)


class Exp(DegreeAngleExp):
    """Change only matcher L1 geometry from direct LE135 to raw qbox8."""

    def __init__(self):
        super().__init__()
        self.exp_name = os.path.splitext(os.path.basename(__file__))[0]

    def get_model(self):
        model = super().get_model()
        matcher = model.head.criterion.matcher
        if matcher.matching_l1_representation != "le135":
            raise ValueError("bridge03 requires the LE135 bridge02 matcher")
        matcher.matching_l1_representation = "qbox8"
        return model
