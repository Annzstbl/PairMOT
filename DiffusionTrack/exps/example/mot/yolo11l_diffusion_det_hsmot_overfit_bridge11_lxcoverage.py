"""Controlled bridge 11: reproduce LX Dynamic-K coverage behavior."""

import os

from yolo11l_diffusion_det_hsmot_overfit_bridge10c_adamwscalar import (
    Exp as ScalarAdamWExp,
)


class Exp(ScalarAdamWExp):
    """Change only post-conflict all-GT coverage repair from on to off."""

    def __init__(self):
        super().__init__()
        self.exp_name = os.path.splitext(os.path.basename(__file__))[0]

    def get_model(self):
        model = super().get_model()
        matcher = model.head.criterion.matcher
        if matcher.coverage_mode != "corrected":
            raise ValueError("bridge11 requires corrected coverage mode")
        matcher.coverage_mode = "lx_stale"
        return model
