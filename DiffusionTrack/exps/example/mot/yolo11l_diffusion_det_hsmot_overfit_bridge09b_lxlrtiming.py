"""Controlled bridge 09B: reproduce LX's post-update LR scheduling."""

import os

from yolo11l_diffusion_det_hsmot_overfit_bridge09_lxbaselr import (
    Exp as LxBaseLrExp,
)


class Exp(LxBaseLrExp):
    """Use constructor LR on step one, then install scheduler LR after steps."""

    def __init__(self):
        super().__init__()
        self.exp_name = os.path.splitext(os.path.basename(__file__))[0]
        self.lr_update_timing = "after"
