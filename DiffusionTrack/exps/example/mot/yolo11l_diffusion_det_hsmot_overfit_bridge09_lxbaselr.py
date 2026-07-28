"""Controlled bridge 09: reproduce LX per-step peak base learning rate."""

import os

from yolo11l_diffusion_det_hsmot_overfit_bridge08d_lxrawmatcher import (
    Exp as LxRawMatcherExp,
)


class Exp(LxRawMatcherExp):
    """Change only scheduler peak LR to LX's 0.001 / 64."""

    def __init__(self):
        super().__init__()
        self.exp_name = os.path.splitext(os.path.basename(__file__))[0]
        lx_base_lr = 0.001 / 64.0
        self.scheduler_base_lr = lx_base_lr
