"""Controlled bridge 12: enable LX's residual single-image augmentation."""

import os

from yolo11l_diffusion_det_hsmot_overfit_bridge11_lxcoverage import (
    Exp as LxCoverageExp,
)


class Exp(LxCoverageExp):
    """Change only image augmentation from none to LX distort+mirror."""

    def __init__(self):
        super().__init__()
        self.exp_name = os.path.splitext(os.path.basename(__file__))[0]
        self.hsmot_augment_mode = "lx"
