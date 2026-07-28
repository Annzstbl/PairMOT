"""Controlled bridge 10: remove the HSMOT ConvMSI stem LR multiplier."""

import os

from yolo11l_diffusion_det_hsmot_overfit_bridge09b_lxlrtiming import (
    Exp as LxLrTimingExp,
)


class Exp(LxLrTimingExp):
    """Change only ConvMSI stem LR scale from 10x to LX-style uniform 1x."""

    def __init__(self):
        super().__init__()
        self.exp_name = os.path.splitext(os.path.basename(__file__))[0]
        self.stem_lr_multiplier = 1.0
