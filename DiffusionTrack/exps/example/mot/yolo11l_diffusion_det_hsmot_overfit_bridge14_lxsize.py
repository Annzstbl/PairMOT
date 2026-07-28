"""Controlled bridge 14: reproduce LX's 800x1440 canvas."""

import os

from yolo11l_diffusion_det_hsmot_overfit_bridge13_randomflip import (
    Exp as RandomFlipExp,
)


class Exp(RandomFlipExp):
    """Change only fixed train/eval canvas from 896x1184 to 800x1440."""

    def __init__(self):
        super().__init__()
        self.exp_name = os.path.splitext(os.path.basename(__file__))[0]
        self.input_size = (800, 1440)
        self.test_size = (800, 1440)
