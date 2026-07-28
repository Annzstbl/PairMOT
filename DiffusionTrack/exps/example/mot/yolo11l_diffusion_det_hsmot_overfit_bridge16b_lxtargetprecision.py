"""Controlled bridge 16B: reproduce LX target-coordinate arithmetic."""

import os

from yolo11l_diffusion_det_hsmot_overfit_bridge16_npy import (
    Exp as NpyExp,
)


class Exp(NpyExp):
    """Keep source annotations in float64 until resize, as native LX does."""

    def __init__(self):
        super().__init__()
        self.exp_name = os.path.splitext(os.path.basename(__file__))[0]
        self.hsmot_target_dtype = "float64"
