# encoding: utf-8
"""Fresh full-HSMOT run with the unit-complete degree refinement fix."""

import os

from yolo11l_diffusion_det_hsmot_full30_degreecore_minimal import (
    Exp as DegreeCoreExp,
)


class Exp(DegreeCoreExp):
    def __init__(self):
        super().__init__()
        self.exp_name = os.path.splitext(os.path.basename(__file__))[0]


