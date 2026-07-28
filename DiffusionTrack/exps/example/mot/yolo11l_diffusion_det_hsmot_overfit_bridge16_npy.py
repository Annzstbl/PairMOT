"""Controlled bridge 16: replace 3-JPG input with the source NPY tensor."""

import os

from yolo11l_diffusion_det_hsmot_overfit_bridge15_lxnormalize import (
    Exp as LxNormalizeExp,
)


class Exp(LxNormalizeExp):
    """Model/config unchanged; launcher selects NPY instead of 3-JPG."""

    def __init__(self):
        super().__init__()
        self.exp_name = os.path.splitext(os.path.basename(__file__))[0]
