"""Controlled bridge 08: model/config endpoint for the FP32 runtime bridge."""

import os

from yolo11l_diffusion_det_hsmot_overfit_bridge07c_minrefined2 import (
    Exp as MinRefinedSideExp,
)


class Exp(MinRefinedSideExp):
    """Keep Bridge07 model unchanged; launcher selects FP32."""

    def __init__(self):
        super().__init__()
        self.exp_name = os.path.splitext(os.path.basename(__file__))[0]
