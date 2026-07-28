"""Bridge 05D: switch only post-initialization RNG to LX's natural state."""

import os

from yolo11l_diffusion_det_hsmot_overfit_bridge05c_lxfinalxavier import (
    Exp as LxFinalXavierExp,
)


class Exp(LxFinalXavierExp):
    """Model tensors remain Bridge05C-identical; later RNG becomes natural."""

    def __init__(self):
        super().__init__()
        self.exp_name = os.path.splitext(os.path.basename(__file__))[0]
        self.align_post_model_rng = False
