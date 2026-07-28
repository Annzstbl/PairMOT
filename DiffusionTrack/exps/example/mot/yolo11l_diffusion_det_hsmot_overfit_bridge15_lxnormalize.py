"""Controlled bridge 15: reproduce LX input mean/std normalization."""

import os

from yolo11l_diffusion_det_hsmot_overfit_bridge14_lxsize import (
    Exp as LxSizeExp,
)


class Exp(LxSizeExp):
    """Change only post-/255 normalization to LX's eight-channel constants."""

    def __init__(self):
        super().__init__()
        self.exp_name = os.path.splitext(os.path.basename(__file__))[0]
        self.input_means = (
            0.274, 0.289, 0.282, 0.270,
            0.284, 0.270, 0.284, 0.272,
        )
        self.input_stds = (
            0.197, 0.174, 0.163, 0.175,
            0.183, 0.153, 0.159, 0.165,
        )
