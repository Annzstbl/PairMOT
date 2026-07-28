"""Bridge 18: remove the one-pixel floor from decoded diffusion sides."""

import os

from yolo11l_diffusion_det_hsmot_overfit_bridge17d_lxtrainrng import (
    Exp as LxTrainRngExp,
)


class Exp(LxTrainRngExp):
    """Match LX's direct normalized-to-pixel diffusion-box conversion."""

    def __init__(self):
        super().__init__()
        self.exp_name = os.path.splitext(os.path.basename(__file__))[0]
        self.min_diffusion_side = 0.0
