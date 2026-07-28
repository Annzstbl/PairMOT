"""Bridge 17D: switch only the post-initialization RNG stream to LX."""

import os

from yolo11l_diffusion_det_hsmot_overfit_bridge17c_lxheadinit import (
    Exp as LxHeadInitExp,
)


class Exp(LxHeadInitExp):
    """Model tensors stay Bridge17C-identical; later RNG state becomes LX."""

    def __init__(self):
        super().__init__()
        self.exp_name = os.path.splitext(os.path.basename(__file__))[0]
        self.align_post_model_rng = False
