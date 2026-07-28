"""Bridge 17B: switch non-stem pretrained tensors to LX's 2D checkpoint."""

import os

from yolo11l_diffusion_det_hsmot_overfit_bridge17a_native2dstem import (
    Exp as NativeStemExp,
)


class Exp(NativeStemExp):
    """Native stem is unchanged; primary/non-stem weight source becomes LX."""

    def __init__(self):
        super().__init__()
        self.exp_name = os.path.splitext(os.path.basename(__file__))[0]
        self.yolo11_weights = self.yolo11_native_stem_weights
