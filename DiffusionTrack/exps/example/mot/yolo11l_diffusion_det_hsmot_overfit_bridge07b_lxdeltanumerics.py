"""Bridge 07B: reproduce LX's unguarded bbox-delta arithmetic."""

import os

from yolo11l_diffusion_det_hsmot_overfit_bridge07_nocanonical import (
    Exp as NoCanonicalExp,
)


class Exp(NoCanonicalExp):
    """Change only decoder numerical guards; geometry semantics stay fixed."""

    def __init__(self):
        super().__init__()
        self.exp_name = os.path.splitext(os.path.basename(__file__))[0]

    def get_model(self):
        model = super().get_model()
        for head in model.head.head.head_series:
            if head.lx_delta_numerics:
                raise ValueError("bridge07B requires guarded delta arithmetic")
            head.lx_delta_numerics = True
        return model
