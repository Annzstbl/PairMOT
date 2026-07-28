"""Bridge 07C: clamp every refined width/height to LX's two-pixel floor."""

import os

from yolo11l_diffusion_det_hsmot_overfit_bridge07b_lxdeltanumerics import (
    Exp as LxDeltaNumericsExp,
)


class Exp(LxDeltaNumericsExp):
    """Change only the post-layer refined-box side floor from 0 to 2 pixels."""

    def __init__(self):
        super().__init__()
        self.exp_name = os.path.splitext(os.path.basename(__file__))[0]

    def get_model(self):
        model = super().get_model()
        for head in model.head.head.head_series:
            if head.min_refined_side != 0:
                raise ValueError("bridge07C requires no refined-side floor")
            head.min_refined_side = 2.0
        return model
