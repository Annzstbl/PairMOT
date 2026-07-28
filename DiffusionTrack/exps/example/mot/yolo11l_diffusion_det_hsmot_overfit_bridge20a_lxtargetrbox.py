"""Bridge 20A: use only LX's scalar GT qbox-to-rbox conversion."""

import os

from yolo11l_diffusion_det_hsmot_overfit_bridge20_internaldegrees import (
    Exp as InternalDegreesExp,
)


class Exp(InternalDegreesExp):
    """Preserve Bridge20 and replace only target-rbox conversion arithmetic."""

    def __init__(self):
        super().__init__()
        self.exp_name = os.path.splitext(os.path.basename(__file__))[0]

    def get_model(self):
        model = super().get_model()
        if model.head.target_rbox_converter != "native":
            raise ValueError("bridge20A requires native target conversion")
        model.head.target_rbox_converter = "lx"
        return model
