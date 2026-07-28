"""Controlled bridge 07: reproduce LX refinement without LE135 canonicalization."""

import os

from yolo11l_diffusion_det_hsmot_overfit_bridge06_globalxy import (
    Exp as GlobalXyExp,
)


class Exp(GlobalXyExp):
    """Change only per-stage refined-box canonicalization."""

    def __init__(self):
        super().__init__()
        self.exp_name = os.path.splitext(os.path.basename(__file__))[0]

    def get_model(self):
        model = super().get_model()
        for head in model.head.head.head_series:
            if not head.canonicalize_refined_boxes:
                raise ValueError(
                    "bridge07 requires bridge06 canonicalization enabled")
            head.canonicalize_refined_boxes = False
        return model
