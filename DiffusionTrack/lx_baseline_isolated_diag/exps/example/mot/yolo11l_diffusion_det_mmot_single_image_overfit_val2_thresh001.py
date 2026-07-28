"""Exact LX val2 replay with the AP evaluation threshold corrected to 0.001."""

from yolo11l_diffusion_det_mmot_single_image_overfit_val2 import (
    Exp as Val2Exp,
)


class Exp(Val2Exp):
    def __init__(self):
        super().__init__()
        self.exp_name = (
            "yolo11l_diffusion_det_mmot_single_image_overfit_val2_thresh001")
        self.conf_thresh = 0.001
        self.det_thresh = 0.001
