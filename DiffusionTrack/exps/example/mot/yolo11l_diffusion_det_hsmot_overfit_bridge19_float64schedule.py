"""Bridge 19: compute and retain the cosine diffusion schedule in float64."""

import os

from yolo11l_diffusion_det_hsmot_overfit_bridge18_nosidefloor import (
    Exp as NoSideFloorExp,
)


class Exp(NoSideFloorExp):
    """Match LX schedule construction dtype; all other variables unchanged."""

    def __init__(self):
        super().__init__()
        self.exp_name = os.path.splitext(os.path.basename(__file__))[0]
        self.diffusion_schedule_float64 = True
