"""Bridge 10B: merge the equal-LR optimizer groups to LX's single group."""

import os

from yolo11l_diffusion_det_hsmot_overfit_bridge10_uniformlr import (
    Exp as UniformLrExp,
)


class Exp(UniformLrExp):
    """Change only AdamW parameter grouping; parameter order stays model order."""

    def __init__(self):
        super().__init__()
        self.exp_name = os.path.splitext(os.path.basename(__file__))[0]
        self.optimizer_group_mode = "single"
