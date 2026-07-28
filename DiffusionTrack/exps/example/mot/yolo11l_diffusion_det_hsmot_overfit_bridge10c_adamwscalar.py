"""Bridge 10C: use scalar-loop AdamW like LX's PyTorch 1.11 optimizer."""

import os

from yolo11l_diffusion_det_hsmot_overfit_bridge10b_singleoptgroup import (
    Exp as SingleOptGroupExp,
)


class Exp(SingleOptGroupExp):
    """Change only Torch-2 AdamW foreach dispatch from automatic to false."""

    def __init__(self):
        super().__init__()
        self.exp_name = os.path.splitext(os.path.basename(__file__))[0]
        self.adamw_foreach = False
