"""Bridge 12B: align the padded target capacity with LX."""

import os

from yolo11l_diffusion_det_hsmot_overfit_bridge12a_lxworkers import (
    Exp as LxWorkersExp,
)


class Exp(LxWorkersExp):
    """Change only zero-padding capacity from 500 to 1000 target rows."""

    def __init__(self):
        super().__init__()
        self.exp_name = os.path.splitext(os.path.basename(__file__))[0]
        self.max_labels = 1000
