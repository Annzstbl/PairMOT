"""Controlled bridge 13: enable LX's model-side duplicated-pair flip."""

import os

from yolo11l_diffusion_det_hsmot_overfit_bridge12b_lxmaxlabels import (
    Exp as LxMaxLabelsExp,
)


class Exp(LxMaxLabelsExp):
    """Change only DiffusionNet random_flip from false to true."""

    def __init__(self):
        super().__init__()
        self.exp_name = os.path.splitext(os.path.basename(__file__))[0]
        self.random_flip = True
