"""Bridge 12A: reproduce LX's two data-loader workers."""

import os

from yolo11l_diffusion_det_hsmot_overfit_bridge12_lximageaug import (
    Exp as LxImageAugExp,
)


class Exp(LxImageAugExp):
    """Change only worker count, which controls augmentation RNG streams."""

    def __init__(self):
        super().__init__()
        self.exp_name = os.path.splitext(os.path.basename(__file__))[0]
        self.data_num_workers = 2
