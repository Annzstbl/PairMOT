# encoding: utf-8
"""Two-GPU real-image smoke for the unit-complete degree refinement fix."""

import os

from yolo11l_diffusion_det_hsmot_full30_degreecore_fixed_v2 import (
    Exp as DegreeCoreFixedExp,
)


class Exp(DegreeCoreFixedExp):
    def __init__(self):
        super().__init__()
        self.exp_name = os.path.splitext(os.path.basename(__file__))[0]
        root = os.environ.get(
            "HSMOT_SMOKE_ROOT",
            "/data4/linxu/PairMOT_DiffusionTrack/data/"
            "hsmot_overfit_data43_2_x20")
        self.train_data_dir = root
        self.val_data_dir = root
        self.max_epoch = 1
        self.eval_interval = 1
        self.no_aug_eval_interval = 1
        self.save_interval = 1
        self.print_interval = 1
        self.train_vis_interval = 0
        self.data_num_workers = 1


