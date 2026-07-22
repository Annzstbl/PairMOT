# encoding: utf-8
"""Stage-1 HSMOT detector with fixed resolution and no Mosaic/MixUp."""

import os

from yolo11l_diffusion_det_hsmot import Exp as HSMOTExp


class Exp(HSMOTExp):
    def __init__(self):
        super().__init__()
        self.exp_name = os.path.splitext(os.path.basename(__file__))[0]
        self.input_size = (896, 1184)
        self.test_size = (896, 1184)
        self.random_size = None
        self.enable_mixup = False

    def get_data_loader(self, batch_size, is_distributed, no_aug=False):
        # This experiment is non-Mosaic from the first iteration.  Preserve
        # the base experiment's final five-epoch LR phase and validation
        # cadence; only force the data loader onto its non-Mosaic branch.
        return super().get_data_loader(
            batch_size=batch_size, is_distributed=is_distributed,
            no_aug=True)
