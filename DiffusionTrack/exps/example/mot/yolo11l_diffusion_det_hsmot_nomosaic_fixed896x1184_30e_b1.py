"""Thirty-epoch full-HSMOT Stage-1 run at physical/effective batch one."""

import os

from yolo11l_diffusion_det_hsmot_nomosaic_fixed896x1184 import (
    Exp as FixedNoMosaicExp,
)


class Exp(FixedNoMosaicExp):
    def __init__(self):
        super().__init__()
        self.exp_name = os.path.splitext(os.path.basename(__file__))[0]
        self.max_epoch = 30
        self.optimizer_base_lr = 2.5e-5
        self.scheduler_base_lr = 2.5e-5
        self.warmup_epochs = 1
        self.no_aug_epochs = 5
        self.min_lr_ratio = 0.05
        self.eval_interval = 3
        self.no_aug_eval_interval = 3
        self.save_interval = 5
        # FP32 rotated validation does not fit at the base config's BS=6 on
        # one 24-GiB GPU. Detection remains batched during training; only the
        # periodic evaluator uses one image at a time.
        self.val_batch_size = 1

    def get_model(self):
        model = super().get_model()
        model.head.criterion.matcher.force_gt_coverage = True
        return model
