"""One-image ablation with the legacy center prior and uncovered GTs."""

import os

from yolo11l_diffusion_det_hsmot_overfit_formal_randomt import (
    Exp as FormalRandomTOverfitExp,
)


class Exp(FormalRandomTOverfitExp):
    def __init__(self):
        super().__init__()
        self.exp_name = os.path.splitext(os.path.basename(__file__))[0]
        self.max_epoch = 100
        self.l1_start_epoch = 15

        # Match Linxu's optimizer/scheduler starting point.  The existing
        # PairMOT requirement that ConvMSI stem uses a 10x group LR remains.
        self.optimizer_base_lr = 2.5e-5
        self.scheduler_base_lr = 2.5e-5
        self.warmup_epochs = 1
        self.no_aug_epochs = 5
        self.min_lr_ratio = 0.05

        # Keep validation comparable to the preceding Linxu diagnostic.
        self.eval_interval = 2
        self.no_aug_eval_interval = 2
        self.val_batch_size = 1

        # No large checkpoints; keep scalar logs and diagnostic images.
        self.save_latest_each_epoch = False
        self.save_interval = self.max_epoch + 1
        self.save_after_eval = False
        self.save_last_mosaic_checkpoint = False

    def get_model(self):
        model = super().get_model()
        matcher = model.head.criterion.matcher
        matcher.center_prior_penalty_weight = 100.0
        matcher.force_gt_coverage = False
        return model
