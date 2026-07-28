"""One-image ablation using corrected LX-style all-GT Dynamic-K coverage."""

import os

from yolo11l_diffusion_det_hsmot_overfit_legacy_penalty_uncovered import (
    Exp as LegacyPenaltyUncoveredExp,
)


class Exp(LegacyPenaltyUncoveredExp):
    def __init__(self):
        super().__init__()
        self.exp_name = os.path.splitext(os.path.basename(__file__))[0]

    def get_model(self):
        model = super().get_model()
        model.head.criterion.matcher.force_gt_coverage = True
        return model
