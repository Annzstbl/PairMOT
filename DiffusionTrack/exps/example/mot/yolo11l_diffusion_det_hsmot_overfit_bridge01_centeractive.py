"""Controlled BS1 bridge 01: activate the configured LX center prior only."""

import os

from yolo11l_diffusion_det_hsmot_overfit_legacy_penalty_covered import (
    Exp as CoveredBaselineExp,
)


class Exp(CoveredBaselineExp):
    """Exact covered BS1 baseline plus active center-prior matching cost."""

    def __init__(self):
        super().__init__()
        self.exp_name = os.path.splitext(os.path.basename(__file__))[0]

    def get_model(self):
        model = super().get_model()
        matcher = model.head.criterion.matcher
        if matcher.center_prior_penalty_weight != 100.0:
            raise ValueError("bridge01 requires center prior weight 100")
        matcher.apply_center_prior_penalty = True
        return model
