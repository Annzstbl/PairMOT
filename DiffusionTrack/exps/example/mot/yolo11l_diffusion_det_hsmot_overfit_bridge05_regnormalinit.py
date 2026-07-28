"""Controlled bridge 05: reproduce LX regression projection initialization."""

import os

import torch
from torch import nn

from yolo11l_diffusion_det_hsmot_overfit_bridge04b_lxregl1 import (
    Exp as LxRegressionL1Exp,
)


class Exp(LxRegressionL1Exp):
    """Change only six bbox-delta weights from zeros to N(0, 0.001)."""

    def __init__(self):
        super().__init__()
        self.exp_name = os.path.splitext(os.path.basename(__file__))[0]

    def get_model(self):
        model = super().get_model()
        # The controlled variable is the six projection tensors, not the
        # subsequent training-noise stream. Restore the global CPU RNG after
        # drawing these initialization values so diffusion timesteps/noise
        # remain identical to Bridge04 under the shared experiment seed.
        rng_state = torch.get_rng_state()
        for head in model.head.head.head_series:
            if head.bboxes_delta.weight.count_nonzero().item() != 0:
                raise ValueError(
                    "bridge05 requires zero-initialized regression weights")
            nn.init.normal_(head.bboxes_delta.weight, std=0.001)
        torch.set_rng_state(rng_state)
        return model
