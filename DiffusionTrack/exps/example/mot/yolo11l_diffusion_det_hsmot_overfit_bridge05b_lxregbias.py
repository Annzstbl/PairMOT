"""Bridge 05B: reproduce LX's hidden regression-module bias prior."""

import math
import os

from torch import nn

from yolo11l_diffusion_det_hsmot_overfit_bridge05_regnormalinit import (
    Exp as RegNormalInitExp,
)


class Exp(RegNormalInitExp):
    """Change only hidden reg-module Linear biases from zero to logit(0.01)."""

    def __init__(self):
        super().__init__()
        self.exp_name = os.path.splitext(os.path.basename(__file__))[0]

    def get_model(self):
        model = super().get_model()
        bias = -math.log((1 - 1e-2) / 1e-2)
        for head in model.head.head.head_series:
            for module in head.reg_module:
                if isinstance(module, nn.Linear):
                    if module.bias.count_nonzero().item() != 0:
                        raise ValueError(
                            "bridge05B requires zero hidden regression bias")
                    nn.init.constant_(module.bias, bias)
        return model
