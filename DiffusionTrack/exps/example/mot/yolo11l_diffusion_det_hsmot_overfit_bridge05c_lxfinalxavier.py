"""Bridge 05C: reproduce LX's actual post-reset regression initialization."""

import gc
import os
import random

import torch

from yolo11l_diffusion_det_hsmot_overfit_bridge04b_lxregl1 import (
    Exp as LxRegressionL1Exp,
)


class Exp(LxRegressionL1Exp):
    """Use LX constructor/Xavier tensors while holding later RNG at Bridge05B."""

    def __init__(self):
        super().__init__()
        self.exp_name = os.path.splitext(os.path.basename(__file__))[0]
        self.lx_regression_init = True
        self.align_post_model_rng = True

    def get_model(self):
        if getattr(self, "model", None) is not None:
            return self.model
        torch_state = torch.get_rng_state()
        python_state = random.getstate()
        reference_state = None
        reference_python_state = None
        if self.align_post_model_rng:
            from yolo11l_diffusion_det_hsmot_overfit_bridge05b_lxregbias import (
                Exp as Bridge05BExp,
            )
            shadow_exp = Bridge05BExp()
            shadow_exp.get_model()
            reference_state = torch.get_rng_state()
            reference_python_state = random.getstate()
            del shadow_exp
            gc.collect()
            torch.set_rng_state(torch_state)
            random.setstate(python_state)

        model = super().get_model()
        if reference_state is not None:
            torch.set_rng_state(reference_state)
            random.setstate(reference_python_state)
        return model
