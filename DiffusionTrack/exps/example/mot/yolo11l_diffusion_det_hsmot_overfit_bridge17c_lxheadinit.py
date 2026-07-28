"""Bridge 17C: reproduce LX's build-before-load head initialization offset."""

import gc
import os
import random

import torch

from yolo11l_diffusion_det_hsmot_overfit_bridge17b_lxpretrain import (
    Exp as LxPretrainExp,
)


class Exp(LxPretrainExp):
    """Change YAML-build/head tensors while preserving Bridge17B later RNG."""

    def __init__(self):
        super().__init__()
        self.exp_name = os.path.splitext(os.path.basename(__file__))[0]
        self.yolo11_align_convmsi_rng = False
        self.yolo11_backbone_load_mode = "yolo_builder"
        self.yolo11_cfg = (
            "/data/users/linxu/code/DiffusionTrack-lx-baseline-isolated/"
            "ultralytics/ultralytics/cfg/models/11/"
            "yolo11l-obb-8ch.yaml")
        self.align_post_model_rng = True

    def get_model(self):
        if getattr(self, "model", None) is not None:
            return self.model
        torch_state = torch.get_rng_state()
        python_state = random.getstate()
        reference_state = None
        reference_python_state = None
        if self.align_post_model_rng:
            from yolo11l_diffusion_det_hsmot_overfit_bridge17b_lxpretrain import (
                Exp as Bridge17BExp,
            )
            shadow_exp = Bridge17BExp()
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
