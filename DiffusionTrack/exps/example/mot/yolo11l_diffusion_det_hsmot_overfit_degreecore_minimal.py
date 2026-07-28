"""Minimal HSMOT baseline plus only the proven degree-refine repair.

This deliberately inherits the goal-time BS1 baseline rather than the
LX-aligned Bridge20 chain.  It keeps HSMOT matching, regression loss,
optimizer, ConvMSI/MMOT initialization, 896x1184 input and augmentation
settings unchanged.  The three changes below are one inseparable geometry
interface: degree boxes throughout DynamicHead, LX's degree-space target
conversion, and the degree-input Detectron2 rotated ROIAlign kernel.
"""

import os

from diffusion.models.diffusion_models import Detectron2RotatedROIPooler
from yolo11l_diffusion_det_hsmot_overfit_legacy_penalty_covered import (
    Exp as GoalTimeBaselineExp,
)


class Exp(GoalTimeBaselineExp):
    """Goal-time BS1 semantics with the internal degree representation only."""

    def __init__(self):
        super().__init__()
        self.exp_name = os.path.splitext(os.path.basename(__file__))[0]

    def get_model(self):
        model = super().get_model()
        diffusion_head = model.head
        if diffusion_head.box_angle_degrees:
            raise ValueError("minimal degree core requires radian baseline")
        diffusion_head.set_internal_box_angle_degrees(True)
        # The goal-time baseline already has raw residual scale 1.0.  In the
        # degree representation that is also LX's native update scale, hence
        # it is intentionally *not* changed by this minimal experiment.
        for head in diffusion_head.head.head_series:
            if head.angle_delta_scale != 1.0:
                raise ValueError("unexpected goal-time angle residual scale")

        # This conversion belongs to the degree interface, not to matcher or
        # regression-loss alignment.  Public targets remain radians after the
        # boundary, so the inherited HSMOT matcher/L1/IoU are unchanged.
        if diffusion_head.target_rbox_converter != "native":
            raise ValueError("minimal degree core requires native baseline target")
        diffusion_head.target_rbox_converter = "lx"

        source = diffusion_head.head.box_pooler
        if isinstance(source, Detectron2RotatedROIPooler):
            raise ValueError("minimal degree core must replace baseline MMCV pooler")
        diffusion_head.head.box_pooler = Detectron2RotatedROIPooler(
            output_size=source.output_size[0],
            scales=source.scales,
            sampling_ratio=source.sampling_ratio,
            input_angles_in_degrees=True,
        )
        return model
