"""Bridge 20: LX-style degree boxes inside the complete refine cascade."""

import math
import os

from yolo11l_diffusion_det_hsmot_overfit_bridge19_float64schedule import (
    Exp as Float64ScheduleExp,
)


class Exp(Float64ScheduleExp):
    """Only move the internal DynamicHead angle unit from radians to degrees.

    The public DiffusionHead boundary converts refined boxes back to radians,
    so target preparation, matching, ordinary rotated IoU and evaluation stay
    byte-for-byte on the established HSMOT convention.  Internally all six
    heads, including MMCV RotatedROIAlign, receive LX-compatible degrees.
    """

    def __init__(self):
        super().__init__()
        self.exp_name = os.path.splitext(os.path.basename(__file__))[0]

    def get_model(self):
        model = super().get_model()
        diffusion_head = model.head
        if diffusion_head.box_angle_degrees:
            raise ValueError("bridge20 requires radian-internal parent")
        diffusion_head.set_internal_box_angle_degrees(True)
        for head in diffusion_head.head.head_series:
            if not math.isclose(head.angle_delta_scale, math.pi / 180.0):
                raise ValueError(
                    "bridge20 requires degree-sized radian parent deltas")
            head.angle_delta_scale = 1.0
        pooler = diffusion_head.head.box_pooler
        if pooler.clockwise:
            raise ValueError("bridge20 requires corrected ROIAlign direction")
        return model
