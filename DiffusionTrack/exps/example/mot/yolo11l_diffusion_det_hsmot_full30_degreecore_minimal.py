# encoding: utf-8
"""Full HSMOT Stage-1: goal-time training recipe plus degree geometry core.

All data, matching, loss, optimizer and scheduler settings are inherited from
the established 30-epoch fixed-resolution Stage-1 recipe.  Only the internal
six-layer rotated-box interface uses degrees, matching the implementation
validated by the isolated degree-core diagnostic.
"""

import os

from diffusion.models.diffusion_models import Detectron2RotatedROIPooler
from yolo11l_diffusion_det_hsmot_nomosaic_fixed896x1184_30e_b1 import (
    Exp as Stage1Exp,
)


class Exp(Stage1Exp):
    def __init__(self):
        super().__init__()
        self.exp_name = os.path.splitext(os.path.basename(__file__))[0]
        # Two ranks, one image per rank.  The launcher supplies global BS=2.
        self.val_batch_size = 2

    def get_model(self):
        model = super().get_model()
        head = model.head
        if head.box_angle_degrees:
            raise ValueError("degree-core full run requires the radian baseline")
        head.set_internal_box_angle_degrees(True)
        if head.target_rbox_converter != "native":
            raise ValueError("degree-core full run requires native baseline target")
        head.target_rbox_converter = "lx"
        source = head.head.box_pooler
        if isinstance(source, Detectron2RotatedROIPooler):
            raise ValueError("expected the baseline MMCV rotated ROIAlign pooler")
        head.head.box_pooler = Detectron2RotatedROIPooler(
            output_size=source.output_size[0],
            scales=source.scales,
            sampling_ratio=source.sampling_ratio,
            input_angles_in_degrees=True,
        )
        return model
