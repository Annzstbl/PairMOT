"""Bridge 20B: replace only the internal ROIAlign kernel with LX's native one."""

import os

from diffusion.models.diffusion_models import Detectron2RotatedROIPooler
from yolo11l_diffusion_det_hsmot_overfit_bridge20a_lxtargetrbox import (
    Exp as LxTargetRboxExp,
)


class Exp(LxTargetRboxExp):
    """Keep degree internal boxes and replace MMCV ROIAlign with Detectron2."""

    def __init__(self):
        super().__init__()
        self.exp_name = os.path.splitext(os.path.basename(__file__))[0]

    def get_model(self):
        model = super().get_model()
        dynamic_head = model.head.head
        source = dynamic_head.box_pooler
        if isinstance(source, Detectron2RotatedROIPooler):
            raise ValueError("bridge20B must replace the MMCV parent pooler")
        dynamic_head.box_pooler = Detectron2RotatedROIPooler(
            output_size=source.output_size[0],
            scales=source.scales,
            sampling_ratio=source.sampling_ratio,
            input_angles_in_degrees=True,
        )
        return model
