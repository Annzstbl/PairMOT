"""Bridge 04B: reproduce LX's per-axis normalized five-vector L1 loss."""

import os

from yolo11l_diffusion_det_hsmot_overfit_bridge04_iouaverage import (
    Exp as IouAverageExp,
)


class Exp(IouAverageExp):
    """Change only criterion L1 encoding; matcher remains raw qbox8."""

    def __init__(self):
        super().__init__()
        self.exp_name = os.path.splitext(os.path.basename(__file__))[0]

    def get_model(self):
        model = super().get_model()
        criterion = model.head.criterion
        if criterion.bbox_l1_representation != "le135_geomean":
            raise ValueError(
                "bridge04B requires geometric-mean LE135 regression L1")
        criterion.bbox_l1_representation = "lx_norm5"
        return model
