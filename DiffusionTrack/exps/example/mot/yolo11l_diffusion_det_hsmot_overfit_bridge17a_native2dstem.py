"""Bridge 17A: replace only ConvMSI with LX's native 8-channel 2D stem."""

import os

from yolo11l_diffusion_det_hsmot_overfit_bridge16b_lxtargetprecision import (
    Exp as LxTargetPrecisionExp,
)


class Exp(LxTargetPrecisionExp):
    """Keep all primary non-stem tensors; source only stem from LX weight."""

    def __init__(self):
        super().__init__()
        self.exp_name = os.path.splitext(os.path.basename(__file__))[0]
        self.yolo11_stem_type = "native"
        self.yolo11_native_stem_weights = (
            "/data/users/linxu/code/DiffusionTrack-lx-baseline-isolated/"
            "ultralytics/weights/yolo11L-8ch-2dstem.pt")
