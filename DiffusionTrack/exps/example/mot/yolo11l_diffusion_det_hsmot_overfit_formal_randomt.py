"""Single-image x20 overfit with the exact formal Stage-1 configuration."""

import os

from yolo11l_diffusion_det_hsmot_nomosaic_fixed896x1184 import (
    Exp as FormalStage1Exp,
)


class Exp(FormalStage1Exp):
    def __init__(self):
        super().__init__()
        self.exp_name = os.path.splitext(os.path.basename(__file__))[0]
        root = os.environ.get(
            "HSMOT_OVERFIT_ROOT",
            "/data4/linxu/PairMOT_DiffusionTrack/data/"
            "hsmot_overfit_data43_2_x20")
        self.train_data_dir = root
        self.val_data_dir = root
        self.max_epoch = 200
        # The 20-epoch diagnostic enabled L1 at epoch 15.  Keep that absolute
        # transition when extending the same run to 200 epochs; otherwise the
        # inherited max_epoch-no_aug_epochs rule would delay L1 to epoch 195.
        self.l1_start_epoch = 15

        # Capture the first batch of every epoch.
        # This only enables detached diagnostic snapshots; model, data,
        # diffusion, optimizer, EMA, LR and validation settings stay inherited
        # from the formal experiment without overrides.
        self.diffusion_debug_interval = 0
        self.diffusion_debug_first_iter_each_epoch = True
        self.diffusion_debug_max_proposals = 60
        self.diffusion_debug_save_snapshot = False
        self.train_feature_vis_first_iter = True
        # The x20 dataset has exactly ten iterations per epoch.  Emit one
        # complete loss line and TensorBoard sample at every epoch end.
        self.print_interval = 10

        # This short diagnostic keeps only images, CSV and scalar logs.  The
        # regular 4-GB model/optimizer checkpoints and compressed tensor
        # snapshots are intentionally disabled.
        self.save_latest_each_epoch = False
        self.save_interval = self.max_epoch + 1
        self.save_after_eval = False
        self.save_last_mosaic_checkpoint = False
