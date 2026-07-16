"""Architecture-only R50 PairMOT target for COCO checkpoint adaptation."""
from mmengine.config import read_base

with read_base():
    from .pair_rtdetr_r18vd_coco_adapt import *  # noqa: F401,F403

model.backbone.update(depth=50, init_cfg=None)
model.neck.in_channels = [512, 1024, 2048]
model.encoder.fpn_cfg.expansion = 1.0
model.decoder.num_layers = 6
