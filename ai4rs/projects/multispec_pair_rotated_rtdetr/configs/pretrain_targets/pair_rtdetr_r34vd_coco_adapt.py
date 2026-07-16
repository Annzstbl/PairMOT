"""Architecture-only R34 PairMOT target for COCO checkpoint adaptation."""
from mmengine.config import read_base

with read_base():
    from .pair_rtdetr_r18vd_coco_adapt import *  # noqa: F401,F403

model.backbone.update(depth=34, init_cfg=None)
model.neck.in_channels = [128, 256, 512]
model.encoder.fpn_cfg.expansion = 0.5
model.decoder.num_layers = 4
