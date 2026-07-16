"""Architecture-only R18 PairMOT target for COCO checkpoint adaptation."""
from mmengine.config import read_base

with read_base():
    from ..o2_pair_rtdetr_r18vd_2xb4_72e_hsmot_pairdn_gap1train_dualcls_nopres_pairtopk_v2_unique_pairdn_allgt import *  # noqa: F401,F403

load_from = None
model.backbone.init_cfg = None
