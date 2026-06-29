"""Same-frame train/val diagnostic without PairDN.

This keeps the same data, pair architecture, pretrain, optimizer, validation,
visualization, and early stopping as the same-frame PairDN diagnostic.  The
only intended variable is removing PairDN to check whether DN suppresses
independent detection AP in the formal full-data setting.
"""
from mmengine.config import read_base

with read_base():
    from .o2_pair_rtdetr_r18vd_2xb4_72e_hsmot_pairdn_sameframe_trainval import *

model['pair_dn_cfg'] = None
model['bbox_head']['dn_loss_weight'] = 0.0

work_dir = (
    '/data/users/litianhao01/PairMmot/workdir/'
    'o2_pair_rtdetr_r18vd_2xb4_72e_hsmot_half_sameframe_trainval_nodn')
