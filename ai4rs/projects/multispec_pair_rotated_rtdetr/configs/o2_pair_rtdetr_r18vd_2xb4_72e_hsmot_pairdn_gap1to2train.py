"""Gap-1-or-2 training ablation for the formal HSMOT half pair run."""
from mmengine.config import read_base

with read_base():
    from .o2_pair_rtdetr_r18vd_2xb4_72e_hsmot_pairdn import *

# Minimum-change ablation: keep architecture, PairDN, pretrain, optimizer,
# validation, and visualization identical to the formal run. Only broaden the
# training partner sampler from adjacent-only to random previous gap 1 or 2.
train_dataloader['dataset']['random_interval_range'] = (1, 2)
train_dataloader['dataset']['sample_seed'] = 3407

work_dir = (
    '/data/users/litianhao01/PairMmot/workdir/'
    'o2_pair_rtdetr_r18vd_2xb4_72e_hsmot_half_pairdn_gap1to2train')
