"""Gap-1-only training ablation for the formal HSMOT half pair run."""
from mmengine.config import read_base

with read_base():
    from .o2_pair_rtdetr_r18vd_2xb4_72e_hsmot_pairdn import *

# Minimum-change ablation: keep architecture, PairDN, pretrain, optimizer,
# validation, and visualization identical to the formal run. Only restrict the
# training partner sampler to adjacent frames, matching the current acceptance
# validation target.
train_dataloader['dataset']['random_interval_range'] = (1, 1)
train_dataloader['dataset']['sample_seed'] = 3407

work_dir = (
    '/data/users/litianhao01/PairMmot/workdir/'
    'o2_pair_rtdetr_r18vd_2xb4_72e_hsmot_half_pairdn_gap1train')
