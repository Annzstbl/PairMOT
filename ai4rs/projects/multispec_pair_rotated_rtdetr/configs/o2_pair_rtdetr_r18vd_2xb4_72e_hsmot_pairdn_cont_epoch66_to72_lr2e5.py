"""Low-LR continuation from epoch 66 weights for degradation diagnosis."""
from mmengine.config import read_base

with read_base():
    from .o2_pair_rtdetr_r18vd_2xb4_72e_hsmot_pairdn_cont_epoch66_to72 import *

# Keep the same model weights/data/metrics as the 1e-4 continuation, but reduce
# the reset optimizer step size to check whether AP50 degradation is LR related.
optim_wrapper.optimizer.lr = 2e-5

work_dir = (
    '/data/users/litianhao01/PairMmot/workdir/'
    'o2_pair_rtdetr_r18vd_2xb4_72e_hsmot_half_pairdn_cont_epoch66_to72_lr2e5')
