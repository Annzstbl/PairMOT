"""0714 full-data COCO365 baseline with BF16 AMP training."""
from mmengine.config import read_base

with read_base():
    from .o2_pair_rtdetr_r18vd_2xb4_72e_hsmot_pairdn_gap1train_dualcls_nopres_pairtopk_v2_unique_pairdn_allgt_coco365_full_252 import *  # noqa: F401,F403

optim_wrapper.update(
    type='AmpOptimWrapper',
    dtype='bfloat16',
    loss_scale=1.0)
model.update(
    fp32_transformer_loss=False,
    fp32_after_encoder_loss=True)

work_dir = (
    '/data4/litianhao/PairMmot/workdir_252/'
    '0714_01_0704_resume_coco365_full_unique_allgt_amp')
