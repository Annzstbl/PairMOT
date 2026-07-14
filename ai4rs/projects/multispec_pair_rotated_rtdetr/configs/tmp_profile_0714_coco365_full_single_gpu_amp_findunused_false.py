"""Temporary single-GPU DDP profile with AMP and find_unused disabled."""
from mmengine.config import read_base

with read_base():
    from .tmp_profile_0714_coco365_full_single_gpu_findunused_false import *  # noqa: F401,F403

optim_wrapper.update(
    type='AmpOptimWrapper',
    dtype='bfloat16',
    loss_scale=1.0)
model.update(
    fp32_transformer_loss=False,
    fp32_after_encoder_loss=True)

work_dir = (
    '/data4/litianhao/PairMmot/workdir_252/'
    'tmp_profile_0714_coco365_full_single_gpu_amp_findunused_false')
