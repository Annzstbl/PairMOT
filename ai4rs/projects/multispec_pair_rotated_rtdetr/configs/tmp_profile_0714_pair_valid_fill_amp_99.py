"""Temporary 99 smoke test for AMP with pair-valid fallback."""
from mmengine.config import read_base

with read_base():
    from .tmp_profile_0714_pair_valid_fill_99 import *  # noqa: F401,F403

optim_wrapper.update(
    type='AmpOptimWrapper',
    dtype='bfloat16',
    loss_scale=1.0)
model.update(
    fp32_transformer_loss=False,
    fp32_after_encoder_loss=True)

work_dir = (
    '/data4/litianhao/PairMmot/workdir_99/'
    'tmp_profile_0714_pair_valid_fill_amp_findunused_false')
