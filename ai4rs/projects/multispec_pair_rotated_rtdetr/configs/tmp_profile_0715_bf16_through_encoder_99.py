"""Profile BF16 through the encoder and keep all later stages in FP32."""
from mmengine.config import read_base

with read_base():
    from .tmp_profile_0714_pair_valid_fill_amp_99 import *  # noqa: F401,F403

model.update(
    fp32_transformer_loss=False,
    fp32_after_encoder_loss=True,
)

work_dir = (
    '/data4/litianhao/PairMmot/workdir_99/'
    'tmp_profile_0715_bf16_through_encoder')
