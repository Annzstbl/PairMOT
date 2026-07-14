"""0710_03 liquid8 with inter-band attention sampler on 252.

This keeps the best plain liquid8 recipe but changes the sampler model itself:
before the recurrent sampler head selects 8 cyclic 3-band groups, each raw
spectral band descriptor attends to the other bands.  The goal is to let the
sampler choose groups from learned inter-band contrast/compatibility instead
of relying only on a one-direction recurrent scan.
"""
from mmengine.config import read_base

with read_base():
    from .o2_pair_rtdetr_r18vd_2xb4_72e_hsmot_pairdn_gap1train_dualcls_nopres_pairtopk_v2_unique_pairdn_allgt_liquid8 import *  # noqa: F401,F403

model['backbone']['liquid_sampler'].update(
    use_band_attention=True,
    band_attention_heads=4,
    band_attention_dropout=0.0,
)

custom_keys['backbone.stem.0.liquid_sampler.band_attn'] = dict(lr_mult=1.0)
custom_keys['backbone.stem.0.liquid_sampler.band_ffn'] = dict(lr_mult=1.0)

work_dir = (
    '/data4/litianhao/PairMmot/workdir_252/'
    '0710_03_o2_pair_rtdetr_r18vd_2xb4_72e_hsmot_half_pairdn_'
    'gap1train_dualcls_nopres_pairtopk_v2_unique_pairdn_allgt_'
    'liquid8_sampler_bandattn')

val_evaluator['metrics'].update(
    track_eval=True,
    track_eval_out_dir=f'{work_dir}/val_track_eval',
)
test_evaluator = val_evaluator
