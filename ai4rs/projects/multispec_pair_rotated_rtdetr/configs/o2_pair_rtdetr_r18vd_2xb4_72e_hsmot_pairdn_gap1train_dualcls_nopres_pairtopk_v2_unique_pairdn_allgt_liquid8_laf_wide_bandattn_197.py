"""0711_02 liquid8 wide LAF with inter-band attention sampler on 197.

Standalone sampler band-attention is weak so far, but it may help when paired
with the currently strongest wide overlap LAF.  This variant lets raw spectral
band descriptors communicate before the sampler head, then uses wide
pattern-aware fusion to consume the resulting group choices.
"""
from mmengine.config import read_base

with read_base():
    from .o2_pair_rtdetr_r18vd_2xb4_72e_hsmot_pairdn_gap1train_dualcls_nopres_pairtopk_v2_unique_pairdn_allgt_liquid8_liquidawarefusion import *  # noqa: F401,F403

model['backbone']['liquid_sampler'].update(
    use_band_attention=True,
    band_attention_heads=4,
    band_attention_dropout=0.0,
)
model['backbone']['liquid_sampler']['liquid_aware_fusion'].update(
    embed_dims=64,
    num_heads=4,
    use_overlap_context=True,
    use_spatial_mixer=True,
)

custom_keys['backbone.stem.0.liquid_sampler.band_attn'] = dict(lr_mult=1.0)
custom_keys['backbone.stem.0.liquid_sampler.band_ffn'] = dict(lr_mult=1.0)

work_dir = (
    '/data4/litianhao/PairMmot/workdir_197/'
    '0711_02_o2_pair_rtdetr_r18vd_2xb4_72e_hsmot_half_pairdn_'
    'gap1train_dualcls_nopres_pairtopk_v2_unique_pairdn_allgt_'
    'liquid8_laf_wide_bandattn')

val_evaluator['metrics'].update(
    track_eval=True,
    track_eval_out_dir=f'{work_dir}/val_track_eval',
)
test_evaluator = val_evaluator
