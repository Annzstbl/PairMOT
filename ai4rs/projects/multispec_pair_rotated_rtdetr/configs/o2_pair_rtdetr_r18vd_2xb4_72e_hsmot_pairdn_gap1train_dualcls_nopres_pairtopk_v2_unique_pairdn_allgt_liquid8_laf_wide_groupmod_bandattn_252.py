"""0711_03 liquid8 wide LAF with group modulation and band attention on 252.

This combines the current best wide overlap LAF with the two follow-up signals:
coverage-aware group modulation for det-side stability, and inter-band
attention before the sampler head for context-aware spectral group selection.
"""
from mmengine.config import read_base

with read_base():
    from .o2_pair_rtdetr_r18vd_2xb4_72e_hsmot_pairdn_gap1train_dualcls_nopres_pairtopk_v2_unique_pairdn_allgt_liquid8_liquidawarefusion import *  # noqa: F401,F403

model['backbone']['liquid_sampler'].update(
    use_band_attention=True,
    band_attention_heads=4,
    band_attention_dropout=0.0,
    liquid_group_modulator=dict(
        hidden_dims=16,
        init_std=1e-3,
    ),
)
model['backbone']['liquid_sampler']['liquid_aware_fusion'].update(
    embed_dims=64,
    num_heads=4,
    use_overlap_context=True,
    use_spatial_mixer=True,
)

custom_keys['backbone.stem.0.liquid_sampler.band_attn'] = dict(lr_mult=1.0)
custom_keys['backbone.stem.0.liquid_sampler.band_ffn'] = dict(lr_mult=1.0)
custom_keys['backbone.stem.0.liquid_group_modulator'] = dict(lr_mult=1.0)

work_dir = (
    '/data4/litianhao/PairMmot/workdir_252/'
    '0711_03_o2_pair_rtdetr_r18vd_2xb4_72e_hsmot_half_pairdn_'
    'gap1train_dualcls_nopres_pairtopk_v2_unique_pairdn_allgt_'
    'liquid8_laf_wide_groupmod_bandattn')

val_evaluator['metrics'].update(
    track_eval=True,
    track_eval_out_dir=f'{work_dir}/val_track_eval',
)
test_evaluator = val_evaluator
