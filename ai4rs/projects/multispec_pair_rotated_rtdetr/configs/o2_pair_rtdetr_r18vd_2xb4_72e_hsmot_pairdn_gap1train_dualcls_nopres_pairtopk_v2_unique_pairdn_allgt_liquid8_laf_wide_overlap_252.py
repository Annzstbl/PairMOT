"""0709_04 liquid8 wide liquid-aware fusion with overlap context on 252."""
from mmengine.config import read_base

with read_base():
    from .o2_pair_rtdetr_r18vd_2xb4_72e_hsmot_pairdn_gap1train_dualcls_nopres_pairtopk_v2_unique_pairdn_allgt_liquid8_liquidawarefusion import *  # noqa: F401,F403

model['backbone']['liquid_sampler']['liquid_aware_fusion'].update(
    embed_dims=64,
    num_heads=4,
    use_overlap_context=True,
    use_spatial_mixer=True,
)

work_dir = (
    '/data4/litianhao/PairMmot/workdir_252/'
    '0709_04_o2_pair_rtdetr_r18vd_2xb4_72e_hsmot_half_pairdn_'
    'gap1train_dualcls_nopres_pairtopk_v2_unique_pairdn_allgt_'
    'liquid8_laf_wide_overlap')

val_evaluator['metrics'].update(
    track_eval=True,
    track_eval_out_dir=f'{work_dir}/val_track_eval',
)
test_evaluator = val_evaluator
