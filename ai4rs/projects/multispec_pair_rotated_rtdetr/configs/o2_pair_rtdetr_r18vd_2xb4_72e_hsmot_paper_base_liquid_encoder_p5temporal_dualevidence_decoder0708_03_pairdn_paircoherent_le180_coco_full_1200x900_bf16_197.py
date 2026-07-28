"""0728_01: 0727_01 encoder plus the 0708_03 decoder.

The full paper Base+Liquid protocol and Dual-Evidence encoder are unchanged.
The only model addition is the tri-state pair decoder with zero-initialized
pointer-to-frame and frame-to-pointer recurrent coupling from 0708_03.
"""
from mmengine.config import read_base

with read_base():
    from .o2_pair_rtdetr_r18vd_2xb4_72e_hsmot_paper_base_liquid_encoder_p5temporal_pyramidlocal_pairdn_paircoherent_le180_coco_full_1200x900_bf16_99 import *  # noqa: F401,F403


model['encoder']['post_pair_temporal_adapter_cfg'].update(
    type='pyramid_dual_evidence',
    use_spatial_evidence=False)
model['decoder'].update(
    tristate_decoder=True,
    tristate_separate_ffn=False,
    tristate_zero_init_coupling=True)

_pairmot_root = '/data/users/litianhao/PairMOT'
_hsmot_root = f'{_pairmot_root}/data/hsmot'
_gmc_root = f'{_pairmot_root}/workdir/aux/gmc_cache'
train_dataloader['dataset'].update(
    data_root=f'{_hsmot_root}/train',
    gmc_cache_dir=f'{_gmc_root}/hsmot_train_gap1')
val_dataloader['dataset'].update(
    data_root=f'{_hsmot_root}/test',
    gmc_cache_dir=f'{_gmc_root}/hsmot_test_gap1')
test_dataloader = val_dataloader

work_dir = (
    '/data4/litianhao/PairMmot/workdir_197/'
    '0728_01_paper_base_liquid_encoder_p5temporal_dualevidence_'
    'decoder0708_03_pairdn_paircoherent_le180_r18_coco_full_'
    '1200x900_bf16_orderedpairs_fresh')
val_evaluator['metrics'].update(
    track_eval_out_dir=f'{work_dir}/val_track_eval',
    track_data_root=f'{_hsmot_root}/test')
test_evaluator = val_evaluator
