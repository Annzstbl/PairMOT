"""0726_03: 0726_02 encoder with pair-mean-preserving local detail.

The P5 global temporal MHA from 0705_01 is retained.  Its post-FPN directional
pyramid-local branch is replaced by an order-equivariant common/detail branch:
an invariant [pair mean, absolute pair detail] descriptor gates an odd local
detail transform, and equal/opposite residuals preserve the pair mean.
"""
from mmengine.config import read_base

with read_base():
    from .o2_pair_rtdetr_r18vd_2xb4_72e_hsmot_paper_base_liquid_encoder_p5temporal_pyramidlocal_pairdn_paircoherent_le180_coco_full_1200x900_bf16_99 import *  # noqa: F401,F403


_pairmot_root = '/data/users/litianhao01/PairMmot'
_hsmot_root = f'{_pairmot_root}/data/hsmot'
_gmc_root = f'{_pairmot_root}/workdir/aux/gmc_cache'

train_dataloader['dataset'].update(
    data_root=f'{_hsmot_root}/train',
    gmc_cache_dir=f'{_gmc_root}/hsmot_train_gap1')
val_dataloader['dataset'].update(
    data_root=f'{_hsmot_root}/test',
    gmc_cache_dir=f'{_gmc_root}/hsmot_test_gap1')
test_dataloader = val_dataloader

model['encoder']['post_pair_temporal_adapter_cfg'].update(
    type='pyramid_common_detail')

work_dir = (
    '/data4/litianhao/PairMmot/workdir_252/'
    '0726_03_paper_base_liquid_encoder_p5temporal_commondetail_'
    'pairdn_paircoherent_le180_r18_coco_full_1200x900_bf16_'
    'orderedpairs_fresh')
val_evaluator['metrics'].update(
    track_eval_out_dir=f'{work_dir}/val_track_eval',
    track_data_root=f'{_hsmot_root}/test')
test_evaluator = val_evaluator
