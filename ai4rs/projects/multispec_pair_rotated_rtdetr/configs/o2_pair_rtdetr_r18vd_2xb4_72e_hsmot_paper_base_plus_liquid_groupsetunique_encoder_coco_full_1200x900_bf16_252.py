"""0716_05 paper Base + Liquid group uniqueness + temporal encoder."""
from mmengine.config import read_base

with read_base():
    from .o2_pair_rtdetr_r18vd_2xb4_72e_hsmot_paper_base_plus_liquid_coco_full_1200x900_bf16_99 import *  # noqa: F401,F403


# Prevent different hard groups from collapsing to the same unordered band set.
model['backbone']['liquid_sampler']['hard_group_unique_sets'] = True

# Historical encoder winner (0705_01): P5 global pair interaction followed by
# local pair interaction over all three post-FPN feature levels.
model['encoder'].update(
    pair_temporal_adapter_cfg=dict(
        num_heads=4,
        dropout=0.0,
        gamma_init=0.0),
    pair_temporal_adapter_idx=-1,
    post_pair_temporal_adapter_cfg=dict(
        type='pyramid_local',
        in_channels=[256, 256, 256],
        level_indices=[0, 1, 2],
        reduction=4,
        pointwise_groups=8,
        gamma_init=0.0))

custom_hooks.append(dict(type='PairTemporalAdapterMonitorHook', interval=50))
optim_wrapper['paramwise_cfg']['custom_keys'].update({
    'encoder.pair_temporal_adapter.gamma': dict(
        lr_mult=20.0, decay_mult=0.0),
    'encoder.pair_temporal_adapter': dict(lr_mult=2.0),
    'encoder.post_pair_temporal_adapter.gamma': dict(
        lr_mult=20.0, decay_mult=0.0),
    'encoder.post_pair_temporal_adapter': dict(lr_mult=2.0),
})

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

work_dir = (
    '/data4/litianhao/PairMmot/workdir_252/'
    '0716_05_paper_base_plus_liquid_groupsetunique_encoder_r18_coco_full_1200x900_bf16_orderedpairs_fresh')
val_evaluator['metrics'].update(
    track_eval_out_dir=f'{work_dir}/val_track_eval',
    track_data_root=f'{_hsmot_root}/test')
test_evaluator = val_evaluator
