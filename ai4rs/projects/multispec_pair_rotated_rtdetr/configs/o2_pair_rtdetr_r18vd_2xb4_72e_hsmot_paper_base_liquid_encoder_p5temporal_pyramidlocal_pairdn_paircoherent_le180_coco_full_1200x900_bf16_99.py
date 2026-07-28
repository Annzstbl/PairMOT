"""0726_02: 0723_01 Liquid plus the 0705_01 temporal encoder.

The ablation keeps the Liquid sampler, PairDN, proposal generator, decoder,
losses, data protocol, and initialization of 0723_01. It adds only the two
zero-gated encoder adapters validated by 0705_01:

1. bidirectional global cross-frame attention on encoded P5 before FPN; and
2. bidirectional local temporal residuals on P3/P4/P5 after FPN.
"""
from mmengine.config import read_base

with read_base():
    from .o2_pair_rtdetr_r18vd_2xb4_72e_hsmot_paper_liquid_independent_diffproduct_pairdn_paircoherent_le180_coco_full_1200x900_bf16_99 import *  # noqa: F401,F403


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

# Match the learning-rate treatment used by the original 0705_01 ablation.
# The zero-initialized gates open quickly, while adapter weights use a modest
# multiplier and all inherited modules retain the 0723_01 optimizer policy.
optim_wrapper['paramwise_cfg']['custom_keys'].update({
    'encoder.pair_temporal_adapter.gamma': dict(
        lr_mult=20.0, decay_mult=0.0),
    'encoder.pair_temporal_adapter': dict(lr_mult=2.0),
    'encoder.post_pair_temporal_adapter.gamma': dict(
        lr_mult=20.0, decay_mult=0.0),
    'encoder.post_pair_temporal_adapter': dict(lr_mult=2.0),
})

work_dir = (
    '/data4/litianhao/PairMmot/workdir_99/'
    '0726_02_paper_base_liquid_encoder_p5temporal_pyramidlocal_'
    'pairdn_paircoherent_le180_r18_coco_full_1200x900_bf16_'
    'orderedpairs_fresh')
val_evaluator['metrics']['track_eval_out_dir'] = f'{work_dir}/val_track_eval'
test_evaluator = val_evaluator
