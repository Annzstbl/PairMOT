"""0716_03 paper Base + final pair-aware Liquid on full HSMOT."""
from mmengine.config import read_base

with read_base():
    from .o2_pair_rtdetr_r18vd_2xb4_72e_hsmot_paper_base_coco_full_1200x900_bf16_99 import *  # noqa: F401,F403


_liquid8_patterns = [
    [7, 0, 1],
    [0, 1, 2],
    [1, 2, 3],
    [2, 3, 4],
    [3, 4, 5],
    [4, 5, 6],
    [5, 6, 7],
    [6, 7, 0],
]

# Final Liquid structure validated by 0715_05 on the full dataset. Each frame
# retains its own sampler while ordered pair context couples routing and fusion.
model['backbone'].update(
    liquid_sampler=dict(
        embed_dims=32,
        tau=2.0,
        hard=False,
        init_logit=2.0,
        head_weight_std=1e-3,
        eval_hard=True,
        lowres_grad_downsample=4,
        use_lowres_grad_correction=True,
        lowres_grad_upsample_mode='nearest',
        num_groups=8,
        init_patterns=_liquid8_patterns,
        pair_sampler_router=dict(
            hidden_dims=64,
            init_std=1e-3,
            zero_init=True,
            relation_mode='pair'),
        liquid_group_modulator=dict(
            hidden_dims=16,
            init_std=1e-3),
        liquid_aware_fusion=dict(
            embed_dims=64,
            num_heads=4,
            spatial_kernel=3,
            dropout=0.0,
            init_std=1e-3,
            use_overlap_context=True,
            use_spatial_mixer=True,
            pair_transport=dict(
                hidden_dims=128,
                temperature=0.25,
                init_std=1e-3,
                zero_init=True,
                relation_mode='pair'))))

custom_hooks.extend([
    dict(
        type='LiquidSamplerAnnealHook',
        tau_start=2.0,
        tau_end=0.5,
        anneal_epochs=36,
        hard_start_epoch=36,
        log_interval=200),
    dict(type='LiquidSamplerMonitorHook', interval=50),
])

# Liquid parameters are newly initialized and train at the base learning rate;
# inherited backbone parameters retain the baseline 0.1 multiplier.
optim_wrapper['paramwise_cfg']['custom_keys'].update({
    'backbone.stem.0.liquid_sampler': dict(lr_mult=1.0),
    'backbone.stem.0.liquid_sampler.pair_sampler_router': dict(lr_mult=1.0),
    'backbone.stem.0.liquid_group_modulator': dict(lr_mult=1.0),
    'backbone.stem.0.liquid_aware_fusion': dict(lr_mult=1.0),
    'backbone.stem.0.liquid_aware_fusion.pair_transport': dict(lr_mult=1.0),
    'backbone.stem.0.se_conv': dict(lr_mult=1.0),
})

work_dir = (
    '/data4/litianhao/PairMmot/workdir_99/'
    '0716_03_paper_base_plus_liquid_r18_coco_full_1200x900_bf16_orderedpairs')
val_evaluator['metrics']['track_eval_out_dir'] = f'{work_dir}/val_track_eval'
test_evaluator = val_evaluator
