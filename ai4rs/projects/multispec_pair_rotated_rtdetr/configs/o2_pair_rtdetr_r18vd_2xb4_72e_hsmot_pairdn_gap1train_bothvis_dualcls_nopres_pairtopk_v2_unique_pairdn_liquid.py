"""0702 baseline with Liquid Spectral Sampling Conv3D stem."""
from mmengine.config import read_base

with read_base():
    from .o2_pair_rtdetr_r18vd_2xb4_72e_hsmot_pairdn_gap1train_bothvis_dualcls_nopres_pairtopk_v2_unique_pairdn import *  # noqa: F401,F403

model['backbone'].update(
    liquid_sampler=dict(
        embed_dims=32,
        tau=1.0,
        hard=False,
        lowres_grad_downsample=4,
        use_lowres_grad_correction=True,
    ))

custom_hooks.append(dict(type='LiquidSamplerMonitorHook', interval=50))
custom_keys['backbone.stem.0.liquid_sampler'] = dict(lr_mult=1.0)
custom_keys['backbone.stem.0.se_conv'] = dict(lr_mult=1.0)
custom_keys['backbone.stem.0.se_conv1'] = dict(lr_mult=1.0)
custom_keys['backbone.stem.0.se_conv2'] = dict(lr_mult=1.0)

work_dir = (
    '/data/users/litianhao01/PairMmot/workdir/'
    '0703_liquid_o2_pair_rtdetr_r18vd_2xb4_72e_hsmot_half_pairdn_'
    'gap1train_bothvis_dualcls_nopres_pairtopk_v2_unique_pairdn')

val_evaluator['metrics'].update(
    track_eval=True,
    track_eval_out_dir=f'{work_dir}/val_track_eval',
)
test_evaluator = val_evaluator
