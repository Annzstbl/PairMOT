"""0715_05 final pair-only Liquid module on the full HSMOT train set."""
from mmengine.config import read_base

with read_base():
    from .o2_pair_rtdetr_r18vd_2xb4_72e_hsmot_pairdn_gap1train_dualcls_nopres_pairtopk_v2_unique_pairdn_allgt_liquid8_laf_wide_groupmod_pairtransport_99 import *  # noqa: F401,F403

_pairmot_root = '/data/users/wangying01/lth/PairMOT'
_hsmot_root = f'{_pairmot_root}/data/hsmot'
_gmc_root = f'{_pairmot_root}/workdir/aux/gmc_cache'

# Pair relations are learned from the ordered pair directly.  The MLP can
# represent directional differences without hand-crafted x-y or x*y inputs.
model['backbone']['liquid_sampler']['pair_sampler_router'].update(
    relation_mode='pair')
model['backbone']['liquid_sampler'].update(
    lowres_grad_upsample_mode='nearest')
model['backbone']['liquid_sampler']['liquid_aware_fusion'][
    'pair_transport'].update(relation_mode='pair')

optim_wrapper.update(
    type='AmpOptimWrapper',
    dtype='bfloat16',
    loss_scale=1.0)
find_unused_parameters = False
model.update(
    fp32_transformer_loss=False,
    fp32_after_encoder_loss=True)

load_from = (
    '/data4/litianhao/PairMmot/pretrained_weights/'
    'rtdetr_r18vd_5x_coco_objects365_pair_unique_allgt_full/'
    'pair_coco365_full_adapted_pretrain.pth')

train_dataloader['dataset'].update(
    data_root=f'{_hsmot_root}/train',
    ann_file=None,
    data_prefix=dict(img_path='npy2jpg'),
    gmc_cache_dir=f'{_gmc_root}/hsmot_train_gap1',
    allow_missing_gmc=False)
val_dataloader['dataset'].update(
    data_root=f'{_hsmot_root}/test',
    data_prefix=dict(img_path='npy2jpg'),
    gmc_cache_dir=f'{_gmc_root}/hsmot_test_gap1',
    allow_missing_gmc=False)
test_dataloader = val_dataloader

work_dir = (
    '/data4/litianhao/PairMmot/workdir_99/'
    '0715_05_liquid8_final_pairtransport_paironly_coco365_full_bf16')

default_hooks['visualization']['draw'] = False
for hook in custom_hooks:
    if hook.get('type') == 'HSMOTPairValVisualizationHook':
        hook.update(draw=False)

val_evaluator['metrics'].update(
    track_eval=True,
    track_eval_out_dir=f'{work_dir}/val_track_eval',
    track_data_root=f'{_hsmot_root}/test')
test_evaluator = val_evaluator

# Explicit entries protect all newly initialized Liquid parameters from the
# generic backbone lr_mult=0.1 rule. Shared baseline parameter groups stay put.
optim_wrapper['paramwise_cfg']['custom_keys'].update({
    'backbone.stem.0.liquid_sampler': dict(lr_mult=1.0),
    'backbone.stem.0.liquid_sampler.pair_sampler_router': dict(lr_mult=1.0),
    'backbone.stem.0.liquid_group_modulator': dict(lr_mult=1.0),
    'backbone.stem.0.liquid_aware_fusion': dict(lr_mult=1.0),
    'backbone.stem.0.liquid_aware_fusion.pair_transport': dict(lr_mult=1.0),
    'backbone.stem.0.se_conv': dict(lr_mult=1.0),
})
