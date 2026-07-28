"""0725_02: DSE with group-centered pair-global CP-DSE, 1xB4 Acc2."""
from mmengine.config import read_base

with read_base():
    from .o2_pair_rtdetr_r18vd_1xb8_72e_hsmot_paper_liquid_pairdn_paircoherent_le180_scpd_coco_full_1200x900_bf16_178 import *  # noqa: F401,F403


model['backbone']['liquid_sampler'].pop(
    'spectral_coordinate_pair_dispersion')
model['backbone']['liquid_sampler'][
    'dispersion_aware_spectral_evidence'] = dict(eps=1e-6)
model['backbone']['liquid_sampler'][
    'consistency_preserving_dispersion_evidence'] = dict(
        mode='pair_global',
        max_logit_delta=0.5,
        center_groups=True,
        eps=1e-6)
_custom_keys = optim_wrapper['paramwise_cfg']['custom_keys']
_custom_keys.pop('backbone.stem.0.spectral_coordinate_pair_dispersion')
_custom_keys['backbone.stem.0.dispersion_aware_spectral_evidence'] = dict(
    lr_mult=1.0)
_custom_keys[
    'backbone.stem.0.consistency_preserving_dispersion_evidence'] = dict(
        lr_mult=1.0)

train_dataloader['batch_size'] = 4
optim_wrapper['accumulative_counts'] = 2
param_scheduler[0]['end'] = 4000
custom_hooks[3].update(interval=2, gamma=4000)

work_dir = (
    '/data4/litianhao/PairMmot/workdir_178/'
    '0725_02_paper_liquid_pairdn_paircoherent_le180_dse_cpdse_centered_'
    'r18_coco_full_1200x900_bf16_1xb4acc2_protocolfix_fresh')
val_evaluator['metrics'].update(
    track_eval_out_dir=f'{work_dir}/val_track_eval')
test_evaluator = val_evaluator
