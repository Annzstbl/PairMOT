"""0725_03: DSE with detection-tangent pair-global CP-DSE."""
from mmengine.config import read_base

with read_base():
    from .o2_pair_rtdetr_r18vd_2xb4_72e_hsmot_paper_liquid_pairdn_paircoherent_le180_pecg_coco_full_1200x900_bf16_252 import *  # noqa: F401,F403


_sampler = model['backbone']['liquid_sampler']
_sampler.pop('pair_evidence_consensus_gate')
_sampler['dispersion_aware_spectral_evidence'] = dict(eps=1e-6)
_sampler['consistency_preserving_dispersion_evidence'] = dict(
    mode='pair_global',
    max_logit_delta=0.5,
    preserve_detection_tangent=True,
    eps=1e-6)
_custom_keys = optim_wrapper['paramwise_cfg']['custom_keys']
_custom_keys.pop('backbone.stem.0.pair_evidence_consensus_gate')
_custom_keys['backbone.stem.0.dispersion_aware_spectral_evidence'] = dict(
    lr_mult=1.0)
_custom_keys[
    'backbone.stem.0.consistency_preserving_dispersion_evidence'] = dict(
        lr_mult=1.0)

work_dir = (
    '/data4/litianhao/PairMmot/workdir_252/'
    '0725_03_paper_liquid_pairdn_paircoherent_le180_dse_cpdse_dettangent_'
    'r18_coco_full_1200x900_bf16_orderedpairs_fast_fresh')
val_evaluator['metrics'].update(
    track_eval_out_dir=f'{work_dir}/val_track_eval')
test_evaluator = val_evaluator
