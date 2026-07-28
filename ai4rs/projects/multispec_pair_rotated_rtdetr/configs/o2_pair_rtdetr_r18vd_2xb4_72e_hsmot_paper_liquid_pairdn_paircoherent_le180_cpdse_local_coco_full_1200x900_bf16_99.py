"""0723_05: pair-coherent PairDN with bounded local dispersion evidence."""
from mmengine.config import read_base

with read_base():
    from .o2_pair_rtdetr_r18vd_2xb4_72e_hsmot_paper_liquid_independent_diffproduct_pairdn_paircoherent_le180_coco_full_1200x900_bf16_99 import *  # noqa: F401,F403


model['backbone']['liquid_sampler'][
    'consistency_preserving_dispersion_evidence'] = dict(
        mode='local', max_logit_delta=0.5, eps=1e-6)
optim_wrapper['paramwise_cfg']['custom_keys'][
    'backbone.stem.0.consistency_preserving_dispersion_evidence'] = dict(
        lr_mult=1.0)

work_dir = (
    '/data4/litianhao/PairMmot/workdir_99/'
    '0723_05_paper_liquid_pairdn_paircoherent_le180_cpdse_local_r18_coco_'
    'full_1200x900_bf16_orderedpairs_fresh')
val_evaluator['metrics'].update(track_eval_out_dir=f'{work_dir}/val_track_eval')
test_evaluator = val_evaluator
