"""0723_07: pair-coherent PairDN with pair evidence consensus gates."""
from mmengine.config import read_base

with read_base():
    from .o2_pair_rtdetr_r18vd_2xb4_72e_hsmot_paper_liquid_independent_diffproduct_pairdn_independent_le180_coco_full_1200x900_bf16_252 import *  # noqa: F401,F403


# Restore the shared relative PairDN noise used by the direct 0723_01 parent.
model['pair_dn_cfg']['share_pair_noise'] = True
model['backbone']['liquid_sampler']['pair_evidence_consensus_gate'] = dict(
    max_strength=1.0, init_logit=-4.0, eps=1e-6)
optim_wrapper['paramwise_cfg']['custom_keys'][
    'backbone.stem.0.pair_evidence_consensus_gate'] = dict(lr_mult=1.0)

work_dir = (
    '/data4/litianhao/PairMmot/workdir_252/'
    '0723_07_paper_liquid_pairdn_paircoherent_le180_pecg_r18_coco_full_'
    '1200x900_bf16_orderedpairs_fresh')
val_evaluator['metrics'].update(track_eval_out_dir=f'{work_dir}/val_track_eval')
test_evaluator = val_evaluator
