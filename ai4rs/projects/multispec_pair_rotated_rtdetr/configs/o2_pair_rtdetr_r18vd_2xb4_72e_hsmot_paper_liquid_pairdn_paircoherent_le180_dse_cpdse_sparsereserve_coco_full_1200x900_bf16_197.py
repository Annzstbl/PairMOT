"""0726_01: DSE + pair-global CP-DSE with sparse evidence reserve."""
from mmengine.config import read_base

with read_base():
    from .o2_pair_rtdetr_r18vd_2xb4_72e_hsmot_paper_liquid_pairdn_paircoherent_le180_dse_cpdse_pairglobal_coco_full_1200x900_bf16_197 import *  # noqa: F401,F403


model['backbone']['liquid_sampler'][
    'consistency_preserving_dispersion_evidence'].update(
        preserve_sparse_detection_evidence=True)

work_dir = (
    '/data4/litianhao/PairMmot/workdir_197/'
    '0726_01_paper_liquid_pairdn_paircoherent_le180_dse_cpdse_'
    'sparsereserve_r18_coco_full_1200x900_bf16_orderedpairs_fresh')
val_evaluator['metrics'].update(
    track_eval_out_dir=f'{work_dir}/val_track_eval')
test_evaluator = val_evaluator
