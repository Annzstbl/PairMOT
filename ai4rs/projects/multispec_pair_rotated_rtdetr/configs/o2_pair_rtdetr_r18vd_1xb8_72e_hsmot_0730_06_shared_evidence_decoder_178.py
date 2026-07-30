"""0730_06: swap-invariant shared-evidence decoder on 0727_01.

Each decoder layer derives relative cross-frame disagreement from the two
deformable cross-attention outputs and injects a zero-start correction into
the shared query.  Unlike the frame-specific decoder variants, the evidence
is symmetric under frame exchange and directly serves both classification
and box refinement.  Its input is detached to preserve the baseline gradient
route into the dual cross-attention modules.
"""
from mmengine.config import read_base

with read_base():
    from .o2_pair_rtdetr_r18vd_1xb8_72e_hsmot_paper_base_liquid_encoder_p5temporal_dualevidence_pairdn_paircoherent_le180_coco_full_1200x900_bf16_178 import *  # noqa: F401,F403


model['decoder'].update(
    tristate_decoder=False,
    dual_output_adapter=False,
    common_motion_decoder=False,
    shared_evidence_decoder=True)

work_dir = (
    '/data4/litianhao/PairMmot/workdir_178/'
    '0730_06_paper_base_liquid_encoder_p5temporal_dualevidence_'
    'decoder_sharedevidence_pairdn_paircoherent_le180_r18_coco_full_'
    '1200x900_bf16_1xb8_fresh')
val_evaluator['metrics']['track_eval_out_dir'] = f'{work_dir}/val_track_eval'
test_evaluator = val_evaluator
