"""0730_07: complementary decoder structures on the 0727_01 baseline.

This combines the two decoder changes that produced complementary epoch-4
signals in isolation:

1. shared evidence improves the shared query used by classification and both
   box paths; and
2. common motion predicts an antisymmetric correction for the two boxes.

Both branches remain zero-started.  PairDN, the dual-evidence encoder, data
protocol, optimizer policy, and all other settings are inherited unchanged.
"""
from mmengine.config import read_base

with read_base():
    from .o2_pair_rtdetr_r18vd_2xb4_72e_hsmot_paper_base_liquid_encoder_p5temporal_dualevidence_decoder_dualoutputresidual_pairdn_paircoherent_le180_coco_full_1200x900_bf16_252 import *  # noqa: F401,F403


model['decoder'].update(
    tristate_decoder=False,
    dual_output_adapter=False,
    common_motion_decoder=True,
    shared_evidence_decoder=True)

work_dir = (
    '/data4/litianhao/PairMmot/workdir_252/'
    '0730_07_paper_base_liquid_encoder_p5temporal_dualevidence_'
    'decoder_commonmotion_sharedevidence_pairdn_paircoherent_le180_'
    'r18_coco_full_1200x900_bf16_2xb4_fresh')
val_evaluator['metrics'].update(
    track_eval_out_dir=f'{work_dir}/val_track_eval',
    trackeval_root='/data/users/litianhao01/PairMmot/TrackEval',
    track_data_root='/data/users/litianhao01/PairMmot/data/hsmot/test')
test_evaluator = val_evaluator
