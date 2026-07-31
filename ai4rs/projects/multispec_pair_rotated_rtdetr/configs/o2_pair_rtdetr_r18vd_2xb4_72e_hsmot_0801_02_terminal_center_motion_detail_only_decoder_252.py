"""0801_02: parent classification plus terminal center-motion detail only.

0731_28 improved both HOTA metrics at epoch 8 but lost that advantage by
epoch 12 while its common classification correction and box detail remained
coupled in one experiment.  This minimal ablation leaves every classification
feature exactly on the Encoder parent path and adds only one terminal,
midpoint-preserving x/y motion correction to the two box heads.  Width,
height, angle, auxiliary outputs, and recurrent references are unchanged.
"""
from mmengine.config import read_base

with read_base():
    from .o2_pair_rtdetr_r18vd_2xb4_72e_hsmot_0731_28_terminal_center_motion_factorized_evidence_decoder_99 import *  # noqa: F401,F403


model['decoder'].update(terminal_factorized_detail_only=True)

work_dir = (
    '/data4/litianhao/PairMmot/workdir_252/'
    '0801_02_paper_base_liquid_encoder_p5temporal_dualevidence_'
    'decoder_terminalcentermotiondetailonly_pairdn_paircoherent_'
    'le180_r18_coco_full_1200x900_bf16_2xb4_fresh')
val_evaluator['metrics'].update(
    track_eval_out_dir=f'{work_dir}/val_track_eval')
test_evaluator = val_evaluator
