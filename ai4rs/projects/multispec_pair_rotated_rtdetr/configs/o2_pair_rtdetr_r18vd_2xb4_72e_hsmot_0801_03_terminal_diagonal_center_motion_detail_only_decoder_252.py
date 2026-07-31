"""0801_03: 256-parameter terminal center-motion detail only.

This is the channel-wise counterpart of 0801_02. Classification, recurrent
references, auxiliary outputs, width, height, and angle stay exactly on the
Encoder path. Only the final x/y antisymmetric detail remains, and its dense
256x256 evidence gate is replaced by one 256-vector diagonal gate.
"""
from mmengine.config import read_base

with read_base():
    from .o2_pair_rtdetr_r18vd_2xb4_72e_hsmot_0801_02_terminal_center_motion_detail_only_decoder_252 import *  # noqa: F401,F403


model['decoder'].update(terminal_factorized_diagonal_gates=True)

work_dir = (
    '/data4/litianhao/PairMmot/workdir_252/'
    '0801_03_paper_base_liquid_encoder_p5temporal_dualevidence_'
    'decoder_terminaldiagonalcentermotiondetailonly_pairdn_paircoherent_'
    'le180_r18_coco_full_1200x900_bf16_2xb4_fresh')
val_evaluator['metrics'].update(
    track_eval_out_dir=f'{work_dir}/val_track_eval')
test_evaluator = val_evaluator
