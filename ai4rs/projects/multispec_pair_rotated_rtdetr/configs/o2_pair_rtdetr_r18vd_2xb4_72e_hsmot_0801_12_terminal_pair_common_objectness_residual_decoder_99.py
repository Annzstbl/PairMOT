"""0801_12: terminal pair-common class-agnostic objectness residual.

The final shared decoder state predicts one zero-initialized scalar per query.
It is broadcast to every category and both frames, preserving both inter-frame
same-class differences and within-frame class margins.  DN queries and all
auxiliary layers retain the original absolute classifiers.
"""
from mmengine.config import read_base

with read_base():
    from .o2_pair_rtdetr_r18vd_2xb4_72e_hsmot_0801_11_terminal_pair_common_cls_residual_decoder_99 import *  # noqa: F401,F403


model['bbox_head'].update(
    terminal_pair_common_cls_residual=False,
    terminal_pair_common_objectness_residual=True)

work_dir = (
    '/data4/litianhao/PairMmot/workdir_99/'
    '0801_12_paper_base_liquid_encoder_p5temporal_dualevidence_'
    'decoder_terminalpaircommonobjectnessresidual_pairdn_paircoherent_le180_'
    'r18_coco_full_1200x900_bf16_2xb4_fresh')
val_evaluator['metrics'].update(track_eval_out_dir=f'{work_dir}/val_track_eval')
test_evaluator = val_evaluator
