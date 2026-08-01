"""0801_13: terminal pair-differential class-agnostic objectness residual.

The final shared decoder state predicts one zero-initialized scalar per query.
Subtracting it from prev logits and adding it to curr logits preserves every
within-frame class margin and the exact pair-mean logit.  Only temporal
objectness skew can change.  DN queries and auxiliary layers remain on the
unchanged absolute classifiers.
"""
from mmengine.config import read_base

with read_base():
    from .o2_pair_rtdetr_r18vd_2xb4_72e_hsmot_0801_12_terminal_pair_common_objectness_residual_decoder_99 import *  # noqa: F401,F403


model['bbox_head'].update(
    terminal_pair_common_objectness_residual=False,
    terminal_pair_differential_objectness_residual=True)

work_dir = (
    '/data4/litianhao/PairMmot/workdir_99/'
    '0801_13_paper_base_liquid_encoder_p5temporal_dualevidence_'
    'decoder_terminalpairdifferentialobjectnessresidual_pairdn_'
    'paircoherent_le180_r18_coco_full_1200x900_bf16_2xb4_fresh')
val_evaluator['metrics'].update(track_eval_out_dir=f'{work_dir}/val_track_eval')
test_evaluator = val_evaluator
