"""252-portable 0801_13 terminal pair-differential objectness residual.

The final shared decoder state predicts one zero-initialized scalar per query.
Subtracting it from prev logits and adding it to curr logits preserves every
within-frame class margin and the exact pair-mean logit. Only temporal
objectness skew can change. DN queries and auxiliary layers remain on the
unchanged absolute classifiers.
"""
from mmengine.config import read_base

with read_base():
    from .o2_pair_rtdetr_r18vd_2xb4_72e_hsmot_0801_07_iterative_cls_residual_decoder_252 import *  # noqa: F401,F403


model['bbox_head'].update(
    iterative_cls_residual=False,
    iterative_cls_dn_absolute=False,
    terminal_encoder_cls_residual=False,
    terminal_pair_common_cls_residual=False,
    terminal_pair_common_objectness_residual=False,
    terminal_pair_differential_objectness_residual=True,
    cls_proto_gate=False,
    cls_residual_adapter=False)

work_dir = (
    '/data4/litianhao/PairMmot/workdir_252/'
    '0801_13_paper_base_liquid_encoder_p5temporal_dualevidence_'
    'decoder_terminalpairdifferentialobjectnessresidual_pairdn_'
    'paircoherent_le180_r18_coco_full_1200x900_bf16_2xb4_fresh')
val_evaluator['metrics'].update(
    track_eval_out_dir=f'{work_dir}/val_track_eval',
    trackeval_root='/data/users/litianhao01/PairMmot/TrackEval',
    track_data_root='/data/users/litianhao01/PairMmot/data/hsmot/test')
test_evaluator = val_evaluator
