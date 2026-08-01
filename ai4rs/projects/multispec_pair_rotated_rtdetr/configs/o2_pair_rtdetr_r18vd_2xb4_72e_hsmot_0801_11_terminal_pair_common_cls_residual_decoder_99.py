"""0801_11: terminal pair-common classification residual decoder.

The complete 0727_01 model and decoder remain unchanged at initialization.
Only the final normal-query classification logits receive one shared,
zero-initialized 256-to-8 residual.  Adding the identical residual to both
frames preserves their per-class logit difference, while the original
absolute heads continue to serve DN queries and all auxiliary decoder layers.
There is no category-conditioned routing, class/loss weighting, residual
scale, extra attention, or added decoder depth.
"""
from mmengine.config import read_base

with read_base():
    from .o2_pair_rtdetr_r18vd_2xb4_72e_hsmot_0801_08_iterative_cls_dn_isolated_decoder_99 import *  # noqa: F401,F403


model['bbox_head'].update(
    iterative_cls_residual=False,
    iterative_cls_dn_absolute=False,
    terminal_encoder_cls_residual=False,
    terminal_pair_common_cls_residual=True,
    cls_proto_gate=False,
    cls_residual_adapter=False)

work_dir = (
    '/data4/litianhao/PairMmot/workdir_99/'
    '0801_11_paper_base_liquid_encoder_p5temporal_dualevidence_'
    'decoder_terminalpaircommonclsresidual_pairdn_paircoherent_le180_r18_'
    'coco_full_1200x900_bf16_2xb4_fresh')
val_evaluator['metrics'].update(
    track_eval_out_dir=f'{work_dir}/val_track_eval',
    trackeval_root='/data/users/wangying01/lth/PairMOT/TrackEval',
    track_data_root='/data/users/wangying01/lth/PairMOT/data/hsmot/test')
test_evaluator = val_evaluator
