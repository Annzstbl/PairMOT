"""0801_11 on 197: terminal pair-common classification residual decoder."""
from mmengine.config import read_base

with read_base():
    from .o2_pair_rtdetr_r18vd_2xb4_72e_hsmot_0801_10_terminal_encoder_cls_residual_decoder_197 import *  # noqa: F401,F403


model['bbox_head'].update(
    terminal_encoder_cls_residual=False,
    terminal_pair_common_cls_residual=True,
    iterative_cls_residual=False,
    iterative_cls_dn_absolute=False,
    cls_proto_gate=False,
    cls_residual_adapter=False)

work_dir = (
    '/data4/litianhao/PairMmot/workdir_197/'
    '0801_11_paper_base_liquid_encoder_p5temporal_dualevidence_'
    'decoder_terminalpaircommonclsresidual_pairdn_paircoherent_le180_r18_'
    'coco_full_1200x900_bf16_2xb4_fresh')
val_evaluator['metrics'].update(track_eval_out_dir=f'{work_dir}/val_track_eval')
test_evaluator = val_evaluator
