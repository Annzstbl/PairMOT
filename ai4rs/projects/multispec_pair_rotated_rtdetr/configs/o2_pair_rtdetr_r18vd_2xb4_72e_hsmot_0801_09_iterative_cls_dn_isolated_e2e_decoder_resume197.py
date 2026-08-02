"""197-portable continuation of 0801_09 from its epoch-56 state.

The scientific model and global batch stay identical to 0801_09.  Only the
physical placement changes from 178 1x8 to 197 2x4 so the mature trajectory
can continue while 178 is unavailable.
"""
from mmengine.config import read_base

with read_base():
    from .o2_pair_rtdetr_r18vd_2xb4_72e_hsmot_0801_10_terminal_encoder_cls_residual_decoder_197 import *  # noqa: F401,F403


model['bbox_head'].update(
    iterative_cls_residual=True,
    iterative_cls_dn_absolute=True,
    iterative_cls_detach_between_layers=False,
    iterative_cls_pair_shared_objectness=False,
    terminal_encoder_cls_residual=False,
    terminal_pair_common_cls_residual=False,
    terminal_pair_common_objectness_residual=False,
    terminal_pair_differential_objectness_residual=False,
    cls_proto_gate=False,
    cls_residual_adapter=False)

work_dir = (
    '/data4/litianhao/PairMmot/workdir_178/'
    '0801_09_paper_base_liquid_encoder_p5temporal_dualevidence_'
    'decoder_iterativeclsdnisolatede2e_pairdn_paircoherent_le180_r18_'
    'coco_full_1200x900_bf16_1xb8_fresh')
val_evaluator['metrics'].update(
    track_eval_out_dir=f'{work_dir}/val_track_eval',
    trackeval_root='/data/users/litianhao/PairMOT_sync_3cb888d/TrackEval',
    track_data_root='/data/users/litianhao/PairMOT/data/hsmot/test')
test_evaluator = val_evaluator
