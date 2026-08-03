"""0803_22: terminal geometry with transported semantic margins."""
from mmengine.config import read_base

with read_base():
    from .o2_pair_rtdetr_r18vd_2xb4_72e_hsmot_0803_18_iterative_cls_terminal_log_size_angle_shared_margins_decoder_197 import *  # noqa: F401,F403


model['bbox_head'].update(
    iterative_cls_terminal_shared_margins=False,
    iterative_cls_terminal_transport_margins=True)

work_dir = (
    '/data4/litianhao/PairMmot/workdir_197/'
    '0803_22_paper_base_liquid_encoder_p5temporal_dualevidence_'
    'decoder_iterativeclsdnisolatede2e_terminalsharedlogsizetangent_'
    'periodicangle_transportsemanticmargins_pairdn_paircoherent_le180_'
    'r18_coco_full_1200x900_bf16_2xb4_fresh')
val_evaluator['metrics'].update(track_eval_out_dir=f'{work_dir}/val_track_eval')
test_evaluator = val_evaluator
