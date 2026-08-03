"""0803_21: terminal class-margin transport on the 99 two-GPU lane."""
from mmengine.config import read_base

with read_base():
    from .o2_pair_rtdetr_r18vd_2xb4_72e_hsmot_0803_17_iterative_cls_terminal_shared_margins_decoder_99 import *  # noqa: F401,F403


model['bbox_head'].update(
    iterative_cls_terminal_shared_margins=False,
    iterative_cls_terminal_transport_margins=True)

work_dir = (
    '/data4/litianhao/PairMmot/workdir_99/'
    '0803_21_paper_base_liquid_encoder_p5temporal_dualevidence_'
    'decoder_iterativeclsdnisolatede2e_terminaltransportsemanticmargins_'
    'pairdn_paircoherent_le180_r18_coco_full_1200x900_bf16_2xb4_fresh')
val_evaluator['metrics'].update(track_eval_out_dir=f'{work_dir}/val_track_eval')
test_evaluator = val_evaluator
