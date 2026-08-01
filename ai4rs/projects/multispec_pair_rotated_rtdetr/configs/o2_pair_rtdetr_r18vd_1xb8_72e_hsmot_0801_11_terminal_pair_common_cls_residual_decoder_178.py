"""Single-GPU 178 variant of the 0801_11 pair-common decoder."""
from mmengine.config import read_base

with read_base():
    from .o2_pair_rtdetr_r18vd_1xb8_72e_hsmot_0801_05_symmetric_feature_decoder_178 import *  # noqa: F401,F403


model['decoder'].update(symmetric_feature_decoder=False)
model['bbox_head'].update(
    iterative_cls_residual=False,
    iterative_cls_dn_absolute=False,
    terminal_encoder_cls_residual=False,
    terminal_pair_common_cls_residual=True,
    cls_proto_gate=False,
    cls_residual_adapter=False)

work_dir = (
    '/data4/litianhao/PairMmot/workdir_178/'
    '0801_11_paper_base_liquid_encoder_p5temporal_dualevidence_'
    'decoder_terminalpaircommonclsresidual_pairdn_paircoherent_le180_r18_'
    'coco_full_1200x900_bf16_1xb8_fresh')
val_dataloader['num_workers'] = 0
val_dataloader['persistent_workers'] = False
val_evaluator['metrics'].update(
    track_eval_out_dir=f'{work_dir}/val_track_eval')
test_evaluator = val_evaluator
