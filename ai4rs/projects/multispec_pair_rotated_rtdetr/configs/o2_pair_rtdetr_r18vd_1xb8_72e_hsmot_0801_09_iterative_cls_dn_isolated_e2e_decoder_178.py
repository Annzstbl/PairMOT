"""0801_09: DN-isolated end-to-end iterative classification refinement.

This strict companion to 0801_08 retains absolute classification for the DN
prefix while allowing the final decoder classification loss to propagate
through earlier normal-query residual stages.  Encoder proposal logits remain
detached.  It changes gradient routing only and adds no parameters, layers,
attention, branches, losses, class weights, or scales relative to 0801_08.
"""
from mmengine.config import read_base

with read_base():
    from .o2_pair_rtdetr_r18vd_1xb8_72e_hsmot_0801_05_symmetric_feature_decoder_178 import *  # noqa: F401,F403


model['decoder'].update(symmetric_feature_decoder=False)
model['bbox_head'].update(
    iterative_cls_residual=True,
    iterative_cls_dn_absolute=True,
    iterative_cls_detach_between_layers=False,
    cls_proto_gate=False,
    cls_residual_adapter=False)

work_dir = (
    '/data4/litianhao/PairMmot/workdir_178/'
    '0801_09_paper_base_liquid_encoder_p5temporal_dualevidence_'
    'decoder_iterativeclsdnisolatede2e_pairdn_paircoherent_le180_r18_'
    'coco_full_1200x900_bf16_1xb8_fresh')
val_evaluator['metrics'].update(
    track_eval_out_dir=f'{work_dir}/val_track_eval')
test_evaluator = val_evaluator
