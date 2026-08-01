"""0801_08: isolate absolute DN classification from normal residual cls.

Normal queries keep 0801_07's lightweight iterative residual refinement from
detached encoder proposal logits.  The DN prefix instead uses the existing
absolute decoder classifiers, because DN queries have no aligned encoder
proposal logits.  This removes the conflicting absolute-versus-residual task
from each new linear head without adding parameters, layers, attention,
losses, class weights, or scales.
"""
from mmengine.config import read_base

with read_base():
    from .o2_pair_rtdetr_r18vd_2xb4_72e_hsmot_0731_19_terminal_classification_common_evidence_decoder_99 import *  # noqa: F401,F403


model['decoder'].update(
    terminal_classification_common_evidence_decoder=False)
model['bbox_head'].update(
    iterative_cls_residual=True,
    iterative_cls_dn_absolute=True,
    iterative_cls_detach_between_layers=True,
    cls_proto_gate=False,
    cls_residual_adapter=False)

work_dir = (
    '/data4/litianhao/PairMmot/workdir_99/'
    '0801_08_paper_base_liquid_encoder_p5temporal_dualevidence_'
    'decoder_iterativeclsdnisolated_pairdn_paircoherent_le180_r18_'
    'coco_full_1200x900_bf16_2xb4_fresh')
val_evaluator['metrics'].update(
    track_eval_out_dir=f'{work_dir}/val_track_eval',
    trackeval_root='/data/users/wangying01/lth/PairMOT/TrackEval',
    track_data_root='/data/users/wangying01/lth/PairMOT/data/hsmot/test')
test_evaluator = val_evaluator
