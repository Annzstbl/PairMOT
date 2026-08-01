"""0801_10: one terminal encoder-anchored classification correction.

The encoder and ordinary pair decoder are identical to 0727_01.  Auxiliary
decoder layers keep their original absolute classifiers.  Only the final
normal-query classification logits receive one zero-initialized residual on
top of detached encoder proposal logits; the DN prefix remains on the
original absolute final classifier because it has no aligned encoder query.
This adds two small linear heads and does not alter boxes, references,
attention, decoder depth, losses, class weights, or inference control flow.
"""
from mmengine.config import read_base

with read_base():
    from .o2_pair_rtdetr_r18vd_2xb4_72e_hsmot_0731_20_sharedattention_terminal_classification_common_evidence_decoder_197 import *  # noqa: F401,F403


# Remove the inherited decoder ablation so the only change from the encoder
# control is the final classification head below.
model['decoder'].update(
    shared_attention_decoder=False,
    terminal_classification_common_evidence_decoder=False)
model['bbox_head'].update(
    terminal_encoder_cls_residual=True,
    iterative_cls_residual=False,
    iterative_cls_dn_absolute=False,
    cls_proto_gate=False,
    cls_residual_adapter=False)

work_dir = (
    '/data4/litianhao/PairMmot/workdir_197/'
    '0801_10_paper_base_liquid_encoder_p5temporal_dualevidence_'
    'decoder_terminalencoderclsresidualdnisolated_pairdn_paircoherent_'
    'le180_r18_coco_full_1200x900_bf16_2xb4_fresh')
val_evaluator['metrics'].update(
    track_eval_out_dir=f'{work_dir}/val_track_eval',
    trackeval_root='/data/users/litianhao/PairMOT_sync_3cb888d/TrackEval',
    track_data_root='/data/users/litianhao/PairMOT/data/hsmot/test')
test_evaluator = val_evaluator
