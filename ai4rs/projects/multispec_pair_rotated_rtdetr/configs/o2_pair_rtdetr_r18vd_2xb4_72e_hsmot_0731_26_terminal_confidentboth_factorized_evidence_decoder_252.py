"""0731_26: object-reliable common and detail evidence.

Both terminal factorized corrections use the same detached bilateral parent
confidence.  There is no threshold, learned scale, loss, or class reweighting.
"""
from mmengine.config import read_base

with read_base():
    from .o2_pair_rtdetr_r18vd_2xb4_72e_hsmot_0731_18_sharedattention_terminal_orthogonal_factorized_evidence_decoder_252 import *  # noqa: F401,F403


model['decoder'].update(
    shared_attention_decoder=False,
    terminal_factorized_evidence_decoder=True,
    terminal_factorized_confidence='both')

work_dir = (
    '/data4/litianhao/PairMmot/workdir_252/'
    '0731_26_paper_base_liquid_encoder_p5temporal_dualevidence_'
    'decoder_terminalconfidentbothfactorizedevidence_pairdn_paircoherent_'
    'le180_r18_coco_full_1200x900_bf16_2xb4_fresh')
val_evaluator['metrics'].update(
    track_eval_out_dir=f'{work_dir}/val_track_eval')
test_evaluator = val_evaluator
