"""0731_24: object-reliable common evidence on the 0731_21 structure.

The independent-attention terminal factorization is unchanged except that its
classification-common correction is multiplied by the detached geometric mean
of the two parent-path object confidences.  Box detail remains unchanged.
"""
from mmengine.config import read_base

with read_base():
    from .o2_pair_rtdetr_r18vd_2xb4_72e_hsmot_0731_19_terminal_classification_common_evidence_decoder_99 import *  # noqa: F401,F403


model['decoder'].update(
    shared_attention_decoder=False,
    terminal_classification_common_evidence_decoder=False,
    terminal_factorized_evidence_decoder=True,
    terminal_factorized_confidence='common')

work_dir = (
    '/data4/litianhao/PairMmot/workdir_99/'
    '0731_24_paper_base_liquid_encoder_p5temporal_dualevidence_'
    'decoder_terminalconfidentcommonfactorizedevidence_pairdn_paircoherent_'
    'le180_r18_coco_full_1200x900_bf16_2xb4_fresh')
val_evaluator['metrics'].update(
    track_eval_out_dir=f'{work_dir}/val_track_eval')
test_evaluator = val_evaluator
