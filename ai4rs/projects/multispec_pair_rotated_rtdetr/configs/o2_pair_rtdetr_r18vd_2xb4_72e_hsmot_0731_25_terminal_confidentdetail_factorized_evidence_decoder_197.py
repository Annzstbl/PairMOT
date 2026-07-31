"""0731_25: object-reliable box detail on the 0731_21 structure.

The independent-attention terminal factorization is unchanged except that its
antisymmetric box detail is multiplied by the detached geometric mean of the
two parent-path object confidences.  Classification common evidence is intact.
"""
from mmengine.config import read_base

with read_base():
    from .o2_pair_rtdetr_r18vd_2xb4_72e_hsmot_0731_20_sharedattention_terminal_classification_common_evidence_decoder_197 import *  # noqa: F401,F403


model['decoder'].update(
    shared_attention_decoder=False,
    terminal_classification_common_evidence_decoder=False,
    terminal_factorized_evidence_decoder=True,
    terminal_factorized_confidence='detail')

work_dir = (
    '/data4/litianhao/PairMmot/workdir_197/'
    '0731_25_paper_base_liquid_encoder_p5temporal_dualevidence_'
    'decoder_terminalconfidentdetailfactorizedevidence_pairdn_paircoherent_'
    'le180_r18_coco_full_1200x900_bf16_2xb4_fresh')
val_evaluator['metrics'].update(
    track_eval_out_dir=f'{work_dir}/val_track_eval')
test_evaluator = val_evaluator
