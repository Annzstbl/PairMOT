"""0731_27: channel-preserving terminal common/detail factorization.

This keeps the successful independent-attention, terminal-only orthogonal
factorization of 0731_21, but replaces each dense 256x256 evidence gate with
a diagonal per-channel gate.  It removes cross-channel evidence mixing and
reduces the added gate parameters from 131,072 to 512 without changing the
loss, data, PairDN, recurrent decoder, or common/detail semantics.
"""
from mmengine.config import read_base

with read_base():
    from .o2_pair_rtdetr_r18vd_2xb4_72e_hsmot_0731_25_terminal_confidentdetail_factorized_evidence_decoder_197 import *  # noqa: F401,F403


model['decoder'].update(
    terminal_factorized_confidence='none',
    terminal_factorized_diagonal_gates=True)

work_dir = (
    '/data4/litianhao/PairMmot/workdir_197/'
    '0731_27_paper_base_liquid_encoder_p5temporal_dualevidence_'
    'decoder_terminaldiagonalfactorizedevidence_pairdn_paircoherent_le180_'
    'r18_coco_full_1200x900_bf16_2xb4_fresh')
val_evaluator['metrics'].update(
    track_eval_out_dir=f'{work_dir}/val_track_eval')
test_evaluator = val_evaluator
