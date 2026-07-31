"""0731_29: lightweight terminal common evidence and center motion.

This is the compact geometric intersection of 0731_27 and 0731_28.  It keeps
only per-channel common/detail gates (512 learned scalars total) and restricts
the antisymmetric terminal box correction to center x/y.  Width, height, and
angle remain on the parent geometry.  No decoder layer, attention block,
branch, loss, confidence weighting, or training-protocol change is added.
"""
from mmengine.config import read_base

with read_base():
    from .o2_pair_rtdetr_r18vd_2xb4_72e_hsmot_0731_26_terminal_confidentboth_factorized_evidence_decoder_252 import *  # noqa: F401,F403


model['decoder'].update(
    terminal_factorized_confidence='none',
    terminal_factorized_diagonal_gates=True,
    terminal_factorized_center_motion_only=True)

work_dir = (
    '/data4/litianhao/PairMmot/workdir_252/'
    '0731_29_paper_base_liquid_encoder_p5temporal_dualevidence_'
    'decoder_terminaldiagonalcentermotionfactorizedevidence_pairdn_'
    'paircoherent_le180_r18_coco_full_1200x900_bf16_2xb4_fresh')
val_evaluator['metrics'].update(
    track_eval_out_dir=f'{work_dir}/val_track_eval')
test_evaluator = val_evaluator
