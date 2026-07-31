"""0731_28: terminal common evidence with center-only frame motion.

The 0731_21 factorization improves association but its full five-dimensional
antisymmetric box correction can trade DetA for AssA.  This strict structural
ablation keeps classification common evidence unchanged and constrains the
terminal frame-detail correction to center x/y.  Width, height, and angle stay
on the parent geometry.  It adds no parameters, decoder layers, attention,
branches, losses, or material computation relative to 0731_21.
"""
from mmengine.config import read_base

with read_base():
    from .o2_pair_rtdetr_r18vd_2xb4_72e_hsmot_0731_24_terminal_confidentcommon_factorized_evidence_decoder_99 import *  # noqa: F401,F403


model['decoder'].update(
    terminal_factorized_confidence='none',
    terminal_factorized_diagonal_gates=False,
    terminal_factorized_center_motion_only=True)

work_dir = (
    '/data4/litianhao/PairMmot/workdir_99/'
    '0731_28_paper_base_liquid_encoder_p5temporal_dualevidence_'
    'decoder_terminalcentermotionfactorizedevidence_pairdn_paircoherent_'
    'le180_r18_coco_full_1200x900_bf16_2xb4_fresh')
val_evaluator['metrics'].update(
    track_eval_out_dir=f'{work_dir}/val_track_eval')
test_evaluator = val_evaluator
