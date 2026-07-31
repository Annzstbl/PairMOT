"""0801_01: coupled channel-wise terminal common/detail factorization.

This isolates the imbalance observed in 0731_27: its independently learned
detail gate stayed about three times larger than the common gate.  A single
256-value channel gate now regulates both terminal routes.  No decoder layer,
attention operation, loss, data protocol, or recurrent reference is added.
"""
from mmengine.config import read_base

with read_base():
    from .o2_pair_rtdetr_r18vd_2xb4_72e_hsmot_0731_27_terminal_diagonal_factorized_evidence_decoder_197 import *  # noqa: F401,F403


model['decoder'].update(terminal_factorized_coupled_gate=True)

work_dir = (
    '/data4/litianhao/PairMmot/workdir_197/'
    '0801_01_paper_base_liquid_encoder_p5temporal_dualevidence_'
    'decoder_terminalcoupleddiagonalfactorizedevidence_pairdn_'
    'paircoherent_le180_r18_coco_full_1200x900_bf16_2xb4_fresh')
val_evaluator['metrics'].update(
    track_eval_out_dir=f'{work_dir}/val_track_eval')
test_evaluator = val_evaluator
