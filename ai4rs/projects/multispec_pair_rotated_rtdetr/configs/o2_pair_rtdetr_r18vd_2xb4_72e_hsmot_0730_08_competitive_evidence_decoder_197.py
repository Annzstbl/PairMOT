"""0730_08: swap-invariant competitive-evidence decoder on 0727_01.

Each decoder layer decomposes the two cross-attention outputs into an
equal-weight common path and signed frame detail.  A zero-start, odd gate
selects detail per channel; because both the gate and detail change sign when
the frame order is swapped, their product remains pair invariant.  The gate is
bounded by tanh and its inputs are detached, so the new path cannot bypass the
baseline cross-attention gradient route.
"""
from mmengine.config import read_base

with read_base():
    from .o2_pair_rtdetr_r18vd_2xb4_72e_hsmot_paper_base_liquid_encoder_p5temporal_dualevidence_decoder0708_03_pairdn_paircoherent_le180_coco_full_1200x900_bf16_197 import *  # noqa: F401,F403


model['decoder'].update(
    tristate_decoder=False,
    tristate_separate_ffn=False,
    tristate_zero_init_coupling=False,
    dual_output_adapter=False,
    common_motion_decoder=False,
    shared_evidence_decoder=False,
    competitive_evidence_decoder=True)

work_dir = (
    '/data4/litianhao/PairMmot/workdir_197/'
    '0730_08_paper_base_liquid_encoder_p5temporal_dualevidence_'
    'decoder_competitiveevidence_pairdn_paircoherent_le180_r18_'
    'coco_full_1200x900_bf16_2xb4_fresh')
val_evaluator['metrics'].update(
    track_eval_out_dir=f'{work_dir}/val_track_eval',
    trackeval_root='/data/users/litianhao/PairMOT_sync_3cb888d/TrackEval',
    track_data_root='/data/users/litianhao/PairMOT/data/hsmot/test')
test_evaluator = val_evaluator
