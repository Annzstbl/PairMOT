"""0730_09: detection-protected motion-trust decoder on 0727_01.

The adapter predicts only a direction for an antisymmetric pair-box update.
Its magnitude is bounded by half of the existing inter-frame reference
displacement and by detached bilateral detection confidence.  Thus it keeps
the pair midpoint fixed, cannot invent motion for coincident references, and
suppresses updates for unilateral/background queries.
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
    competitive_evidence_decoder=False,
    motion_trust_decoder=True)

work_dir = (
    '/data4/litianhao/PairMmot/workdir_197/'
    '0730_09_paper_base_liquid_encoder_p5temporal_dualevidence_'
    'decoder_motiontrust_pairdn_paircoherent_le180_r18_'
    'coco_full_1200x900_bf16_2xb4_fresh')
val_evaluator['metrics'].update(
    track_eval_out_dir=f'{work_dir}/val_track_eval',
    trackeval_root='/data/users/litianhao/PairMOT_sync_3cb888d/TrackEval',
    track_data_root='/data/users/litianhao/PairMOT/data/hsmot/test')
test_evaluator = val_evaluator
