"""0730_05: motion-decomposed decoder on the 0727_01 encoder baseline.

The shared decoder state and classification path remain identical to the
Liquid + Dual-Evidence encoder baseline.  Each decoder layer additionally
uses the signed difference between its two cross-attention outputs and the
periodic dual-reference displacement to predict one antisymmetric 5D motion
correction.  The correction is zero-initialized and only changes the two box
refinement paths (prev -= delta, curr += delta).
"""
from mmengine.config import read_base

with read_base():
    from .o2_pair_rtdetr_r18vd_2xb4_72e_hsmot_paper_base_liquid_encoder_p5temporal_pyramidlocal_pairdn_paircoherent_le180_coco_full_1200x900_bf16_99 import *  # noqa: F401,F403


model['encoder']['post_pair_temporal_adapter_cfg'].update(
    type='pyramid_dual_evidence',
    use_spatial_evidence=False)
model['decoder'].update(
    tristate_decoder=False,
    dual_output_adapter=False,
    common_motion_decoder=True)

work_dir = (
    '/data4/litianhao/PairMmot/workdir_99/'
    '0730_05_paper_base_liquid_encoder_p5temporal_dualevidence_'
    'decoder_commonmotion_pairdn_paircoherent_le180_r18_coco_full_'
    '1200x900_bf16_orderedpairs_fresh')
val_evaluator['metrics'].update(
    track_eval_out_dir=f'{work_dir}/val_track_eval',
    trackeval_root='/data/users/wangying01/lth/PairMOT/TrackEval',
    track_data_root='/data/users/wangying01/lth/PairMOT/data/hsmot/test')
test_evaluator = val_evaluator
