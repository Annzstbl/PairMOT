"""0730_12: motion-trust plus swap-invariant shared evidence.

The shared-evidence path contributes pair-disagreement information to the
common decoder query, while motion-trust applies a separately bounded,
detection-confident antisymmetric correction only to the two box branches.
Both adapters are zero-started and use detached evidence, so the experiment
tests complementary query and geometry structure without changing the parent
gradient paths, losses, or training protocol.
"""
from mmengine.config import read_base

with read_base():
    from .o2_pair_rtdetr_r18vd_1xb8_72e_hsmot_paper_base_liquid_encoder_p5temporal_dualevidence_pairdn_paircoherent_le180_coco_full_1200x900_bf16_178 import *  # noqa: F401,F403


model['decoder'].update(
    tristate_decoder=False,
    tristate_separate_ffn=False,
    tristate_zero_init_coupling=False,
    dual_output_adapter=False,
    common_motion_decoder=False,
    shared_evidence_decoder=True,
    competitive_evidence_decoder=False,
    motion_trust_decoder=True,
    symmetric_pair_decoder=False,
    shared_routing_decoder=False)

work_dir = (
    '/data4/litianhao/PairMmot/workdir_178/'
    '0730_12_paper_base_liquid_encoder_p5temporal_dualevidence_'
    'decoder_motiontrust_sharedevidence_pairdn_paircoherent_le180_r18_'
    'coco_full_1200x900_bf16_1xb8_fresh')
val_evaluator['metrics']['track_eval_out_dir'] = f'{work_dir}/val_track_eval'
test_evaluator = val_evaluator
