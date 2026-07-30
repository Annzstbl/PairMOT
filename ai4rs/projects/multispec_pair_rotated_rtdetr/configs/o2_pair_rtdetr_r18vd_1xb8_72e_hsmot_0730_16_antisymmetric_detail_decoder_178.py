"""0730_16: detection-preserving antisymmetric frame-detail heads.

The recurrent decoder query remains exactly the 0727_01 shared-query path.
Each layer extracts detached signed detail from its two frame cross-attention
outputs and applies one shared, zero-started, tanh-bounded adapter.  Only the
frame-specific classification/regression heads receive ``-detail`` and
``+detail`` features, so their midpoint remains the common representation.
This tests whether real frame evidence can improve detection without the
DetA-to-AssA transfer seen in motion-trust and shared-evidence decoders.
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
    shared_evidence_decoder=False,
    competitive_evidence_decoder=False,
    motion_trust_decoder=False,
    symmetric_pair_decoder=False,
    shared_routing_decoder=False,
    shared_attention_decoder=False,
    antisymmetric_detail_decoder=True)

work_dir = (
    '/data4/litianhao/PairMmot/workdir_178/'
    '0730_16_paper_base_liquid_encoder_p5temporal_dualevidence_'
    'decoder_antisymmetricdetail_pairdn_paircoherent_le180_r18_'
    'coco_full_1200x900_bf16_1xb8_fresh')
val_evaluator['metrics']['track_eval_out_dir'] = f'{work_dir}/val_track_eval'
test_evaluator = val_evaluator
