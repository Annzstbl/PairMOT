"""0727_04: detail-energy-conserving common/detail temporal encoder.

This is a strict successor of 0726_03. The learned signed detail direction is
unchanged, but each channel's applied residual RMS cannot exceed the original
pair-detail RMS. The parameter-free cap prevents late temporal overcorrection
without changing Liquid, losses, or pair-mean preservation.
"""
from mmengine.config import read_base

with read_base():
    from .o2_pair_rtdetr_r18vd_2xb4_72e_hsmot_paper_base_liquid_encoder_p5temporal_commondetail_pairdn_paircoherent_le180_coco_full_1200x900_bf16_252 import *  # noqa: F401,F403


model['encoder']['post_pair_temporal_adapter_cfg'].update(
    conserve_detail_energy=True)

work_dir = (
    '/data4/litianhao/PairMmot/workdir_252/'
    '0727_04_paper_base_liquid_encoder_p5temporal_detailenergy_'
    'pairdn_paircoherent_le180_r18_coco_full_1200x900_bf16_'
    'orderedpairs_fresh')
val_evaluator['metrics'].update(
    track_eval_out_dir=f'{work_dir}/val_track_eval')
test_evaluator = val_evaluator
