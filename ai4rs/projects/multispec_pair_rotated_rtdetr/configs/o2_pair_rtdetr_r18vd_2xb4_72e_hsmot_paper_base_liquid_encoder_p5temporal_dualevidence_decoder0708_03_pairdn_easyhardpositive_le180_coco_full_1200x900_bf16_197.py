"""0729_01: 0728_03 easy/hard-positive DN plus 0708_03 decoder.

This is the strict decoder successor to 0728_03. The Liquid stem,
Dual-Evidence encoder, data protocol, losses, and easy/hard-positive PairDN
remain unchanged; only the zero-initialized tri-state decoder is enabled.
"""
from mmengine.config import read_base

with read_base():
    from .o2_pair_rtdetr_r18vd_2xb4_72e_hsmot_paper_base_liquid_encoder_p5temporal_dualevidence_decoder0708_03_pairdn_paircoherent_le180_coco_full_1200x900_bf16_197 import *  # noqa: F401,F403


model['pair_dn_cfg'].update(
    dn_target_mode='easy_hard_positive',
    share_pair_noise=False,
    positive_hard_min_magnitude=0.5,
    positive_hard_max_magnitude=1.25)

work_dir = (
    '/data4/litianhao/PairMmot/workdir_197/'
    '0729_01_paper_base_liquid_encoder_p5temporal_dualevidence_'
    'decoder0708_03_pairdn_easyhardpositive_le180_r18_coco_full_'
    '1200x900_bf16_orderedpairs_fresh')
val_evaluator['metrics']['track_eval_out_dir'] = f'{work_dir}/val_track_eval'
test_evaluator = val_evaluator
