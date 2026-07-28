"""0728_03 on 178: easy/hard-positive PairDN plus 0727_01 encoder."""
from mmengine.config import read_base

with read_base():
    from .o2_pair_rtdetr_r18vd_1xb8_72e_hsmot_paper_base_liquid_encoder_p5temporal_dualevidence_pairdn_paircoherent_le180_coco_full_1200x900_bf16_178 import *  # noqa: F401,F403


model['pair_dn_cfg'].update(
    dn_target_mode='easy_hard_positive',
    share_pair_noise=False,
    positive_hard_min_magnitude=0.5,
    positive_hard_max_magnitude=1.25)

work_dir = (
    '/data4/litianhao/PairMmot/workdir_178/'
    '0728_03_paper_base_liquid_encoder_p5temporal_dualevidence_'
    'pairdn_easyhardpositive_le180_r18_coco_full_1200x900_bf16_'
    '1xb8_fresh')
val_evaluator['metrics']['track_eval_out_dir'] = f'{work_dir}/val_track_eval'
test_evaluator = val_evaluator
