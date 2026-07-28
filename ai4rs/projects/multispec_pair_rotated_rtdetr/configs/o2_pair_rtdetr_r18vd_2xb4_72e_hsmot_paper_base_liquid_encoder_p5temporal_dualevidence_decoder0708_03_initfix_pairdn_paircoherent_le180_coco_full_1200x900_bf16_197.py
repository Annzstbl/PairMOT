"""0729_04: strict old-DN rerun of 0728_01 after decoder init fixes.

The Liquid Base, Dual-Evidence encoder, pair-coherent positive/negative DN,
data protocol, and tri-state decoder configuration are identical to 0728_01.
Only the globally corrected structural initialization in the current source
distinguishes this fresh run from the historical invalid-initialization run.
"""
from mmengine.config import read_base

with read_base():
    from .o2_pair_rtdetr_r18vd_2xb4_72e_hsmot_paper_base_liquid_encoder_p5temporal_dualevidence_decoder0708_03_pairdn_paircoherent_le180_coco_full_1200x900_bf16_197 import *  # noqa: F401,F403


work_dir = (
    '/data4/litianhao/PairMmot/workdir_197/'
    '0729_04_paper_base_liquid_encoder_p5temporal_dualevidence_'
    'decoder0708_03_initfix_pairdn_paircoherent_le180_r18_coco_full_'
    '1200x900_bf16_orderedpairs_fresh')
val_evaluator['metrics']['track_eval_out_dir'] = f'{work_dir}/val_track_eval'
test_evaluator = val_evaluator
