"""0716_02 paper Base fresh restart after the local server reboot."""
from mmengine.config import read_base

with read_base():
    from .o2_pair_rtdetr_r18vd_2xb4_72e_hsmot_paper_base_coco_full_1200x900_bf16_99 import *  # noqa: F401,F403


work_dir = (
    '/data4/litianhao/PairMmot/workdir_99/'
    '0716_02_paper_base_r18_coco_full_1200x900_bf16_orderedpairs_reboot_fresh')
val_evaluator['metrics']['track_eval_out_dir'] = f'{work_dir}/val_track_eval'
test_evaluator = val_evaluator
