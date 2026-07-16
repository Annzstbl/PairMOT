"""Four-iteration DDP smoke test for the 0716_02 paper baseline."""
from mmengine.config import read_base

with read_base():
    from ..o2_pair_rtdetr_r18vd_2xb4_72e_hsmot_paper_base_coco_full_1200x900_bf16_99 import *  # noqa: F401,F403

work_dir = (
    '/data4/litianhao/PairMmot/workdir_99/'
    'smoke_0716_02_paper_base_1200x900_4iter')
train_dataloader['dataset']['indices'] = 32
train_cfg.update(max_epochs=1, val_interval=999)
default_hooks['checkpoint'].update(interval=1, max_keep_ckpts=1)
val_evaluator['metrics'].update(
    track_eval=False,
    track_eval_out_dir=f'{work_dir}/val_track_eval')
test_evaluator = val_evaluator
