"""0719_06 Paper Base with validated long-tail positive reweighting."""
from mmengine.config import read_base

with read_base():
    from .o2_pair_rtdetr_r18vd_2xb4_72e_hsmot_paper_base_coco_full_1200x900_bf16_99 import *  # noqa: F401,F403


# Class order: car, bike, pedestrian, van, truck, bus, tricycle, awning-bike.
# These positive-only weights are the single-variable 0713_01 setting that
# improved both cls HOTA and det HOTA on the historical controlled protocol.
model['bbox_head'].update(
    cls_pos_loss_weights=[
        1.00,
        1.30,
        1.00,
        1.25,
        1.80,
        1.60,
        1.70,
        1.25,
    ])

work_dir = (
    '/data4/litianhao/PairMmot/workdir_99/'
    '0719_06_paper_base_longtail_reweight_r18_coco_full_1200x900_bf16_'
    'orderedpairs_fresh')
val_evaluator['metrics']['track_eval_out_dir'] = f'{work_dir}/val_track_eval'
test_evaluator = val_evaluator
