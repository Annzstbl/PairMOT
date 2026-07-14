"""0713_01 long-tail positive class reweighting on 252.

PairMOT 0704_resume has strong class-agnostic det HOTA but weak class-aware
HOTA on fine-grained long-tail classes.  This experiment keeps the 0704_01
model and tracker path unchanged, and only increases positive classification
loss weight for the classes that lost most against MOTRv2.
"""
from mmengine.config import read_base

with read_base():
    from .o2_pair_rtdetr_r18vd_2xb4_72e_hsmot_pairdn_gap1train_dualcls_nopres_pairtopk_v2_unique_pairdn_allgt import *  # noqa: F401,F403

# Class order: car, bike, pedestrian, van, truck, bus, tricycle, awning-bike.
model.bbox_head.update(
    cls_pos_loss_weights=[
        1.00,  # car: already strong
        1.30,  # bike
        1.00,  # pedestrian: already ahead of MOTRv2
        1.25,  # van
        1.80,  # truck
        1.60,  # bus
        1.70,  # tricycle
        1.25,  # awning-bike
    ],
)

work_dir = (
    '/data4/litianhao/PairMmot/workdir_252/'
    '0713_01_o2_pair_rtdetr_r18vd_2xb4_72e_hsmot_half_pairdn_'
    'gap1train_dualcls_nopres_pairtopk_v2_unique_pairdn_allgt_'
    'longtail_reweight')

val_evaluator['metrics'].update(
    track_eval=True,
    track_eval_out_dir=f'{work_dir}/val_track_eval',
)
test_evaluator = val_evaluator
