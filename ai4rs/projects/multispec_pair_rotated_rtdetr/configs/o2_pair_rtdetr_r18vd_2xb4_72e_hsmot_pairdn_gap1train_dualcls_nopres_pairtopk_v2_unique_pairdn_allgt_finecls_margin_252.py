"""0713_02 fine-grained class margin on 252.

This isolates the class-confusion hypothesis from the threshold diagnosis:
the training loss subtracts an additive margin from positive logits of the
fine-grained vehicle/bike-like long-tail classes, forcing a larger true-class
logit gap without changing box, proposal or association losses.
"""
from mmengine.config import read_base

with read_base():
    from .o2_pair_rtdetr_r18vd_2xb4_72e_hsmot_pairdn_gap1train_dualcls_nopres_pairtopk_v2_unique_pairdn_allgt import *  # noqa: F401,F403

# Class order: car, bike, pedestrian, van, truck, bus, tricycle, awning-bike.
model.bbox_head.update(
    cls_pos_loss_weights=[
        1.00,
        1.15,
        1.00,
        1.10,
        1.35,
        1.25,
        1.30,
        1.10,
    ],
    cls_pos_logit_margins=[
        0.00,
        0.08,
        0.00,
        0.06,
        0.15,
        0.12,
        0.14,
        0.06,
    ],
)

work_dir = (
    '/data4/litianhao/PairMmot/workdir_252/'
    '0713_02_o2_pair_rtdetr_r18vd_2xb4_72e_hsmot_half_pairdn_'
    'gap1train_dualcls_nopres_pairtopk_v2_unique_pairdn_allgt_'
    'finecls_margin')

val_evaluator['metrics'].update(
    track_eval=True,
    track_eval_out_dir=f'{work_dir}/val_track_eval',
)
test_evaluator = val_evaluator
