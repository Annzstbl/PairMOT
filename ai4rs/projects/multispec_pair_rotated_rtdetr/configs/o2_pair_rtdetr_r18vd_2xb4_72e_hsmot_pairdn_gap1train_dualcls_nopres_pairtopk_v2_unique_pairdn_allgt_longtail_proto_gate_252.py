"""0713_03 long-tail prototype-gated classification head on 252.

Builds on 0713_01 longtail reweight.  The extra structure adds a lightweight
class-prototype similarity bias to each classification logit so long-tail
classes can learn a class-aware decision direction instead of relying only on
static loss weights.
"""
from mmengine.config import read_base

with read_base():
    from .o2_pair_rtdetr_r18vd_2xb4_72e_hsmot_pairdn_gap1train_dualcls_nopres_pairtopk_v2_unique_pairdn_allgt_longtail_reweight_252 import *  # noqa: F401,F403

# Class order: car, bike, pedestrian, van, truck, bus, tricycle, awning-bike.
model.bbox_head.update(
    cls_proto_gate=True,
    cls_proto_gate_scale=0.12,
    cls_proto_gate_weights=[
        0.20,  # car: keep stable, avoid disturbing the head class
        0.70,  # bike
        0.25,  # pedestrian
        0.60,  # van
        1.00,  # truck
        0.90,  # bus
        1.00,  # tricycle
        0.70,  # awning-bike
    ],
)

work_dir = (
    '/data4/litianhao/PairMmot/workdir_252/'
    '0713_03_o2_pair_rtdetr_r18vd_2xb4_72e_hsmot_half_pairdn_'
    'gap1train_dualcls_nopres_pairtopk_v2_unique_pairdn_allgt_'
    'longtail_proto_gate')

val_evaluator['metrics'].update(
    track_eval=True,
    track_eval_out_dir=f'{work_dir}/val_track_eval',
)
test_evaluator = val_evaluator
