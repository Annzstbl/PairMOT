"""0717_01 paper Liquid with soft set-transport routing."""
from mmengine.config import read_base

with read_base():
    from .o2_pair_rtdetr_r18vd_2xb4_72e_hsmot_paper_base_plus_liquid_coco_full_1200x900_bf16_99 import *  # noqa: F401,F403


model['backbone']['liquid_sampler'].update(
    hard_group_unique_sets=True,
    soft_group_set_transport=dict(
        initial_strength=0.0,
        num_iters=16,
        temperature=1.0))

# Open set competition before the hard switch. This preserves the exact
# baseline path at startup and aligns soft routing with hard unique assignment
# by epoch 12.
for _hook in custom_hooks:
    if _hook.get('type') == 'LiquidSamplerAnnealHook':
        _hook.update(
            set_transport_start=0.0,
            set_transport_end=1.0,
            set_transport_anneal_epochs=12)

work_dir = (
    '/data4/litianhao/PairMmot/workdir_99/'
    '0717_01_paper_base_plus_liquid_settransport_r18_coco_full_1200x900_bf16_orderedpairs_fresh')
val_evaluator['metrics']['track_eval_out_dir'] = f'{work_dir}/val_track_eval'
test_evaluator = val_evaluator
