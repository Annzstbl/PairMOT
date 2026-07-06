"""0704 liquid typed proposal experiment.

This keeps the Liquid Spectral Sampling Conv3D stem and switches the pair
proposal generator to typed survival/curr-only/prev-only queries.
"""
from mmengine.config import read_base

with read_base():
    from .o2_pair_rtdetr_r18vd_2xb4_72e_hsmot_pairdn_gap1train_bothvis_dualcls_nopres_pairtopk_v2_unique_pairdn_liquid import *  # noqa: F401,F403

model.update(
    query_init='typed_pair_topk_v1',
    num_queries=360,
)
model.decoder.update(num_queries=360)
model.bbox_head.update(
    train_both_visible_only=False,
)
model.test_cfg.update(max_per_img=360)
model['pair_proposal_cfg'].update(
    num_survival_queries=300,
    num_curr_only_queries=30,
    num_prev_only_queries=30,
    unique_pair_selection=True,
    affinity_thr=0.15,
    proposal_quality_weight=0.85,
    learned_quality_weight=0.0,
    affinity_rank_weight=0.15,
)

max_epochs = 72
train_cfg.update(max_epochs=max_epochs)
default_hooks.checkpoint.update(interval=4, max_keep_ckpts=12)

for hook in custom_hooks:
    if hook.get('type') in ('EarlyStoppingHook', 'PairTrackEarlyStoppingHook'):
        hook.update(
            type='PairTrackEarlyStoppingHook',
            min_delta=0.001,
            patience=4,
            strict=False)

work_dir = (
    '/data4/litianhao/PairMmot/workdir_197/'
    '0704_02_liquid_o2_pair_rtdetr_r18vd_2xb4_72e_hsmot_half_pairdn_'
    'gap1train_dualcls_nopres_typed_pairtopk_v1_pairdn')

val_evaluator['metrics'].update(
    track_eval=True,
    track_eval_out_dir=f'{work_dir}/val_track_eval',
)
test_evaluator = val_evaluator
