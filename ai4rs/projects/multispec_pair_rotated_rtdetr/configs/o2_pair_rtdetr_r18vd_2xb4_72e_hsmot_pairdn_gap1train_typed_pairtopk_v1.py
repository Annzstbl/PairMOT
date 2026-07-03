"""0703 baseline: typed survival/curr-only/prev-only pair-topk proposals."""
from mmengine.config import read_base

with read_base():
    from .o2_pair_rtdetr_r18vd_2xb4_72e_hsmot_pairdn_gap1train_bothvis_dualcls_nopres_pairtopk_v2_unique_pairdn import *  # noqa: F401,F403

load_from = (
    '/data/users/litianhao01/PairMmot/pretrained_weights/'
    'o2_r18_hsmot_3dse_r2_e72_pair_dualcls_pairdn_adapted/'
    'pair_dualcls_pairdn_adapted_pretrain.pth')

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

work_dir = (
    '/data/users/litianhao01/PairMmot/workdir/'
    '0703_baseline_o2_pair_rtdetr_r18vd_2xb4_72e_hsmot_half_pairdn_'
    'gap1train_dualcls_nopres_typed_pairtopk_v1_pairdn')

val_evaluator['metrics'].update(
    track_eval=True,
    track_eval_out_dir=f'{work_dir}/val_track_eval',
)
test_evaluator = val_evaluator

for hook in custom_hooks:
    if hook.get('type') == 'EarlyStoppingHook':
        hook.update(
            monitor='pair/pair_mAP50_95',
            rule='greater',
            min_delta=0.001,
            patience=4,
            strict=False)
