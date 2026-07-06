"""Gap-1 PairTopK-v1 proposal ablation for HSMOT half training."""
from mmengine.config import read_base

with read_base():
    from .o2_pair_rtdetr_r18vd_2xb4_72e_hsmot_pairdn_gap1train import *

model.update(
    query_init='pair_topk_v1',
    pair_proposal_cfg=dict(
        pre_topk=600,
        sim_weight=1.0,
        geom_weight=1.0,
        score_weight=1.0,
        geom_sigma=0.08,
        max_center_dist=0.35,
        max_log_scale=1.2,
        match_score_thr=0.0,
        birth_score_thr=0.35,
        death_score_thr=0.35,
        enable_birth=True,
        enable_death=True,
    ),
)

# Shorter innovation runs: validate more frequently and stop once the relaxed
# AP50 signal reaches a small plateau.
max_epochs = 48
train_cfg.update(max_epochs=max_epochs, val_interval=4)
default_hooks.checkpoint.update(interval=4, max_keep_ckpts=8)
for hook in custom_hooks:
    if hook.get('type') in ('EarlyStoppingHook', 'PairTrackEarlyStoppingHook'):
        hook.update(
            type='PairTrackEarlyStoppingHook',
            min_delta=0.001,
            patience=3,
            strict=False)

work_dir = (
    '/data/users/litianhao01/PairMmot/workdir/'
    'o2_pair_rtdetr_r18vd_2xb4_72e_hsmot_half_pairdn_'
    'gap1train_pairtopk_v1')
