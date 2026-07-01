"""Gap-1 conservative dual-frame same-index proposal ablation."""
from mmengine.config import read_base

with read_base():
    from .o2_pair_rtdetr_r18vd_2xb4_72e_hsmot_pairdn_gap1train import *

model.update(
    query_init='pair_topk_sameidx_v1',
    pair_proposal_cfg=dict(
        sameidx_score_mode='sqrt',
        sameidx_ref_source='frame',
    ),
)

# Innovation runs validate frequently and stop once the relaxed AP50 signal
# reaches a small plateau.
max_epochs = 48
train_cfg.update(max_epochs=max_epochs, val_interval=4)
default_hooks.checkpoint.update(interval=4, max_keep_ckpts=8)
for hook in custom_hooks:
    if hook.get('type') == 'EarlyStoppingHook':
        hook.update(
            monitor='pair/independent_AP50',
            rule='greater',
            min_delta=0.001,
            patience=3,
            strict=False)

work_dir = (
    '/data/users/litianhao01/PairMmot/workdir/'
    'o2_pair_rtdetr_r18vd_2xb4_72e_hsmot_half_pairdn_'
    'gap1train_pairtopk_sameidx_v1')
