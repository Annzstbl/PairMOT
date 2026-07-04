"""0704 liquid unique proposal with all-GT supervision.

This keeps the Liquid Spectral Sampling Conv3D stem and the non-typed unique
pair-topk proposal generator.  The training target change is disabling the
both-visible GT filter, so decoder loss, encoder/proposal loss, and PairDN
direct targets all use track-union GT.
"""
from mmengine.config import read_base

with read_base():
    from .o2_pair_rtdetr_r18vd_2xb4_72e_hsmot_pairdn_gap1train_bothvis_dualcls_nopres_pairtopk_v2_unique_pairdn_liquid import *  # noqa: F401,F403

model.update(
    num_queries=300,
)
model.decoder.update(num_queries=300)
model.bbox_head.update(
    train_both_visible_only=False,
)
model.test_cfg.update(max_per_img=300)

max_epochs = 72
train_cfg.update(max_epochs=max_epochs)
default_hooks.checkpoint.update(interval=4, max_keep_ckpts=12)

for hook in custom_hooks:
    if hook.get('type') == 'EarlyStoppingHook':
        hook.update(
            monitor='pair/pair_mAP50_95',
            rule='greater',
            min_delta=0.001,
            patience=4,
            strict=False)

work_dir = (
    '/data4/litianhao/PairMmot/workdir_197/'
    '0704_03_liquid_o2_pair_rtdetr_r18vd_2xb4_72e_hsmot_half_pairdn_'
    'gap1train_dualcls_nopres_pairtopk_v2_unique_pairdn_allgt')

val_evaluator['metrics'].update(
    track_eval=True,
    track_eval_out_dir=f'{work_dir}/val_track_eval',
)
test_evaluator = val_evaluator
