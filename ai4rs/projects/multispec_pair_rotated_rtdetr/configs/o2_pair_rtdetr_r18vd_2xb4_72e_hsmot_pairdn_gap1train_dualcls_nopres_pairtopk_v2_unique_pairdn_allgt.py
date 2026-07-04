"""0704_01: unique pair-topk proposal with all-GT supervision.

This keeps the 0702 non-typed unique proposal generator.  The only training
target change is disabling the both-visible GT filter in the head, so decoder
loss, encoder/proposal loss, and PairDN direct targets all use track-union GT.
For dual-cls/no-presence targets, invisible sides are supervised as background
classification and receive no box regression loss.
"""
from mmengine.config import read_base

with read_base():
    from .o2_pair_rtdetr_r18vd_2xb4_72e_hsmot_pairdn_gap1train_bothvis_dualcls_nopres_pairtopk_v2_unique_pairdn import *  # noqa: F401,F403

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
    '/data/users/litianhao01/PairMmot/workdir/'
    '0704_01_o2_pair_rtdetr_r18vd_2xb4_72e_hsmot_half_pairdn_'
    'gap1train_dualcls_nopres_pairtopk_v2_unique_pairdn_allgt')

val_evaluator['metrics'].update(
    track_eval=True,
    track_eval_out_dir=f'{work_dir}/val_track_eval',
)
test_evaluator = val_evaluator
