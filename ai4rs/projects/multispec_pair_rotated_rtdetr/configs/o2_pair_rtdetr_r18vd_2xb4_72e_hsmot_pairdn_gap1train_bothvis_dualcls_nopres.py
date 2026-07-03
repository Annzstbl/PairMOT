"""Gap-1 both-visible pair baseline with dual cls and no presence branch."""
from mmengine.config import read_base

with read_base():
    from .o2_pair_rtdetr_r18vd_2xb4_72e_hsmot_pairdn_gap1train import *

load_from = (
    '/data/users/litianhao01/PairMmot/pretrained_weights/'
    'o2_r18_hsmot_3dse_r2_e72_pair_dualcls_adapted/'
    'pair_dualcls_adapted_pretrain.pth')

# Keep the stable baseline proposal generator.  This experiment isolates the
# GT/head formulation change from PairTopK-v1 proposal exploration.
model.update(
    query_init='dual_topk',
    pair_dn_cfg=None,
)

_pair_assigner = dict(
    type='PairHungarianAssigner',
    match_costs=[
        dict(type='mmdet.FocalLossCost', weight=2.0),
        dict(type='PairChamferCost', side='prev', weight=5.0),
        dict(type='PairChamferCost', side='curr', weight=5.0),
        dict(type='PairGDCost', side='prev', loss_type='kld', fun='log1p',
             tau=1, sqrt=False, weight=2.0),
        dict(type='PairGDCost', side='curr', loss_type='kld', fun='log1p',
             tau=1, sqrt=False, weight=2.0),
    ],
)
model.train_cfg = dict(assigner=_pair_assigner)
model.bbox_head.update(
    type='PairRotatedRTDETRHead',
    use_presence=False,
    dual_cls=True,
    train_both_visible_only=True,
    dn_loss_weight=0.0,
)

test_evaluator = val_evaluator

max_epochs = 48
train_cfg.update(max_epochs=max_epochs, val_interval=4)
default_hooks.checkpoint.update(interval=4, max_keep_ckpts=8)

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
    '0702_o2_pair_rtdetr_r18vd_2xb4_72e_hsmot_half_pairdn_'
    'gap1train_bothvis_dualcls_nopres')

val_evaluator['metrics'].update(
    track_eval=True,
    track_eval_out_dir=f'{work_dir}/val_track_eval',
    track_data_root='/data/users/litianhao01/PairMmot/data/hsmot/test',
    track_new_born_th=0.6,
    track_track_th=0.2,
    track_match_iou_th=0.25,
    track_new_birth_iou_th=0.5,
    track_max_age=30,
)
test_evaluator = val_evaluator
