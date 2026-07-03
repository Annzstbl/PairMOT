"""Gap-1 both-visible dual-cls/no-presence pair baseline with PairDN."""
from mmengine.config import read_base

with read_base():
    from .o2_pair_rtdetr_r18vd_2xb4_72e_hsmot_pairdn_gap1train_bothvis_dualcls_nopres import *  # noqa: F401,F403

load_from = (
    '/data/users/litianhao01/PairMmot/pretrained_weights/'
    'o2_r18_hsmot_3dse_r2_e72_pair_dualcls_pairdn_adapted/'
    'pair_dualcls_pairdn_adapted_pretrain.pth')

model.update(
    pair_dn_cfg=dict(
        label_noise_scale=0.5,
        box_noise_scale=0.4,
        group_cfg=dict(dynamic=True, num_dn_queries=100),
    ),
)

model.bbox_head.update(
    dn_loss_weight=0.2,
)

work_dir = (
    '/data/users/litianhao01/PairMmot/workdir/'
    '0702_o2_pair_rtdetr_r18vd_2xb4_72e_hsmot_half_pairdn_'
    'gap1train_bothvis_dualcls_nopres_pairdn')

val_evaluator['metrics'].update(
    track_eval=True,
    track_eval_out_dir=f'{work_dir}/val_track_eval',
)
test_evaluator = val_evaluator
