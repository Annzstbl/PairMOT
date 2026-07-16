"""Zero-shot elliptical-motion and spectral-affinity evaluation on 0714_01."""
from mmengine.config import read_base

with read_base():
    from .o2_pair_rtdetr_r18vd_2xb4_72e_hsmot_pairdn_gap1train_dualcls_nopres_pairtopk_v2_unique_pairdn_allgt_coco365_full_252 import *  # noqa: F401,F403

work_dir = (
    '/data4/litianhao/PairMmot/workdir_99/'
    '0715_07_full_baseline_elliptical_spectral_zeroshot')

# Keep proposal pools, legal-pair gates, unique selection, and final top-k
# unchanged. Only redistribute the existing affinity weights to add spectrum.
model['pair_proposal_cfg'].update(
    elliptical_motion=True,
    ellipse_max_aspect_sqrt=1.6,
    sim_weight=0.15,
    geom_weight=0.50,
    score_weight=0.25,
    spectral_weight=0.10,
    spectral_sample_offset=0.20,
)

_data_root = '/data/users/wangying01/lth/PairMOT/data/hsmot/test'
_gmc_root = (
    '/data/users/wangying01/lth/PairMOT/workdir/aux/gmc_cache/'
    'hsmot_test_gap1')
val_dataloader['dataset'].update(
    data_root=_data_root,
    data_prefix=dict(img_path='npy2jpg'),
    gmc_cache_dir=_gmc_root,
    allow_missing_gmc=False,
)
test_dataloader = val_dataloader

val_evaluator['metrics'].update(
    track_eval=True,
    track_eval_out_dir=f'{work_dir}/val_track_eval',
)
test_evaluator = val_evaluator

if 'default_hooks' in globals() and 'visualization' in default_hooks:
    default_hooks['visualization'].update(draw=False)

for hook in custom_hooks:
    if hook.get('type') == 'HSMOTPairValVisualizationHook':
        hook.update(draw=False)
