"""0714: future BF16 launch config for the 0704_01 half baseline on 197."""
from mmengine.config import read_base

with read_base():
    from .o2_pair_rtdetr_r18vd_2xb4_72e_hsmot_pairdn_gap1train_dualcls_nopres_pairtopk_v2_unique_pairdn_allgt import *  # noqa: F401,F403

_pairmot_root = '/data/users/litianhao/PairMOT'
_gmc_root = f'{_pairmot_root}/workdir/aux/gmc_cache'

optim_wrapper.update(
    type='AmpOptimWrapper',
    dtype='bfloat16',
    loss_scale=1.0)
find_unused_parameters = False

model.update(
    fp32_transformer_loss=False,
    fp32_after_encoder_loss=True)

load_from = (
    '/data4/litianhao/PairMmot/pretrained_weights/'
    'o2_r18_hsmot_3dse_r2_e72_pair_dualcls_pairdn_adapted/'
    'pair_dualcls_pairdn_adapted_pretrain.pth')

train_dataloader['dataset'].update(
    data_root=f'{_pairmot_root}/data/hsmot/train',
    gmc_cache_dir=f'{_gmc_root}/hsmot_train_gap1',
)
val_dataloader['dataset'].update(
    data_root=f'{_pairmot_root}/data/hsmot/test',
    gmc_cache_dir=f'{_gmc_root}/hsmot_test_gap1',
)
test_dataloader = val_dataloader

work_dir = (
    '/data4/litianhao/PairMmot/workdir_197/'
    '0714_03_0704_01_half_unique_allgt_hybrid_amp_fixed')

val_evaluator['metrics'].update(
    track_eval=True,
    track_eval_out_dir=f'{work_dir}/val_track_eval',
    track_data_root=f'{_pairmot_root}/data/hsmot/test',
)
test_evaluator = val_evaluator
