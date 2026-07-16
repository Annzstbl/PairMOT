"""0715 half-data baseline with BF16 through encoder on local 99."""
from mmengine.config import read_base

with read_base():
    from .o2_pair_rtdetr_r18vd_2xb4_72e_hsmot_pairdn_gap1train_dualcls_nopres_pairtopk_v2_unique_pairdn_allgt import *  # noqa: F401,F403

_pairmot_root = '/data/users/wangying01/lth/PairMOT'
_hsmot_root = '/data/users/wangying01/lth/data/hsmot'
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
    data_root=f'{_hsmot_root}/train',
    gmc_cache_dir=f'{_gmc_root}/hsmot_train_gap1',
)
val_dataloader['dataset'].update(
    data_root=f'{_hsmot_root}/test',
    gmc_cache_dir=f'{_gmc_root}/hsmot_test_gap1',
)
test_dataloader = val_dataloader

work_dir = (
    '/data4/litianhao/PairMmot/workdir_99/'
    '0715_01_0704_01_half_unique_allgt_bf16_encoder_findfalse')

default_hooks['visualization']['draw'] = False
val_evaluator['metrics'].update(
    track_eval=True,
    track_eval_out_dir=f'{work_dir}/val_track_eval',
    track_data_root=f'{_hsmot_root}/test',
)
test_evaluator = val_evaluator
