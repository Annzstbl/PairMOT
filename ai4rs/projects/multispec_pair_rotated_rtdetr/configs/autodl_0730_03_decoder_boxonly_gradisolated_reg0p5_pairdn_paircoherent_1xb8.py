"""AutoDL 0730_03 box-only, gradient-isolated decoder residual."""
from mmengine.config import read_base

with read_base():
    from .o2_pair_rtdetr_r18vd_1xb8_72e_hsmot_paper_base_liquid_encoder_p5temporal_dualevidence_pairdn_paircoherent_le180_coco_full_1200x900_bf16_178 import *  # noqa: F401,F403


model['decoder'].update(
    tristate_decoder=False,
    dual_output_adapter=True,
    dual_output_cls_scale=0.0,
    dual_output_reg_scale=0.5,
    dual_output_detach_adapter_input=True)

_hsmot_root = '/root/autodl-tmp/data/hsmot'
_pretrain_root = '/root/autodl-fs/PairMOT_assets/pretrained_weights'
_gmc_root = '/root/autodl-tmp/PairMOT_assets/gmc_cache'

work_dir = (
    '/root/autodl-tmp/work_dirs/'
    '0730_03_decoder_boxonly_gradisolated_reg0p5_'
    'pairdn_paircoherent_1xb8_fresh')
load_from = (
    f'{_pretrain_root}/rtdetr_r18vd_dec3_6x_coco_from_paddle_pair_adapted/'
    'pair_coco_adapted_pretrain.pth')
resume = False

train_dataloader.update(
    batch_size=8,
    num_workers=8,
    persistent_workers=True)
train_dataloader['dataset'].update(
    data_root=f'{_hsmot_root}/train',
    ann_file=None,
    data_prefix=dict(img_path='npy2jpg'),
    gmc_cache_dir=f'{_gmc_root}/hsmot_train_gap1',
    allow_missing_gmc=False)
val_dataloader.update(
    batch_size=8,
    num_workers=8,
    persistent_workers=True)
val_dataloader['dataset'].update(
    data_root=f'{_hsmot_root}/test',
    data_prefix=dict(img_path='npy2jpg'),
    gmc_cache_dir=f'{_gmc_root}/hsmot_test_gap1',
    allow_missing_gmc=False)
test_dataloader = val_dataloader
val_evaluator['metrics'].update(
    track_eval_out_dir=f'{work_dir}/val_track_eval',
    track_data_root=f'{_hsmot_root}/test')
test_evaluator = val_evaluator
