"""0729_02: baseline-preserving dual-output residual decoder.

The shared decoder recurrence remains unchanged. Each layer exposes zero-start
prev/curr residual adapters only to the prediction branches, so iteration zero
is exactly the 0728_03 encoder decoder path while both frame branches can
specialize immediately.
"""
from mmengine.config import read_base

with read_base():
    from .o2_pair_rtdetr_r18vd_2xb4_72e_hsmot_paper_base_liquid_encoder_p5temporal_dualevidence_pairdn_easyhardpositive_le180_coco_full_1200x900_bf16_99 import *  # noqa: F401,F403


model['decoder'].update(
    tristate_decoder=False,
    dual_output_adapter=True)

_pairmot_root = '/data/users/litianhao01/PairMmot'
_hsmot_root = f'{_pairmot_root}/data/hsmot'
_gmc_root = f'{_pairmot_root}/workdir/aux/gmc_cache'

train_dataloader['dataset'].update(
    data_root=f'{_hsmot_root}/train',
    gmc_cache_dir=f'{_gmc_root}/hsmot_train_gap1')
val_dataloader['dataset'].update(
    data_root=f'{_hsmot_root}/test',
    gmc_cache_dir=f'{_gmc_root}/hsmot_test_gap1')
test_dataloader = val_dataloader

work_dir = (
    '/data4/litianhao/PairMmot/workdir_252/'
    '0729_02_paper_base_liquid_encoder_p5temporal_dualevidence_'
    'decoder_dualoutputresidual_pairdn_easyhardpositive_le180_'
    'r18_coco_full_1200x900_bf16_orderedpairs_fresh')
val_evaluator['metrics'].update(
    track_eval_out_dir=f'{work_dir}/val_track_eval',
    track_data_root=f'{_hsmot_root}/test')
test_evaluator = val_evaluator
