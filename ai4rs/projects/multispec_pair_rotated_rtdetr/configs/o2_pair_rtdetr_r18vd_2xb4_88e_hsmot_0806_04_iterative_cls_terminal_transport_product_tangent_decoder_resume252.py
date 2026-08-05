"""0806_04: extend the mature product-tangent trajectory from e80 to e88.

Epoch 80 still improves cls/det HOTA and both AP protocols over epoch 76, but
misses the strict det and sum thresholds by 0.051 and 0.542.  This duration-only
continuation preserves model, optimizer, EMA, LR, global batch, data, losses,
decoder geometry, and four-epoch evaluation cadence.
"""
from mmengine.config import read_base

with read_base():
    from .o2_pair_rtdetr_r18vd_2xb4_80e_hsmot_0806_01_iterative_cls_terminal_transport_product_tangent_decoder_resume252 import *  # noqa: F401,F403


max_epochs = 88
train_cfg.update(max_epochs=max_epochs, val_interval=4)

work_dir = (
    '/data4/litianhao/PairMmot/workdir_252/'
    '0806_04_terminal_transport_product_tangent_resume252_e80_to_e88')
val_evaluator['metrics'].update(
    track_eval_out_dir=f'{work_dir}/val_track_eval')
test_evaluator = val_evaluator
