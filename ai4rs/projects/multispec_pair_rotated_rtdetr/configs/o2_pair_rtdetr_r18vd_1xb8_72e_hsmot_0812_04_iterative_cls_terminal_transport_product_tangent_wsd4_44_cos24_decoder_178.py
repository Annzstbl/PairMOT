"""0812_04: standard WSD with a longer, smoother cosine decay.

The terminal-only product-tangent model, parameters, data, losses, EMA,
global batch, optimizer groups, and inference graph are unchanged.  The only
scientific change is a conventional warmup-stable-decay schedule: four warmup
epochs, 44 stable epochs, and 24 cosine-decay epochs.  The peak is selected so
the nominal LR integral remains 0.0096, matching the 96-epoch parent budget.
"""
from mmengine.config import read_base

with read_base():
    from .o2_pair_rtdetr_r18vd_1xb8_72e_hsmot_0811_02_iterative_cls_terminal_transport_product_tangent_warmup4_cosine2667_decoder_178 import *  # noqa: F401,F403


_peak_lr = 1.6551724137931035e-4
optim_wrapper['optimizer']['lr'] = _peak_lr
param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=1.0e-7 / _peak_lr,
        end_factor=1.0,
        begin=0,
        end=4,
        by_epoch=True),
    dict(
        type='CosineAnnealingLR',
        T_max=24,
        eta_min_ratio=1.0e-4,
        begin=48,
        end=72,
        by_epoch=True),
]

work_dir = (
    '/data4/litianhao/PairMmot/workdir_178/'
    '0812_04_final_product_tangent_wsd4_44_cos24_72e_1xb8_fresh')
val_evaluator['metrics'].update(track_eval_out_dir=f'{work_dir}/val_track_eval')
test_evaluator = val_evaluator
