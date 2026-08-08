"""0808_03: accelerate only decoder and prediction-head optimization.

The final product-tangent model and global base LR remain unchanged.  Decoder
and bbox-head parameters receive a 4/3 LR multiplier, isolating whether the
late convergence is local to decoder refinement instead of the encoder or
Liquid backbone.  No parameter, state, loss, or inference operation is added.
"""
from mmengine.config import read_base

with read_base():
    from .o2_pair_rtdetr_r18vd_1xb8_72e_hsmot_0804_01_iterative_cls_terminal_transport_product_tangent_decoder_178 import *  # noqa: F401,F403


optim_wrapper['paramwise_cfg']['custom_keys'].update({
    'decoder': dict(lr_mult=4.0 / 3.0),
    'bbox_head': dict(lr_mult=4.0 / 3.0),
})

work_dir = (
    '/data4/litianhao/PairMmot/workdir_178/'
    '0808_03_final_product_tangent_decoderheadlr133_72e_1xb8_fresh')
val_evaluator['metrics'].update(track_eval_out_dir=f'{work_dir}/val_track_eval')
test_evaluator = val_evaluator
