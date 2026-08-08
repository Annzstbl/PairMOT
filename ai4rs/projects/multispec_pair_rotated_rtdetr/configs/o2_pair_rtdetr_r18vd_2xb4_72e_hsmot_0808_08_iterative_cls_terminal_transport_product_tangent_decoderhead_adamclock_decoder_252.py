"""0808_08: compress only the decoder/head optimizer clock.

The final product-tangent model and all global training clocks remain
unchanged.  Building on the successful decoder/head LR x4/3 intervention,
only those parameter groups also receive Adam memory decays compressed from
96 to 72 epochs: beta' = beta ** (96 / 72).  Backbone and encoder parameters
retain the parent LR and Adam betas.  No class-aware operation, reweighting,
parameter, state, loss, or inference computation is introduced.
"""
from mmengine.config import read_base

with read_base():
    from .o2_pair_rtdetr_r18vd_2xb4_72e_hsmot_0804_01_iterative_cls_terminal_transport_product_tangent_decoder_resume252 import *  # noqa: F401,F403


_clock_ratio = 96.0 / 72.0
_local_betas = (0.9 ** _clock_ratio, 0.999 ** _clock_ratio)
optim_wrapper['paramwise_cfg']['custom_keys'].update({
    'decoder': dict(lr_mult=4.0 / 3.0, betas=_local_betas),
    'bbox_head': dict(lr_mult=4.0 / 3.0, betas=_local_betas),
})

work_dir = (
    '/data4/litianhao/PairMmot/workdir_252/'
    '0808_08_final_product_tangent_decoderhead_adamclock_72e_2xb4_fresh')
val_evaluator['metrics'].update(track_eval_out_dir=f'{work_dir}/val_track_eval')
test_evaluator = val_evaluator
