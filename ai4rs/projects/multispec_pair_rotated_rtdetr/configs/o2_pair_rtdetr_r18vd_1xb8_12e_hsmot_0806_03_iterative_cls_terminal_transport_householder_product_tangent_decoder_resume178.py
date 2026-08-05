"""0806_03 on 178: mature Householder transport from epoch 8 to 12.

This is the exact 0804_09 zero-state Householder product-tangent decoder moved
from the CPU-throttled 197 host to one 178 GPU.  Physical batch changes from
2x4 to 1x8, preserving global batch eight without accumulation.  Only the
runtime horizon, data paths, and isolated work directory differ.
"""
from mmengine.config import read_base

with read_base():
    from .o2_pair_rtdetr_r18vd_1xb8_72e_hsmot_0804_01_iterative_cls_terminal_transport_product_tangent_decoder_178 import *  # noqa: F401,F403


model['decoder'].update(
    pair_shared_terminal_log_size_refinement_decoder=False,
    pair_shared_terminal_transport_center_tangent_refinement_decoder=False,
    pair_shared_terminal_transport_shape_tangent_refinement_decoder=False,
    pair_shared_terminal_transport_product_tangent_refinement_decoder=False,
    pair_shared_terminal_transport_shared_metric_product_tangent_refinement_decoder=False,
    pair_shared_terminal_transport_householder_product_tangent_refinement_decoder=True,
    pair_shared_terminal_transport_body_frame_product_tangent_refinement_decoder=False,
    pair_shared_terminal_transport_se2_product_tangent_refinement_decoder=False,
    pair_shared_terminal_transport_frenet_product_tangent_refinement_decoder=False,
    pair_shared_terminal_transport_axis_frenet_product_tangent_refinement_decoder=False,
    pair_shared_terminal_transport_tangent_refinement_decoder=False,
    pair_shared_terminal_transport_plane_refinement_decoder=False)

train_cfg['max_epochs'] = 12
work_dir = (
    '/data4/litianhao/PairMmot/workdir_178/'
    '0806_03_paper_base_liquid_encoder_p5temporal_dualevidence_'
    'decoder_iterativeclsdnisolatede2e_pairsharedterminaltransporthouseholder_'
    'producttangentrefinement_pairdn_paircoherent_le180_r18_coco_full_'
    '1200x900_bf16_1xb8_resume_from_e8_to_e12')
val_evaluator['metrics'].update(track_eval_out_dir=f'{work_dir}/val_track_eval')
test_evaluator = val_evaluator
