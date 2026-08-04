"""0804_08 on 99: shared-metric terminal product tangent.

Both frame center updates, the reference chord, and center reconstruction use
one geometric-mean width/height metric. This removes the coordinate-connection
mismatch of comparing per-frame-normalized updates inside a pair-normalized
projector. Shape transport and all other model/training choices are unchanged.
The operation is parameter-free, swap-equivariant, class agnostic, and has no
reweighting.
"""
from mmengine.config import read_base

with read_base():
    from .o2_pair_rtdetr_r18vd_2xb4_72e_hsmot_0804_05_iterative_cls_terminal_transport_se2_product_tangent_decoder_99 import *  # noqa: F401,F403


model['decoder'].update(
    pair_shared_terminal_transport_product_tangent_refinement_decoder=False,
    pair_shared_terminal_transport_shared_metric_product_tangent_refinement_decoder=True,
    pair_shared_terminal_transport_body_frame_product_tangent_refinement_decoder=False,
    pair_shared_terminal_transport_se2_product_tangent_refinement_decoder=False,
    pair_shared_terminal_transport_frenet_product_tangent_refinement_decoder=False,
    pair_shared_terminal_transport_axis_frenet_product_tangent_refinement_decoder=False)

work_dir = (
    '/data4/litianhao/PairMmot/workdir_99/'
    '0804_08_paper_base_liquid_encoder_p5temporal_dualevidence_'
    'decoder_iterativeclsdnisolatede2e_pairsharedterminaltransportsharedmetric_'
    'producttangentrefinement_pairdn_paircoherent_le180_r18_coco_full_'
    '1200x900_bf16_2xb4_fresh')
val_evaluator['metrics'].update(track_eval_out_dir=f'{work_dir}/val_track_eval')
test_evaluator = val_evaluator
