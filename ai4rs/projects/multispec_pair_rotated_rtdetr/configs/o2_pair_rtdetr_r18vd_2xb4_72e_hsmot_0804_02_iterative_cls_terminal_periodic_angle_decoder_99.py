"""0804_02 on 99: terminal-only periodic-angle quotient consensus.

Only the final normal-query angle increment is replaced by its pi-periodic
pair midpoint. Center and size outputs, recursive references, classification,
DN, and auxiliary outputs remain frame-specific. This isolates whether the
mature benefit of terminal shape consensus comes from orientation while
avoiding scale over-constraint. It is parameter-free, swap-equivariant, class
agnostic, and adds no layer, attention, loss, or reweighting.
"""
from mmengine.config import read_base

with read_base():
    from .o2_pair_rtdetr_r18vd_2xb4_72e_hsmot_0803_14_iterative_cls_terminal_log_area_periodic_angle_decoder_99 import *  # noqa: F401,F403


model['decoder'].update(
    frame_evidence_cls_decoder=False,
    frame_detail_cls_decoder=False,
    pair_shared_shape_refinement_decoder=False,
    pair_shared_angle_refinement_decoder=False,
    pair_shared_periodic_angle_refinement_decoder=False,
    pair_shared_log_size_periodic_angle_refinement_decoder=False,
    pair_shared_log_area_periodic_angle_refinement_decoder=False,
    pair_shared_late_log_size_periodic_angle_refinement_decoder=False,
    pair_shared_terminal_log_size_periodic_angle_refinement_decoder=False,
    pair_shared_terminal_log_area_periodic_angle_refinement_decoder=False,
    pair_shared_terminal_periodic_angle_refinement_decoder=True,
    pair_shared_progressive_log_shape_periodic_angle_refinement_decoder=False,
    pair_shared_normalized_center_refinement_decoder=False)

work_dir = (
    '/data4/litianhao/PairMmot/workdir_99/'
    '0804_02_paper_base_liquid_encoder_p5temporal_dualevidence_'
    'decoder_iterativeclsdnisolatede2e_pairsharedterminalperiodicangle_'
    'refinement_pairdn_paircoherent_le180_r18_coco_full_1200x900_'
    'bf16_2xb4_fresh')
val_evaluator['metrics'].update(track_eval_out_dir=f'{work_dir}/val_track_eval')
test_evaluator = val_evaluator
