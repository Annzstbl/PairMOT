"""0803_20: terminal full-tangent geometry plus semantic margins.

The final normal-query box update shares reference-local center displacement,
log-ratio size, and pi-periodic orientation. The final classification update
preserves each frame's class mean and shares only centered margins. Recurrent
references and DN remain independent. Both projections are parameter free.
"""
from mmengine.config import read_base

with read_base():
    from .o2_pair_rtdetr_r18vd_2xb4_72e_hsmot_0803_18_iterative_cls_terminal_log_size_angle_shared_margins_decoder_197 import *  # noqa: F401,F403


model['decoder'].update(
    pair_shared_terminal_log_size_periodic_angle_refinement_decoder=False,
    pair_shared_terminal_full_tangent_refinement_decoder=True)
model['bbox_head'].update(
    iterative_cls_residual=True,
    iterative_cls_dn_absolute=True,
    iterative_cls_detach_between_layers=False,
    iterative_cls_pair_shared_objectness=False,
    iterative_cls_terminal_shared_margins=True)

work_dir = (
    '/data4/litianhao/PairMmot/workdir_197/'
    '0803_20_paper_base_liquid_encoder_p5temporal_dualevidence_'
    'decoder_iterativeclsdnisolatede2e_terminalfulltangent_'
    'semanticmargins_pairdn_paircoherent_le180_r18_coco_full_'
    '1200x900_bf16_2xb4_fresh')
val_evaluator['metrics'].update(track_eval_out_dir=f'{work_dir}/val_track_eval')
test_evaluator = val_evaluator
