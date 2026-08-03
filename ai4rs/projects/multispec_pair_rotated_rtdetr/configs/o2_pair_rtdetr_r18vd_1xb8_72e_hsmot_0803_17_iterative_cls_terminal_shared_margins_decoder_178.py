"""0803_17: terminal-only semantic-margin consensus.

Each frame preserves the mean of its final classification residual exactly.
Only the centered class-margin direction is averaged across the aligned query
pair.  Earlier iterative stages and DN logits remain independent.  The
projection is parameter-free, class-permutation-equivariant, and introduces
no class identity, weighting, attention, or additional decoder layer.
"""
from mmengine.config import read_base

with read_base():
    from .o2_pair_rtdetr_r18vd_1xb8_72e_hsmot_0801_09_iterative_cls_dn_isolated_e2e_decoder_178 import *  # noqa: F401,F403


model['bbox_head'].update(
    iterative_cls_residual=True,
    iterative_cls_dn_absolute=True,
    iterative_cls_detach_between_layers=False,
    iterative_cls_pair_shared_objectness=False,
    iterative_cls_terminal_shared_margins=True)

work_dir = (
    '/data4/litianhao/PairMmot/workdir_178/'
    '0803_17_paper_base_liquid_encoder_p5temporal_dualevidence_'
    'decoder_iterativeclsdnisolatede2e_terminalsharedsemanticmargins_'
    'pairdn_paircoherent_le180_r18_coco_full_1200x900_bf16_1xb8_fresh')
val_evaluator['metrics'].update(
    track_eval_out_dir=f'{work_dir}/val_track_eval')
test_evaluator = val_evaluator
