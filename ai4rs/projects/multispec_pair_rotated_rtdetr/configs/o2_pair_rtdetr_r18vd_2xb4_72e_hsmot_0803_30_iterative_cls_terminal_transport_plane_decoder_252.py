"""0803_30 on 252: geometry-only terminal osculating-plane transport.

This is the two-GPU, global-batch-eight realization of the 178 candidate.
Relative to full transported tangent geometry, only the final normal-query
box-detail projection changes from a one-dimensional motion tangent to the
at-most two-dimensional plane spanned by established motion and the detached
pair-common terminal correction. Classification, DN, auxiliary outputs, and
recursive references remain unchanged. The operation is parameter-free,
class agnostic, and does not reweight predictions.
"""
from mmengine.config import read_base

with read_base():
    from .o2_pair_rtdetr_r18vd_2xb4_72e_hsmot_0803_24_iterative_cls_terminal_transport_shape_tangent_decoder_197 import *  # noqa: F401,F403


model['decoder'].update(
    pair_shared_terminal_transport_shape_tangent_refinement_decoder=False,
    pair_shared_terminal_transport_plane_refinement_decoder=True)

# The scientific parent is the validated 197 two-GPU realization.  Only its
# physical dataset/cache placement must be rebound for the 252 account.
_hsmot_root = '/data/users/litianhao01/PairMmot/data/hsmot'
_gmc_root = '/data/users/litianhao01/PairMmot/workdir/aux/gmc_cache'
train_dataloader['dataset'].update(
    data_root=f'{_hsmot_root}/train',
    gmc_cache_dir=f'{_gmc_root}/hsmot_train_gap1')
val_dataloader['dataset'].update(
    data_root=f'{_hsmot_root}/test',
    gmc_cache_dir=f'{_gmc_root}/hsmot_test_gap1')
test_dataloader = val_dataloader

work_dir = (
    '/data4/litianhao/PairMmot/workdir_252/'
    '0803_30_paper_base_liquid_encoder_p5temporal_dualevidence_'
    'decoder_iterativeclsdnisolatede2e_pairsharedterminaltransportplane_'
    'refinement_pairdn_paircoherent_le180_r18_coco_full_1200x900_'
    'bf16_2xb4_fresh')
val_evaluator['metrics'].update(
    track_eval_out_dir=f'{work_dir}/val_track_eval',
    trackeval_root='/data/users/litianhao01/PairMmot/TrackEval',
    track_data_root=f'{_hsmot_root}/test')
test_evaluator = val_evaluator
