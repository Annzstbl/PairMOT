"""99 port of 0803_14 on any free two-GPU pair."""
from mmengine.config import read_base

with read_base():
    from .o2_pair_rtdetr_r18vd_2xb4_72e_hsmot_0803_14_iterative_cls_terminal_log_area_periodic_angle_decoder_252 import *  # noqa: F401,F403


train_dataloader['dataset'].update(
    data_root='/data/users/wangying01/lth/PairMOT/data/hsmot/train',
    gmc_cache_dir=(
        '/data/users/wangying01/lth/PairMOT/workdir/aux/gmc_cache/'
        'hsmot_train_gap1'))
val_dataloader['dataset'].update(
    data_root='/data/users/wangying01/lth/PairMOT/data/hsmot/test',
    gmc_cache_dir=(
        '/data/users/wangying01/lth/PairMOT/workdir/aux/gmc_cache/'
        'hsmot_test_gap1'))
test_dataloader['dataset'].update(
    data_root='/data/users/wangying01/lth/PairMOT/data/hsmot/test',
    gmc_cache_dir=(
        '/data/users/wangying01/lth/PairMOT/workdir/aux/gmc_cache/'
        'hsmot_test_gap1'))

work_dir = (
    '/data4/litianhao/PairMmot/workdir_99/'
    '0803_14_paper_base_liquid_encoder_p5temporal_dualevidence_'
    'decoder_iterativeclsdnisolatede2e_pairsharedterminallogarea_'
    'periodicanglerefinement_pairdn_paircoherent_le180_'
    'r18_coco_full_1200x900_bf16_2xb4_fresh')
val_evaluator['metrics'].update(
    track_eval_out_dir=f'{work_dir}/val_track_eval',
    trackeval_root='/data/users/wangying01/lth/PairMOT/TrackEval',
    track_data_root='/data/users/wangying01/lth/PairMOT/data/hsmot/test')
test_evaluator = val_evaluator
