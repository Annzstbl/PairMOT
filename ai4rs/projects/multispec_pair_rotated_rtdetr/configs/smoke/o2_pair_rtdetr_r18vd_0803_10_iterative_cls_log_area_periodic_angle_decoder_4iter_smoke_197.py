"""Four-iteration real-data DDP smoke for 0803_10 on 197."""
from mmengine.config import read_base

with read_base():
    from ..o2_pair_rtdetr_r18vd_2xb4_72e_hsmot_0801_09_iterative_cls_dn_isolated_e2e_decoder_resume197 import *  # noqa: F401,F403


model['decoder'].update(
    frame_evidence_cls_decoder=False,
    frame_detail_cls_decoder=False,
    pair_shared_shape_refinement_decoder=False,
    pair_shared_angle_refinement_decoder=False,
    pair_shared_periodic_angle_refinement_decoder=False,
    pair_shared_log_size_periodic_angle_refinement_decoder=False,
    pair_shared_log_area_periodic_angle_refinement_decoder=True,
    pair_shared_normalized_center_refinement_decoder=False)
model['bbox_head'].update(
    iterative_cls_pair_shared_objectness=False,
    iterative_cls_residual=True,
    iterative_cls_dn_absolute=True,
    iterative_cls_detach_between_layers=False)

train_dataloader['batch_size'] = 4
train_cfg = dict(type='IterBasedTrainLoop', max_iters=4, val_interval=1000)
param_scheduler = []
val_cfg = None
test_cfg = None
val_dataloader = None
test_dataloader = None
val_evaluator = None
test_evaluator = None
default_hooks['checkpoint'].update(
    by_epoch=False, interval=4, save_last=True, max_keep_ckpts=1,
    filename_tmpl='iter_{}.pth')
default_hooks['logger'].update(interval=1)
custom_hooks = [
    hook for hook in custom_hooks
    if hook.get('type') != 'AsyncPairTrackEvalHook'
]
work_dir = (
    '/data4/litianhao/PairMmot/workdir_197/'
    'smoke_0803_10_iterative_cls_log_area_periodic_angle_4iter')
