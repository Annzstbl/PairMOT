default_scope = 'mmrotate'

runner_type = 'MultispecRunner'

default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=50),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(type='CheckpointHook', interval=6, max_keep_ckpts=99999),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='HSMOTVisualizationHook'))

env_cfg = dict(
    cudnn_benchmark=False,
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    dist_cfg=dict(backend='nccl'),
)

vis_backends = [dict(type='LocalVisBackend')]
visualizer = dict(
    type='RotLocalVisualizer', vis_backends=vis_backends, name='visualizer')
log_processor = dict(type='LogProcessor', window_size=50, by_epoch=True)

# File + console logging (see MultispecMMLogger / MultispecRunner).
# Omit console_log_level / file_log_level to mirror stock MMLogger behaviour.
log_level = 'INFO'
# file_log_level = None
# console_log_level = None
# console_suppress_patterns = []

load_from = None
resume = False

custom_hooks = [
    dict(type='mmdet.NumClassCheckHook'),
    dict(type='TrainingCurveHook'),
]
