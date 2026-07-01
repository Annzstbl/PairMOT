# HSMOT pair overfit: prev/curr use the same frame + InfiniteSampler.
dataset_type = 'HSMOTPairDataset'
backend_args = None

hsmot_mean = [m * 255 for m in [
    0.27358221, 0.28804452, 0.28133921, 0.26906377,
    0.28309119, 0.26928305, 0.28372527, 0.27149373,
]]
hsmot_std = [s * 255 for s in [
    0.19756629, 0.17432339, 0.16413284, 0.17581682,
    0.18366176, 0.1536845, 0.15964683, 0.16557951,
]]

img_format = 'npy'

train_pipeline = [
    dict(type='LoadHSMOTPairImages', to_float32=False, backend_args=backend_args),
    dict(type='HSMOTPairLoadAnnotations', box_type='qbox'),
    dict(type='ConvertPairBoxType', dst_box_type='rbox'),
    dict(type='PairSharedResize', scale=(800, 1200), keep_ratio=True,
         clip_object_border=False),
    dict(type='PackHSMOTPairInputs'),
]

test_pipeline = train_pipeline

train_dataloader = dict(
    batch_size=4,
    num_workers=2,
    persistent_workers=False,
    sampler=dict(type='InfiniteSampler', shuffle=True),
    batch_sampler=None,
    dataset=dict(
        type=dataset_type,
        data_root='data/HSMOT_pair_overfit/train',
        ann_subdir='mot',
        ann_file='ImageSets/train.txt',
        data_prefix=dict(img_path='npy'),
        img_format=img_format,
        same_frame=True,
        filter_cfg=dict(filter_empty_gt=False),
        pipeline=train_pipeline,
        backend_args=backend_args))

val_dataloader = dict(
    batch_size=1,
    num_workers=0,
    persistent_workers=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    batch_sampler=None,
    dataset=dict(
        type=dataset_type,
        data_root='data/HSMOT_pair_overfit/train',
        ann_subdir='mot',
        ann_file='ImageSets/train.txt',
        data_prefix=dict(img_path='npy'),
        img_format=img_format,
        same_frame=True,
        filter_cfg=dict(filter_empty_gt=False),
        pipeline=test_pipeline,
        backend_args=backend_args))

val_evaluator = dict(
    type='Evaluator',
    metrics=dict(
        type='HSMOTPairOverfitMetric',
        score_thr=0.35,
        iou_thr=0.5,
        pres_thr=0.5))

test_dataloader = val_dataloader
test_evaluator = val_evaluator
