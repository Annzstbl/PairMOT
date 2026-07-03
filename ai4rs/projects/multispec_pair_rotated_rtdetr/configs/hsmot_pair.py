"""HSMOT pair-frame training and evaluation data configuration."""

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

train_pipeline = [
    dict(type='LoadHSMOTPairImages', to_float32=False, backend_args=backend_args),
    dict(type='HSMOTPairLoadAnnotations', box_type='qbox'),
    dict(type='ConvertPairBoxType', dst_box_type='rbox'),
    dict(type='PairSharedResize', scale=(800, 1200), keep_ratio=True,
         clip_object_border=False),
    dict(type='PairSharedRandomFlip', prob=0.5,
         direction=['horizontal', 'vertical']),
    dict(type='PairSharedRandomRotate', prob=0.5, angle_range=180),
    dict(type='PackHSMOTPairInputs'),
]

val_pipeline = [
    dict(type='LoadHSMOTPairImages', to_float32=False, backend_args=backend_args),
    dict(type='HSMOTPairLoadAnnotations', box_type='qbox'),
    dict(type='ConvertPairBoxType', dst_box_type='rbox'),
    dict(type='PairSharedResize', scale=(800, 1200), keep_ratio=True,
         clip_object_border=False),
    dict(type='PackHSMOTPairInputs'),
]

train_dataloader = dict(
    batch_size=4,
    num_workers=2,
    # Required: epoch-specific pairs are rebuilt before every iterator.
    persistent_workers=False,
    sampler=dict(type='DefaultSampler', shuffle=True),
    batch_sampler=None,
    dataset=dict(
        type=dataset_type,
        data_root='../data/hsmot/train',
        ann_subdir='mot',
        # BaseDataset resolves relative annotation files under data_root.
        ann_file='../train_half.txt',
        data_prefix=dict(img_path='npy2jpg'),
        img_format='3jpg',
        random_interval_range=(1, 1),
        sample_seed=3407,
        filter_cfg=dict(filter_empty_gt=False),
        serialize_data=False,
        pipeline=train_pipeline,
        backend_args=backend_args,
    ),
)

val_dataloader = dict(
    batch_size=4,
    num_workers=8,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root='../data/hsmot/test',
        ann_subdir='mot',
        data_prefix=dict(img_path='npy2jpg'),
        img_format='3jpg',
        # Formal acceptance is sequential adjacent-frame performance only.
        frame_intervals=(1,),
        filter_cfg=dict(filter_empty_gt=False),
        test_mode=True,
        pipeline=val_pipeline,
        backend_args=backend_args,
    ),
)

test_dataloader = val_dataloader
val_evaluator = dict(
    type='Evaluator',
    metrics=dict(
        type='HSMOTPairAPMetric',
        score_thr=0.35,
        iou_thr=0.5,
        pres_thr=0.5,
        report_gaps=(1,),
    ),
)
test_evaluator = val_evaluator
