# Real HSMOT subset for integration test (3jpg, splits under PairMmot/tmp).
dataset_type = 'HSMOTDataset'
backend_args = None

_train_ann_file = '../tmp/hsmot_splits/train_mini.txt'
_test_ann_file = '../tmp/hsmot_splits/test_mini.txt'

hsmot_mean = [m * 255 for m in [
    0.27358221, 0.28804452, 0.28133921, 0.26906377,
    0.28309119, 0.26928305, 0.28372527, 0.27149373,
]]
hsmot_std = [s * 255 for s in [
    0.19756629, 0.17432339, 0.16413284, 0.17581682,
    0.18366176, 0.1536845, 0.15964683, 0.16557951,
]]

img_format = '3jpg'
img_loader = dict(
    type='LoadMultichannelImageFrom3JPG', backend_args=backend_args)

train_pipeline = [
    img_loader,
    dict(
        type='HSMOTLoadAnnotations',
        with_bbox=True,
        with_track_id=False,
        box_type='qbox'),
    dict(type='ConvertBoxType', box_type_mapping=dict(gt_bboxes='rbox')),
    dict(type='mmdet.Resize', scale=(640, 960), keep_ratio=True),
    dict(
        type='mmdet.RandomFlip',
        prob=0.5,
        direction=['horizontal', 'vertical']),
    dict(type='mmdet.PackDetInputs')
]

val_pipeline = [
    img_loader,
    dict(type='mmdet.Resize', scale=(640, 960), keep_ratio=True),
    dict(
        type='HSMOTLoadAnnotations',
        with_bbox=True,
        with_track_id=False,
        box_type='qbox'),
    dict(type='ConvertBoxType', box_type_mapping=dict(gt_bboxes='rbox')),
    dict(
        type='mmdet.PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor', 'seq_name', 'frame_id'))
]

test_pipeline = [
    img_loader,
    dict(type='mmdet.Resize', scale=(640, 960), keep_ratio=True),
    dict(
        type='mmdet.PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor', 'seq_name', 'frame_id'))
]

train_dataloader = dict(
    batch_size=1,
    num_workers=0,
    persistent_workers=False,
    sampler=dict(type='DefaultSampler', shuffle=True),
    batch_sampler=None,
    dataset=dict(
        type=dataset_type,
        data_root='../data/hsmot/train',
        ann_subdir='mot',
        ann_file=_train_ann_file,
        data_prefix=dict(img_path='npy2jpg'),
        img_format=img_format,
        with_track_id=False,
        filter_cfg=dict(filter_empty_gt=True),
        pipeline=train_pipeline,
        backend_args=backend_args))

val_dataloader = dict(
    batch_size=1,
    num_workers=0,
    persistent_workers=False,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root='../data/hsmot/test',
        ann_subdir='mot',
        ann_file=_test_ann_file,
        data_prefix=dict(img_path='npy2jpg'),
        img_format=img_format,
        with_track_id=False,
        test_mode=True,
        pipeline=val_pipeline,
        backend_args=backend_args))

test_dataloader = val_dataloader

val_evaluator = dict(type='HSMOTDetMetric')
test_evaluator = val_evaluator
