"""Single-frame O2-RTDETR eval config for HSMOT + rotated BoT-SORT."""
from pathlib import Path

from mmengine.config import read_base

with read_base():
    from projects.multispec_rotated_rtdetr.configs.o2_rtdetr_r18vd_2xb4_72e_hsmot_coco_pretrain_3dse_reduction2 import *  # noqa: F401,F403

_pairmot_root = Path(__file__).resolve().parents[4]
_data_root = f'{_pairmot_root}/data/hsmot/test'

img_format = '3jpg'
img_loader = dict(type='LoadMultichannelImageFrom3JPG', backend_args=None)

test_pipeline = [
    img_loader,
    dict(type='mmdet.Resize', scale=(800, 1200), keep_ratio=True),
    dict(
        type='mmdet.PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor', 'seq_name', 'frame_id'))
]

val_pipeline = [
    img_loader,
    dict(type='mmdet.Resize', scale=(800, 1200), keep_ratio=True),
    dict(
        type='HSMOTLoadAnnotations',
        with_bbox=True,
        with_track_id=True,
        box_type='qbox'),
    dict(type='ConvertBoxType', box_type_mapping=dict(gt_bboxes='rbox')),
    dict(
        type='mmdet.PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor', 'seq_name', 'frame_id'))
]

val_dataloader['dataset'].update(
    data_root=_data_root,
    ann_subdir='mot',
    ann_file='ImageSets/test.txt',
    data_prefix=dict(img_path='npy2jpg'),
    img_format=img_format,
    with_track_id=True,
    test_mode=True,
    pipeline=val_pipeline,
)
test_dataloader = val_dataloader

model.test_cfg = dict(max_per_img=300, rescale=False)
work_dir = '/data4/litianhao/PairMmot/workdir_99/single_botsort_eval'
