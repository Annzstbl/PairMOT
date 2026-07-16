"""HSMOT pair dataset and batch adapter for standalone CTracker training."""

import math

import torch
import torch.nn.functional as F

from mmrotate.datasets.hsmot import HSMOT_MEAN, HSMOT_STD
from mmrotate.datasets.hsmot_pair import HSMOTPairDataset
import mmrotate.datasets.transforms  # noqa: F401 - register pair transforms


HSMOT_CLASSES = (
    'car', 'bike', 'pedestrian', 'van', 'truck', 'bus', 'tricycle',
    'awning-bike')


def build_hsmot_pair_dataset(data_root, ann_file='', ann_subdir='mot',
                             img_subdir='npy2jpg', img_format='3jpg',
                             training=True, image_scale=(900, 1200),
                             augment=True, same_frame=False):
    # The CLI uses the conventional (height, width) order, while MMDet
    # Resize expects (width, height).
    resize_scale = (image_scale[1], image_scale[0])
    pipeline = [
        dict(type='mmrotate.LoadHSMOTPairImages', to_float32=False),
        dict(type='mmrotate.HSMOTPairLoadAnnotations', box_type='qbox'),
        dict(type='mmrotate.ConvertPairBoxType', dst_box_type='rbox'),
        dict(type='mmrotate.PairSharedResize', scale=resize_scale, keep_ratio=True,
             clip_object_border=False),
    ]
    if training and augment:
        pipeline.extend([
            dict(type='mmrotate.PairSharedRandomFlip', prob=0.5,
                 direction=['horizontal', 'vertical']),
            dict(type='mmrotate.PairSharedRandomRotate', prob=0.5,
                 angle_range=180),
        ])
    pipeline.append(dict(type='mmrotate.PackHSMOTPairInputs'))
    kwargs = dict(
        data_root=data_root,
        ann_subdir=ann_subdir,
        data_prefix=dict(img_path=img_subdir),
        img_format=img_format,
        same_frame=same_frame,
        require_prev_image=True,
        filter_cfg=dict(filter_empty_gt=False),
        serialize_data=False,
        test_mode=not training,
        pipeline=pipeline,
    )
    if ann_file:
        kwargs['ann_file'] = ann_file
    if training:
        kwargs.update(frame_intervals=(1,), sample_seed=3407)
    else:
        kwargs.update(frame_intervals=(1,))
    return HSMOTPairDataset(**kwargs)


def _pad_image(image, height, width):
    return F.pad(image, (0, width - image.size(-1), 0,
                         height - image.size(-2)))


def ctracker_collate(samples, pad_divisor=32):
    if not samples:
        raise ValueError('Cannot collate an empty HSMOT batch')
    max_height = max(sample['inputs'].size(-2) for sample in samples)
    max_width = max(sample['inputs'].size(-1) for sample in samples)
    max_height = int(math.ceil(max_height / pad_divisor) * pad_divisor)
    max_width = int(math.ceil(max_width / pad_divisor) * pad_divisor)
    mean = torch.tensor(HSMOT_MEAN).view(1, 8, 1, 1)
    std = torch.tensor(HSMOT_STD).view(1, 8, 1, 1)

    pairs = []
    targets = []
    metas = []
    for sample in samples:
        pair = sample['inputs'].float() / 255.0
        pair = (pair - mean) / std
        pairs.append(_pad_image(pair, max_height, max_width))

        data_sample = sample['data_samples']
        gt = data_sample.pair_gt_instances
        targets.append(dict(
            bboxes_prev=gt.bboxes_prev.tensor.float(),
            bboxes_curr=gt.bboxes_curr.tensor.float(),
            labels=gt.labels.long(),
            track_ids=gt.track_ids.long(),
            valid_prev=gt.valid_prev.bool(),
            valid_curr=gt.valid_curr.bool(),
        ))
        metas.append(dict(data_sample.metainfo))

    pairs = torch.stack(pairs, dim=0)
    max_objects = max(len(target['labels']) for target in targets)
    batch_size = len(targets)
    padded_targets = dict(
        bboxes_prev=torch.zeros(batch_size, max_objects, 5),
        bboxes_curr=torch.zeros(batch_size, max_objects, 5),
        labels=torch.zeros(batch_size, max_objects, dtype=torch.long),
        track_ids=torch.full(
            (batch_size, max_objects), -1, dtype=torch.long),
        valid_prev=torch.zeros(batch_size, max_objects, dtype=torch.bool),
        valid_curr=torch.zeros(batch_size, max_objects, dtype=torch.bool),
    )
    for batch_index, target in enumerate(targets):
        count = len(target['labels'])
        for key in padded_targets:
            padded_targets[key][batch_index, :count] = target[key]
    return dict(
        img_prev=pairs[:, 0],
        img_curr=pairs[:, 1],
        targets=padded_targets,
        img_metas=metas,
    )
