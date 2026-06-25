# Copyright (c) AI4RS. All rights reserved.
import os.path as osp
from typing import Optional

import mmcv
import numpy as np
import torch
from mmcv.transforms import to_tensor
from mmcv.transforms.base import BaseTransform
from mmengine.fileio import get
from mmdet.structures.bbox import get_box_type

from mmrotate.datasets.pair_gt import build_pair_gt_from_instances
from mmrotate.registry import TRANSFORMS


def _load_npy_image(filename: str, to_float32: bool) -> np.ndarray:
    img = np.load(filename)
    if to_float32:
        img = img.astype(np.float32)
    return img


def _load_3jpg_image(filename: str, to_float32: bool,
                     backend_args: Optional[dict]) -> np.ndarray:
    stem, ext = osp.splitext(filename)
    if stem.endswith(('_p1', '_p2', '_p3')):
        base_stem = stem.rsplit('_', 1)[0]
    else:
        base_stem = stem

    part_paths = [
        f'{base_stem}_p1{ext}',
        f'{base_stem}_p2{ext}',
        f'{base_stem}_p3{ext}',
    ]
    part_imgs = []
    for part_path in part_paths:
        img_bytes = get(part_path, backend_args=backend_args)
        img = mmcv.imfrombytes(img_bytes, channel_order='rgb')
        if img is None:
            raise FileNotFoundError(
                f'Failed to load HSMOT 3-JPG part: {part_path}')
        part_imgs.append(img)

    img = np.concatenate(
        [part_imgs[0], part_imgs[1], part_imgs[2][:, :, :2]], axis=2)
    if to_float32:
        img = img.astype(np.float32)
    return img


def _load_multichannel_image(filename: str,
                             to_float32: bool = False,
                             backend_args: Optional[dict] = None) -> np.ndarray:
    if filename.endswith('.npy'):
        return _load_npy_image(filename, to_float32)
    return _load_3jpg_image(filename, to_float32, backend_args)


@TRANSFORMS.register_module()
class LoadHSMOTPairImages(BaseTransform):
    """Load previous and current 8-channel HSMOT images into ``results['img']``.

    Required Keys:

    - img_path_prev
    - img_path

    Modified Keys:

    - img (list[np.ndarray]): ``[img_prev, img_curr]``, each (H, W, 8)
    - img_shape, ori_shape
  """

    def __init__(self,
                 to_float32: bool = False,
                 backend_args: Optional[dict] = None) -> None:
        self.to_float32 = to_float32
        self.backend_args = backend_args

    def transform(self, results: dict) -> dict:
        img_prev = _load_multichannel_image(
            results['img_path_prev'], self.to_float32, self.backend_args)
        img_curr = _load_multichannel_image(
            results['img_path'], self.to_float32, self.backend_args)
        if img_prev.shape != img_curr.shape:
            raise ValueError(
                f'Pair image shape mismatch: prev {img_prev.shape} vs '
                f'curr {img_curr.shape} '
                f'({results["img_path_prev"]} vs {results["img_path"]}).')

        results['img'] = [img_prev, img_curr]
        results['img_shape'] = img_curr.shape[:2]
        results['ori_shape'] = img_curr.shape[:2]
        return results


@TRANSFORMS.register_module()
class HSMOTPairLoadAnnotations(BaseTransform):
    """Build pair GT and per-frame qboxes for geometric transforms.

    Populates aligned pair fields and temporary ``gt_bboxes_prev`` /
    ``gt_bboxes_curr`` (QuadriBoxes) for shared augmentation.

    Required Keys:

    - instances_prev, instances_curr
    - video_id, frame_id, frame_id_prev (optional, for error messages)

    Modified Keys:

    - pair_labels, pair_track_ids, pair_valid_prev, pair_valid_curr
    - gt_bboxes_prev, gt_bboxes_curr (QuadriBoxes)
    """

    def __init__(self, box_type: str = 'qbox') -> None:
        self.box_type, box_type_cls = get_box_type(box_type)

    def transform(self, results: dict) -> dict:
        pair_gt = build_pair_gt_from_instances(
            results.get('instances_prev', []),
            results.get('instances_curr', []),
            video_id=results.get('video_id', results.get('seq_name', '')),
            frame_id_prev=int(results.get('frame_id_prev', -1)),
            frame_id_curr=int(results.get('frame_id', -1)),
        )

        results['pair_labels'] = pair_gt['labels']
        results['pair_track_ids'] = pair_gt['track_ids']
        results['pair_valid_prev'] = pair_gt['valid_prev']
        results['pair_valid_curr'] = pair_gt['valid_curr']

        num = len(pair_gt['track_ids'])
        if num > 0:
            box_type_cls = get_box_type(self.box_type)[1]
            results['gt_bboxes_prev'] = box_type_cls(
                pair_gt['bboxes_prev'], dtype=torch.float32)
            results['gt_bboxes_curr'] = box_type_cls(
                pair_gt['bboxes_curr'], dtype=torch.float32)
        else:
            box_type_cls = get_box_type(self.box_type)[1]
            results['gt_bboxes_prev'] = box_type_cls(
                np.zeros((0, 8), dtype=np.float32), dtype=torch.float32)
            results['gt_bboxes_curr'] = box_type_cls(
                np.zeros((0, 8), dtype=np.float32), dtype=torch.float32)

        return results


@TRANSFORMS.register_module()
class ConvertPairBoxType(BaseTransform):
    """Convert ``gt_bboxes_prev`` / ``gt_bboxes_curr`` to a target box type."""

    def __init__(self, dst_box_type: str = 'rbox') -> None:
        from mmrotate.datasets.transforms.transforms import ConvertBoxType
        self.inner = ConvertBoxType(
            box_type_mapping={'gt_bboxes': dst_box_type})

    def transform(self, results: dict) -> dict:
        for key in ('gt_bboxes_prev', 'gt_bboxes_curr'):
            if key not in results:
                continue
            sub = {'gt_bboxes': results[key]}
            self.inner.transform(sub)
            results[key] = sub['gt_bboxes']
        return results


@TRANSFORMS.register_module()
class PackHSMOTPairInputs(BaseTransform):
    """Pack HSMOT pair inputs and ``pair_gt_instances`` into training tensors.

    Output ``inputs`` has shape ``(2, C, H, W)`` which collates to
    ``(B, 2, C, H, W)``.
    """

    def __init__(
        self,
        meta_keys=('img_id', 'img_path', 'img_path_prev', 'ori_shape',
                   'img_shape', 'scale_factor', 'flip', 'flip_direction',
                   'video_id', 'seq_name', 'frame_id', 'frame_id_prev')):
        self.meta_keys = meta_keys

    def transform(self, results: dict) -> dict:
        imgs = results['img']
        if not isinstance(imgs, list) or len(imgs) != 2:
            raise ValueError(
                f'PackHSMOTPairInputs expects results["img"] to be a list of '
                f'two images, got {type(imgs)} with len '
                f'{len(imgs) if isinstance(imgs, list) else "n/a"}.')

        packed_imgs = []
        for img in imgs:
            if img.ndim < 3:
                img = np.expand_dims(img, -1)
            if not img.flags.c_contiguous:
                chw = np.ascontiguousarray(img.transpose(2, 0, 1))
                packed_imgs.append(to_tensor(chw))
            else:
                packed_imgs.append(
                    to_tensor(img).permute(2, 0, 1).contiguous())
        inputs = torch.stack(packed_imgs, dim=0)

        from mmdet.structures import DetDataSample
        from mmengine.structures import InstanceData

        pair_instance = InstanceData()
        pair_instance.labels = to_tensor(results['pair_labels'])
        pair_instance.track_ids = to_tensor(results['pair_track_ids'])
        pair_instance.bboxes_prev = results['gt_bboxes_prev']
        pair_instance.bboxes_curr = results['gt_bboxes_curr']
        pair_instance.valid_prev = to_tensor(results['pair_valid_prev'])
        pair_instance.valid_curr = to_tensor(results['pair_valid_curr'])

        data_sample = DetDataSample()
        data_sample.pair_gt_instances = pair_instance

        img_meta = {}
        for key in self.meta_keys:
            if key in results:
                img_meta[key] = results[key]
        data_sample.set_metainfo(img_meta)

        return {'inputs': inputs, 'data_samples': data_sample}
