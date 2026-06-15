# Copyright (c) AI4RS. All rights reserved.
import os.path as osp
from typing import Optional

import mmcv
import numpy as np
import torch
from mmcv.transforms import BaseTransform
from mmengine.fileio import get
from mmdet.datasets.transforms import LoadAnnotations as MMDetLoadAnnotations

from mmrotate.registry import TRANSFORMS


@TRANSFORMS.register_module()
class LoadMultichannelImageFromNpy(BaseTransform):
    """Load 8-channel image from ``.npy`` or a regular image file.

    Required Keys:

    - img_path

    Modified Keys:

    - img
    - img_shape
    - ori_shape
    """

    def __init__(self,
                 to_float32: bool = False,
                 color_type: str = 'unchanged',
                 backend_args: Optional[dict] = None) -> None:
        self.to_float32 = to_float32
        self.color_type = color_type
        self.backend_args = backend_args

    def transform(self, results: dict) -> dict:
        filename = results['img_path']
        if filename.endswith('.npy'):
            img = np.load(filename)
        else:
            img_bytes = get(filename, backend_args=self.backend_args)
            img = mmcv.imfrombytes(
                img_bytes, flag=self.color_type, channel_order='rgb')

        if self.to_float32:
            img = img.astype(np.float32)

        results['img'] = img
        results['img_shape'] = img.shape[:2]
        results['ori_shape'] = img.shape[:2]
        return results


@TRANSFORMS.register_module()
class LoadMultichannelImageFrom3JPG(BaseTransform):
    """Load 8-channel image from three JPG files.

    Expected naming convention under the same directory:

    - ``{stem}_p1.jpg``: 3 channels
    - ``{stem}_p2.jpg``: 3 channels
    - ``{stem}_p3.jpg``: 2 channels (only first two bands are used)

    ``img_path`` may point to any one of the three files, or to the base
    ``{stem}_p1.jpg`` path produced by :class:`HSMOTDataset`.
    """

    def __init__(self,
                 to_float32: bool = False,
                 backend_args: Optional[dict] = None) -> None:
        self.to_float32 = to_float32
        self.backend_args = backend_args

    def transform(self, results: dict) -> dict:
        filename = results['img_path']
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
            img_bytes = get(part_path, backend_args=self.backend_args)
            img = mmcv.imfrombytes(img_bytes, channel_order='rgb')
            if img is None:
                raise FileNotFoundError(
                    f'Failed to load HSMOT 3-JPG part: {part_path}')
            part_imgs.append(img)

        img = np.concatenate(
            [part_imgs[0], part_imgs[1], part_imgs[2][:, :, :2]], axis=2)
        if self.to_float32:
            img = img.astype(np.float32)

        results['img'] = img
        results['img_shape'] = img.shape[:2]
        results['ori_shape'] = img.shape[:2]
        return results


@TRANSFORMS.register_module()
class HSMOTLoadAnnotations(MMDetLoadAnnotations):
    """Load HSMOT rotated-box annotations with optional track IDs.

    When ``with_track_id=True``, parsed track IDs are stored in
    ``results['gt_instances'].track_ids``.
    """

    def __init__(self, with_track_id: bool = False, **kwargs) -> None:
        self.with_track_id = with_track_id
        super().__init__(**kwargs)

    def transform(self, results: dict) -> dict:
        track_ids = None
        if self.with_track_id:
            track_ids = [
                inst.get('track_id', -1)
                for inst in results.get('instances', [])
            ]
        results = super().transform(results)
        if self.with_track_id and track_ids is not None:
            if 'gt_instances' in results:
                results['gt_instances'].track_ids = torch.tensor(
                    track_ids, dtype=torch.int64)
        return results
