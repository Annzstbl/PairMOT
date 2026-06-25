# Copyright (c) AI4RS. All rights reserved.
import copy
import os.path as osp
from typing import Dict, List, Optional, Tuple, Union

import numpy as np

from mmengine.dataset import BaseDataset
from mmengine.fileio import list_from_file

from mmrotate.datasets.hsmot import (
    HSMOTDataset,
    load_hsmot_sequence_ann,
)
from mmrotate.registry import DATASETS


@DATASETS.register_module()
class HSMOTPairDataset(BaseDataset):
    """HSMOT image-pair dataset for temporal rotated-box learning.

    Each sample is a pair of frames from the same sequence separated by
    ``frame_interval``. Pair-level ground truth is built from the union of
    track ids on both frames (see ``build_pair_gt_from_instances``).

    Args:
        frame_interval (int): Frame gap between previous and current frame.
            ``frame_id_prev = frame_id_curr - frame_interval``. Defaults to 1.
        ann_subdir (str): MOT annotation subdirectory under ``data_root``.
        img_format (str): ``'npy'`` or ``'3jpg'``.
        require_prev_image (bool): If ``True``, skip pairs whose previous
            frame image file is missing. Defaults to ``True``.
        same_frame (bool): If ``True``, use the current frame for both prev
            and curr images (and duplicate GT). For overfit / sanity checks.
        backend_args (dict, optional): File I/O backend arguments.
    """

    METAINFO = HSMOTDataset.METAINFO

    def __init__(self,
                 frame_interval: int = 1,
                 ann_subdir: str = 'mot',
                 img_format: str = 'npy',
                 require_prev_image: bool = True,
                 same_frame: bool = False,
                 file_client_args: dict = None,
                 backend_args: dict = None,
                 **kwargs) -> None:
        assert frame_interval >= 1, (
            f'frame_interval must be >= 1, got {frame_interval}')
        assert img_format in ('npy', '3jpg'), (
            f"img_format must be 'npy' or '3jpg', got {img_format}")
        self.frame_interval = frame_interval
        self.ann_subdir = ann_subdir
        self.img_format = img_format
        self.require_prev_image = require_prev_image
        self.same_frame = same_frame
        self.backend_args = backend_args
        if file_client_args is not None:
            raise RuntimeError(
                'The `file_client_args` is deprecated, '
                'please use `backend_args` instead.')
        super().__init__(**kwargs)

    def _get_ann_dir(self) -> str:
        if self.ann_file and osp.isdir(self.ann_file):
            return self.ann_file
        return osp.join(self.data_root, self.ann_subdir)

    def _get_sequence_list(self, ann_dir: str) -> List[str]:
        if self.ann_file and not osp.isdir(self.ann_file):
            seq_list = list_from_file(
                self.ann_file, backend_args=self.backend_args)
            return [seq.strip() for seq in seq_list if seq.strip()]
        import glob
        txt_files = sorted(glob.glob(osp.join(ann_dir, '*.txt')))
        if not txt_files:
            raise FileNotFoundError(
                f'No MOT annotation files found in {ann_dir}')
        return [osp.splitext(osp.basename(p))[0] for p in txt_files]

    def _get_img_filename(self, frame_id: int) -> str:
        if self.img_format == 'npy':
            return f'{frame_id:06d}.npy'
        return f'{frame_id:06d}_p1.jpg'

    def _img_path(self, img_root: str, seq_name: str, frame_id: int) -> str:
        return osp.join(img_root, seq_name, self._get_img_filename(frame_id))

    def _instances_from_frame(
            self, frame_anns: Dict[int, List[dict]], frame_id: int) -> List[dict]:
        instances = []
        for ann in frame_anns.get(frame_id, []):
            instances.append({
                'bbox': np.array(ann['polygon'], dtype=np.float32),
                'bbox_label': ann['class_id'],
                'ignore_flag': ann['ignore_flag'],
                'track_id': ann['track_id'],
            })
        return instances

    def load_data_list(self) -> List[dict]:
        ann_dir = self._get_ann_dir()
        seq_list = self._get_sequence_list(ann_dir)
        img_root = self.data_prefix.get('img_path', '')

        data_list: List[dict] = []
        for seq_name in seq_list:
            ann_path = osp.join(ann_dir, f'{seq_name}.txt')
            if not osp.isfile(ann_path):
                raise FileNotFoundError(
                    f'MOT annotation not found: {ann_path}')
            frame_anns = load_hsmot_sequence_ann(ann_path)
            curr_frame_ids = sorted(frame_anns.keys())

            for frame_id_curr in curr_frame_ids:
                img_path_curr = self._img_path(img_root, seq_name, frame_id_curr)
                if not osp.isfile(img_path_curr):
                    continue

                if self.same_frame:
                    instances_curr = self._instances_from_frame(
                        frame_anns, frame_id_curr)
                    data_list.append({
                        'img_id': f'{seq_name}_{frame_id_curr:06d}_same',
                        'video_id': seq_name,
                        'seq_name': seq_name,
                        'frame_id': frame_id_curr,
                        'frame_id_prev': frame_id_curr,
                        'img_path': img_path_curr,
                        'img_path_prev': img_path_curr,
                        'file_name': self._get_img_filename(frame_id_curr),
                        'file_name_prev': self._get_img_filename(frame_id_curr),
                        'instances_prev': copy.deepcopy(instances_curr),
                        'instances_curr': instances_curr,
                    })
                    continue

                frame_id_prev = frame_id_curr - self.frame_interval
                if frame_id_prev < 1:
                    continue

                img_path_prev = self._img_path(img_root, seq_name, frame_id_prev)

                if self.require_prev_image and not osp.isfile(img_path_prev):
                    continue

                instances_prev = self._instances_from_frame(
                    frame_anns, frame_id_prev)
                instances_curr = self._instances_from_frame(
                    frame_anns, frame_id_curr)

                data_list.append({
                    'img_id': f'{seq_name}_{frame_id_curr:06d}_p{frame_id_prev:06d}',
                    'video_id': seq_name,
                    'seq_name': seq_name,
                    'frame_id': frame_id_curr,
                    'frame_id_prev': frame_id_prev,
                    'img_path': img_path_curr,
                    'img_path_prev': img_path_prev,
                    'file_name': self._get_img_filename(frame_id_curr),
                    'file_name_prev': self._get_img_filename(frame_id_prev),
                    'instances_prev': instances_prev,
                    'instances_curr': instances_curr,
                })
        return data_list

    def parse_data_info(self, raw_data_info: dict) -> Union[dict, List[dict]]:
        return raw_data_info

    def filter_data(self) -> List[dict]:
        if self.test_mode:
            return self.data_list

        filter_empty_gt = False
        if self.filter_cfg is not None:
            filter_empty_gt = self.filter_cfg.get('filter_empty_gt', False)

        if not filter_empty_gt:
            return self.data_list

        filtered = []
        for data_info in self.data_list:
            has_prev = len(data_info['instances_prev']) > 0
            has_curr = len(data_info['instances_curr']) > 0
            if has_prev or has_curr:
                filtered.append(data_info)
        return filtered

    def get_cat_ids(self, idx: int) -> List[int]:
        info = self.get_data_info(idx)
        labels = [
            inst['bbox_label'] for inst in info['instances_prev']
        ] + [
            inst['bbox_label'] for inst in info['instances_curr']
        ]
        return labels
