# Copyright (c) AI4RS. All rights reserved.
import glob
import os.path as osp
from collections import defaultdict
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
from mmengine.dataset import BaseDataset
from mmengine.fileio import list_from_file

from mmrotate.registry import DATASETS

# 8-channel spectral normalization (0-1), multiply by 255 for mmdet.Normalize.
HSMOT_MEAN = [
    0.27358221, 0.28804452, 0.28133921, 0.26906377,
    0.28309119, 0.26928305, 0.28372527, 0.27149373,
]
HSMOT_STD = [
    0.19756629, 0.17432339, 0.16413284, 0.17581682,
    0.18366176, 0.1536845, 0.15964683, 0.16557951,
]


def parse_hsmot_mot_line(line: str) -> Optional[Tuple[int, int, List[float], int, int]]:
    """Parse one HSMOT MOT annotation line.

    Format:
        frame_id, track_id, x1, y1, x2, y2, x3, y3, x4, y4, score, class_id, truncation

    Returns:
        tuple or None: (frame_id, track_id, polygon, class_id, ignore_flag)
    """
    line = line.strip()
    if not line:
        return None
    parts = line.split(',')
    if len(parts) < 12:
        return None
    frame_id = int(float(parts[0]))
    track_id = int(float(parts[1]))
    polygon = [float(v) for v in parts[2:10]]
    class_id = int(float(parts[11]))
    truncation = int(float(parts[12])) if len(parts) > 12 else 0
    ignore_flag = 1 if truncation > 0 else 0
    return frame_id, track_id, polygon, class_id, ignore_flag


def load_hsmot_sequence_ann(
        ann_path: str) -> Dict[int, List[dict]]:
    """Load one sequence MOT file and group annotations by frame_id."""
    frame_anns: Dict[int, List[dict]] = defaultdict(list)
    with open(ann_path, 'r', encoding='utf-8') as f:
        for line in f:
            parsed = parse_hsmot_mot_line(line)
            if parsed is None:
                continue
            frame_id, track_id, polygon, class_id, ignore_flag = parsed
            frame_anns[frame_id].append({
                'track_id': track_id,
                'polygon': polygon,
                'class_id': class_id,
                'ignore_flag': ignore_flag,
            })
    return frame_anns


@DATASETS.register_module()
class HSMOTDataset(BaseDataset):
    """HSMOT rotated-box multi-spectral MOT dataset.

    Supports 8-channel images stored as ``.npy`` or three ``.jpg`` files,
    and annotations in HSMOT MOT text format (13 columns).

    Args:
        ann_subdir (str): Sub-directory under ``data_root`` that stores
            sequence-level ``*.txt`` MOT files. Defaults to ``'mot'``.
        ann_file (str, optional): Split file listing sequence names, one per
            line. If ``None``, all ``*.txt`` files under ``ann_subdir`` are
            used. Defaults to ``None``.
        img_format (str): Image storage format. Options are ``'npy'`` and
            ``'3jpg'``. Defaults to ``'npy'``.
        with_track_id (bool): Whether to attach ``track_id`` to each instance.
            Set to ``False`` for detection-only training. Defaults to ``True``.
        backend_args (dict, optional): Backend arguments for file I/O.
            Defaults to ``None``.
        file_client_args (dict): Deprecated, use ``backend_args`` instead.
    """

    METAINFO = {
        'classes':
        ('car', 'bike', 'pedestrian', 'van', 'truck', 'bus', 'tricycle',
         'awning-bike'),
        'palette': [(220, 20, 60), (119, 11, 32), (0, 0, 142), (0, 0, 230),
                    (106, 0, 228), (0, 60, 100), (0, 80, 100), (0, 0, 70)],
    }

    def __init__(self,
                 ann_subdir: str = 'mot',
                 img_format: str = 'npy',
                 with_track_id: bool = True,
                 file_client_args: dict = None,
                 backend_args: dict = None,
                 **kwargs) -> None:
        assert img_format in ('npy', '3jpg'), \
            f"img_format must be 'npy' or '3jpg', but got {img_format}"
        self.ann_subdir = ann_subdir
        self.img_format = img_format
        self.with_track_id = with_track_id
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
        txt_files = sorted(glob.glob(osp.join(ann_dir, '*.txt')))
        if not txt_files:
            raise FileNotFoundError(
                f'No MOT annotation files found in {ann_dir}')
        return [osp.splitext(osp.basename(p))[0] for p in txt_files]

    def _get_img_filename(self, frame_id: int) -> str:
        if self.img_format == 'npy':
            return f'{frame_id:06d}.npy'
        return f'{frame_id:06d}_p1.jpg'

    def load_data_list(self) -> List[dict]:
        ann_dir = self._get_ann_dir()
        seq_list = self._get_sequence_list(ann_dir)
        # ``data_prefix`` is already joined with ``data_root`` in
        # ``BaseDataset._join_prefix()``; do not join again here.
        img_root = self.data_prefix.get('img_path', '')

        data_list: List[dict] = []
        for seq_name in seq_list:
            ann_path = osp.join(ann_dir, f'{seq_name}.txt')
            if not osp.isfile(ann_path):
                raise FileNotFoundError(
                    f'MOT annotation not found: {ann_path}')
            frame_anns = load_hsmot_sequence_ann(ann_path)
            for frame_id in sorted(frame_anns.keys()):
                file_name = self._get_img_filename(frame_id)
                img_path = osp.join(img_root, seq_name, file_name)
                instances = []
                for ann in frame_anns[frame_id]:
                    instance = {
                        'bbox':
                        np.array(ann['polygon'], dtype=np.float32),
                        'bbox_label':
                        ann['class_id'],
                        'ignore_flag':
                        ann['ignore_flag'],
                    }
                    if self.with_track_id:
                        instance['track_id'] = ann['track_id']
                    instances.append(instance)

                data_list.append({
                    'img_id': f'{seq_name}_{frame_id:06d}',
                    'seq_name': seq_name,
                    'frame_id': frame_id,
                    'file_name': file_name,
                    'img_path': img_path,
                    'instances': instances,
                })
        return data_list

    def parse_data_info(self, raw_data_info: dict) -> Union[dict, List[dict]]:
        """Keep compatibility with BaseDataset hooks."""
        return raw_data_info

    def filter_data(self) -> List[dict]:
        if self.test_mode:
            return self.data_list

        filter_empty_gt = False
        if self.filter_cfg is not None:
            filter_empty_gt = self.filter_cfg.get('filter_empty_gt', False)

        if not filter_empty_gt:
            return self.data_list

        return [
            data_info for data_info in self.data_list
            if len(data_info['instances']) > 0
        ]

    def get_cat_ids(self, idx: int) -> List[int]:
        instances = self.get_data_info(idx)['instances']
        return [instance['bbox_label'] for instance in instances]
