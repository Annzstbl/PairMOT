# Copyright (c) AI4RS. All rights reserved.
import os
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from mmrotate.datasets.hsmot_pair import HSMOTPairDataset
from mmrotate.datasets.pair_gt import (
    TrackIdClassMismatchError,
    build_pair_gt_from_instances,
    INVALID_QBOX_PLACEHOLDER,
)
from mmrotate.datasets.transforms.loading_hsmot_pair import (
    ConvertPairBoxType,
    HSMOTPairLoadAnnotations,
    LoadHSMOTPairImages,
    PackHSMOTPairInputs,
)
from mmrotate.datasets.transforms.transforms_hsmot_pair import (
    PairSharedRandomFlip,
    PairSharedRandomRotate,
    PairSharedResize,
)
from mmrotate.datasets.transforms.visualize_hsmot_pair import visualize_hsmot_pair
from mmrotate.structures.bbox import RotatedBoxes

# 可视化输出目录（PairMmot/tmp/hsmot_pair_test_vis）
_DEFAULT_VIS_DIR = Path(__file__).resolve().parents[3] / 'tmp' / 'hsmot_pair_test_vis'


def _checkerboard(h: int, w: int, c: int = 8, cell: int = 8) -> np.ndarray:
    """合成可分辨的 8 通道测试图（非全黑）。"""
    img = np.zeros((h, w, c), dtype=np.uint8)
    for ch in range(c):
        for y in range(h):
            for x in range(w):
                if ((x // cell) + (y // cell) + ch) % 2 == 0:
                    img[y, x, ch] = 80 + ch * 15
                else:
                    img[y, x, ch] = 180 - ch * 10
    return img


def _export_vis(name: str, img_prev, img_curr, bboxes_prev, bboxes_curr,
                track_ids, valid_prev, valid_curr, out_dir: Path) -> str:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f'{name}.jpg'
    visualize_hsmot_pair(
        img_prev,
        img_curr,
        bboxes_prev,
        bboxes_curr,
        track_ids=track_ids,
        valid_prev=valid_prev,
        valid_curr=valid_curr,
        save_path=str(path),
    )
    return str(path)


def _qbox(cx, cy, w, h, angle_deg=0):
    x1, y1 = cx - w / 2, cy - h / 2
    x2, y2 = cx + w / 2, cy + h / 2
    return np.array([x1, y1, x2, y1, x2, y2, x1, y2], dtype=np.float32)


class TestBuildPairGT(unittest.TestCase):

    def test_persistent_new_disappearing(self):
        instances_prev = [
            {
                'track_id': 1,
                'bbox': _qbox(10, 10, 4, 4),
                'bbox_label': 0,
            },
            {
                'track_id': 2,
                'bbox': _qbox(30, 30, 4, 4),
                'bbox_label': 1,
            },
        ]
        instances_curr = [
            {
                'track_id': 1,
                'bbox': _qbox(12, 12, 4, 4),
                'bbox_label': 0,
            },
            {
                'track_id': 3,
                'bbox': _qbox(50, 50, 4, 4),
                'bbox_label': 2,
            },
        ]
        pair_gt = build_pair_gt_from_instances(instances_prev, instances_curr)
        self.assertEqual(pair_gt['track_ids'].tolist(), [1, 2, 3])
        self.assertTrue(pair_gt['valid_prev'][0])
        self.assertTrue(pair_gt['valid_curr'][0])
        self.assertTrue(pair_gt['valid_prev'][1])
        self.assertFalse(pair_gt['valid_curr'][1])
        self.assertFalse(pair_gt['valid_prev'][2])
        self.assertTrue(pair_gt['valid_curr'][2])
        self.assertTrue(
            np.allclose(pair_gt['bboxes_curr'][1], INVALID_QBOX_PLACEHOLDER))

    def test_empty_frames(self):
        pair_gt = build_pair_gt_from_instances([], [])
        self.assertEqual(len(pair_gt['track_ids']), 0)
        self.assertEqual(pair_gt['bboxes_prev'].shape, (0, 8))

    def test_class_mismatch_raises(self):
        prev = [{
            'track_id': 1,
            'bbox': _qbox(10, 10, 4, 4),
            'bbox_label': 0,
        }]
        curr = [{
            'track_id': 1,
            'bbox': _qbox(10, 10, 4, 4),
            'bbox_label': 1,
        }]
        with self.assertRaises(TrackIdClassMismatchError):
            build_pair_gt_from_instances(prev, curr, video_id='seqA', frame_id_prev=1,
                                         frame_id_curr=2)

    def test_different_target_counts(self):
        prev = [
            {'track_id': i, 'bbox': _qbox(10 * i, 10, 4, 4), 'bbox_label': 0}
            for i in range(1, 4)
        ]
        curr = [
            {'track_id': i, 'bbox': _qbox(10 * i, 10, 4, 4), 'bbox_label': 0}
            for i in range(1, 6)
        ]
        pair_gt = build_pair_gt_from_instances(prev, curr)
        self.assertEqual(len(pair_gt['track_ids']), 5)


class TestHSMOTPairDataset(unittest.TestCase):

    def _build_root(self, tmpdir):
        mot_dir = os.path.join(tmpdir, 'mot')
        img_dir = os.path.join(tmpdir, 'npy', 'seq01')
        os.makedirs(mot_dir)
        os.makedirs(img_dir)

        with open(os.path.join(mot_dir, 'seq01.txt'), 'w', encoding='utf-8') as f:
            # frame 1: tracks 1,2; frame 2: tracks 1,3; frame 3: empty
            f.write('1,1,10,10,20,10,20,20,10,20,-1,0,0\n')
            f.write('1,2,30,30,40,30,40,40,30,40,-1,1,0\n')
            f.write('2,1,12,12,22,12,22,22,12,22,-1,0,0\n')
            f.write('2,3,50,50,60,50,60,60,50,60,-1,2,0\n')

        for frame_id in (1, 2, 3):
            np.save(os.path.join(img_dir, f'{frame_id:06d}.npy'),
                    np.zeros((64, 64, 8), dtype=np.uint8))
        return tmpdir

    def test_pair_indexing(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = self._build_root(tmpdir)
            dataset = HSMOTPairDataset(
                data_root=root,
                ann_subdir='mot',
                data_prefix=dict(img_path='npy'),
                img_format='npy',
                frame_interval=1,
                pipeline=[],
                lazy_init=True)
            dataset.full_init()
            # frame 2 has annotations -> one pair (prev=1, curr=2)
            self.assertEqual(len(dataset), 1)
            info = dataset.get_data_info(0)
            self.assertEqual(info['frame_id'], 2)
            self.assertEqual(info['frame_id_prev'], 1)
            self.assertEqual(info['video_id'], 'seq01')


class TestHSMOTPairPipeline(unittest.TestCase):

    def _make_results(self):
        instances_prev = [{
            'track_id': 1,
            'bbox': _qbox(20, 32, 8, 16),
            'bbox_label': 0,
            'ignore_flag': 0,
        }]
        instances_curr = [
            {
                'track_id': 1,
                'bbox': _qbox(20, 32, 8, 16),
                'bbox_label': 0,
                'ignore_flag': 0,
            },
            {
                'track_id': 2,
                'bbox': _qbox(48, 48, 8, 8),
                'bbox_label': 1,
                'ignore_flag': 0,
            },
        ]
        return {
            'img_path_prev': '',
            'img_path': '',
            'video_id': 'seq01',
            'frame_id_prev': 1,
            'frame_id': 2,
            'instances_prev': instances_prev,
            'instances_curr': instances_curr,
        }

    def test_pack_shape_and_fields(self):
        results = self._make_results()
        results['img'] = [
            np.zeros((64, 64, 8), dtype=np.uint8),
            np.zeros((64, 64, 8), dtype=np.uint8),
        ]
        results['img_shape'] = (64, 64)
        results['ori_shape'] = (64, 64)

        ann = HSMOTPairLoadAnnotations()
        results = ann.transform(results)
        convert = ConvertPairBoxType(dst_box_type='rbox')
        results = convert.transform(results)

        pack = PackHSMOTPairInputs()
        packed = pack.transform(results)

        self.assertEqual(packed['inputs'].shape, (2, 8, 64, 64))
        pair_gt = packed['data_samples'].pair_gt_instances
        self.assertEqual(pair_gt.labels.shape[0], 2)
        self.assertEqual(pair_gt.track_ids.tolist(), [1, 2])
        self.assertTrue(pair_gt.valid_prev[0])
        self.assertFalse(pair_gt.valid_prev[1])
        self.assertTrue(pair_gt.valid_curr[0])
        self.assertTrue(pair_gt.valid_curr[1])

    def test_shared_flip_horizontal(self):
        results = self._make_results()
        results['img'] = [
            np.zeros((64, 64, 8), dtype=np.uint8),
            np.zeros((64, 64, 8), dtype=np.uint8),
        ]
        results['img_shape'] = (64, 64)
        results['ori_shape'] = (64, 64)
        ann = HSMOTPairLoadAnnotations()
        results = ann.transform(results)
        convert = ConvertPairBoxType(dst_box_type='rbox')
        results = convert.transform(results)

        before_prev = results['gt_bboxes_prev'].tensor.clone()
        before_curr = results['gt_bboxes_curr'].tensor.clone()

        flip = PairSharedRandomFlip(prob=1.0, direction='horizontal')
        results = flip.transform(results)

        self.assertTrue(results['flip'])
        self.assertEqual(results['flip_direction'], 'horizontal')
        w = 64
        before_x = before_prev[0, 0].item()
        after_x = results['gt_bboxes_prev'].tensor[0, 0].item()
        self.assertAlmostEqual(after_x, w - before_x, places=3)
        self.assertAlmostEqual(
            after_x,
            results['gt_bboxes_curr'].tensor[0, 0].item(),
            places=4)

    def test_shared_resize_same_scale(self):
        results = self._make_results()
        results['img'] = [
            np.zeros((64, 64, 8), dtype=np.uint8),
            np.zeros((64, 64, 8), dtype=np.uint8),
        ]
        results['img_shape'] = (64, 64)
        results['ori_shape'] = (64, 64)
        ann = HSMOTPairLoadAnnotations()
        results = ann.transform(results)
        convert = ConvertPairBoxType(dst_box_type='rbox')
        results = convert.transform(results)

        resize = PairSharedResize(scale=(32, 32), keep_ratio=False)
        results = resize.transform(results)
        self.assertEqual(results['img'][0].shape[:2], (32, 32))
        self.assertEqual(results['img'][1].shape[:2], (32, 32))
        self.assertEqual(tuple(results['scale_factor']), (0.5, 0.5))

    def test_shared_rotate_same_angle(self):
        results = self._make_results()
        results['img'] = [
            np.zeros((64, 64, 8), dtype=np.uint8),
            np.zeros((64, 64, 8), dtype=np.uint8),
        ]
        results['img_shape'] = (64, 64)
        results['ori_shape'] = (64, 64)
        ann = HSMOTPairLoadAnnotations()
        results = ann.transform(results)
        convert = ConvertPairBoxType(dst_box_type='rbox')
        results = convert.transform(results)

        rotate = PairSharedRandomRotate(prob=1.0, angle_range=90)
        np.random.seed(0)
        results = rotate.transform(results)
        self.assertEqual(results['img'][0].shape, results['img'][1].shape)

    def test_visualize_runs(self):
        img = np.zeros((64, 64, 8), dtype=np.uint8)
        bboxes_prev = RotatedBoxes(torch.tensor([[32., 32., 16., 16., 0.]]))
        bboxes_curr = RotatedBoxes(torch.tensor([[32., 32., 16., 16., 0.],
                                                 [48., 48., 8., 8., 0.]]))
        vis = visualize_hsmot_pair(
            img,
            img,
            bboxes_prev,
            bboxes_curr,
            track_ids=[1, 2],
            valid_prev=[True, False],
            valid_curr=[True, True],
        )
        self.assertEqual(vis.shape[0], 64)
        self.assertGreater(vis.shape[1], 64)


class TestHSMOTPairVisualExport(unittest.TestCase):
    """导出各测试场景可视化到 tmp/hsmot_pair_test_vis/。"""

    def test_export_all_scenarios(self):
        out_dir = Path(os.environ.get('HSMOT_PAIR_TEST_VIS_DIR',
                                      str(_DEFAULT_VIS_DIR)))
        paths = export_hsmot_pair_test_visualizations(out_dir)
        self.assertGreaterEqual(len(paths), 6)
        for p in paths:
            self.assertTrue(os.path.isfile(p), f'missing vis file: {p}')


def export_hsmot_pair_test_visualizations(
        out_dir: Path = _DEFAULT_VIS_DIR) -> list[str]:
    """将 test_hsmot_pair 各场景绘制并保存为 JPG。

    Returns:
        list[str]: 已保存图片路径列表。
    """
    out_dir = Path(out_dir)
    saved: list[str] = []
    h, w = 128, 128
    img_prev = _checkerboard(h, w)
    img_curr = _checkerboard(h, w)

    # 1) 持续 / 新生 / 消失
    instances_prev = [
        {'track_id': 1, 'bbox': _qbox(32, 40, 20, 16), 'bbox_label': 0},
        {'track_id': 2, 'bbox': _qbox(80, 80, 16, 16), 'bbox_label': 1},
    ]
    instances_curr = [
        {'track_id': 1, 'bbox': _qbox(36, 44, 20, 16), 'bbox_label': 0},
        {'track_id': 3, 'bbox': _qbox(100, 30, 12, 12), 'bbox_label': 2},
    ]
    pair_gt = build_pair_gt_from_instances(instances_prev, instances_curr)
    results = {
        'video_id': 'seq01', 'frame_id_prev': 1, 'frame_id': 2,
        'instances_prev': instances_prev,
        'instances_curr': instances_curr,
        'img': [img_prev.copy(), img_curr.copy()],
        'img_shape': (h, w), 'ori_shape': (h, w),
    }
    results = HSMOTPairLoadAnnotations().transform(results)
    results = ConvertPairBoxType(dst_box_type='rbox').transform(results)
    saved.append(_export_vis(
        '01_persistent_new_disappear',
        results['img'][0], results['img'][1],
        results['gt_bboxes_prev'], results['gt_bboxes_curr'],
        pair_gt['track_ids'].tolist(),
        pair_gt['valid_prev'].tolist(),
        pair_gt['valid_curr'].tolist(),
        out_dir))

    # 2) 空帧
    empty = build_pair_gt_from_instances([], [])
    results_empty = {
        'video_id': 'seq01', 'frame_id_prev': 1, 'frame_id': 2,
        'instances_prev': [], 'instances_curr': [],
        'img': [img_prev.copy(), img_curr.copy()],
        'img_shape': (h, w), 'ori_shape': (h, w),
    }
    results_empty = HSMOTPairLoadAnnotations().transform(results_empty)
    results_empty = ConvertPairBoxType(dst_box_type='rbox').transform(results_empty)
    saved.append(_export_vis(
        '02_empty_frames',
        results_empty['img'][0], results_empty['img'][1],
        results_empty['gt_bboxes_prev'], results_empty['gt_bboxes_curr'],
        empty['track_ids'].tolist(),
        empty['valid_prev'].tolist(),
        empty['valid_curr'].tolist(),
        out_dir))

    # 3) 不同目标数量
    prev_many = [
        {'track_id': i, 'bbox': _qbox(20 * i, 64, 10, 10), 'bbox_label': 0}
        for i in range(1, 4)
    ]
    curr_many = [
        {'track_id': i, 'bbox': _qbox(15 * i, 64, 10, 10), 'bbox_label': 0}
        for i in range(1, 6)
    ]
    gt_many = build_pair_gt_from_instances(prev_many, curr_many)
    res_many = {
        'video_id': 'seq01', 'frame_id_prev': 1, 'frame_id': 2,
        'instances_prev': prev_many, 'instances_curr': curr_many,
        'img': [img_prev.copy(), img_curr.copy()],
        'img_shape': (h, w), 'ori_shape': (h, w),
    }
    res_many = HSMOTPairLoadAnnotations().transform(res_many)
    res_many = ConvertPairBoxType(dst_box_type='rbox').transform(res_many)
    saved.append(_export_vis(
        '03_different_target_counts',
        res_many['img'][0], res_many['img'][1],
        res_many['gt_bboxes_prev'], res_many['gt_bboxes_curr'],
        gt_many['track_ids'].tolist(),
        gt_many['valid_prev'].tolist(),
        gt_many['valid_curr'].tolist(),
        out_dir))

    # 4) 水平翻转前后对比
    base = TestHSMOTPairPipeline()._make_results()
    base['img'] = [img_prev.copy(), img_curr.copy()]
    base['img_shape'] = (h, w)
    base['ori_shape'] = (h, w)
    base = HSMOTPairLoadAnnotations().transform(base)
    base = ConvertPairBoxType(dst_box_type='rbox').transform(base)
    before = {
        'img': [base['img'][0].copy(), base['img'][1].copy()],
        'gt_bboxes_prev': base['gt_bboxes_prev'].clone(),
        'gt_bboxes_curr': base['gt_bboxes_curr'].clone(),
        'pair_labels': base['pair_labels'],
        'pair_track_ids': base['pair_track_ids'],
        'pair_valid_prev': base['pair_valid_prev'],
        'pair_valid_curr': base['pair_valid_curr'],
    }
    saved.append(_export_vis(
        '04_flip_before',
        before['img'][0], before['img'][1],
        before['gt_bboxes_prev'], before['gt_bboxes_curr'],
        base['pair_track_ids'].tolist(),
        base['pair_valid_prev'].tolist(),
        base['pair_valid_curr'].tolist(),
        out_dir))
    flipped = PairSharedRandomFlip(prob=1.0, direction='horizontal').transform(base)
    saved.append(_export_vis(
        '04_flip_after_horizontal',
        flipped['img'][0], flipped['img'][1],
        flipped['gt_bboxes_prev'], flipped['gt_bboxes_curr'],
        flipped['pair_track_ids'].tolist(),
        flipped['pair_valid_prev'].tolist(),
        flipped['pair_valid_curr'].tolist(),
        out_dir))

    # 5) Resize
    res_resize = TestHSMOTPairPipeline()._make_results()
    res_resize['img'] = [img_prev.copy(), img_curr.copy()]
    res_resize['img_shape'] = (h, w)
    res_resize['ori_shape'] = (h, w)
    res_resize = HSMOTPairLoadAnnotations().transform(res_resize)
    res_resize = ConvertPairBoxType(dst_box_type='rbox').transform(res_resize)
    res_resize = PairSharedResize(scale=(64, 64), keep_ratio=False).transform(res_resize)
    saved.append(_export_vis(
        '05_resize_64x64',
        res_resize['img'][0], res_resize['img'][1],
        res_resize['gt_bboxes_prev'], res_resize['gt_bboxes_curr'],
        res_resize['pair_track_ids'].tolist(),
        res_resize['pair_valid_prev'].tolist(),
        res_resize['pair_valid_curr'].tolist(),
        out_dir))

    # 6) Rotate
    res_rot = TestHSMOTPairPipeline()._make_results()
    res_rot['img'] = [img_prev.copy(), img_curr.copy()]
    res_rot['img_shape'] = (h, w)
    res_rot['ori_shape'] = (h, w)
    res_rot = HSMOTPairLoadAnnotations().transform(res_rot)
    res_rot = ConvertPairBoxType(dst_box_type='rbox').transform(res_rot)
    np.random.seed(42)
    res_rot = PairSharedRandomRotate(prob=1.0, angle_range=45).transform(res_rot)
    saved.append(_export_vis(
        '06_rotate_45deg',
        res_rot['img'][0], res_rot['img'][1],
        res_rot['gt_bboxes_prev'], res_rot['gt_bboxes_curr'],
        res_rot['pair_track_ids'].tolist(),
        res_rot['pair_valid_prev'].tolist(),
        res_rot['pair_valid_curr'].tolist(),
        out_dir))

    # 7) 完整 pipeline pack 后张量（resize 后打包，避免 rotate 过滤占位框导致长度不一致）
    packed = PackHSMOTPairInputs().transform(res_resize)
    pair_gt = packed['data_samples'].pair_gt_instances
    saved.append(_export_vis(
        '07_packed_pipeline',
        packed['inputs'][0], packed['inputs'][1],
        pair_gt.bboxes_prev, pair_gt.bboxes_curr,
        pair_gt.track_ids.tolist(),
        pair_gt.valid_prev.tolist(),
        pair_gt.valid_curr.tolist(),
        out_dir))

    index_path = out_dir / 'index.txt'
    with open(index_path, 'w', encoding='utf-8') as f:
        f.write('HSMOT pair test visualizations\n')
        f.write(f'output_dir: {out_dir.resolve()}\n\n')
        for p in saved:
            f.write(f'{p}\n')
    saved.append(str(index_path))
    return saved


if __name__ == '__main__':
    import sys
    if '--vis' in sys.argv:
        sys.argv.remove('--vis')
        paths = export_hsmot_pair_test_visualizations()
        print(f'Exported {len(paths) - 1} images to {_DEFAULT_VIS_DIR}')
        for p in paths:
            print(p)
    else:
        unittest.main()
