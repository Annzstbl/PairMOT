# Copyright (c) AI4RS. All rights reserved.
import os
import tempfile
import unittest

import numpy as np

from mmrotate.datasets.hsmot import (HSMOTDataset, load_hsmot_sequence_ann,
                                     parse_hsmot_mot_line)


class TestHSMOTParsing(unittest.TestCase):

    def test_parse_mot_line(self):
        line = '1,1,976,904,976,858,859,858,859,904,-1,0,1'
        parsed = parse_hsmot_mot_line(line)
        self.assertIsNotNone(parsed)
        frame_id, track_id, polygon, class_id, ignore_flag = parsed
        self.assertEqual(frame_id, 1)
        self.assertEqual(track_id, 1)
        self.assertEqual(len(polygon), 8)
        self.assertEqual(class_id, 0)
        self.assertEqual(ignore_flag, 1)

    def test_load_sequence_ann(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            ann_path = os.path.join(tmpdir, 'data23-1.txt')
            with open(ann_path, 'w', encoding='utf-8') as f:
                f.write('1,1,976,904,976,858,859,858,859,904,-1,0,0\n')
                f.write('1,2,1028,568,1030,517,913,511,910,561,-1,1,0\n')
                f.write('2,1,976,904,976,858,859,858,859,904,-1,0,0\n')
            frame_anns = load_hsmot_sequence_ann(ann_path)
            self.assertEqual(len(frame_anns[1]), 2)
            self.assertEqual(len(frame_anns[2]), 1)
            self.assertEqual(frame_anns[1][0]['track_id'], 1)
            self.assertEqual(frame_anns[1][1]['class_id'], 1)


class TestHSMOTDataset(unittest.TestCase):

    def _build_dataset_root(self, tmpdir, img_format='npy'):
        mot_dir = os.path.join(tmpdir, 'mot')
        img_dir = os.path.join(tmpdir, 'npy', 'data23-1')
        os.makedirs(mot_dir)
        os.makedirs(img_dir)

        with open(os.path.join(mot_dir, 'data23-1.txt'), 'w',
                  encoding='utf-8') as f:
            f.write('1,1,10,10,20,10,20,20,10,20,-1,0,0\n')
            f.write('1,2,30,30,40,30,40,40,30,40,-1,2,0\n')
            f.write('2,1,10,10,20,10,20,20,10,20,-1,0,0\n')

        if img_format == 'npy':
            np.save(os.path.join(img_dir, '000001.npy'),
                    np.zeros((64, 64, 8), dtype=np.uint8))
            np.save(os.path.join(img_dir, '000002.npy'),
                    np.zeros((64, 64, 8), dtype=np.uint8))
        else:
            jpg_dir = os.path.join(tmpdir, 'npy2jpg', 'data23-1')
            os.makedirs(jpg_dir)
            try:
                import cv2
                for frame_id in (1, 2):
                    stem = f'{frame_id:06d}'
                    cv2.imwrite(os.path.join(jpg_dir, f'{stem}_p1.jpg'),
                                np.zeros((64, 64, 3), dtype=np.uint8))
                    cv2.imwrite(os.path.join(jpg_dir, f'{stem}_p2.jpg'),
                                np.zeros((64, 64, 3), dtype=np.uint8))
                    cv2.imwrite(os.path.join(jpg_dir, f'{stem}_p3.jpg'),
                                np.zeros((64, 64, 3), dtype=np.uint8))
            except ImportError:
                raise unittest.SkipTest('opencv-python is required for 3jpg test')

        return tmpdir

    def test_npy_dataset(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = self._build_dataset_root(tmpdir, img_format='npy')
            dataset = HSMOTDataset(
                data_root=root,
                ann_subdir='mot',
                data_prefix=dict(img_path='npy'),
                img_format='npy',
                with_track_id=True,
                pipeline=[],
                lazy_init=True)
            dataset.full_init()
            self.assertEqual(len(dataset), 2)
            info = dataset.get_data_info(0)
            self.assertTrue(info['img_path'].endswith('000001.npy'))
            self.assertEqual(len(info['instances']), 2)
            self.assertIn('track_id', info['instances'][0])

    def test_detection_only(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = self._build_dataset_root(tmpdir, img_format='npy')
            dataset = HSMOTDataset(
                data_root=root,
                ann_subdir='mot',
                data_prefix=dict(img_path='npy'),
                img_format='npy',
                with_track_id=False,
                pipeline=[],
                lazy_init=True)
            dataset.full_init()
            info = dataset.get_data_info(0)
            self.assertNotIn('track_id', info['instances'][0])

    def test_filter_empty_gt(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = self._build_dataset_root(tmpdir, img_format='npy')
            mot_path = os.path.join(root, 'mot', 'data23-1.txt')
            with open(mot_path, 'a', encoding='utf-8') as f:
                f.write('3,1,10,10,20,10,20,20,10,20,-1,0,0\n')
            dataset = HSMOTDataset(
                data_root=root,
                ann_subdir='mot',
                data_prefix=dict(img_path='npy'),
                img_format='npy',
                filter_cfg=dict(filter_empty_gt=True),
                pipeline=[],
                lazy_init=True)
            dataset.full_init()
            self.assertEqual(len(dataset), 2)


if __name__ == '__main__':
    unittest.main()