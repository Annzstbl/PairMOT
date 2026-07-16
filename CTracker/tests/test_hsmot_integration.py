import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from hsmot_adapter import build_hsmot_pair_dataset, ctracker_collate


class TestHSMOTIntegration(unittest.TestCase):
    def test_onepic_pair(self):
        root = os.path.abspath(os.path.join(
            os.path.dirname(__file__), '..', '..', 'data', 'hsmot',
            'OnePic'))
        if not os.path.isdir(root):
            self.skipTest(f'HSMOT OnePic is unavailable: {root}')
        dataset = build_hsmot_pair_dataset(
            root, ann_subdir='train/mot', img_subdir='npy2jpg',
            training=True, image_scale=(256, 256), augment=False)
        batch = ctracker_collate([dataset[0]])
        self.assertEqual(batch['img_prev'].shape[1], 8)
        self.assertEqual(batch['img_curr'].shape, batch['img_prev'].shape)
        # Native HSMOT is 1200x900 (W x H); keep-ratio resize to a 256x256
        # canvas therefore produces 256x192 before divisor padding.
        self.assertEqual(batch['img_prev'].shape[-2:], (192, 256))
        target = {
            key: value[0] for key, value in batch['targets'].items()
        }
        self.assertEqual(target['bboxes_prev'].shape[1], 5)
        self.assertEqual(len(target['track_ids']), len(target['labels']))
        self.assertTrue(target['valid_prev'].any())


if __name__ == '__main__':
    unittest.main()
