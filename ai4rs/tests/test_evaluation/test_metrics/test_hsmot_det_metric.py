# Copyright (c) AI4RS. All rights reserved.
import unittest

import numpy as np
import torch

from mmrotate.evaluation import HSMOTDetMetric


class TestHSMOTDetMetric(unittest.TestCase):

    def _create_dummy_data_sample(self):
        bboxes = np.array([[23, 31, 10.0, 20.0, 0.0],
                           [100, 120, 10.0, 20.0, 0.1]])
        labels = np.array([0, 1])
        bboxes_ignore = np.array([[0] * 5])
        labels_ignore = np.array([0])
        pred_bboxes = np.array([[23, 31, 10.0, 20.0, 0.0],
                                [100, 120, 10.0, 20.0, 0.1]])
        pred_scores = np.array([1.0, 0.98])
        pred_labels = np.array([0, 1])
        return [
            dict(
                img_id='seq1/000001',
                gt_instances=dict(
                    bboxes=torch.from_numpy(bboxes),
                    labels=torch.from_numpy(labels)),
                ignored_instances=dict(
                    bboxes=torch.from_numpy(bboxes_ignore),
                    labels=torch.from_numpy(labels_ignore)),
                pred_instances=dict(
                    bboxes=torch.from_numpy(pred_bboxes),
                    scores=torch.from_numpy(pred_scores),
                    labels=torch.from_numpy(pred_labels)))
        ]

    def test_default_iou_thrs(self):
        metric = HSMOTDetMetric()
        self.assertEqual(len(metric.iou_thrs), 10)
        self.assertTrue(metric.classwise)

    def test_eval(self):
        metric = HSMOTDetMetric(iou_thrs=[0.5, 0.75], classwise=True)
        metric.dataset_meta = {'classes': ('car', 'bike')}
        metric.process({}, self._create_dummy_data_sample())
        results = metric.evaluate(size=1)

        self.assertIn('hsmot/mAP', results)
        self.assertIn('hsmot/AP50', results)
        self.assertIn('hsmot/AP75', results)
        self.assertIn('hsmot/car_AP', results)
        self.assertIn('hsmot/bike_AP', results)
        self.assertIn('hsmot/car_recall', results)
        self.assertIn('hsmot/mean_recall', results)


if __name__ == '__main__':
    unittest.main()
