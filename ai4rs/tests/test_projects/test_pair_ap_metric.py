import math
import unittest

import torch
from mmengine.structures import InstanceData

from projects.multispec_pair_rotated_rtdetr.multispec_pair_rotated_rtdetr.pair_ap_metric import (
    HSMOTPairAPMetric,
    _format_pair_metric_table,
)


class TestPairAPMetricDiagnosticMode(unittest.TestCase):

    def _sample(self):
        gt = InstanceData()
        gt.labels = torch.tensor([0, 1])
        gt.bboxes_prev = torch.rand(2, 5)
        gt.bboxes_curr = torch.rand(2, 5)
        gt.valid_prev = torch.ones(2, dtype=torch.bool)
        gt.valid_curr = torch.ones(2, dtype=torch.bool)
        pred = InstanceData()
        pred.scores = torch.rand(4)
        pred.labels = torch.tensor([0, 1, 0, 1])
        pred.bboxes_prev = torch.rand(4, 5)
        pred.bboxes_curr = torch.rand(4, 5)
        pred.scores_prev = torch.rand(4)
        pred.scores_curr = torch.rand(4)
        sample = InstanceData()
        sample.set_field(gt, 'pair_gt_instances')
        sample.set_field(pred, 'pred_pair_instances')
        sample.set_metainfo(dict(frame_gap=1))
        return sample

    def test_fast_mode_skips_diagnostics(self):
        metric = HSMOTPairAPMetric(diagnostic_mode=False)
        metric.process({}, [self._sample()])
        self.assertNotIn('matched_queries', metric.results[0])
        metrics = metric.compute_metrics(metric.results)
        self.assertTrue(math.isnan(metrics['match_ratio']))
        self.assertIn('pair_AP50', metrics)
        table = _format_pair_metric_table(metrics)
        self.assertIn('match_ratio', table)
        self.assertIn('n/a', table)

    def test_diagnostic_mode_collects_counters(self):
        metric = HSMOTPairAPMetric(diagnostic_mode=True)
        metric.process({}, [self._sample()])
        self.assertIn('matched_queries', metric.results[0])
        metrics = metric.compute_metrics(metric.results)
        self.assertFalse(math.isnan(metrics['match_ratio']))

    def test_main_ap_always_uses_all_gt_and_outputs_breakdown(self):
        sample = self._sample()
        sample.pair_gt_instances.valid_prev = torch.tensor([True, False])
        sample.pair_gt_instances.valid_curr = torch.tensor([True, True])
        metric = HSMOTPairAPMetric(
            diagnostic_mode=False,
            both_visible_gt_only=True)
        metric.process({}, [sample])
        ap_sample = metric.results[0]['ap_sample']
        self.assertEqual(ap_sample['gt_labels'].numel(), 2)
        metrics = metric.compute_metrics(metric.results)
        self.assertIn('pair_AP50', metrics)
        self.assertIn('both_pair_AP50', metrics)
        self.assertIn('new_pair_AP50', metrics)
        self.assertIn('disappear_pair_AP50', metrics)
        table = _format_pair_metric_table(metrics)
        self.assertIn('GT Filter AP Breakdown', table)


if __name__ == '__main__':
    unittest.main()
