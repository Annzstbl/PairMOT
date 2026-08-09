import unittest
from collections import OrderedDict

import torch

from projects.multispec_pair_rotated_rtdetr.tools \
    .make_ema_lag_corrected_checkpoint import interpolate_state_dict


class TestEmaLagCorrectedCheckpoint(unittest.TestCase):

    def setUp(self):
        self.averaged = OrderedDict([
            ('weight', torch.tensor([0.0, 2.0])),
            ('counter', torch.tensor(3, dtype=torch.int64)),
        ])
        self.online = OrderedDict([
            ('steps', torch.tensor(8)),
            ('module.weight', torch.tensor([2.0, 4.0])),
            ('module.counter', torch.tensor(5, dtype=torch.int64)),
        ])

    def test_halfway_interpolation(self):
        output, stats = interpolate_state_dict(
            self.averaged, self.online, 0.5)
        self.assertTrue(torch.equal(output['weight'], torch.tensor([1.0, 3.0])))
        self.assertEqual(output['counter'].item(), 3)
        self.assertEqual(stats['floating_tensors'], 1)
        self.assertEqual(stats['nonfloating_tensors'], 1)

    def test_online_endpoint_uses_online_nonfloating_state(self):
        output, _ = interpolate_state_dict(self.averaged, self.online, 1.0)
        self.assertTrue(torch.equal(output['weight'], self.online['module.weight']))
        self.assertEqual(output['counter'].item(), 5)

    def test_excluded_prefix_stays_at_ema(self):
        averaged = OrderedDict([
            ('decoder.weight', torch.tensor([0.0, 2.0])),
            ('bbox_head.weight', torch.tensor([1.0, 3.0])),
        ])
        online = OrderedDict([
            ('steps', torch.tensor(8)),
            ('module.decoder.weight', torch.tensor([2.0, 4.0])),
            ('module.bbox_head.weight', torch.tensor([3.0, 5.0])),
        ])
        output, stats = interpolate_state_dict(
            averaged, online, 0.25, ('decoder.', ))
        self.assertTrue(torch.equal(
            output['decoder.weight'], averaged['decoder.weight']))
        self.assertTrue(torch.equal(
            output['bbox_head.weight'], torch.tensor([1.5, 3.5])))
        self.assertEqual(stats['interpolated_floating_tensors'], 1)
        self.assertEqual(stats['excluded_floating_tensors'], 1)

    def test_excluded_nonfloating_stays_at_ema_at_online_endpoint(self):
        output, _ = interpolate_state_dict(
            self.averaged, self.online, 1.0, ('counter', ))
        self.assertEqual(output['counter'].item(), 3)

    def test_rejects_empty_exclude_prefix(self):
        with self.assertRaises(ValueError):
            interpolate_state_dict(
                self.averaged, self.online, 0.25, ('', ))

    def test_input_is_not_mutated(self):
        before_averaged = self.averaged['weight'].clone()
        before_online = self.online['module.weight'].clone()
        interpolate_state_dict(self.averaged, self.online, 0.25)
        self.assertTrue(torch.equal(self.averaged['weight'], before_averaged))
        self.assertTrue(torch.equal(self.online['module.weight'], before_online))

    def test_rejects_invalid_fraction(self):
        for value in (-0.1, 1.1, float('nan')):
            with self.assertRaises(ValueError):
                interpolate_state_dict(self.averaged, self.online, value)

    def test_rejects_missing_online_key(self):
        online = OrderedDict([('steps', torch.tensor(8))])
        with self.assertRaises(KeyError):
            interpolate_state_dict(self.averaged, online, 0.5)


if __name__ == '__main__':
    unittest.main()
