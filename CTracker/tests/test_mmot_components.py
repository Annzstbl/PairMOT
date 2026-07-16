import os
import sys
import unittest

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from rotated_losses import RotatedCTrackerLoss
from rotated_ops import (decode_hboxes_to_rboxes,
                         encode_hboxes_to_rboxes,
                         multiclass_rotated_soft_nms)
from stem_conv3d_se import SpectralStemConv3dSE
import model


class TestSpectralStem(unittest.TestCase):
    def test_shape_uniform_gate_and_rgb_mapping(self):
        stem = SpectralStemConv3dSE(
            out_channels=16, num_spectral=8, reduction=4)
        rgb_weight = torch.randn(16, 3, 7, 7)
        stem.load_rgb_weight(rgb_weight)
        torch.testing.assert_close(
            stem.conv3d.weight, rgb_weight.unsqueeze(1))
        output, gate = stem(
            torch.randn(2, 8, 32, 48), return_gate=True)
        self.assertEqual(output.shape, (2, 16, 16, 24))
        torch.testing.assert_close(gate, torch.full_like(gate, 1 / 8))


class TestRotatedOperations(unittest.TestCase):
    def test_coder_round_trip(self):
        anchors = torch.tensor([
            [10.0, 20.0, 50.0, 100.0],
            [30.0, 40.0, 90.0, 70.0],
        ])
        targets = torch.tensor([
            [30.0, 60.0, 80.0, 35.0, 0.3],
            [60.0, 55.0, 60.0, 30.0, -1.2],
        ])
        decoded = decode_hboxes_to_rboxes(
            anchors, encode_hboxes_to_rboxes(anchors, targets))
        from mmcv.ops import box_iou_rotated
        iou = box_iou_rotated(decoded, targets, aligned=True)
        self.assertTrue(torch.all(iou > 0.999))

    def test_multiclass_rotated_soft_nms(self):
        paired = torch.tensor([
            [10., 10., 8., 4., 0., 11., 10., 8., 4., 0.],
            [10., 10., 8., 4., 0., 12., 10., 8., 4., 0.],
            [10., 10., 8., 4., 0., 11., 10., 8., 4., 0.],
        ])
        scores = torch.tensor([0.9, 0.8, 0.7])
        labels = torch.tensor([0, 0, 1])
        boxes, kept_scores, kept_labels = multiclass_rotated_soft_nms(
            paired, scores, labels)
        self.assertEqual(boxes.size(1), 10)
        self.assertEqual(set(kept_labels.tolist()), {0, 1})
        self.assertTrue(torch.all(kept_scores[:-1] >= kept_scores[1:]))


class TestRotatedLoss(unittest.TestCase):
    def test_finite_backward(self):
        anchors = torch.tensor([[[0., 0., 32., 64.],
                                 [32., 0., 64., 64.]]])
        classification = torch.full(
            (1, 2, 8), 0.01, requires_grad=True)
        regression = torch.zeros((1, 2, 10), requires_grad=True)
        association = torch.full(
            (1, 2, 1), 0.01, requires_grad=True)
        target = dict(
            bboxes_prev=torch.tensor([[16., 32., 64., 32., 1.5707]]),
            bboxes_curr=torch.tensor([[18., 32., 64., 32., 1.5707]]),
            labels=torch.tensor([3]),
            track_ids=torch.tensor([7]),
            valid_prev=torch.tensor([True]),
            valid_curr=torch.tensor([True]),
        )
        losses = RotatedCTrackerLoss(8)(
            classification, regression, association, anchors, [target])
        total = sum(losses.values())
        self.assertTrue(torch.isfinite(total))
        total.backward()
        self.assertTrue(torch.isfinite(classification.grad).all())
        self.assertTrue(torch.isfinite(regression.grad).all())
        self.assertTrue(torch.isfinite(association.grad).all())


class TestLegacyCTrackerPretrain(unittest.TestCase):
    def test_original_r50_adaptation(self):
        path = ('/data4/litianhao/PairMmot/pretrained_weights/'
                'ctracker_model_final.pt')
        if not os.path.isfile(path):
            self.skipTest(f'Legacy CTracker checkpoint unavailable: {path}')
        network = model.resnet50(
            num_classes=8, num_spectral=8,
            use_3d_se_stem=True, rotated=True)
        report = model.load_legacy_ctracker(network, path)
        self.assertGreaterEqual(report['copied'], 300)
        stem = network.conv1.conv3d.weight.detach()
        self.assertEqual(stem.shape, (64, 1, 3, 7, 7))
        cls = network.classificationModel.output.weight.detach()
        torch.testing.assert_close(cls, cls[:1].expand_as(cls))
        reg = network.regressionModel.output.weight.detach()
        self.assertEqual(reg.shape[0], 10)
        self.assertTrue(torch.count_nonzero(reg[4]) == 0)
        self.assertTrue(torch.count_nonzero(reg[9]) == 0)


if __name__ == '__main__':
    unittest.main()
