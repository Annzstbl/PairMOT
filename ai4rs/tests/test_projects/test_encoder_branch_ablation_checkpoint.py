import unittest

import torch

from projects.multispec_pair_rotated_rtdetr.tools \
    .make_encoder_branch_ablation_checkpoint import ablate_gamma


class TestEncoderBranchAblationCheckpoint(unittest.TestCase):

    def setUp(self):
        self.gamma = torch.tensor([
            [1.0, 2.0],
            [3.0, 4.0],
            [5.0, 6.0],
        ])

    def test_no_common(self):
        expected = torch.tensor([
            [0.0, 2.0],
            [0.0, 4.0],
            [0.0, 6.0],
        ])
        self.assertTrue(torch.equal(
            ablate_gamma(self.gamma, 'no_common'), expected))

    def test_no_detail(self):
        expected = torch.tensor([
            [1.0, 0.0],
            [3.0, 0.0],
            [5.0, 0.0],
        ])
        self.assertTrue(torch.equal(
            ablate_gamma(self.gamma, 'no_detail'), expected))

    def test_no_p4_common(self):
        expected = self.gamma.clone()
        expected[1, 0] = 0
        self.assertTrue(torch.equal(
            ablate_gamma(self.gamma, 'no_p4_common'), expected))

    def test_no_post(self):
        self.assertTrue(torch.equal(
            ablate_gamma(self.gamma, 'no_post'),
            torch.zeros_like(self.gamma)))

    def test_input_is_not_mutated(self):
        before = self.gamma.clone()
        ablate_gamma(self.gamma, 'no_common')
        self.assertTrue(torch.equal(self.gamma, before))


if __name__ == '__main__':
    unittest.main()
