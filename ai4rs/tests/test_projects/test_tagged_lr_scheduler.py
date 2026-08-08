import unittest

import torch

from projects.multispec_pair_rotated_rtdetr.multispec_pair_rotated_rtdetr import (
    TaggedMultiStepLR,
)


class TaggedMultiStepLRTest(unittest.TestCase):

    def test_only_tagged_group_changes_at_milestone(self):
        tagged = torch.nn.Parameter(torch.ones(()))
        untagged = torch.nn.Parameter(torch.ones(()))
        optimizer = torch.optim.AdamW([
            dict(params=[tagged], lr=1e-4, decoder_head_delayed_lr=True),
            dict(params=[untagged], lr=1e-4),
        ])
        scheduler = TaggedMultiStepLR(
            optimizer,
            milestones=[2],
            gamma=1.4,
            tag_key='decoder_head_delayed_lr')

        self.assertEqual([group['lr'] for group in optimizer.param_groups],
                         [1e-4, 1e-4])
        optimizer.step()
        scheduler.step()
        self.assertEqual([group['lr'] for group in optimizer.param_groups],
                         [1e-4, 1e-4])
        optimizer.step()
        scheduler.step()
        self.assertAlmostEqual(optimizer.param_groups[0]['lr'], 1.4e-4)
        self.assertAlmostEqual(optimizer.param_groups[1]['lr'], 1e-4)

    def test_rejects_empty_tag_key(self):
        parameter = torch.nn.Parameter(torch.ones(()))
        optimizer = torch.optim.AdamW([parameter], lr=1e-4)
        with self.assertRaisesRegex(ValueError, 'nonempty'):
            TaggedMultiStepLR(
                optimizer, milestones=[1], gamma=1.4, tag_key='')


if __name__ == '__main__':
    unittest.main()
