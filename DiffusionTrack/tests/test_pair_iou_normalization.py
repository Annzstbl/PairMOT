import pathlib
import sys

import torch


ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from diffusion.models.diffusion_losses import SetCriterionDynamicK


def make_criterion():
    return SetCriterionDynamicK(
        num_classes=1,
        matcher=None,
        weight_dict={},
        eos_coef=0.1,
        losses=[],
        use_focal=True,
        use_fed_loss=False,
    )


def test_pair_iou_loss_keeps_historical_sum_by_default():
    criterion = make_criterion()
    result = criterion.normalize_pair_iou_sum(torch.tensor(6.0), matched=3)
    torch.testing.assert_close(result, torch.tensor(2.0))


def test_pair_iou_loss_can_match_lx_frame_average():
    criterion = make_criterion()
    criterion.average_pair_iou_loss = True
    result = criterion.normalize_pair_iou_sum(torch.tensor(6.0), matched=3)
    torch.testing.assert_close(result, torch.tensor(1.0))
