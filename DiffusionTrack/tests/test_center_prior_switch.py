import pathlib
import sys

import torch


ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from diffusion.models.diffusion_losses import HungarianMatcherDynamicK


def test_center_prior_is_off_by_default_for_reproducibility():
    matcher = HungarianMatcherDynamicK(2.0, 5.0, 2.0, True, False)
    bbox = torch.tensor([[1.0]])
    classification = torch.tensor([[0.20]])
    riou = torch.tensor([[0.05]])
    center = torch.tensor([[100.0]])
    assert torch.equal(
        matcher.compose_matching_cost(
            bbox, classification, riou, center),
        torch.tensor([[1.25]]))


def test_center_prior_can_be_activated_as_single_variable():
    matcher = HungarianMatcherDynamicK(2.0, 5.0, 2.0, True, False)
    matcher.center_prior_penalty_weight = 100.0
    matcher.apply_center_prior_penalty = True
    bbox = torch.tensor([[1.0]])
    classification = torch.tensor([[0.20]])
    riou = torch.tensor([[0.05]])
    center = torch.tensor([[100.0]])
    assert torch.equal(
        matcher.compose_matching_cost(
            bbox, classification, riou, center),
        torch.tensor([[101.25]]))
