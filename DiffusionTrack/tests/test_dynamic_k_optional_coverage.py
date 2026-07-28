import torch

from diffusion.models.diffusion_losses import HungarianMatcherDynamicK


def _conflicting_inputs():
    # Both GTs select query 0 when Dynamic-K=1. Conflict resolution keeps it
    # for GT 0 because its cost is lower, leaving GT 1 uncovered.
    cost = torch.tensor([
        [0.1, 0.2],
        [5.0, 6.0],
    ])
    pair_iou = torch.tensor([
        [0.9, 0.9],
        [0.0, 0.0],
    ])
    return cost, pair_iou


def test_dynamic_k_can_leave_gt_uncovered_when_requested():
    matcher = HungarianMatcherDynamicK()
    matcher.force_gt_coverage = False
    (selected, gt_indices), best_query = matcher.dynamic_k_matching(
        *_conflicting_inputs(), num_gt=2)

    torch.testing.assert_close(selected, torch.tensor([True, False]))
    torch.testing.assert_close(gt_indices, torch.tensor([0]))
    torch.testing.assert_close(best_query, torch.tensor([0, -1]))


def test_dynamic_k_repaired_default_still_covers_every_gt():
    matcher = HungarianMatcherDynamicK()
    (selected, gt_indices), best_query = matcher.dynamic_k_matching(
        *_conflicting_inputs(), num_gt=2)

    assert selected.sum().item() == 2
    assert sorted(gt_indices.tolist()) == [0, 1]
    assert (best_query >= 0).all()


def test_dynamic_k_lx_stale_mode_can_collapse_repaired_gts():
    # All GTs initially choose q0. After the first conflict pass, GT1/GT2
    # independently choose q1. LX then reuses the old q0 conflict mask, so q1
    # remains multi-assigned and only one of GT1/GT2 survives in the returned
    # query->GT view.
    cost = torch.tensor([
        [0.1, 0.2, 0.3],
        [5.0, 5.1, 5.2],
        [9.0, 9.1, 9.2],
    ])
    pair_iou = torch.tensor([
        [0.9, 0.9, 0.9],
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
    ])
    matcher = HungarianMatcherDynamicK()
    matcher.coverage_mode = "lx_stale"
    (selected, gt_indices), best_query = matcher.dynamic_k_matching(
        cost, pair_iou, num_gt=3)

    assert selected.sum().item() == 2
    assert len(set(gt_indices.tolist())) == 2
    assert best_query.tolist() == [0, 1, 1]
