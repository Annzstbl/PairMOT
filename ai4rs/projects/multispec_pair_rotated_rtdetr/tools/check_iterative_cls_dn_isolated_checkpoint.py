"""Verify trained residual and DN-absolute classification branches."""
from __future__ import annotations

import sys

import torch


checkpoint = torch.load(sys.argv[1], map_location='cpu', weights_only=False)
state = checkpoint.get('state_dict', checkpoint)

for side in ('prev', 'curr'):
    for layer in range(3):
        prefix = f'bbox_head.iterative_cls_residual_branches_{side}.{layer}'
        weight = state[f'{prefix}.weight']
        bias = state[f'{prefix}.bias']
        assert torch.isfinite(weight).all() and torch.isfinite(bias).all()
        assert float(weight.abs().max()) > 0.0
        assert float(bias.abs().max()) > 0.0

for layer in range(3):
    prev_weight = state[f'bbox_head.cls_branches.{layer}.weight']
    curr_weight = state[f'bbox_head.cls_branches_curr.{layer}.weight']
    prev_bias = state[f'bbox_head.cls_branches.{layer}.bias']
    curr_bias = state[f'bbox_head.cls_branches_curr.{layer}.bias']
    for tensor in (prev_weight, curr_weight, prev_bias, curr_bias):
        assert torch.isfinite(tensor).all()
    assert max(
        float((prev_weight - curr_weight).abs().max()),
        float((prev_bias - curr_bias).abs().max())) > 0.0

print('iterative classification residual and DN absolute heads are finite '
      'and trained')
