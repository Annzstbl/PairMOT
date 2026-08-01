"""Verify the trained pair-common terminal classifier residual."""
from __future__ import annotations

import sys

import torch


checkpoint = torch.load(sys.argv[1], map_location='cpu', weights_only=False)
state = checkpoint.get('state_dict', checkpoint)
prefix = 'bbox_head.terminal_pair_common_cls_residual_branch'
weight = state[f'{prefix}.weight']
bias = state[f'{prefix}.bias']
assert torch.isfinite(weight).all() and torch.isfinite(bias).all()
assert float(weight.abs().max()) > 0.0
assert float(bias.abs().max()) > 0.0
assert tuple(weight.shape) == (8, 256)
assert tuple(bias.shape) == (8, )
print('terminal pair-common classification residual is finite and trained')

