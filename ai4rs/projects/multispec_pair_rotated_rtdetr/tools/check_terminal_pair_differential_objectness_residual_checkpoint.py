"""Verify the trained 0801_13 differential objectness residual."""
from __future__ import annotations

import sys

import torch


checkpoint = torch.load(sys.argv[1], map_location='cpu', weights_only=False)
state = checkpoint.get('state_dict', checkpoint)
prefix = 'bbox_head.terminal_pair_differential_objectness_residual_branch'
weight = state[f'{prefix}.weight']
bias = state[f'{prefix}.bias']
assert torch.isfinite(weight).all() and torch.isfinite(bias).all()
assert float(weight.abs().max()) > 0.0
assert float(bias.abs().max()) > 0.0
assert tuple(weight.shape) == (1, 256)
assert tuple(bias.shape) == (1, )
print('terminal pair-differential objectness residual is finite and trained')
