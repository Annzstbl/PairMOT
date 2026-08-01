"""Verify finite, trained iterative classification residual projections."""
from __future__ import annotations

import sys

import torch


checkpoint = torch.load(sys.argv[1], map_location='cpu', weights_only=False)
state = checkpoint.get('state_dict', checkpoint)
prefixes = (
    'bbox_head.iterative_cls_residual_branches_prev.',
    'bbox_head.iterative_cls_residual_branches_curr.',
)
residual_tensors = {
    key: value for key, value in state.items()
    if key.startswith(prefixes)
}
assert len(residual_tensors) == 12, sorted(residual_tensors)
assert all(torch.isfinite(value).all()
           for value in residual_tensors.values())
for side in ('prev', 'curr'):
    for layer in range(3):
        weight = state[
            f'bbox_head.iterative_cls_residual_branches_{side}.'
            f'{layer}.weight']
        bias = state[
            f'bbox_head.iterative_cls_residual_branches_{side}.'
            f'{layer}.bias']
        assert float(weight.abs().max()) > 0.0
        assert float(bias.abs().max()) > 0.0

print('ITERATIVE_CLS_RESIDUAL_CHECKPOINT_OK', {
    'tensors': len(residual_tensors),
    'max_abs': max(float(value.abs().max())
                   for value in residual_tensors.values()),
})
