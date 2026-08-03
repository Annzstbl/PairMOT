"""Assert that every floating tensor in a checkpoint is finite."""
from __future__ import annotations

import sys

import torch


checkpoint = torch.load(sys.argv[1], map_location='cpu', weights_only=False)
state = checkpoint.get('state_dict', checkpoint)
floating = {
    name: tensor for name, tensor in state.items()
    if torch.is_tensor(tensor) and tensor.is_floating_point()
}
assert floating, 'checkpoint contains no floating tensors'
nonfinite = [
    name for name, tensor in floating.items()
    if not torch.isfinite(tensor).all()
]
assert not nonfinite, f'non-finite tensors: {nonfinite[:10]}'
print(f'{len(floating)} floating checkpoint tensors are finite')
