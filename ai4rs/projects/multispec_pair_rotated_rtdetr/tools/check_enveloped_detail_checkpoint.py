"""Verify the three bounded frame-detail gates receive finite updates."""
from __future__ import annotations

import sys

import torch


checkpoint = torch.load(
    sys.argv[1], map_location='cpu', weights_only=False)
state = checkpoint.get('state_dict', checkpoint)
weights = {
    key: value
    for key, value in state.items()
    if ('decoder.enveloped_detail_gates' in key
        and key.endswith('.weight'))
}
assert len(weights) == 3, sorted(weights)
maxima = {key: float(value.abs().max()) for key, value in weights.items()}
assert all(torch.isfinite(value).all() for value in weights.values()), maxima
assert all(value > 0.0 for value in maxima.values()), maxima
print('ENVELOPED_DETAIL_CHECKPOINT_OK', maxima)
