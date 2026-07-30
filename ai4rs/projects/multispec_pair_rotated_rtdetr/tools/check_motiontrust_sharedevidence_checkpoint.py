"""Verify both 0730_12 structural paths receive finite updates."""
from __future__ import annotations

import sys

import torch


checkpoint = torch.load(sys.argv[1], map_location='cpu')
state = checkpoint.get('state_dict', checkpoint)
groups = {}
for name in ('motion_trust_adapters', 'shared_evidence_adapters'):
    weights = {
        key: value
        for key, value in state.items()
        if f'decoder.{name}' in key and key.endswith('.weight')
    }
    assert len(weights) == 3, (name, sorted(weights))
    maxima = {key: float(value.abs().max()) for key, value in weights.items()}
    assert all(torch.isfinite(value).all() for value in weights.values()), name
    assert all(value > 0.0 for value in maxima.values()), (name, maxima)
    groups[name] = maxima
print('MOTIONTRUST_SHAREDEVIDENCE_CHECKPOINT_OK', groups)
