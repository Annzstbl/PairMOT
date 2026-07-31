"""Verify the learned terminal classification-only common-evidence gate."""
from __future__ import annotations

import sys

import torch


checkpoint = torch.load(
    sys.argv[1], map_location='cpu', weights_only=False)
state = checkpoint.get('state_dict', checkpoint)
weights = {
    key: value
    for key, value in state.items()
    if ('decoder.terminal_common_evidence_bypass_gates' in key
        and key.endswith('.weight'))
}
assert len(weights) == 1, sorted(weights)
assert all(torch.isfinite(value).all() for value in weights.values())
maxima = {key: float(value.abs().max()) for key, value in weights.items()}
assert all(value > 0.0 for value in maxima.values()), maxima
print('TERMINAL_CLASSIFICATION_COMMON_EVIDENCE_CHECKPOINT_OK', maxima)
