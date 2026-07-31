"""Verify independent attention and the sole diagonal detail-only gate."""
from __future__ import annotations

import sys

import torch


checkpoint = torch.load(sys.argv[1], map_location='cpu', weights_only=False)
state = checkpoint.get('state_dict', checkpoint)
attention_pairs = []
for key, value in state.items():
    marker = '.cross_attn_prev.attention_weights.'
    if marker in key:
        curr_key = key.replace(
            marker, '.cross_attn_curr.attention_weights.')
        if curr_key in state:
            attention_pairs.append((value, state[curr_key]))

assert len(attention_pairs) == 6, len(attention_pairs)
max_attention_difference = max(
    float((prev - curr).abs().max()) for prev, curr in attention_pairs)
assert max_attention_difference > 0.0, max_attention_difference

detail_weights = {
    key: value
    for key, value in state.items()
    if ('decoder.terminal_enveloped_detail_gates' in key
        and not key.endswith('.weight'))
}
common_weights = {
    key: value
    for key, value in state.items()
    if 'decoder.terminal_common_evidence_bypass_gates' in key
}
assert len(detail_weights) == 1, sorted(detail_weights)
assert not common_weights, sorted(common_weights)
detail_weight = next(iter(detail_weights.values()))
assert detail_weight.ndim == 1, detail_weight.shape
assert detail_weight.numel() == 256, detail_weight.shape
assert torch.isfinite(detail_weight).all()
detail_maximum = float(detail_weight.abs().max())
assert detail_maximum > 0.0, detail_maximum

print('TERMINAL_DIAGONAL_DETAIL_ONLY_CHECKPOINT_OK', {
    'attention_pairs': len(attention_pairs),
    'max_attention_difference': max_attention_difference,
    'detail_gate_shape': tuple(detail_weight.shape),
    'detail_gate_maximum': detail_maximum,
})
