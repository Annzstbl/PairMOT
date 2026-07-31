"""Verify independent attention and learned diagonal factorized gates."""
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

gate_keys = (
    'decoder.terminal_common_evidence_bypass_gates.0',
    'decoder.terminal_enveloped_detail_gates.0',
)
gate_maxima = {}
for key in gate_keys:
    value = state[key]
    assert tuple(value.shape) == (256,), (key, tuple(value.shape))
    assert torch.isfinite(value).all(), key
    maximum = float(value.abs().max())
    assert maximum > 0.0, (key, maximum)
    gate_maxima[key] = maximum

print('TERMINAL_DIAGONAL_FACTORIZED_EVIDENCE_CHECKPOINT_OK', {
    'attention_pairs': len(attention_pairs),
    'max_attention_difference': max_attention_difference,
    'gate_numel': sum(state[key].numel() for key in gate_keys),
    'gate_maxima': gate_maxima,
})
