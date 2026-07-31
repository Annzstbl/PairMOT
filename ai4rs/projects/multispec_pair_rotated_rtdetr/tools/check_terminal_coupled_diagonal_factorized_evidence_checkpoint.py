"""Verify independent attention and one learned coupled diagonal gate."""
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

gate_key = 'decoder.terminal_coupled_evidence_gate'
gate_keys = [key for key in state if 'terminal_' in key and 'gate' in key]
assert gate_keys == [gate_key], gate_keys
gate = state[gate_key]
assert tuple(gate.shape) == (256,), tuple(gate.shape)
assert gate.numel() == 256, gate.numel()
assert torch.isfinite(gate).all(), gate_key
gate_maximum = float(gate.abs().max())
assert gate_maximum > 0.0, gate_maximum

print('TERMINAL_COUPLED_DIAGONAL_FACTORIZED_EVIDENCE_CHECKPOINT_OK', {
    'attention_pairs': len(attention_pairs),
    'max_attention_difference': max_attention_difference,
    'gate_numel': gate.numel(),
    'gate_maximum': gate_maximum,
})
