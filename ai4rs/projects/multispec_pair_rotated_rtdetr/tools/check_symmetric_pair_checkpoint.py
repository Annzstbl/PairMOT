"""Verify the exact parameter tying invariants of 0730_10."""
from __future__ import annotations

import sys

import torch


checkpoint = torch.load(sys.argv[1], map_location='cpu')
state = checkpoint.get('state_dict', checkpoint)
allow_missing_fusion = '--allow-missing-fusion' in sys.argv[2:]

attention_errors = {}
for key, value in state.items():
    marker = '.cross_attn_prev.'
    if marker not in key:
        continue
    curr_key = key.replace(marker, '.cross_attn_curr.')
    if curr_key in state:
        attention_errors[key] = float(
            (value - state[curr_key]).abs().max())

fusion_errors = {}
for key, value in state.items():
    if not (
            key.endswith('decoder.pair_pos_fusion.weight')
            or key.endswith('.cross_fusion.weight')):
        continue
    half = value.shape[1] // 2
    fusion_errors[key] = float(
        (value[:, :half] - value[:, half:]).abs().max())

assert attention_errors, 'No prev/curr cross-attention parameter pairs found'
assert fusion_errors or allow_missing_fusion, (
    'No symmetric fusion weights found')
assert max(attention_errors.values()) <= 1e-7, attention_errors
assert not fusion_errors or max(fusion_errors.values()) <= 1e-7, fusion_errors
print('SYMMETRIC_PAIR_CHECKPOINT_OK', {
    'attention_pairs': len(attention_errors),
    'max_attention_error': max(attention_errors.values()),
    'fusion_matrices': len(fusion_errors),
    'max_fusion_error': max(fusion_errors.values()) if fusion_errors else None,
})
