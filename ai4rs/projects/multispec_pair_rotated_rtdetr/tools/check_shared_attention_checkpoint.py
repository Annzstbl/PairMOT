"""Verify shared attention and frame-specific localization invariants."""
from __future__ import annotations

import sys

import torch


try:
    checkpoint = torch.load(
        sys.argv[1], map_location='cpu', weights_only=False)
except TypeError:
    checkpoint = torch.load(sys.argv[1], map_location='cpu')
state = checkpoint.get('state_dict', checkpoint)
allow_equal_independent = '--allow-equal-independent' in sys.argv[2:]

attention_errors = {}
independent_differences = {}
for key, value in state.items():
    marker = '.cross_attn_prev.'
    if marker not in key:
        continue
    curr_key = key.replace(marker, '.cross_attn_curr.')
    if curr_key not in state:
        continue
    suffix = key.split(marker, 1)[1]
    difference = float((value - state[curr_key]).abs().max())
    if suffix.startswith('attention_weights.'):
        attention_errors[key] = difference
    elif suffix.startswith(
            ('sampling_offsets.', 'value_proj.', 'output_proj.')):
        independent_differences[key] = difference

assert attention_errors, 'No shared attention parameter pairs found'
assert independent_differences, 'No frame-specific parameter pairs found'
assert max(attention_errors.values()) <= 1e-7, attention_errors
if not allow_equal_independent:
    assert max(independent_differences.values()) > 1e-7, (
        'Frame-specific parameters did not diverge', independent_differences)
for key, value in state.items():
    if '.cross_attn_' in key:
        assert torch.isfinite(value).all(), key
print('SHARED_ATTENTION_CHECKPOINT_OK', {
    'attention_pairs': len(attention_errors),
    'max_attention_error': max(attention_errors.values()),
    'independent_pairs': len(independent_differences),
    'max_independent_difference': max(independent_differences.values()),
})
