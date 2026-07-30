"""Verify shared routing and independent projection invariants."""
from __future__ import annotations

import sys

import torch


checkpoint = torch.load(sys.argv[1], map_location='cpu')
state = checkpoint.get('state_dict', checkpoint)
allow_equal_projections = '--allow-equal-projections' in sys.argv[2:]

routing_errors = {}
projection_differences = {}
for key, value in state.items():
    marker = '.cross_attn_prev.'
    if marker not in key:
        continue
    curr_key = key.replace(marker, '.cross_attn_curr.')
    if curr_key not in state:
        continue
    suffix = key.split(marker, 1)[1]
    difference = float((value - state[curr_key]).abs().max())
    if suffix.startswith(('sampling_offsets.', 'attention_weights.')):
        routing_errors[key] = difference
    elif suffix.startswith(('value_proj.', 'output_proj.')):
        projection_differences[key] = difference

assert routing_errors, 'No shared routing parameter pairs found'
assert projection_differences, 'No frame-specific projection pairs found'
assert max(routing_errors.values()) <= 1e-7, routing_errors
if not allow_equal_projections:
    assert max(projection_differences.values()) > 1e-7, (
        'Frame-specific projections did not diverge', projection_differences)
for key, value in state.items():
    if '.cross_attn_' in key:
        assert torch.isfinite(value).all(), key
print('SHARED_ROUTING_CHECKPOINT_OK', {
    'routing_pairs': len(routing_errors),
    'max_routing_error': max(routing_errors.values()),
    'projection_pairs': len(projection_differences),
    'max_projection_difference': max(projection_differences.values()),
})
