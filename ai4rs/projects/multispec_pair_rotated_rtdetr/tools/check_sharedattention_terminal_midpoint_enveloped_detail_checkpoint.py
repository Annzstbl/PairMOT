"""Verify shared attention and the learned terminal-midpoint detail gate."""
from __future__ import annotations

import sys

import torch


checkpoint = torch.load(
    sys.argv[1], map_location='cpu', weights_only=False)
state = checkpoint.get('state_dict', checkpoint)
attention_pairs = []
independent_pairs = []
for key, value in state.items():
    if '.cross_attn_prev.attention_weights.' in key:
        curr_key = key.replace(
            '.cross_attn_prev.attention_weights.',
            '.cross_attn_curr.attention_weights.')
        if curr_key in state:
            attention_pairs.append((key, curr_key, value, state[curr_key]))
    for component in ('sampling_offsets', 'value_proj', 'output_proj'):
        marker = f'.cross_attn_prev.{component}.'
        if marker in key:
            curr_key = key.replace(
                marker, f'.cross_attn_curr.{component}.')
            if curr_key in state:
                independent_pairs.append(
                    (key, curr_key, value, state[curr_key]))

assert len(attention_pairs) == 6, len(attention_pairs)
assert len(independent_pairs) == 18, len(independent_pairs)
max_attention_error = max(
    float((prev - curr).abs().max())
    for _, _, prev, curr in attention_pairs)
max_independent_difference = max(
    float((prev - curr).abs().max())
    for _, _, prev, curr in independent_pairs)
assert max_attention_error == 0.0, max_attention_error
assert max_independent_difference > 0.0, max_independent_difference

weights = {
    key: value
    for key, value in state.items()
    if ('decoder.terminal_enveloped_detail_gates' in key
        and key.endswith('.weight'))
}
assert len(weights) == 1, sorted(weights)
maxima = {key: float(value.abs().max()) for key, value in weights.items()}
assert all(torch.isfinite(value).all() for value in weights.values()), maxima
assert all(value > 0.0 for value in maxima.values()), maxima
print('SHAREDATTENTION_TERMINAL_MIDPOINT_DETAIL_CHECKPOINT_OK', {
    'attention_pairs': len(attention_pairs),
    'max_attention_error': max_attention_error,
    'independent_pairs': len(independent_pairs),
    'max_independent_difference': max_independent_difference,
    'terminal_midpoint_detail_maxima': maxima,
})
