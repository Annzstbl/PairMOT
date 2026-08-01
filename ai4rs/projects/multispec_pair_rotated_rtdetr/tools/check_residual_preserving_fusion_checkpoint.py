"""Verify independent frame attention and trained finite fusion tensors."""
from __future__ import annotations

import sys

import torch


checkpoint = torch.load(sys.argv[1], map_location='cpu', weights_only=False)
state = checkpoint.get('state_dict', checkpoint)
independent_pairs = []
for key, value in state.items():
    for component in (
            'attention_weights', 'sampling_offsets', 'value_proj',
            'output_proj'):
        marker = f'.cross_attn_prev.{component}.'
        if marker in key:
            curr_key = key.replace(marker, f'.cross_attn_curr.{component}.')
            if curr_key in state:
                independent_pairs.append((value, state[curr_key]))

assert len(independent_pairs) == 24, len(independent_pairs)
assert all(
    torch.isfinite(prev).all() and torch.isfinite(curr).all()
    for prev, curr in independent_pairs)
max_independent_difference = max(
    float((prev - curr).abs().max()) for prev, curr in independent_pairs)
assert max_independent_difference > 0.0, max_independent_difference

fusion_tensors = {
    key: value for key, value in state.items() if '.cross_fusion.' in key
}
assert len(fusion_tensors) == 6, sorted(fusion_tensors)
assert all(torch.isfinite(value).all() for value in fusion_tensors.values())

print('RESIDUAL_PRESERVING_FUSION_CHECKPOINT_OK', {
    'independent_attention_pairs': len(independent_pairs),
    'max_independent_difference': max_independent_difference,
    'fusion_tensors': len(fusion_tensors),
})
