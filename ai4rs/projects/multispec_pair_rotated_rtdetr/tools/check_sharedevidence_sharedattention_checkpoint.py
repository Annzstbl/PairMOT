import sys

import torch


try:
    checkpoint = torch.load(
        sys.argv[1], map_location='cpu', weights_only=False)
except TypeError:
    checkpoint = torch.load(sys.argv[1], map_location='cpu')
state = checkpoint['state_dict']

evidence = {
    key: value
    for key, value in state.items()
    if 'decoder.shared_evidence_adapters' in key
}
assert len(evidence) == 3, sorted(evidence)
evidence_maxima = {
    key: value.abs().max().item() for key, value in evidence.items()
}
assert all(value > 0.0 for value in evidence_maxima.values()), evidence_maxima
assert all(torch.isfinite(value).all() for value in evidence.values())

attention_errors = []
independent_differences = []
for layer in range(3):
    prefix = f'decoder.layers.{layer}.'
    for suffix in ('attention_weights.weight', 'attention_weights.bias'):
        prev = state[prefix + 'cross_attn_prev.' + suffix]
        curr = state[prefix + 'cross_attn_curr.' + suffix]
        attention_errors.append((prev - curr).abs().max().item())
    for module in ('sampling_offsets', 'value_proj', 'output_proj'):
        for suffix in ('weight', 'bias'):
            prev = state[prefix + f'cross_attn_prev.{module}.{suffix}']
            curr = state[prefix + f'cross_attn_curr.{module}.{suffix}']
            independent_differences.append((prev - curr).abs().max().item())

assert max(attention_errors) == 0.0, attention_errors
assert max(independent_differences) > 1e-7, independent_differences
print(
    'SHAREDEVIDENCE_SHAREDATTENTION_CHECKPOINT_OK',
    {
        'evidence_maxima': evidence_maxima,
        'attention_pairs': len(attention_errors),
        'max_attention_error': max(attention_errors),
        'independent_pairs': len(independent_differences),
        'max_independent_difference': max(independent_differences),
    })
