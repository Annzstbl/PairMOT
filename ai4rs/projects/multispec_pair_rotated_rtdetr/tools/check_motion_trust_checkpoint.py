import sys

import torch


checkpoint = torch.load(sys.argv[1], map_location='cpu')
state = checkpoint['state_dict']
weights = {
    key: value
    for key, value in state.items()
    if 'decoder.motion_trust_adapters' in key
}
assert len(weights) == 3, sorted(weights)
maxima = {key: value.abs().max().item() for key, value in weights.items()}
assert all(value > 0.0 for value in maxima.values()), maxima
assert all(torch.isfinite(value).all() for value in weights.values())
print('MOTION_TRUST_CHECKPOINT_OK', maxima)
