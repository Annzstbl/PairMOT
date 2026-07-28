#!/usr/bin/env python
"""Verify a short dual-output smoke learned from its exact zero start."""

import argparse
from collections import OrderedDict

import torch


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('checkpoint')
    parser.add_argument('--max-adapter', type=float, default=0.02)
    parser.add_argument('--max-fusion-error', type=float, default=0.05)
    return parser.parse_args()


def _average_expected(weight):
    width = weight.shape[0]
    expected = torch.zeros_like(weight)
    identity = torch.eye(width, dtype=weight.dtype)
    expected[:, :width] = 0.5 * identity
    expected[:, width:2 * width] = 0.5 * identity
    return expected


def main():
    args = _parse_args()
    checkpoint = torch.load(
        args.checkpoint, map_location='cpu', weights_only=False)
    raw_state = checkpoint.get('state_dict', checkpoint)
    state = OrderedDict(
        (key.removeprefix('module.'), value.float())
        for key, value in raw_state.items()
        if isinstance(value, torch.Tensor))

    adapter_weights = [
        (key, value) for key, value in state.items()
        if ('decoder.dual_output_prev_adapters.' in key
            or 'decoder.dual_output_curr_adapters.' in key)
        and key.endswith('.weight')
    ]
    if len(adapter_weights) != 6:
        raise RuntimeError(
            f'Expected six dual-output adapter weights, '
            f'got {len(adapter_weights)}')
    maxima = [value.abs().max().item() for _, value in adapter_weights]
    print('adapter_maxima', ' '.join(f'{value:.8f}' for value in maxima))
    if not all(torch.isfinite(value).all() for _, value in adapter_weights):
        raise RuntimeError('A dual-output adapter contains non-finite values')
    if min(maxima) <= 0.0:
        raise RuntimeError('At least one dual-output adapter did not learn')
    if max(maxima) >= args.max_adapter:
        raise RuntimeError(
            'Adapter scale is incompatible with four updates from zero')

    fusion_weights = [
        (key, value) for key, value in state.items()
        if (key.endswith('decoder.pair_pos_fusion.weight')
            or ('decoder.layers.' in key
                and key.endswith('cross_fusion.weight')))
    ]
    if len(fusion_weights) != 4:
        raise RuntimeError(
            f'Expected four pair fusion weights, got {len(fusion_weights)}')
    fusion_error = max(
        (value - _average_expected(value)).abs().max().item()
        for _, value in fusion_weights)
    print(f'fusion_global_max_error={fusion_error:.8f}')
    if fusion_error >= args.max_fusion_error:
        raise RuntimeError('Pair-average fusion initialization was not kept')

    tristate_keys = [
        key for key in state
        if 'decoder.query_to_prev' in key
        or 'decoder.pointer_to_prev' in key
    ]
    if tristate_keys:
        raise RuntimeError('Unexpected tri-state parameters in residual smoke')
    print('DUAL_OUTPUT_ADAPTER_SMOKE_OK')


if __name__ == '__main__':
    main()
