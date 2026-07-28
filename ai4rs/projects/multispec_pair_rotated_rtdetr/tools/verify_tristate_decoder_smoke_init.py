#!/usr/bin/env python
"""Verify that a short tri-state smoke started from structural initialization."""

import argparse
from collections import OrderedDict
from typing import Dict

import torch
from torch import Tensor


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('checkpoint')
    parser.add_argument('--max-coupling', type=float, default=0.02)
    parser.add_argument('--max-identity-error', type=float, default=0.05)
    return parser.parse_args()


def _state_dict(checkpoint_path: str) -> Dict[str, Tensor]:
    checkpoint = torch.load(
        checkpoint_path, map_location='cpu', weights_only=False)
    state = checkpoint.get('state_dict', checkpoint)
    return OrderedDict(
        (key.removeprefix('module.'), value)
        for key, value in state.items()
        if isinstance(value, Tensor))


def _find(state: Dict[str, Tensor], suffix: str) -> Tensor:
    matches = [value for key, value in state.items() if key.endswith(suffix)]
    if len(matches) != 1:
        raise RuntimeError(
            f'Expected one tensor ending in {suffix!r}, got {len(matches)}')
    return matches[0].float()


def _average_fusion_expected(weight: Tensor) -> Tensor:
    width = weight.shape[0]
    expected = torch.zeros_like(weight)
    identity = torch.eye(width, dtype=weight.dtype)
    expected[:, :width] = 0.5 * identity
    expected[:, width:2 * width] = 0.5 * identity
    return expected


def main() -> None:
    args = _parse_args()
    state = _state_dict(args.checkpoint)

    query_names = ('query_to_prev', 'query_to_curr', 'query_to_pointer')
    for name in query_names:
        weight = _find(state, f'decoder.{name}.weight')
        bias = _find(state, f'decoder.{name}.bias')
        identity = torch.eye(weight.shape[0], dtype=weight.dtype)
        error = (weight - identity).abs().max().item()
        bias_max = bias.abs().max().item()
        print(f'{name}: identity_max_error={error:.8f} bias_max={bias_max:.8f}')
        if not torch.isfinite(weight).all() or not torch.isfinite(bias).all():
            raise RuntimeError(f'{name} contains a non-finite value')
        if max(error, bias_max) >= args.max_identity_error:
            raise RuntimeError(
                f'{name} does not retain the fixed identity initialization')

    pair_pos = _find(state, 'decoder.pair_pos_fusion.weight')
    pair_pos_bias = _find(state, 'decoder.pair_pos_fusion.bias')
    pair_pos_error = (
        pair_pos - _average_fusion_expected(pair_pos)).abs().max().item()
    pair_pos_bias_max = pair_pos_bias.abs().max().item()
    print(
        'pair_pos_fusion: '
        f'max_error={pair_pos_error:.8f} bias_max={pair_pos_bias_max:.8f}')
    if max(pair_pos_error, pair_pos_bias_max) >= args.max_identity_error:
        raise RuntimeError('pair_pos_fusion did not start as pair averaging')

    cross_fusions = [
        (key, value.float()) for key, value in state.items()
        if 'decoder.layers.' in key and key.endswith('cross_fusion.weight')
    ]
    cross_fusion_errors = [
        (weight - _average_fusion_expected(weight)).abs().max().item()
        for _, weight in cross_fusions
    ]
    if not cross_fusion_errors:
        raise RuntimeError('No decoder cross_fusion tensors found')
    cross_fusion_error = max(cross_fusion_errors)
    print(
        f'cross_fusion_tensors={len(cross_fusions)} '
        f'global_max_error={cross_fusion_error:.8f}')
    if cross_fusion_error >= args.max_identity_error:
        raise RuntimeError('cross_fusion did not retain pair averaging')

    fusion = _find(state, 'decoder.pointer_init_fusion.weight')
    fusion_bias = _find(state, 'decoder.pointer_init_fusion.bias')
    width = fusion.shape[0]
    fusion_expected = torch.zeros_like(fusion)
    fusion_expected[:, :width] = torch.eye(width, dtype=fusion.dtype)
    fusion_error = (fusion - fusion_expected).abs().max().item()
    fusion_bias_max = fusion_bias.abs().max().item()
    print(
        'pointer_init_fusion: '
        f'max_error={fusion_error:.8f} bias_max={fusion_bias_max:.8f}')
    if max(fusion_error, fusion_bias_max) >= args.max_identity_error:
        raise RuntimeError('pointer_init_fusion did not start structurally')

    coupling_suffixes = (
        'pointer_to_prev.weight',
        'pointer_to_prev.bias',
        'pointer_to_curr.weight',
        'pointer_to_curr.bias',
        'pointer_update.weight',
        'pointer_update.bias',
    )
    coupling_tensors = [
        (key, value.float()) for key, value in state.items()
        if 'decoder.layers.' in key
        and key.endswith(coupling_suffixes)
    ]
    if not coupling_tensors:
        raise RuntimeError('No tri-state coupling tensors found')
    coupling_max = max(
        tensor.abs().max().item() for _, tensor in coupling_tensors)
    print(
        f'coupling_tensors={len(coupling_tensors)} '
        f'global_max_abs={coupling_max:.8f}')
    if not all(torch.isfinite(tensor).all() for _, tensor in coupling_tensors):
        raise RuntimeError('A coupling tensor contains a non-finite value')
    if coupling_max >= args.max_coupling:
        raise RuntimeError(
            'Coupling scale is incompatible with four updates from zero')

    print('TRISTATE_SMOKE_INIT_OK')


if __name__ == '__main__':
    main()
