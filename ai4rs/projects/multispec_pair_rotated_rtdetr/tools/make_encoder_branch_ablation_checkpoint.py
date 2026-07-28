#!/usr/bin/env python3
"""Create an analysis-only checkpoint with selected encoder gates disabled."""

from __future__ import annotations

import argparse
import copy
import os
from pathlib import Path
from typing import MutableMapping

import torch
from torch import Tensor


MODES = ('no_common', 'no_detail', 'no_p4_common', 'no_post')
GAMMA_SUFFIX = 'encoder.post_pair_temporal_adapter.gamma'


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--mode', required=True, choices=MODES)
    return parser.parse_args()


def find_gamma_key(state_dict: MutableMapping[str, Tensor]) -> str:
    matches = [key for key in state_dict if key.endswith(GAMMA_SUFFIX)]
    if len(matches) != 1:
        raise KeyError(
            f'expected one key ending in {GAMMA_SUFFIX!r}, got {matches}')
    return matches[0]


def ablate_gamma(gamma: Tensor, mode: str) -> Tensor:
    if gamma.ndim != 2 or gamma.shape[1] != 2:
        raise ValueError(
            f'dual-evidence gamma must have shape [levels, 2], got '
            f'{tuple(gamma.shape)}')
    if gamma.shape[0] < 2:
        raise ValueError('no_p4_common requires at least two pyramid levels')
    output = gamma.detach().clone()
    if mode == 'no_common':
        output[:, 0] = 0
    elif mode == 'no_detail':
        output[:, 1] = 0
    elif mode == 'no_p4_common':
        output[1, 0] = 0
    elif mode == 'no_post':
        output.zero_()
    else:
        raise ValueError(f'unsupported mode: {mode}')
    return output


def main() -> None:
    args = parse_args()
    input_path = args.input.resolve()
    output_path = args.output.resolve()
    if output_path.exists():
        raise FileExistsError(f'refusing to overwrite {output_path}')
    checkpoint = torch.load(input_path, map_location='cpu')
    if not isinstance(checkpoint, dict) or 'state_dict' not in checkpoint:
        raise ValueError(f'{input_path} is not an MMEngine checkpoint')

    state_dict = copy.deepcopy(checkpoint['state_dict'])
    gamma_key = find_gamma_key(state_dict)
    before = state_dict[gamma_key].detach().clone()
    after = ablate_gamma(before, args.mode)
    state_dict[gamma_key] = after

    meta = copy.deepcopy(checkpoint.get('meta', {}))
    meta['analysis_only_encoder_ablation'] = {
        'mode': args.mode,
        'source_checkpoint': str(input_path),
        'gamma_key': gamma_key,
        'gamma_before': before.tolist(),
        'gamma_after': after.tolist(),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + '.tmp')
    torch.save({'meta': meta, 'state_dict': state_dict}, temporary)
    os.replace(temporary, output_path)
    print(f'mode={args.mode}')
    print(f'gamma_key={gamma_key}')
    print(f'before={before.tolist()}')
    print(f'after={after.tolist()}')
    print(f'output={output_path}')


if __name__ == '__main__':
    main()
