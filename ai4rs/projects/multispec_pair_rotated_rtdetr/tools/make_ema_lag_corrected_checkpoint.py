#!/usr/bin/env python3
"""Interpolate a saved EMA checkpoint toward its same-step online weights.

MMEngine's EMAHook stores the averaged weights in ``state_dict`` and the
corresponding online weights in the ``module.*`` entries of
``ema_state_dict``.  This tool creates a deployment/evaluation checkpoint
using only those two same-iteration states.  It does not use a later
checkpoint or change the model graph.
"""

from __future__ import annotations

import argparse
import copy
import math
import os
from collections import OrderedDict
from pathlib import Path
from typing import Mapping

import torch
from torch import Tensor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument(
        '--online-fraction', type=float, required=True,
        help='0 keeps the saved EMA weights; 1 uses the online weights.')
    parser.add_argument(
        '--exclude-prefix', action='append', default=[],
        help=(
            'State-dict prefix to keep at the saved EMA value. May be '
            'repeated; useful for a single blockwise lag-correction '
            'ablation.'))
    return parser.parse_args()


def interpolate_state_dict(
        averaged_state: Mapping[str, Tensor],
        ema_state: Mapping[str, Tensor],
        online_fraction: float,
        exclude_prefixes: tuple[str, ...] = ()) \
        -> tuple[OrderedDict[str, Tensor], dict]:
    """Return EMA-to-online interpolation with strict key/finite checks."""
    if not math.isfinite(online_fraction) or not 0 <= online_fraction <= 1:
        raise ValueError('online_fraction must be finite and in [0, 1]')
    if any(not prefix for prefix in exclude_prefixes):
        raise ValueError('exclude prefixes must be non-empty')

    output: OrderedDict[str, Tensor] = OrderedDict()
    floating = 0
    nonfloating = 0
    interpolated_floating = 0
    excluded_floating = 0
    squared_delta = 0.0
    squared_ema = 0.0
    expected_online_keys = set()

    for key, averaged in averaged_state.items():
        if not isinstance(averaged, Tensor):
            raise TypeError(f'averaged state {key!r} is not a tensor')
        online_key = f'module.{key}'
        if online_key not in ema_state:
            raise KeyError(f'missing online state {online_key!r}')
        expected_online_keys.add(online_key)
        online = ema_state[online_key]
        if not isinstance(online, Tensor):
            raise TypeError(f'online state {online_key!r} is not a tensor')
        if averaged.shape != online.shape or averaged.dtype != online.dtype:
            raise ValueError(
                f'incompatible states for {key!r}: '
                f'{averaged.shape}/{averaged.dtype} versus '
                f'{online.shape}/{online.dtype}')

        excluded = key.startswith(exclude_prefixes)
        if torch.is_floating_point(averaged):
            if not torch.isfinite(averaged).all():
                raise ValueError(f'non-finite averaged state {key!r}')
            if not torch.isfinite(online).all():
                raise ValueError(f'non-finite online state {online_key!r}')
            averaged_float = averaged.detach().float()
            online_float = online.detach().float()
            delta = online_float - averaged_float
            if excluded:
                mixed = averaged_float
                excluded_floating += 1
            else:
                mixed = averaged_float + online_fraction * delta
                interpolated_floating += 1
                squared_delta += float(torch.sum(delta * delta))
                squared_ema += float(torch.sum(averaged_float * averaged_float))
            if not torch.isfinite(mixed).all():
                raise ValueError(f'non-finite interpolated state {key!r}')
            output[key] = mixed.to(dtype=averaged.dtype)
            floating += 1
        else:
            # Integer counters are not meaningfully interpolated.  Keep the
            # deployed EMA value except at the exact-online endpoint.
            source = (
                online if online_fraction == 1 and not excluded else averaged)
            output[key] = source.detach().clone()
            nonfloating += 1

    extra_online = set(ema_state) - expected_online_keys - {'steps'}
    if extra_online:
        preview = sorted(extra_online)[:5]
        raise KeyError(f'unexpected online EMA keys: {preview}')

    relative_delta = (
        math.sqrt(squared_delta / squared_ema)
        if squared_ema > 0 else 0.0)
    stats = {
        'floating_tensors': floating,
        'nonfloating_tensors': nonfloating,
        'interpolated_floating_tensors': interpolated_floating,
        'excluded_floating_tensors': excluded_floating,
        'relative_online_to_ema_l2': relative_delta,
    }
    return output, stats


def main() -> None:
    args = parse_args()
    input_path = args.input.resolve()
    output_path = args.output.resolve()
    if output_path.exists():
        raise FileExistsError(f'refusing to overwrite {output_path}')

    checkpoint = torch.load(input_path, map_location='cpu', weights_only=False)
    if not isinstance(checkpoint, dict):
        raise ValueError(f'{input_path} is not an MMEngine checkpoint')
    if 'state_dict' not in checkpoint or 'ema_state_dict' not in checkpoint:
        raise ValueError('checkpoint must contain state_dict and ema_state_dict')

    state_dict, stats = interpolate_state_dict(
        checkpoint['state_dict'], checkpoint['ema_state_dict'],
        args.online_fraction, tuple(args.exclude_prefix))
    meta = copy.deepcopy(checkpoint.get('meta', {}))
    meta['ema_lag_correction'] = {
        'source_checkpoint': str(input_path),
        'online_fraction': args.online_fraction,
        'excluded_prefixes': list(args.exclude_prefix),
        **stats,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + '.tmp')
    torch.save({'meta': meta, 'state_dict': state_dict}, temporary)
    os.replace(temporary, output_path)
    print(f'online_fraction={args.online_fraction}')
    for key, value in stats.items():
        print(f'{key}={value}')
    print(f'output={output_path}')


if __name__ == '__main__':
    main()
