# Copyright (c) AI4RS. All rights reserved.
"""Convert a 3-channel O2-RTDETR / RT-DETR checkpoint to 8-channel input."""

import argparse
import os

import torch

from projects.multispec_rotated_rtdetr.multispec_rotated_rtdetr.pretrain_utils import (
    adapt_state_dict_in_channels, load_checkpoint_state_dict)


def parse_args():
    parser = argparse.ArgumentParser(
        description='Expand stem conv weights from 3 channels to 8 channels.')
    parser.add_argument('src', help='Source checkpoint path')
    parser.add_argument('dst', help='Destination checkpoint path')
    parser.add_argument(
        '--in-channels',
        type=int,
        default=8,
        help='Target input channels')
    parser.add_argument(
        '--expand-mode',
        choices=['rgbrepeat', 'interpolate'],
        default='rgbrepeat',
        help='Stem weight expansion strategy')
    return parser.parse_args()


def main():
    args = parse_args()
    state_dict = load_checkpoint_state_dict(args.src)
    adapted = adapt_state_dict_in_channels(
        state_dict,
        in_channels=args.in_channels,
        expand_mode=args.expand_mode)

    checkpoint = torch.load(args.src, map_location='cpu')
    if isinstance(checkpoint, dict):
        if 'state_dict' in checkpoint:
            checkpoint['state_dict'] = adapted
        elif 'model' in checkpoint and isinstance(checkpoint['model'], dict):
            checkpoint['model'] = adapted
        else:
            checkpoint = adapted
    else:
        checkpoint = adapted

    os.makedirs(os.path.dirname(os.path.abspath(args.dst)), exist_ok=True)
    torch.save(checkpoint, args.dst)
    print(f'Saved 8-channel checkpoint to {args.dst}')


if __name__ == '__main__':
    main()
