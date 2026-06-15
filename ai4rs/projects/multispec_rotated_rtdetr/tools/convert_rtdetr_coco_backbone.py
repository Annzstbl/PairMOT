"""Convert RT-DETR COCO checkpoints to O2-RTDETR backbone checkpoints."""
import argparse
from collections import OrderedDict

import torch


def _load_state_dict(path):
    checkpoint = torch.load(path, map_location='cpu')
    if isinstance(checkpoint, dict):
        if 'state_dict' in checkpoint:
            return checkpoint['state_dict']
        if 'ema' in checkpoint and isinstance(checkpoint['ema'], dict):
            return checkpoint['ema']['module']
        if 'model' in checkpoint:
            return checkpoint['model']
    return checkpoint


def _map_norm(src_key, target_prefix):
    suffix = src_key.rsplit('.norm.', 1)[1]
    return f'{target_prefix}.{suffix}'


def convert_backbone(state_dict, depth):
    if depth in (18, 34):
        block_type = 'basic'
    elif depth in (50, 101):
        block_type = 'bottleneck'
    else:
        raise ValueError(f'Unsupported depth: {depth}')

    converted = OrderedDict()

    stem_map = {
        'backbone.conv1.conv1_1': ('backbone.stem.0', 'backbone.stem.1'),
        'backbone.conv1.conv1_2': ('backbone.stem.3', 'backbone.stem.4'),
        'backbone.conv1.conv1_3': ('backbone.stem.6', 'backbone.stem.7'),
    }
    for key, value in state_dict.items():
        for src_prefix, (conv_prefix, norm_prefix) in stem_map.items():
            if key == f'{src_prefix}.conv.weight':
                converted[f'{conv_prefix}.weight'] = value
            elif key.startswith(f'{src_prefix}.norm.'):
                converted[_map_norm(key, norm_prefix)] = value

    for key, value in state_dict.items():
        if not key.startswith('backbone.res_layers.'):
            continue
        parts = key.split('.')
        stage = int(parts[2])
        block = int(parts[4])
        layer_prefix = f'backbone.layer{stage + 1}.{block}'

        if '.branch2a.conv.weight' in key:
            converted[f'{layer_prefix}.conv1.weight'] = value
        elif '.branch2a.norm.' in key:
            converted[_map_norm(key, f'{layer_prefix}.bn1')] = value
        elif '.branch2b.conv.weight' in key:
            converted[f'{layer_prefix}.conv2.weight'] = value
        elif '.branch2b.norm.' in key:
            converted[_map_norm(key, f'{layer_prefix}.bn2')] = value
        elif '.branch2c.conv.weight' in key:
            converted[f'{layer_prefix}.conv3.weight'] = value
        elif '.branch2c.norm.' in key:
            converted[_map_norm(key, f'{layer_prefix}.bn3')] = value

        if block_type == 'basic' and stage == 0:
            downsample_conv_idx, downsample_norm_idx = 0, 1
        else:
            downsample_conv_idx, downsample_norm_idx = 1, 2

        if '.short.conv.weight' in key or '.short.conv.conv.weight' in key:
            converted[
                f'{layer_prefix}.downsample.{downsample_conv_idx}.weight'] = value
        elif '.short.norm.' in key or '.short.conv.norm.' in key:
            converted[_map_norm(
                key, f'{layer_prefix}.downsample.{downsample_norm_idx}')] = value

    return converted


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('src')
    parser.add_argument('dst')
    parser.add_argument('--depth', type=int, required=True)
    args = parser.parse_args()

    converted = convert_backbone(_load_state_dict(args.src), args.depth)
    torch.save(
        dict(
            state_dict=converted,
            meta=dict(
                source=args.src,
                depth=args.depth,
                note='RT-DETR COCO backbone converted for O2-RTDETR.')),
        args.dst)
    print(f'Saved {len(converted)} backbone tensors to {args.dst}')


if __name__ == '__main__':
    main()
