#!/usr/bin/env python3
"""Adapt official COCO RT-DETR R18/R34/R50 weights to PairMOT models."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from projects.multispec_pair_rotated_rtdetr.tools.load_pair_pretrain import (
    ensure_rtdetr_pair_adapted_checkpoint,
)


PRETRAIN_ROOT = Path('/data4/litianhao/PairMmot/pretrained_weights')
RELEASE_ROOT = 'https://github.com/lyuwenyu/storage/releases/download/v0.1'
FAMILY = {
    'r18': ('rtdetr_r18vd_dec3_6x_coco_from_paddle.pth',
            'pair_rtdetr_r18vd_coco_adapt.py', 3),
    'r34': ('rtdetr_r34vd_dec4_6x_coco_from_paddle.pth',
            'pair_rtdetr_r34vd_coco_adapt.py', 4),
    'r50': ('rtdetr_r50vd_6x_coco_from_paddle.pth',
            'pair_rtdetr_r50vd_coco_adapt.py', 6),
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as file:
        for chunk in iter(lambda: file.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--backbones', nargs='+', choices=FAMILY,
                        default=list(FAMILY))
    parser.add_argument('--pretrain-root', type=Path, default=PRETRAIN_ROOT)
    parser.add_argument('--force', action='store_true')
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_root = Path(
        'projects/multispec_pair_rotated_rtdetr/configs/pretrain_targets')
    manifest = {'format_version': 1, 'models': {}}
    for backbone in args.backbones:
        source_name, config_name, decoder_layers = FAMILY[backbone]
        source = args.pretrain_root / source_name
        output_dir = args.pretrain_root / f'{source_name[:-4]}_pair_adapted'
        output = ensure_rtdetr_pair_adapted_checkpoint(
            src_ckpt=str(source),
            target_config=str(config_root / config_name),
            cache_dir=str(output_dir),
            force=args.force)
        digest = sha256_file(source)
        stats_path = Path(f'{output}.json')
        stats = json.loads(stats_path.read_text(encoding='utf-8'))
        manifest['models'][backbone] = {
            'official_url': f'{RELEASE_ROOT}/{source_name}',
            'source_checkpoint': str(source),
            'source_size_bytes': source.stat().st_size,
            'source_sha256': digest,
            'decoder_layers': decoder_layers,
            'target_config': str(config_root / config_name),
            'adapted_checkpoint': output,
            'adaptation_stats': stats,
        }
        print(f'{backbone}: {output}')
    manifest_path = args.pretrain_root / 'rtdetr_coco_pair_family_manifest.json'
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding='utf-8')
    print(f'manifest: {manifest_path}')


if __name__ == '__main__':
    main()
