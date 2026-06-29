#!/usr/bin/env python3
"""Create the pair-adapted HSMOT R18 checkpoint used by formal training."""

from pathlib import Path

from projects.multispec_pair_rotated_rtdetr.tools.load_pair_pretrain import (
    ensure_pair_adapted_checkpoint,
)
from projects.multispec_rotated_rtdetr.configs.pretrain_paths import (
    O2_R18_HSMOT_3DSE_R2_E72,
)


def main() -> None:
    root = Path('/data/users/litianhao01/PairMmot/pretrained_weights')
    output = ensure_pair_adapted_checkpoint(
        O2_R18_HSMOT_3DSE_R2_E72,
        str(root / 'o2_r18_hsmot_3dse_r2_e72_pair_adapted'))
    print(output)


if __name__ == '__main__':
    main()
