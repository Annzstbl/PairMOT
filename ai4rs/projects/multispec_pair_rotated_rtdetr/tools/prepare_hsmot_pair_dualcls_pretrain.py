#!/usr/bin/env python3
"""Create the pair-adapted HSMOT R18 checkpoint for dual-cls pair heads."""

from pathlib import Path

from projects.multispec_pair_rotated_rtdetr.tools.load_pair_pretrain import (
    ensure_pair_adapted_checkpoint,
)
from projects.multispec_rotated_rtdetr.configs.pretrain_paths import (
    O2_R18_HSMOT_3DSE_R2_E72,
)


def main() -> None:
    root = Path('/data/users/litianhao01/PairMmot/pretrained_weights')
    src_ckpt = Path(O2_R18_HSMOT_3DSE_R2_E72)
    if not src_ckpt.is_file():
        src_ckpt = Path(
            '/data/users/litianhao01/PairMmot/workdir/01_01_single_detection/'
            'o2_rtdetr_r18vd_2xb4_72e_hsmot_coco_pretrain_'
            '3dse_reduction2/epoch_72.pth')
    output = ensure_pair_adapted_checkpoint(
        str(src_ckpt),
        str(root / 'o2_r18_hsmot_3dse_r2_e72_pair_dualcls_adapted'),
        copy_cls_branches_curr=True,
        output_name='pair_dualcls_adapted_pretrain.pth')
    print(output)


if __name__ == '__main__':
    main()
