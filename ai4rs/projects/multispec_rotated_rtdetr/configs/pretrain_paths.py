# Local O2-RTDETR checkpoints under PairMmot/pretrained_weights.
_PAIRMMOT_ROOT = '/data/users/litianhao01/PairMmot'
PRETRAIN_ROOT = f'{_PAIRMMOT_ROOT}/pretrained_weights'

O2_R18_DOTA_E72 = (
    f'{PRETRAIN_ROOT}/o2_rtdetr_r18vd_2xb4_72e_dota_epoch_72.pth')
O2_DEIM_R18_DOTA_E29 = (
    f'{PRETRAIN_ROOT}/o2_rtdetr_r18vd_2xb4_coco_pretrain_72e_dota_ms_epoch_29.pth')
O2_R18_COCO_BACKBONE = (
    f'{PRETRAIN_ROOT}/rtdetr_r18vd_dec3_6x_coco_backbone_o2.pth')
O2_R34_COCO_BACKBONE = (
    f'{PRETRAIN_ROOT}/rtdetr_r34vd_dec4_6x_coco_backbone_o2.pth')
O2_R50_COCO_BACKBONE = (
    f'{PRETRAIN_ROOT}/rtdetr_r50vd_6x_coco_backbone_o2.pth')
O2_R34_DOTA_E72 = (
    f'{PRETRAIN_ROOT}/o2_rtdetr_r34vd_2xb4_72e_dota_epoch_72.pth')
O2_R50_DOTA_E72 = (
    f'{PRETRAIN_ROOT}/o2_rtdetr_r50vd_2xb4_72e_dota_epoch_72.pth')
