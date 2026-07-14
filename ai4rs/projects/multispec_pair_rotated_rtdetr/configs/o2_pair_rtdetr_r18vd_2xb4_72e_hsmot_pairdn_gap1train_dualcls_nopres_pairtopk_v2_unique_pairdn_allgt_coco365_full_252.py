"""0714 module-ablation baseline: 0704_01 structure, COCO365 init, full train."""
from mmengine.config import read_base

with read_base():
    from .o2_pair_rtdetr_r18vd_2xb4_72e_hsmot_pairdn_gap1train_dualcls_nopres_pairtopk_v2_unique_pairdn_allgt import *  # noqa: F401,F403

load_from = (
    '/data4/litianhao/PairMmot/pretrained_weights/'
    'rtdetr_r18vd_5x_coco_objects365_pair_unique_allgt_full/'
    'pair_coco365_full_adapted_pretrain.pth')

work_dir = (
    '/data4/litianhao/PairMmot/workdir_252/'
    '0714_01_0704_resume_coco365_full_unique_allgt')

# Full-data training: HSMOTPairDataset scans all train/mot/*.txt sequences
# when ann_file is None.  On 252 this is 75 train sequences instead of 29.
train_dataloader['dataset'].update(
    data_root='/data/users/litianhao01/PairMmot/data/hsmot/train',
    ann_file=None,
    data_prefix=dict(img_path='npy2jpg'),
)
val_dataloader['dataset'].update(
    data_root='/data/users/litianhao01/PairMmot/data/hsmot/test',
    data_prefix=dict(img_path='npy2jpg'),
)
test_dataloader = val_dataloader

val_evaluator['metrics'].update(
    track_eval=True,
    track_eval_out_dir=f'{work_dir}/val_track_eval',
)
test_evaluator = val_evaluator

if 'default_hooks' in globals() and 'visualization' in default_hooks:
    default_hooks['visualization'].update(draw=False)

for hook in custom_hooks:
    if hook.get('type') == 'HSMOTPairValVisualizationHook':
        hook.update(draw=False)
