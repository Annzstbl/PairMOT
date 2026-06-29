"""Same-frame train/val diagnostic for the formal HSMOT half pair run."""
from mmengine.config import read_base

with read_base():
    from .o2_pair_rtdetr_r18vd_2xb4_72e_hsmot_pairdn import *

# Diagnostic target:
# - train prev/curr on the identical image, so the pair architecture should be
#   able to recover single-frame-like detection quality if the pair formulation
#   is not the bottleneck.
# - validate on identical prev/curr too; report gap0 metrics explicitly.
train_dataloader['dataset']['same_frame'] = True
train_dataloader['dataset'].pop('random_interval_range', None)
train_dataloader['dataset'].pop('sample_seed', None)

val_dataloader['dataset']['same_frame'] = True
val_dataloader['dataset'].pop('frame_intervals', None)
val_evaluator['metrics']['report_gaps'] = (0,)

test_dataloader = val_dataloader
test_evaluator = val_evaluator

work_dir = (
    '/data/users/litianhao01/PairMmot/workdir/'
    'o2_pair_rtdetr_r18vd_2xb4_72e_hsmot_half_pairdn_sameframe_trainval')
