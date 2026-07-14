"""Temporary single-GPU DDP profile with find_unused_parameters disabled."""
from mmengine.config import read_base

with read_base():
    from .tmp_profile_0714_coco365_full_single_gpu import *  # noqa: F401,F403

find_unused_parameters = False

work_dir = (
    '/data4/litianhao/PairMmot/workdir_252/'
    'tmp_profile_0714_coco365_full_single_gpu_findunused_false')
