"""Build 0803_18 and prove that both terminal projections add no state."""
from __future__ import annotations

import copy
import gc
import sys

from mmengine.config import Config
from mmengine.utils import import_modules_from_strings
from mmdet.registry import MODELS
from mmrotate.utils import register_all_modules


DEFAULT_NEW_CONFIG = (
    'projects/multispec_pair_rotated_rtdetr/configs/'
    'o2_pair_rtdetr_r18vd_2xb4_72e_hsmot_0803_18_iterative_cls_'
    'terminal_log_size_angle_shared_margins_decoder_197.py')
DEFAULT_SMOKE_CONFIG = (
    'projects/multispec_pair_rotated_rtdetr/configs/smoke/'
    'o2_pair_rtdetr_r18vd_0803_18_iterative_cls_terminal_log_size_'
    'angle_shared_margins_decoder_4iter_smoke_197.py')


def load_config(path: str) -> Config:
    config = Config.fromfile(path)
    copy.deepcopy(config)
    if config.get('custom_imports'):
        import_modules_from_strings(**config.custom_imports)
    return config


register_all_modules()
new_config = load_config(sys.argv[1] if len(sys.argv) > 1 else DEFAULT_NEW_CONFIG)
smoke_config = load_config(sys.argv[2] if len(sys.argv) > 2 else DEFAULT_SMOKE_CONFIG)
parent_config = copy.deepcopy(new_config)
parent_config.model.bbox_head.iterative_cls_terminal_shared_margins = False
parent_config.model.decoder.pair_shared_terminal_log_size_periodic_angle_refinement_decoder = False

assert new_config.model.bbox_head.iterative_cls_terminal_shared_margins
assert not new_config.model.bbox_head.iterative_cls_pair_shared_objectness
assert new_config.model.decoder.pair_shared_terminal_log_size_periodic_angle_refinement_decoder
assert not new_config.model.decoder.pair_shared_late_log_size_periodic_angle_refinement_decoder
assert smoke_config.train_cfg.max_iters == 4
assert smoke_config.val_cfg is None

parent_model = MODELS.build(parent_config.model)
parent_parameters = sum(p.numel() for p in parent_model.parameters())
parent_state = {k: tuple(v.shape) for k, v in parent_model.state_dict().items()}
del parent_model
gc.collect()
new_model = MODELS.build(new_config.model)
new_parameters = sum(p.numel() for p in new_model.parameters())
new_state = {k: tuple(v.shape) for k, v in new_model.state_dict().items()}
assert new_parameters == parent_parameters
assert new_state == parent_state
print('TERMINAL_GEOMETRY_SHARED_MARGINS_BUILD_OK', {
    'parameters': new_parameters,
    'parameter_delta': 0,
    'state_tensors': len(new_state),
    'smoke_iters': smoke_config.train_cfg.max_iters,
})
