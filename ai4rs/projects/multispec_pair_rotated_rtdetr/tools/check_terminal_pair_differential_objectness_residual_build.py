"""Build and compare 0801_13 against its unchanged parent."""
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
    'o2_pair_rtdetr_r18vd_2xb4_72e_hsmot_0801_13_terminal_pair_'
    'differential_objectness_residual_decoder_99.py')
DEFAULT_SMOKE_CONFIG = (
    'projects/multispec_pair_rotated_rtdetr/configs/smoke/'
    'o2_pair_rtdetr_r18vd_0801_13_terminal_pair_differential_'
    'objectness_residual_decoder_4iter_smoke_99.py')


def load_config(path: str) -> Config:
    config = Config.fromfile(path)
    copy.deepcopy(config)
    if config.get('custom_imports'):
        import_modules_from_strings(**config.custom_imports)
    return config


register_all_modules()
NEW_CONFIG = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_NEW_CONFIG
SMOKE_CONFIG = sys.argv[2] if len(sys.argv) > 2 else DEFAULT_SMOKE_CONFIG
new_config = load_config(NEW_CONFIG)
smoke_config = load_config(SMOKE_CONFIG)
parent_config = copy.deepcopy(new_config)
parent_config.model.bbox_head.terminal_pair_differential_objectness_residual = False

assert new_config.model.bbox_head.terminal_pair_differential_objectness_residual
assert not new_config.model.bbox_head.terminal_pair_common_objectness_residual
assert not new_config.model.bbox_head.terminal_pair_common_cls_residual
assert smoke_config.train_cfg.max_iters == 4
assert smoke_config.val_cfg is None

parent_model = MODELS.build(parent_config.model)
parent_parameters = sum(parameter.numel() for parameter in parent_model.parameters())
parent_state = set(parent_model.state_dict())
del parent_model
gc.collect()

new_model = MODELS.build(new_config.model)
new_parameters = sum(parameter.numel() for parameter in new_model.parameters())
new_keys = sorted(set(new_model.state_dict()) - parent_state)
expected_parameter_delta = new_model.bbox_head.embed_dims + 1
assert new_parameters - parent_parameters == expected_parameter_delta
assert new_keys == [
    'bbox_head.terminal_pair_differential_objectness_residual_branch.bias',
    'bbox_head.terminal_pair_differential_objectness_residual_branch.weight',
]
print('TERMINAL_PAIR_DIFFERENTIAL_OBJECTNESS_RESIDUAL_BUILD_OK', {
    'parent_parameters': parent_parameters,
    'new_parameters': new_parameters,
    'parameter_delta': new_parameters - parent_parameters,
    'parameter_delta_pct': (
        100.0 * (new_parameters - parent_parameters) / parent_parameters),
    'new_state_keys': new_keys,
})
