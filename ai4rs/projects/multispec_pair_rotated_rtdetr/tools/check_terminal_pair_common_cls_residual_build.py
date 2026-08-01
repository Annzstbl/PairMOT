"""Build and compare 0801_11 against its unchanged Encoder parent."""
from __future__ import annotations

import copy
import gc

from mmengine.config import Config
from mmengine.utils import import_modules_from_strings
from mmdet.registry import MODELS
from mmrotate.utils import register_all_modules


NEW_CONFIG = (
    'projects/multispec_pair_rotated_rtdetr/configs/'
    'o2_pair_rtdetr_r18vd_2xb4_72e_hsmot_0801_11_terminal_'
    'pair_common_cls_residual_decoder_99.py')
SMOKE_CONFIG = (
    'projects/multispec_pair_rotated_rtdetr/configs/smoke/'
    'o2_pair_rtdetr_r18vd_0801_11_terminal_pair_common_cls_'
    'residual_decoder_4iter_smoke_99.py')


def load_config(path: str) -> Config:
    config = Config.fromfile(path)
    copy.deepcopy(config)
    if config.get('custom_imports'):
        import_modules_from_strings(**config.custom_imports)
    return config


register_all_modules()
new_config = load_config(NEW_CONFIG)
smoke_config = load_config(SMOKE_CONFIG)
parent_config = copy.deepcopy(new_config)
parent_config.model.bbox_head.terminal_pair_common_cls_residual = False

assert new_config.model.bbox_head.terminal_pair_common_cls_residual is True
assert new_config.model.bbox_head.iterative_cls_residual is False
assert new_config.model.bbox_head.terminal_encoder_cls_residual is False
assert new_config.model.decoder.terminal_classification_common_evidence_decoder is False
assert smoke_config.train_cfg.max_iters == 4
assert smoke_config.val_cfg is None
assert smoke_config.val_dataloader is None
assert smoke_config.val_evaluator is None

parent_model = MODELS.build(parent_config.model)
parent_parameters = sum(parameter.numel() for parameter in parent_model.parameters())
parent_state = {
    key: tuple(value.shape) for key, value in parent_model.state_dict().items()
}
del parent_model
gc.collect()

new_model = MODELS.build(new_config.model)
new_parameters = sum(parameter.numel() for parameter in new_model.parameters())
new_state = {
    key: tuple(value.shape) for key, value in new_model.state_dict().items()
}
new_keys = sorted(set(new_state) - set(parent_state))

expected_parameter_delta = (
    (new_model.bbox_head.embed_dims + 1)
    * new_model.bbox_head.cls_out_channels)
assert new_parameters - parent_parameters == expected_parameter_delta, (
    new_parameters, parent_parameters, expected_parameter_delta)
assert new_keys == [
    'bbox_head.terminal_pair_common_cls_residual_branch.bias',
    'bbox_head.terminal_pair_common_cls_residual_branch.weight',
]

print('TERMINAL_PAIR_COMMON_CLS_RESIDUAL_BUILD_OK', {
    'parent_parameters': parent_parameters,
    'new_parameters': new_parameters,
    'parameter_delta': new_parameters - parent_parameters,
    'parameter_delta_pct': (
        100.0 * (new_parameters - parent_parameters) / parent_parameters),
    'new_state_keys': new_keys,
    'smoke_iters': smoke_config.train_cfg.max_iters,
})

