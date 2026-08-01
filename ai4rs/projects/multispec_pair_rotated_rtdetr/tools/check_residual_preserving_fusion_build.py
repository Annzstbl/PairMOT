"""Build and compare the 0801_06 decoder against its Encoder parent."""
from __future__ import annotations

import copy
import gc

from mmengine.config import Config
from mmengine.utils import import_modules_from_strings
from mmdet.registry import MODELS
from mmrotate.utils import register_all_modules


NEW_CONFIG = (
    'projects/multispec_pair_rotated_rtdetr/configs/'
    'o2_pair_rtdetr_r18vd_2xb4_72e_hsmot_0801_06_symmetric_position_'
    'residual_preserving_fusion_decoder_252.py')
SMOKE_CONFIG = (
    'projects/multispec_pair_rotated_rtdetr/configs/smoke/'
    'o2_pair_rtdetr_r18vd_0801_06_symmetric_position_residual_'
    'preserving_fusion_decoder_4iter_smoke_252.py')


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
parent_config.model.decoder.symmetric_position_decoder = False
parent_config.model.decoder.residual_preserving_fusion_decoder = False

decoder_config = new_config.model.decoder
assert decoder_config.symmetric_position_decoder is True
assert decoder_config.residual_preserving_fusion_decoder is True
assert decoder_config.symmetric_feature_decoder is False
assert decoder_config.symmetric_pair_decoder is False
assert smoke_config.train_cfg.max_iters == 4

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

assert new_parameters == parent_parameters, (new_parameters, parent_parameters)
assert new_state == parent_state
assert new_model.decoder.symmetric_position_decoder is True
assert new_model.decoder.residual_preserving_fusion_decoder is True
assert all(
    layer.residual_preserving_fusion_decoder
    for layer in new_model.decoder.layers)
assert all(
    layer.cross_attn_prev is not layer.cross_attn_curr
    for layer in new_model.decoder.layers)

print('RESIDUAL_PRESERVING_FUSION_BUILD_OK', {
    'parameters': new_parameters,
    'state_tensors': len(new_state),
    'layers': len(new_model.decoder.layers),
    'smoke_iters': smoke_config.train_cfg.max_iters,
})
