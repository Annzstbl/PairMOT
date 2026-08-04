"""Build 0804_04 and prove body-frame product tangent adds no state."""
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
    'o2_pair_rtdetr_r18vd_1xb8_72e_hsmot_0804_04_iterative_cls_terminal_'
    'body_frame_product_tangent_decoder_178.py')
DEFAULT_SMOKE_CONFIG = (
    'projects/multispec_pair_rotated_rtdetr/configs/smoke/'
    'o2_pair_rtdetr_r18vd_0804_04_iterative_cls_terminal_body_frame_'
    'product_tangent_decoder_4iter_smoke_178.py')


def load_config(path: str) -> Config:
    config = Config.fromfile(path)
    copy.deepcopy(config)
    if config.get('custom_imports'):
        import_modules_from_strings(**config.custom_imports)
    return config


register_all_modules()
new_path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_NEW_CONFIG
smoke_path = sys.argv[2] if len(sys.argv) > 2 else DEFAULT_SMOKE_CONFIG
new_config = load_config(new_path)
smoke_config = load_config(smoke_path)
parent_config = copy.deepcopy(new_config)
parent_config.model.decoder.update(
    pair_shared_terminal_transport_body_frame_product_tangent_refinement_decoder=False,
    pair_shared_terminal_transport_product_tangent_refinement_decoder=True)

decoder = new_config.model.decoder
assert decoder.pair_shared_terminal_transport_body_frame_product_tangent_refinement_decoder
assert not decoder.pair_shared_terminal_transport_product_tangent_refinement_decoder
assert not decoder.pair_shared_terminal_transport_center_tangent_refinement_decoder
assert not decoder.pair_shared_terminal_transport_shape_tangent_refinement_decoder
assert not decoder.pair_shared_terminal_transport_tangent_refinement_decoder
assert not decoder.pair_shared_terminal_transport_plane_refinement_decoder
assert not decoder.pair_shared_terminal_full_tangent_refinement_decoder
assert not decoder.pair_shared_terminal_log_size_periodic_angle_refinement_decoder
assert new_config.model.bbox_head.iterative_cls_residual
assert new_config.model.bbox_head.iterative_cls_dn_absolute
assert not new_config.model.bbox_head.iterative_cls_detach_between_layers
assert not new_config.model.bbox_head.iterative_cls_pair_shared_objectness
assert smoke_config.train_cfg.max_iters == 4
assert smoke_config.val_cfg is None

parent_model = MODELS.build(parent_config.model)
parent_parameters = sum(
    parameter.numel() for parameter in parent_model.parameters())
parent_state = {
    key: tuple(value.shape) for key, value in parent_model.state_dict().items()
}
del parent_model
gc.collect()

new_model = MODELS.build(new_config.model)
new_parameters = sum(
    parameter.numel() for parameter in new_model.parameters())
new_state = {
    key: tuple(value.shape) for key, value in new_model.state_dict().items()
}
assert new_parameters == parent_parameters
assert new_state == parent_state
print('ITERATIVE_CLS_TERMINAL_BODY_FRAME_PRODUCT_TANGENT_BUILD_OK', {
    'parameters': new_parameters,
    'parameter_delta': 0,
    'state_tensors': len(new_state),
    'smoke_iters': smoke_config.train_cfg.max_iters,
})
