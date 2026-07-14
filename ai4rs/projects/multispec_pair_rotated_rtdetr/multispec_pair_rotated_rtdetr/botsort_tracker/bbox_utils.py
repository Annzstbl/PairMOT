"""BBox conversion helpers used by the vendored RMMOT BoT-SORT tracker."""
from __future__ import annotations

import numpy as np
import torch

from mmrotate.structures.bbox import qbox2rbox, rbox2qbox


def poly2obb_np_woscore(polys, version='le135'):
    """Convert polygon boxes ``[..., 8]`` to rotated boxes ``[..., 5]``."""
    assert version == 'le135'
    polys = np.asarray(polys, dtype=np.float32)
    assert polys.shape[-1] == 8
    was_1d = polys.ndim == 1
    tensor = torch.from_numpy(polys.reshape(-1, 8))
    rboxes = qbox2rbox(tensor).detach().cpu().numpy()
    return rboxes[0] if was_1d else rboxes


def obb2poly_np_woscore(obbs, version='le135'):
    """Convert rotated boxes ``[..., 5]`` to polygon boxes ``[..., 8]``."""
    assert version == 'le135'
    obbs = np.asarray(obbs, dtype=np.float32)
    assert obbs.shape[-1] == 5
    was_1d = obbs.ndim == 1
    tensor = torch.from_numpy(obbs.reshape(-1, 5))
    qboxes = rbox2qbox(tensor).detach().cpu().numpy()
    return qboxes[0] if was_1d else qboxes
