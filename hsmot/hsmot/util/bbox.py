import numpy as np
import torch

from hsmot.mmlab.hs_mmrotate import obb2poly_np, poly2obb_np
from hsmot.util.iou import box_iou_rotated


def poly2obb_np_woscore(polys, version="le135"):
    """将多边形顶点转为 OBB（无 score 列）。

    Args:
        polys: (N, 8) 或 (8,) 四边形顶点 [x1,y1,...,x4,y4]。
        version: 角度编码，当前仅支持 ``le135``。

    Returns:
        (N, 5) 或 (5,) xywhr；退化框时宽高置为 -1。
    """
    # polys支持两个维度 一个是 [N, 8] 一个是 [8]
    assert polys.shape[-1] == 8
    if polys.ndim == 1:
        return np.array(poly2obb_np(polys, version=version))
    else:
        ret = []
        for poly in polys:
            result = poly2obb_np(poly, version=version)
            if result is None:
                cx = (poly[0] + poly[2] + poly[4] + poly[6]) / 4
                cy = (poly[1] + poly[3] + poly[5] + poly[7]) / 4
                result = [cx, cy, -1, -1, 0]  # TODO框太小时候的处理
            ret.append(result)
        return np.array(ret)


def obb2poly_np_woscore(obbs, version="le135"):
    """将 OBB 转为多边形顶点（无 score 列）。

    Args:
        obbs: (N, 5) 或 (5,) xywhr。
        version: 角度编码，当前仅支持 ``le135``。

    Returns:
        (N, 8) 或 (8,) 顶点坐标。
    """
    # obbs支持两个维度 一个是 [N, 5] 一个是 [5]
    assert obbs.shape[-1] == 5

    # 需要先加一个score,最后再删掉
    obbs = obbs.reshape(-1, 5)
    obbs = np.concatenate([obbs, np.ones((obbs.shape[0], 1))], axis=1)

    assert version == "le135"
    ret = obb2poly_np(obbs, version=version)

    if obbs.ndim == 1:
        return ret[0][:-1]
    else:
        return ret[:, :-1]


def poly_iou(apolys, bpolys, aligned=False):
    """计算两组四边形多边形的几何 IoU（评测用，非可微）。

    内部将多边形转为 OBB 后调用 ``box_iou_rotated``。

    Args:
        apolys: (N, 8) 多边形 A。
        bpolys: (M, 8) 或 aligned=True 时 (N, 8) 多边形 B。
        aligned: 是否逐对计算。

    Returns:
        numpy 数组，(N,) 或 (N, M)。
    """
    if aligned:
        assert apolys.shape[0] == bpolys.shape[0]
    apolys = apolys[:, :8]
    bpolys = bpolys[:, :8]

    aobbs = poly2obb_np_woscore(apolys, version="le135")
    bobbs = poly2obb_np_woscore(bpolys, version="le135")

    aobbs_tensor = torch.tensor(aobbs, dtype=torch.float32)
    bobbs_tensor = torch.tensor(bobbs, dtype=torch.float32)

    ret = box_iou_rotated(aobbs_tensor, bobbs_tensor, aligned=aligned)

    return ret.cpu().numpy()
