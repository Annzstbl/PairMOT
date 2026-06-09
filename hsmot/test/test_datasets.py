import numpy as np

from hsmot.datasets.pipelines.channel import MmrotateToMotr, MotrToMmrotate
from hsmot.datasets.pipelines.compose import MotCompose
from hsmot.datasets.pipelines.formatting import MotCollect, MotDefaultFormatBundle
from hsmot.datasets.pipelines.loading import (
    MotLoadAnnotations,
    MotLoadMultichannelImageFromNpy,
)
from hsmot.datasets.pipelines.transforms import MotNormalize, MotPad, MotRRandomCrop, MotRRandomFlip, MotRRsize
from hsmot.mmlab.hs_mmrotate import poly2obb_np

HSMOT_SAMPLE_NPY = "/data/users/litianhao/data/HSMOT/npy/data23-1/000001.npy"

ANN_TEXT = """
    1,1,976,904,976,858,859,858,859,904,-1,0,1
    1,2,1028,568,1030,517,913,511,910,561,-1,0,0
    1,3,924,85,1047,79,1045,29,922,34,-1,0,0
    1,4,880,825,1003,820,1001,768,878,773,-1,0,0
    1,5,1006,762,1008,709,887,705,885,758,-1,0,0
    1,6,1028,632,1028,577,910,574,910,629,-1,0,0
    1,7,1034,503,1035,452,906,447,905,498,-1,0,0
    1,8,910,421,1029,416,1028,364,909,368,-1,0,0
    1,9,905,695,1016,693,1015,637,904,639,-1,0,0
    1,10,1069,225,1069,171,931,170,931,224,-1,0,0
    1,11,910,361,1046,358,1044,305,908,308,-1,0,0
    1,12,942,142,1074,136,1072,85,941,90,-1,0,0
    1,13,929,213,930,194,911,193,910,212,-1,2,0
    1,14,925,435,925,421,905,421,905,435,-1,2,0
    1,15,853,250,872,250,872,233,853,233,-1,2,0
    1,16,573,479,583,352,522,348,512,475,-1,0,0
    1,17,936,293,1059,286,1057,241,934,247,-1,0,0
    1,18,883,219,900,219,900,204,883,204,-1,2,0
"""


def _resolve_sample_npy(tmp_path) -> str:
    """Prefer real HSMOT sample; otherwise create a synthetic 8-channel npy."""
    import os

    if os.path.exists(HSMOT_SAMPLE_NPY):
        return HSMOT_SAMPLE_NPY

    npy_path = tmp_path / "000001.npy"
    # Coordinates in ANN_TEXT reach ~1100x900; use a slightly larger canvas.
    img = np.random.randint(0, 255, size=(960, 1280, 8), dtype=np.uint8)
    np.save(npy_path, img)
    return str(npy_path)


def _prepare_data(img_path: str, version: str = "le135"):
    data_info = {"filename": img_path, "ann": {}}
    gt_bboxes, gt_labels, gt_ids, gt_polygons = [], [], [], []

    for line in ANN_TEXT.strip().split("\n"):
        frame_id, track_id, *xyxyxyxy, _, cls, _ = line.strip().split(",")
        x, y, w, h, a = poly2obb_np(np.array(xyxyxyxy, dtype=np.float32), version)
        gt_bboxes.append([x, y, w, h, a])
        gt_labels.append(cls)
        gt_polygons.append(xyxyxyxy)
        gt_ids.append(track_id)

    data_info["ann"]["bboxes"] = np.array(gt_bboxes, dtype=np.float32)
    data_info["ann"]["labels"] = np.array(gt_labels, dtype=np.int64)
    data_info["ann"]["polygons"] = np.array(gt_polygons, dtype=np.float32)
    data_info["ann"]["trackids"] = np.array(gt_ids, dtype=np.int64)

    results = dict(img_info=data_info, ann_info=data_info["ann"])
    results["img_prefix"] = None
    results["seg_prefix"] = None
    results["proposal_file"] = None
    results["bbox_fields"] = []
    results["mask_fields"] = []
    results["seg_fields"] = []
    return [results]


def _build_transform():
    scales_w = [608, 640, 672, 704, 736, 768, 800, 832, 864, 896, 928, 960, 992, 1024, 1056, 1088, 1120, 1152, 1184]
    scales_h = [int(w / 4 * 3) for w in scales_w]
    scales = list(zip(scales_h, scales_w))
    mean = [0.27358221, 0.28804452, 0.28133921, 0.26906377, 0.28309119, 0.26928305, 0.28372527, 0.27149373]
    std = [0.19756629, 0.17432339, 0.16413284, 0.17581682, 0.18366176, 0.1536845, 0.15964683, 0.16557951]
    mean = [m * 255 for m in mean]
    std = [s * 255 for s in std]

    return MotCompose(
        [
            MotrToMmrotate(),
            MotLoadMultichannelImageFromNpy(),
            MotLoadAnnotations(poly2mask=False),
            MotRRandomFlip(direction=["horizontal", "vertical"], flip_ratio=[0.25, 0.25], version="le135"),
            MotRRandomCrop(
                crop_size=(800, 1200),
                allow_negative_crop=True,
                iof_thr=0.5,
                version="le135",
            ),
            MotRRsize(multiscale_mode="value", img_scale=scales, bbox_clip_border=False),
            MotNormalize(mean=mean, std=std, to_rgb=False),
            MotPad(size_divisor=32),
            MotDefaultFormatBundle(),
            MotCollect(keys=["img", "gt_bboxes", "gt_labels", "gt_trackids"]),
            MmrotateToMotr(),
        ]
    )


def test_transform_pipeline(tmp_path):
    np.random.seed(0)
    img_path = _resolve_sample_npy(tmp_path)
    input_sequence = _prepare_data(img_path)
    transform = _build_transform()
    result = transform(input_sequence)
    assert result is not None
    images, targets, img_metas = result
    assert len(images) == 1
    assert len(targets) == 1
    assert images[0].shape[0] == 8
