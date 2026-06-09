# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp

import numpy as np

from hsmot.mmlab import hs_mmcv as mmcv
from hsmot.mmlab.hs_mmdet import LoadAnnotations, LoadImageFromFile


class LoadMultichannelImageFromNpy:
    """从 npy 文件或图像文件加载多通道图像。"""

    def __init__(
        self,
        to_float32=False,
        color_type="color",
        channel_order="bgr",
        backend_args=dict(backend="disk"),
    ):
        self.to_float32 = to_float32
        self.color_type = color_type
        self.channel_order = channel_order
        self.backend_args = backend_args.copy()

    def __call__(self, results):
        if results["img_prefix"] is not None:
            filename = osp.join(results["img_prefix"], results["img_info"]["filename"])
        else:
            filename = results["img_info"]["filename"]

        if filename.endswith("npy"):
            img = np.load(filename)
        else:
            img = mmcv.imread(filename, flag=self.color_type, channel_order="rgb")

        if self.to_float32:
            img = img.astype(np.float32)

        results["filename"] = filename
        results["ori_filename"] = results["img_info"]["filename"]
        results["img"] = img
        results["img_shape"] = img.shape
        results["ori_shape"] = img.shape
        results["img_fields"] = ["img"]
        return results

    def __repr__(self):
        return (
            f"{self.__class__.__name__}("
            f"to_float32={self.to_float32}, "
            f"color_type='{self.color_type}', "
            f"channel_order='{self.channel_order}', "
            f"backend_args={self.backend_args})"
        )


class LoadRgbImageFromNpy:
    """从 npy 或图像文件加载 RGB 图像。"""

    def __init__(
        self,
        to_float32=False,
        color_type="color",
        channel_order="bgr",
        file_client_args=dict(backend="disk"),
    ):
        self.to_float32 = to_float32
        self.color_type = color_type
        self.channel_order = channel_order
        self.file_client_args = file_client_args.copy()
        self.file_client = None

    def __call__(self, results):
        if self.file_client is None:
            self.file_client = mmcv.FileClient(**self.file_client_args)

        if results["img_prefix"] is not None:
            filename = osp.join(results["img_prefix"], results["img_info"]["filename"])
        else:
            filename = results["img_info"]["filename"]

        if filename.endswith("npy"):
            img = np.load(filename)
            if img.shape[2] == 8:
                img = img[:, :, :3]
        else:
            img_bytes = self.file_client.get(filename)
            img = mmcv.imfrombytes(img_bytes, flag=self.color_type, channel_order=self.channel_order)
        if self.to_float32:
            img = img.astype(np.float32)

        results["filename"] = filename
        results["ori_filename"] = results["img_info"]["filename"]
        results["img"] = img
        results["img_shape"] = img.shape
        results["ori_shape"] = img.shape
        results["img_fields"] = ["img"]
        return results

    def __repr__(self):
        return (
            f"{self.__class__.__name__}("
            f"to_float32={self.to_float32}, "
            f"color_type='{self.color_type}', "
            f"channel_order='{self.channel_order}', "
            f"file_client_args={self.file_client_args})"
        )


class MotLoadAnnotations(LoadAnnotations):
    def __init__(
        self,
        with_bbox=True,
        with_label=True,
        with_mask=False,
        with_seg=False,
        poly2mask=False,
        denorm_bbox=False,
        file_client_args=dict(backend="disk"),
        with_trackid=True,
    ):
        super(MotLoadAnnotations, self).__init__(
            with_bbox=with_bbox,
            with_label=with_label,
            with_mask=with_mask,
            with_seg=with_seg,
            poly2mask=poly2mask,
            denorm_bbox=denorm_bbox,
            file_client_args=file_client_args,
        )
        self.with_trackid = with_trackid

    def __call__(self, results_list):
        for results in results_list:
            super(MotLoadAnnotations, self).__call__(results)
            if self.with_trackid:
                results = self._load_trackids(results)
        return results_list

    def _load_trackids(self, results):
        results["gt_trackids"] = results["ann_info"]["trackids"].copy()


class MotLoadImageFromFile(LoadImageFromFile):
    def __init__(
        self,
        to_float32=False,
        color_type="color",
        channel_order="bgr",
        file_client_args=dict(backend="disk"),
    ):
        super(MotLoadImageFromFile, self).__init__(
            to_float32=to_float32,
            color_type=color_type,
            channel_order=channel_order,
            file_client_args=file_client_args,
        )

    def __call__(self, results_list):
        for results in results_list:
            super(MotLoadImageFromFile, self).__call__(results)
        return results_list


class MotLoadMultichannelImageFromNpy(LoadMultichannelImageFromNpy):
    def __init__(
        self,
        to_float32=False,
        color_type="color",
        backend_args=dict(backend="disk"),
    ):
        super(MotLoadMultichannelImageFromNpy, self).__init__(
            to_float32=to_float32,
            color_type=color_type,
            backend_args=backend_args,
        )

    def __call__(self, result_list):
        for results in result_list:
            super(MotLoadMultichannelImageFromNpy, self).__call__(results)
        return result_list


class LoadMultichannelImageFrom3JPG(LoadMultichannelImageFromNpy):
    """从 3 张 JPG 还原 8 通道图像。"""

    def __call__(self, results):
        if results["img_prefix"] is not None:
            filename = osp.join(results["img_prefix"], results["img_info"]["filename"])
        else:
            filename = results["img_info"]["filename"]

        stem, ext = osp.splitext(filename)
        if stem.endswith(("_p1", "_p2", "_p3")):
            base_stem = stem.rsplit("_", 1)[0]
        else:
            base_stem = stem

        part_paths = [f"{base_stem}_p1{ext}", f"{base_stem}_p2{ext}", f"{base_stem}_p3{ext}"]
        part_imgs = []
        for part_path in part_paths:
            img = mmcv.imread(part_path, channel_order="rgb")
            assert img is not None, f"Failed to load image: {part_path}"
            part_imgs.append(img)

        img = np.concatenate([part_imgs[0], part_imgs[1], part_imgs[2][:, :, :2]], axis=2)
        if self.to_float32:
            img = img.astype(np.float32)

        results["filename"] = filename
        results["ori_filename"] = results["img_info"]["filename"]
        results["img"] = img
        results["img_shape"] = img.shape
        results["ori_shape"] = img.shape
        results["img_fields"] = ["img"]
        return results


class MotLoadMultichannelImageFrom3JPG(LoadMultichannelImageFrom3JPG):
    def __init__(
        self,
        to_float32=False,
        color_type="color",
        backend_args=dict(backend="disk"),
    ):
        super(MotLoadMultichannelImageFrom3JPG, self).__init__(
            to_float32=to_float32,
            color_type=color_type,
            backend_args=backend_args,
        )

    def __call__(self, result_list):
        for results in result_list:
            super(MotLoadMultichannelImageFrom3JPG, self).__call__(results)
        return result_list


class LoadRgbIrImageFromJPG(LoadMultichannelImageFromNpy):
    """从 RGB(/00/) 与 IR(/01/) JPG 加载 4 通道图像。"""

    def __call__(self, results):
        if results["img_prefix"] is not None:
            rgb_path = osp.join(results["img_prefix"], results["img_info"]["filename"])
        else:
            rgb_path = results["img_info"]["filename"]

        ir_path = rgb_path.replace("/00/", "/01/")
        assert osp.exists(rgb_path), f"RGB image not found: {rgb_path}"
        assert osp.exists(ir_path), f"IR image not found: {ir_path}"

        rgb = mmcv.imread(rgb_path, channel_order="rgb")
        assert rgb is not None, f"Failed to load RGB image: {rgb_path}"

        ir = mmcv.imread(ir_path, flag="grayscale")
        assert ir is not None, f"Failed to load IR image: {ir_path}"
        if ir.ndim == 2:
            ir = ir[..., np.newaxis]

        img = np.concatenate([rgb, ir], axis=2)
        if self.to_float32:
            img = img.astype(np.float32)

        results["filename"] = rgb_path
        results["ori_filename"] = results["img_info"]["filename"]
        results["img"] = img
        results["img_shape"] = img.shape
        results["ori_shape"] = img.shape
        results["img_fields"] = ["img"]
        return results


class MotLoadRgbIrImageFromJPG(LoadRgbIrImageFromJPG):
    def __init__(
        self,
        to_float32=False,
        color_type="color",
        backend_args=dict(backend="disk"),
    ):
        super(MotLoadRgbIrImageFromJPG, self).__init__(
            to_float32=to_float32,
            color_type=color_type,
            backend_args=backend_args,
        )

    def __call__(self, result_list):
        for results in result_list:
            super(MotLoadRgbIrImageFromJPG, self).__call__(results)
        return result_list


class RLoadProposalsScores:
    """Load proposal pipeline for rotated boxes."""

    def __init__(self, num_max_proposals=None):
        self.num_max_proposals = num_max_proposals

    def __call__(self, results):
        proposals = results["proposals"]
        if proposals.shape[1] not in (5, 6):
            raise AssertionError(f"proposals should have shapes (n, 5) or (n, 6), but found {proposals.shape}")
        proposals = proposals[:, :5]

        if self.num_max_proposals is not None:
            proposals = proposals[: self.num_max_proposals]

        if len(proposals) == 0:
            proposals = np.array([[0, 0, 0, 0, 0]], dtype=np.float32)
        results["proposals"] = proposals
        results["bbox_fields"].append("proposals")
        return results

    def __repr__(self):
        return self.__class__.__name__ + f"(num_max_proposals={self.num_max_proposals})"


class MotRLoadProposals(RLoadProposalsScores):
    def __init__(self, num_max_proposals=None):
        super(MotRLoadProposals, self).__init__(num_max_proposals)

    def __call__(self, results_list):
        for results in results_list:
            super(MotRLoadProposals, self).__call__(results)
        return results_list
