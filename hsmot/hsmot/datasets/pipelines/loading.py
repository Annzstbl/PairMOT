# Copyright (c) OpenMMLab. All rights reserved.
import mmcv
import numpy as np
from hsmot.mmlab.hs_mmdet import LoadImageFromFile, LoadAnnotations
import mmengine
import mmengine.fileio as fileio
import cv2

import os.path as osp


# @ROTATED_PIPELINES.register_module()
# 已经在自己的mmrotate实现过 没找到
class LoadMultichannelImageFromNpy:
    """从npy文件加载一个多光谱图像
    Load an image from file.

    Required keys are "img_prefix" and "img_info" (a dict that must contain the
    key "filename"). Added or updated keys are "filename", "img", "img_shape",
    "ori_shape" (same as `img_shape`), "pad_shape" (same as `img_shape`),
    "scale_factor" (1.0) and "img_norm_cfg" (means=0 and stds=1).

    Args:
        to_float32 (bool): Whether to convert the loaded image to a float32
            numpy array. If set to False, the loaded image is an uint8 array.
            Defaults to False.
        color_type (str): The flag argument for :func:`mmcv.imfrombytes`.
            Defaults to 'color'.
        file_client_args (dict): Arguments to instantiate a FileClient.
            See :class:`mmcv.fileio.FileClient` for details.
            Defaults to ``dict(backend='disk')``.
    """

    def __init__(self,
                 to_float32=False,
                 color_type='color',
                 channel_order='bgr',
                 backend_args= dict(backend='disk')):

        self.to_float32 = to_float32
        self.color_type = color_type
        self.channel_order = channel_order
        # self.file_client_args = file_client_args.copy()
        # self.file_client = None
        self.backend_args = backend_args.copy()

    def __call__(self, results):
        """Call functions to load image and get image meta information.

        Args:
            results (dict): Result dict from :obj:`mmdet.CustomDataset`.

        Returns:
            dict: The dict contains loaded image and meta information.
        """

        # if self.file_client is None:
        #     self.file_client = mmengine.FileClient(**self.file_client_args)

        if results['img_prefix'] is not None:
            filename = osp.join(results['img_prefix'],
                                results['img_info']['filename'])
        else:
            filename = results['img_info']['filename']

        # 如果filename是以npy结尾的
        if filename.endswith('npy'):
            img = np.load(filename)
        else:
            img = mmcv.imread(filename)
            cv2.cvtColor(img, cv2.COLOR_BGR2RGB, img)
            #TODO
            # 不知道为什么以下提示 backends 不包含disk key
            # img_bytes = fileio.get(
            #     filename, backend_args=self.backend_args)
            # # img_bytes = self.file_client.get(filename)
            # img = mmcv.imfrombytes(
            #     img_bytes, flag=self.color_type, channel_order=self.channel_order)
        if self.to_float32:
            img = img.astype(np.float32)

        results['filename'] = filename
        results['ori_filename'] = results['img_info']['filename']
        results['img'] = img
        results['img_shape'] = img.shape
        results['ori_shape'] = img.shape
        results['img_fields'] = ['img']
        return results

    def __repr__(self):
        repr_str = (f'{self.__class__.__name__}('
                    f'to_float32={self.to_float32}, '
                    f"color_type='{self.color_type}', "
                    f"channel_order='{self.channel_order}', "
                    f'file_client_args={self.file_client_args})')
        return repr_str


# @ROTATED_PIPELINES.register_module()
# 已经在自己的mmrotate实现过
class LoadRgbImageFromNpy:
    """从npy文件加载一个多光谱图像
    Load an image from file.

    Required keys are "img_prefix" and "img_info" (a dict that must contain the
    key "filename"). Added or updated keys are "filename", "img", "img_shape",
    "ori_shape" (same as `img_shape`), "pad_shape" (same as `img_shape`),
    "scale_factor" (1.0) and "img_norm_cfg" (means=0 and stds=1).

    Args:
        to_float32 (bool): Whether to convert the loaded image to a float32
            numpy array. If set to False, the loaded image is an uint8 array.
            Defaults to False.
        color_type (str): The flag argument for :func:`mmcv.imfrombytes`.
            Defaults to 'color'.
        file_client_args (dict): Arguments to instantiate a FileClient.
            See :class:`mmcv.fileio.FileClient` for details.
            Defaults to ``dict(backend='disk')``.
    """

    def __init__(self,
                 to_float32=False,
                 color_type='color',
                 channel_order='bgr',
                 file_client_args=dict(backend='disk')):
        self.to_float32 = to_float32
        self.color_type = color_type
        self.channel_order = channel_order
        self.file_client_args = file_client_args.copy()
        self.file_client = None

    def __call__(self, results):
        """Call functions to load image and get image meta information.

        Args:
            results (dict): Result dict from :obj:`mmdet.CustomDataset`.

        Returns:
            dict: The dict contains loaded image and meta information.
        """

        if self.file_client is None:
            self.file_client = mmengine.FileClient(**self.file_client_args)

        if results['img_prefix'] is not None:
            filename = osp.join(results['img_prefix'],
                                results['img_info']['filename'])
        else:
            filename = results['img_info']['filename']

        # 如果filename是以npy结尾的
        if filename.endswith('npy'):
            img = np.load(filename)
            # 如果img是8通道
            if img.shape[2] == 8:
                img = img[:, :, :3]
        else:
            img_bytes = self.file_client.get(filename)
            img = mmcv.imfrombytes(
                img_bytes, flag=self.color_type, channel_order=self.channel_order)
        if self.to_float32:
            img = img.astype(np.float32)

        results['filename'] = filename
        results['ori_filename'] = results['img_info']['filename']
        results['img'] = img
        results['img_shape'] = img.shape
        results['ori_shape'] = img.shape
        results['img_fields'] = ['img']
        return results

    def __repr__(self):
        repr_str = (f'{self.__class__.__name__}('
                    f'to_float32={self.to_float32}, '
                    f"color_type='{self.color_type}', "
                    f"channel_order='{self.channel_order}', "
                    f'file_client_args={self.file_client_args})')
        return repr_str


class MotLoadAnnotations(LoadAnnotations):
    def __init__(self,
            with_bbox=True,
            with_label=True,
            with_mask=False,
            with_seg=False,
            poly2mask=False,
            denorm_bbox=False,
            file_client_args=dict(backend='disk'),
            with_trackid=True):
        super(MotLoadAnnotations, self).__init__(
            with_bbox=with_bbox,
            with_label=with_label,
            with_mask=with_mask,
            with_seg=with_seg,
            poly2mask=poly2mask,
            denorm_bbox=denorm_bbox,
            file_client_args=file_client_args
        )
        self.with_trackid = with_trackid
        
    def __call__(self, results_list):
        for results in results_list:
            # process with_bbox with_label with_mask with_seg
            super(MotLoadAnnotations, self).__call__(results)
            # process with_trackids
            if self.with_trackid:
                results = self._load_trackids(results)
            
        return results_list
    
    def _load_trackids(self, results):
        results['gt_trackids'] = results['ann_info']['trackids'].copy()

    
class MotLoadImageFromFile(LoadImageFromFile):
    def __init__(self,
                to_float32=False,
                color_type='color',
                channel_order='bgr',
                file_client_args=dict(backend='disk')):
        super(MotLoadImageFromFile, self).__init__(
            to_float32 = to_float32,
            color_type = color_type,
            channel_order = channel_order,
            file_client_args = file_client_args
        )
    
    def __call__(self, results_list):
        for results in results_list:
            super(MotLoadImageFromFile, self).__call__(results)
        return results_list


class MotLoadMultichannelImageFromNpy(LoadMultichannelImageFromNpy):
    def __init__(self, 
                 to_float32=False,
                 color_type='color',
                 backend_args= dict(backend='disk')):
        super(MotLoadMultichannelImageFromNpy, self).__init__(
            to_float32 = to_float32,
            color_type = color_type,
            backend_args = backend_args
        )
    
    def __call__(self, result_list):
        for results in result_list:
            super(MotLoadMultichannelImageFromNpy, self).__call__(results)
        return result_list


class LoadMultichannelImageFrom3JPG(LoadMultichannelImageFromNpy):
    """从 3 张 JPG 还原 8 通道图像。

    约定文件命名为 ``000001_p1.jpg``、``000001_p2.jpg``、``000001_p3.jpg``，
    分别对应通道 ``[0,1,2]``、``[3,4,5]``、``[6,7,0]``，因此最后一张只保留前两通道。
    """

    def __call__(self, results):
        if results['img_prefix'] is not None:
            filename = osp.join(results['img_prefix'], results['img_info']['filename'])
        else:
            filename = results['img_info']['filename']

        stem, ext = osp.splitext(filename)
        if stem.endswith(('_p1', '_p2', '_p3')):
            base_stem = stem.rsplit('_', 1)[0]
        else:
            base_stem = stem

        part_paths = [f'{base_stem}_p1{ext}', f'{base_stem}_p2{ext}', f'{base_stem}_p3{ext}']
        part_imgs = []
        for part_path in part_paths:
            img = mmcv.imread(part_path)
            assert img is not None, f'Failed to load image: {part_path}'
            cv2.cvtColor(img, cv2.COLOR_BGR2RGB, img)
            part_imgs.append(img)

        img = np.concatenate([part_imgs[0], part_imgs[1], part_imgs[2][:, :, :2]], axis=2)
        if self.to_float32:
            img = img.astype(np.float32)

        results['filename'] = filename
        results['ori_filename'] = results['img_info']['filename']
        results['img'] = img
        results['img_shape'] = img.shape
        results['ori_shape'] = img.shape
        results['img_fields'] = ['img']
        return results


class MotLoadMultichannelImageFrom3JPG(LoadMultichannelImageFrom3JPG):
    def __init__(self,
                 to_float32=False,
                 color_type='color',
                 backend_args=dict(backend='disk')):
        super(MotLoadMultichannelImageFrom3JPG, self).__init__(
            to_float32=to_float32,
            color_type=color_type,
            backend_args=backend_args
        )

    def __call__(self, result_list):
        for results in result_list:
            super(MotLoadMultichannelImageFrom3JPG, self).__call__(results)
        return result_list


class LoadRgbIrImageFromJPG(LoadMultichannelImageFromNpy):
    """从 RGB(/00/) 与 IR(/01/) JPG 加载 4 通道图像。"""

    def __call__(self, results):
        if results['img_prefix'] is not None:
            rgb_path = osp.join(results['img_prefix'], results['img_info']['filename'])
        else:
            rgb_path = results['img_info']['filename']

        ir_path = rgb_path.replace('/00/', '/01/')
        assert osp.exists(rgb_path), f'RGB image not found: {rgb_path}'
        assert osp.exists(ir_path), f'IR image not found: {ir_path}'

        rgb = mmcv.imread(rgb_path)
        assert rgb is not None, f'Failed to load RGB image: {rgb_path}'
        cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB, rgb)

        ir = mmcv.imread(ir_path, flag='grayscale')
        assert ir is not None, f'Failed to load IR image: {ir_path}'
        if ir.ndim == 2:
            ir = ir[..., np.newaxis]

        img = np.concatenate([rgb, ir], axis=2)
        if self.to_float32:
            img = img.astype(np.float32)

        results['filename'] = rgb_path
        results['ori_filename'] = results['img_info']['filename']
        results['img'] = img
        results['img_shape'] = img.shape
        results['ori_shape'] = img.shape
        results['img_fields'] = ['img']
        return results


class MotLoadRgbIrImageFromJPG(LoadRgbIrImageFromJPG):
    def __init__(self,
                 to_float32=False,
                 color_type='color',
                 backend_args=dict(backend='disk')):
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
    """Load proposal pipeline.
        旋转目标框
        scores以proposal_scores的形式存在

    Required key is "proposals". Updated keys are "proposals", "bbox_fields".

    Args:
        num_max_proposals (int, optional): Maximum number of proposals to load.
            If not specified, all proposals will be loaded.
    """

    def __init__(self, num_max_proposals=None):
        self.num_max_proposals = num_max_proposals

    def __call__(self, results):
        """Call function to load proposals from file.

        Args:
            results (dict): Result dict from :obj:`mmdet.CustomDataset`.

        Returns:
            dict: The dict contains loaded proposal annotations.
        """

        proposals = results['proposals']
        if proposals.shape[1] not in (5, 6):# 5是旋转框，6是旋转框+score
            raise AssertionError(
                'proposals should have shapes (n, 5) or (n, 6), '
                f'but found {proposals.shape}')
        proposals = proposals[:, :5]

        if self.num_max_proposals is not None:
            proposals = proposals[:self.num_max_proposals]

        if len(proposals) == 0:
            proposals = np.array([[0, 0, 0, 0, 0]], dtype=np.float32)
        results['proposals'] = proposals
        results['bbox_fields'].append('proposals')
        return results

    def __repr__(self):
        return self.__class__.__name__ + \
            f'(num_max_proposals={self.num_max_proposals})'

   
class MotRLoadProposals(RLoadProposalsScores):
    def __init__(self, num_max_proposals=None):
        super(MotRLoadProposals, self).__init__(num_max_proposals)
    
    def __call__(self, results_list):
        for results in results_list:
            super(MotRLoadProposals, self).__call__(results)
        return results_list