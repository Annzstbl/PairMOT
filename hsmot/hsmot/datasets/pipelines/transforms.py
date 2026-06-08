from hsmot.mmlab.hs_mmrotate import RResize, RRandomFlip, RRandomCrop
from hsmot.mmlab.hs_mmdet import Normalize, Pad, Resize, RandomFlip, RandomCrop

import numpy as np 
import mmcv
from mmcv.ops import box_iou_rotated
import torch
import copy as cp


class MotRRsize(RResize):
    
    def __init__(self,
                 img_scale=None,
                 multiscale_mode='range',
                 ratio_range=None, 
                 bbox_clip_border=True):
        super(MotRRsize, self).__init__(
            img_scale=img_scale,
            multiscale_mode=multiscale_mode,
            ratio_range=ratio_range)
        # override class Resize parameter
        self.bbox_clip_border = bbox_clip_border
    
    def __call__(self, results_list):
        if len(results_list) == 0:
            return results_list

        # 保证对于一个序列中的每个图像做相同的处理
        results = results_list[0]
        if 'scale' not in results:
            if 'scale_factor' in results:
                img_shape = results['img'].shape[:2]
                scale_factor = results['scale_factor']
                assert isinstance(scale_factor, float)
                results['scale'] = tuple(
                    [int(x * scale_factor) for x in img_shape][::-1])
            else:
                self._random_scale(results)
        else:
            if not self.override:
                assert 'scale_factor' not in results, (
                    'scale and scale_factor cannot be both set.')
            else:
                results.pop('scale')
                if 'scale_factor' in results:
                    results.pop('scale_factor')
                self._random_scale(results)

        for r in results_list[1:]:
            r['scale'] = results_list[0]['scale']
            r['scale_idx'] = results_list[0]['scale_idx']

        for r in results_list:
            self._resize_img(r)# 修改了'img_fileds'指向的字段
            self._resize_bboxes(r) #修改了'box_fields'指向的字段
            self._resize_masks(r)
            self._resize_seg(r)
        return results_list
    

class MotRRandomFlip(RRandomFlip):
    def __init__(self, flip_ratio=None, direction='horizontal', version='oc'):
        super(MotRRandomFlip, self).__init__(flip_ratio, direction, version)
    
    def __call__(self, results_list):

        if len(results_list) == 0:
            return results_list
        
        results_0 = results_list[0]
        if 'flip' not in results_0:
            if isinstance(self.direction, list):
                # None means non-flip
                direction_list = self.direction + [None]
            else:
                # None means non-flip
                direction_list = [self.direction, None]

            if isinstance(self.flip_ratio, list):
                non_flip_ratio = 1 - sum(self.flip_ratio)
                flip_ratio_list = self.flip_ratio + [non_flip_ratio]
            else:
                non_flip_ratio = 1 - self.flip_ratio
                # exclude non-flip
                single_ratio = self.flip_ratio / (len(direction_list) - 1)
                flip_ratio_list = [single_ratio] * (len(direction_list) -
                                                    1) + [non_flip_ratio]

            cur_dir = np.random.choice(direction_list, p=flip_ratio_list)

            results_0['flip'] = cur_dir is not None
        if 'flip_direction' not in results_0:
            results_0['flip_direction'] = cur_dir

        for r in results_list[1:]:
            r['flip_direction'] = results_list[0]['flip_direction']
            r['flip'] = results_list[0]['flip']

        if results_0['flip']:
            for results in results_list:
                # flip image
                for key in results.get('img_fields', ['img']):
                    results[key] = mmcv.imflip(
                        results[key], direction=results['flip_direction'])
                # flip bboxes
                for key in results.get('bbox_fields', []):
                    results[key] = self.bbox_flip(results[key],
                                                results['img_shape'],
                                                results['flip_direction'])
                # flip masks
                for key in results.get('mask_fields', []):
                    results[key] = results[key].flip(results['flip_direction'])

                # flip segs
                for key in results.get('seg_fields', []):
                    results[key] = mmcv.imflip(
                        results[key], direction=results['flip_direction'])
                    
        return results_list


class MotRRandomCrop(RRandomCrop):
    """
    Random crop for MOT pipeline.

    Constraints in this project:
    - only supports crop_type = 'absolute_w_range'
    - keep_ratio must be True
    - crop_size is interpreted as (w_min, w_max)
    """

    # iof_thr=0.5表示中心点在crop区域内的bbox才会被保留
    def __init__(self, crop_size, crop_type='absolute_w_range', allow_negative_crop=False,
                 iof_thr=0.5, version='oc', keep_ratio=True):
        if crop_type != 'absolute_w_range':
            raise NotImplementedError(
                f"MotRRandomCrop only supports crop_type='absolute_w_range', got '{crop_type}'."
            )
        if keep_ratio is not True:
            raise ValueError("MotRRandomCrop requires keep_ratio=True.")
        if len(crop_size) != 2:
            raise ValueError(f"crop_size must be (w_min, w_max), got {crop_size}.")
        w_min, w_max = int(crop_size[0]), int(crop_size[1])
        if w_min <= 0 or w_max <= 0 or w_min > w_max:
            raise ValueError(f"Invalid crop_size (w_min, w_max): {crop_size}.")

        self.keep_ratio = keep_ratio
        # Parent RandomCrop validates crop_type and does not know 'absolute_w_range'.
        # Use a supported type for parent init, then keep our own mode flag.
        super().__init__(crop_size=(w_min, w_max), crop_type='absolute_range',
                         allow_negative_crop=allow_negative_crop, iof_thr=iof_thr, version=version)
        self.crop_type = 'absolute_w_range'
        self.bbox2track_id = {
            'gt_bboxes': 'gt_trackids',
            'gt_bboxes_ignore': 'gt_trackids_ignore'
        }
        self.box2scores = {
            'proposals': "proposal_scores"
        }

    def _get_crop_size(self, image_size):
        """
        Sample crop size in absolute pixels.

        For absolute_w_range mode:
        1) Randomly sample crop_w from [w_min, w_max] clipped by image width.
        2) Keep original aspect ratio and compute crop_h from crop_w.
        """
        if self.crop_type != 'absolute_w_range':
            raise NotImplementedError(
                f"MotRRandomCrop only supports crop_type='absolute_w_range', got '{self.crop_type}'."
            )
        if self.keep_ratio is not True:
            raise ValueError("MotRRandomCrop requires keep_ratio=True.")

        h, w = image_size
        w_min, w_max = int(self.crop_size[0]), int(self.crop_size[1])
        low_w = min(w, w_min)
        high_w = min(w, w_max)
        if low_w > high_w:
            low_w = high_w

        crop_w = int(np.random.randint(low_w, high_w + 1))
        aspect = float(h) / float(w)
        crop_h = int(round(crop_w * aspect))
        crop_h = max(1, min(h, crop_h))
        crop_w = max(1, min(w, crop_w))
        return crop_h, crop_w

    def _crop_data(self, results, crop_size, allow_negative_crop):
        """Function to randomly crop images, bounding boxes.

        Args:
            results (dict): Result dict from loading pipeline.
            crop_size (tuple): Expected absolute size after cropping, (h, w).
            allow_negative_crop (bool): Whether to allow a crop that does not
                contain any bbox area. Default to False.

        Returns:
            dict: Randomly cropped results, 'img_shape' key in result dict is
                updated according to crop size.
        """
        assert crop_size[0] > 0 and crop_size[1] > 0
        for key in results.get('bbox_fields', []):
            assert results[key].shape[-1] % 5 == 0

        for key in results.get('img_fields', ['img']):
            img = results[key]
            margin_h = max(img.shape[0] - crop_size[0], 0)
            margin_w = max(img.shape[1] - crop_size[1], 0)
            offset_h = np.random.randint(0, margin_h + 1)
            offset_w = np.random.randint(0, margin_w + 1)
            crop_y1, crop_y2 = offset_h, offset_h + crop_size[0]
            crop_x1, crop_x2 = offset_w, offset_w + crop_size[1]

            # crop the image
            img = img[crop_y1:crop_y2, crop_x1:crop_x2, ...]
            img_shape = img.shape
            results[key] = img
        results['img_shape'] = img_shape

        height, width, _ = img_shape

        # crop bboxes accordingly and clip to the image boundary
        for key in results.get('bbox_fields', []):
            # e.g. gt_bboxes and gt_bboxes_ignore
            bbox_offset = np.array([offset_w, offset_h, 0, 0, 0],
                                   dtype=np.float32)
            bboxes = results[key] - bbox_offset

            windows = np.array([width / 2, height / 2, width, height, 0],
                               dtype=np.float32).reshape(-1, 5)

            valid_inds = box_iou_rotated(
                torch.tensor(bboxes), torch.tensor(windows),
                mode='iof').numpy().reshape(-1) > self.iof_thr

            # If the crop does not contain any gt-bbox area and
            # allow_negative_crop is False, skip this image.
            if (key == 'gt_bboxes' and not valid_inds.any()
                    and not allow_negative_crop):
                return None
            results[key] = bboxes[valid_inds, :]
            # label fields. e.g. gt_labels and gt_labels_ignore
            label_key = self.bbox2label.get(key)
            if label_key in results:
                results[label_key] = results[label_key][valid_inds]
            trackid_key = self.bbox2track_id.get(key)
            if trackid_key in results:
                results[trackid_key] = results[trackid_key][valid_inds]
            score_key = self.box2scores.get(key)
            if score_key in results:
                results[score_key] = results[score_key][valid_inds]

        return results

    def __call__(self, results_list):
        """Call function to randomly crop images, bounding boxes, masks,
        semantic segmentation maps.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: Randomly cropped results, 'img_shape' key in result dict is
                updated according to crop size.
        """
        if len(results_list) == 0:
            return results_list
        results_0 = results_list[0]

        image_size = results_0['img'].shape[:2]
        crop_size = self._get_crop_size(image_size)
        # results = self._crop_data(results, crop_size, self.allow_negative_crop)

        if not self.allow_negative_crop: # don't allow negative crop, when crop failed, return uncroped results
            unchanged_results = cp.deepcopy(results_list)

        for i in range(len(results_list)):
            results_list[i] = self._crop_data(results_list[i], crop_size, self.allow_negative_crop)
            if results_list[i]:
                results_list[i]['crop_size'] = crop_size
            else: # crop failed when allow_negative_crop is False
                for i in range(len(unchanged_results)):
                    unchanged_results[i]['crop_size'] = (-1, -1)
                return unchanged_results
        return results_list
        

class MotHResize(Resize):
    """水平正框 Resize，clip 内各帧共享 scale。

    resize_hw_range 非空时，在 (w_min, w_max) 与 (h_min, h_max) 内独立随机采样目标尺寸。
    """

    def __init__(self,
                 img_scale=None,
                 multiscale_mode='range',
                 ratio_range=None,
                 keep_ratio=False,
                 bbox_clip_border=True,
                 resize_hw_range=None):
        super(MotHResize, self).__init__(
            img_scale=img_scale,
            multiscale_mode=multiscale_mode,
            ratio_range=ratio_range,
            keep_ratio=keep_ratio,
            bbox_clip_border=bbox_clip_border,
        )
        self.resize_hw_range = resize_hw_range

    def _random_scale(self, results):
        if self.resize_hw_range is not None:
            w_min, w_max, h_min, h_max = self.resize_hw_range
            resize_w = int(np.random.randint(w_min, w_max + 1))
            resize_h = int(np.random.randint(h_min, h_max + 1))
            results['scale'] = (resize_w, resize_h)
            results['scale_idx'] = None
            return
        super(MotHResize, self)._random_scale(results)

    def __call__(self, results_list):
        if len(results_list) == 0:
            return results_list

        results = results_list[0]
        if 'scale' not in results:
            if 'scale_factor' in results:
                img_shape = results['img'].shape[:2]
                scale_factor = results['scale_factor']
                assert isinstance(scale_factor, float)
                results['scale'] = tuple(
                    [int(x * scale_factor) for x in img_shape][::-1])
            else:
                self._random_scale(results)
        else:
            if not self.override:
                assert 'scale_factor' not in results, (
                    'scale and scale_factor cannot be both set.')
            else:
                results.pop('scale')
                if 'scale_factor' in results:
                    results.pop('scale_factor')
                self._random_scale(results)

        for r in results_list[1:]:
            r['scale'] = results_list[0]['scale']
            r['scale_idx'] = results_list[0].get('scale_idx')

        for r in results_list:
            self._resize_img(r)
            self._resize_bboxes(r)
            self._resize_masks(r)
            self._resize_seg(r)
        return results_list


class MotHRandomFlip(RandomFlip):
    """水平正框 RandomFlip，clip 内各帧同步。"""

    def __call__(self, results_list):
        if len(results_list) == 0:
            return results_list

        results_0 = results_list[0]
        if 'flip' not in results_0:
            if isinstance(self.direction, list):
                direction_list = self.direction + [None]
            else:
                direction_list = [self.direction, None]

            if isinstance(self.flip_ratio, list):
                non_flip_ratio = 1 - sum(self.flip_ratio)
                flip_ratio_list = self.flip_ratio + [non_flip_ratio]
            else:
                non_flip_ratio = 1 - self.flip_ratio
                single_ratio = self.flip_ratio / (len(direction_list) - 1)
                flip_ratio_list = [single_ratio] * (len(direction_list) - 1) + [non_flip_ratio]

            cur_dir = np.random.choice(direction_list, p=flip_ratio_list)
            results_0['flip'] = cur_dir is not None
            results_0['flip_direction'] = cur_dir
        elif 'flip_direction' not in results_0:
            results_0['flip_direction'] = None

        for r in results_list[1:]:
            r['flip'] = results_list[0]['flip']
            r['flip_direction'] = results_list[0]['flip_direction']

        if results_0['flip']:
            for results in results_list:
                for key in results.get('img_fields', ['img']):
                    results[key] = mmcv.imflip(
                        results[key], direction=results['flip_direction'])
                for key in results.get('bbox_fields', []):
                    results[key] = self.bbox_flip(
                        results[key], results['img_shape'], results['flip_direction'])
                for key in results.get('mask_fields', []):
                    results[key] = results[key].flip(results['flip_direction'])
                for key in results.get('seg_fields', []):
                    results[key] = mmcv.imflip(
                        results[key], direction=results['flip_direction'])
        return results_list


class MotHRandomCrop(RandomCrop):
    """水平正框 RandomCrop，clip 内各帧共享 crop。

    仅支持 crop_type='absolute_w_range'，与 MotRRandomCrop 参数语义一致。
    keep_ratio=True 时 crop 高宽比与原图一致；False 时宽高独立采样。
    """

    def __init__(self, crop_size, crop_type='absolute_w_range', allow_negative_crop=False,
                 iof_thr=0.5, keep_ratio=False):
        if crop_type != 'absolute_w_range':
            raise NotImplementedError(
                f"MotHRandomCrop only supports crop_type='absolute_w_range', got '{crop_type}'."
            )
        if len(crop_size) == 4:
            w_min, w_max, h_min, h_max = (int(crop_size[0]), int(crop_size[1]),
                                          int(crop_size[2]), int(crop_size[3]))
        elif len(crop_size) == 2:
            w_min, w_max = int(crop_size[0]), int(crop_size[1])
            h_min, h_max = w_min, w_max
        else:
            raise ValueError(
                f'crop_size must be (w_min, w_max) or (w_min, w_max, h_min, h_max), got {crop_size}.')

        if w_min <= 0 or w_max <= 0 or w_min > w_max:
            raise ValueError(f'Invalid crop width range: ({w_min}, {w_max}).')
        if h_min <= 0 or h_max <= 0 or h_min > h_max:
            raise ValueError(f'Invalid crop height range: ({h_min}, {h_max}).')

        self.keep_ratio = keep_ratio
        self.iof_thr = iof_thr
        super().__init__(
            crop_size=(w_min, w_max),
            crop_type='absolute_range',
            allow_negative_crop=allow_negative_crop,
        )
        self.crop_hw_range = (h_min, h_max)
        self.crop_type = 'absolute_w_range'
        self.bbox2track_id = {
            'gt_bboxes': 'gt_trackids',
            'gt_bboxes_ignore': 'gt_trackids_ignore',
        }

    def _get_crop_size(self, image_size):
        h, w = image_size
        w_min, w_max = int(self.crop_size[0]), int(self.crop_size[1])
        h_min, h_max = int(self.crop_hw_range[0]), int(self.crop_hw_range[1])
        low_w = min(w, w_min)
        high_w = min(w, w_max)
        if low_w > high_w:
            low_w = high_w

        crop_w = int(np.random.randint(low_w, high_w + 1))
        if self.keep_ratio:
            aspect = float(h) / float(w)
            crop_h = int(round(crop_w * aspect))
        else:
            low_h = min(h, h_min)
            high_h = min(h, h_max)
            if low_h > high_h:
                low_h = high_h
            crop_h = int(np.random.randint(low_h, high_h + 1))

        crop_h = max(1, min(h, crop_h))
        crop_w = max(1, min(w, crop_w))
        return crop_h, crop_w

    def _crop_data(self, results, crop_size, allow_negative_crop):
        assert crop_size[0] > 0 and crop_size[1] > 0
        for key in results.get('bbox_fields', []):
            assert results[key].shape[-1] % 4 == 0

        for key in results.get('img_fields', ['img']):
            img = results[key]
            margin_h = max(img.shape[0] - crop_size[0], 0)
            margin_w = max(img.shape[1] - crop_size[1], 0)
            offset_h = np.random.randint(0, margin_h + 1)
            offset_w = np.random.randint(0, margin_w + 1)
            crop_y1, crop_y2 = offset_h, offset_h + crop_size[0]
            crop_x1, crop_x2 = offset_w, offset_w + crop_size[1]

            img = img[crop_y1:crop_y2, crop_x1:crop_x2, ...]
            img_shape = img.shape
            results[key] = img
        results['img_shape'] = img_shape

        height, width = img_shape[:2]
        for key in results.get('bbox_fields', []):
            bbox_offset = np.array([offset_w, offset_h, offset_w, offset_h], dtype=np.float32)
            bboxes = results[key] - bbox_offset

            cx = (bboxes[:, 0] + bboxes[:, 2]) * 0.5
            cy = (bboxes[:, 1] + bboxes[:, 3]) * 0.5
            valid_inds = (cx >= 0) & (cx < width) & (cy >= 0) & (cy < height)

            if key == 'gt_bboxes' and not valid_inds.any() and not allow_negative_crop:
                return None

            results[key] = bboxes[valid_inds, :]
            label_key = self.bbox2label.get(key)
            if label_key in results:
                results[label_key] = results[label_key][valid_inds]
            trackid_key = self.bbox2track_id.get(key)
            if trackid_key in results:
                results[trackid_key] = results[trackid_key][valid_inds]

        for key in results.get('seg_fields', []):
            results[key] = results[key][crop_y1:crop_y2, crop_x1:crop_x2]

        return results

    def __call__(self, results_list):
        if len(results_list) == 0:
            return results_list

        crop_size = self._get_crop_size(results_list[0]['img'].shape[:2])
        if not self.allow_negative_crop:
            unchanged_results = cp.deepcopy(results_list)

        for i in range(len(results_list)):
            results_list[i] = self._crop_data(
                results_list[i], crop_size, self.allow_negative_crop)
            if results_list[i]:
                results_list[i]['crop_size'] = crop_size
            else:
                for j in range(len(unchanged_results)):
                    unchanged_results[j]['crop_size'] = (-1, -1)
                return unchanged_results
        return results_list


class MotNormalize(Normalize):

    def __call__(self, results_list):
        for results in results_list:
            super(MotNormalize, self).__call__(results)
        return results_list


class MotPad(Pad):

    def __call__(self, results_list):
        for results in results_list:
            super(MotPad, self).__call__(results)
        return results_list
