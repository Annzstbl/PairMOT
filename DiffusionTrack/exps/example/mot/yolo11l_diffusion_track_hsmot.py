# encoding: utf-8
"""Stage-2 HSMOT Pair Diffusion tracking, preserving original training logic."""

import torch.distributed as dist

from exps.example.mot.yolo11l_diffusion_det_hsmot import (
    Exp as DetectionExp, HSMOT_MEAN, HSMOT_STD)


class Exp(DetectionExp):
    def __init__(self):
        super().__init__()
        self.exp_name = "yolo11l_diffusion_track_hsmot"
        self.task = "tracking"
        # Stage two is initialized from the complete stage-one checkpoint.
        self.yolo11_weights = ""
        self.freeze_yolo11_backbone = True

    def get_data_loader(self, batch_size, is_distributed, no_aug=False):
        from yolox.data import (DataLoader, DiffusionMosaicDetection,
                                DiffusionTrainTransform, HSMOTDataset,
                                InfiniteSampler, YoloBatchSampler)
        transform = DiffusionTrainTransform(
            rgb_means=HSMOT_MEAN, std=HSMOT_STD, max_labels=500)
        dataset = HSMOTDataset(
            data_dir=self.train_data_dir, img_size=self.input_size,
            preproc=None, ann_subdir="mot", img_subdir="npy")
        dataset = DiffusionMosaicDetection(
            dataset, mosaic=not no_aug, img_size=self.input_size,
            preproc=transform, degrees=self.degrees, translate=self.translate,
            scale=self.scale, shear=self.shear, perspective=self.perspective,
            enable_mixup=self.enable_mixup)
        self.dataset = dataset
        if is_distributed:
            batch_size //= dist.get_world_size()
        sampler = InfiniteSampler(len(dataset), seed=self.seed or 0)
        batch_sampler = YoloBatchSampler(
            sampler=sampler, batch_size=batch_size, drop_last=False,
            input_dimension=self.input_size, mosaic=not no_aug)
        return DataLoader(
            dataset, batch_sampler=batch_sampler,
            num_workers=self.data_num_workers, pin_memory=True)
