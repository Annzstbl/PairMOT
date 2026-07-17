# encoding: utf-8
"""Stage-1 HSMOT detector: YOLO11L + baseline Conv3D-SE + rotated diffusion."""

import os

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.optim import AdamW

from yolox.exp import Exp as MyExp


_PAIRMOT_ROOT = os.path.abspath(os.path.join(
    os.path.dirname(__file__), "..", "..", "..", ".."))
HSMOT_MEAN = (0.27358221, 0.28804452, 0.28133921, 0.26906377,
              0.28309119, 0.26928305, 0.28372527, 0.27149373)
HSMOT_STD = (0.19756629, 0.17432339, 0.16413284, 0.17581682,
             0.18366176, 0.15368450, 0.15964683, 0.16557951)


class Exp(MyExp):
    def __init__(self):
        super().__init__()
        self.num_classes = 8
        self.target_dim = 9
        self.depth = 1.33
        self.width = 1.25
        self.exp_name = os.path.splitext(os.path.basename(__file__))[0]
        self.input_size = (900, 1200)
        self.test_size = (900, 1200)
        # Keep the original multi-scale span while matching HSMOT's 4:3 ratio.
        self.random_size = (18, 28)
        self.max_epoch = 20
        self.print_interval = 20
        self.save_interval = 5
        self.eval_interval = 5
        # Keep Stage-1 validation/checkpoint cadence unchanged after mosaic is
        # disabled; the upstream trainer otherwise validates every epoch.
        self.no_aug_eval_interval = 5
        self.no_aug_epochs = 5
        self.basic_lr_per_img = 0.001 / 64.0
        self.warmup_epochs = 1
        self.task = "detection"
        self.enable_mixup = True
        self.seed = 8823
        self.conf_thresh = 0.4
        self.det_thresh = 0.6
        self.nms_thresh2d = 0.75
        self.nms_thresh3d = 0.7
        self.interval = 5
        self.data_num_workers = 4
        self.train_data_dir = os.environ.get(
            "HSMOT_TRAIN_ROOT", os.path.join(_PAIRMOT_ROOT, "data/hsmot/train"))
        self.val_data_dir = os.environ.get(
            "HSMOT_VAL_ROOT", os.path.join(_PAIRMOT_ROOT, "data/hsmot/test"))
        self.yolo11_cfg = "yolo11l-obb.yaml"
        self.yolo11_weights = os.environ.get(
            "YOLO11_WEIGHTS",
            "/data4/litianhao/PairMmot/pretrained_weights/mmot_official/"
            "yolo11L-8ch-3dstem.pt")
        self.freeze_yolo11_backbone = False

    def _dataset(self, root, transform):
        from yolox.data import HSMOTDataset
        return HSMOTDataset(
            data_dir=root, img_size=self.input_size, preproc=transform,
            ann_subdir="mot", img_subdir="npy")

    def get_data_loader(self, batch_size, is_distributed, no_aug=False):
        from yolox.data import (DataLoader, InfiniteSampler, MosaicDetection,
                                TrainTransform, YoloBatchSampler)
        transform = TrainTransform(
            rgb_means=HSMOT_MEAN, std=HSMOT_STD, max_labels=500)
        dataset = self._dataset(self.train_data_dir, transform)
        dataset = MosaicDetection(
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

    def get_eval_loader(self, batch_size, is_distributed, testdev=False):
        from yolox.data import DiffusionValTransform
        dataset = self._dataset(
            self.val_data_dir,
            DiffusionValTransform(
                rgb_means=HSMOT_MEAN, std=HSMOT_STD, max_labels=500))
        sampler = (torch.utils.data.distributed.DistributedSampler(
            dataset, shuffle=False) if is_distributed else
            torch.utils.data.SequentialSampler(dataset))
        if is_distributed:
            batch_size //= dist.get_world_size()
        return torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, sampler=sampler,
            num_workers=self.data_num_workers, pin_memory=True)

    def get_model(self):
        from diffusion.models.diffusion_head import DiffusionHead
        from diffusion.models.diffusionnet import DiffusionNet
        from yolox.models import YOLO11BackboneAdapter

        def init_bn(module):
            if isinstance(module, nn.BatchNorm2d):
                module.eps, module.momentum = 1e-3, 0.03

        if getattr(self, "model", None) is None:
            weights = self.yolo11_weights
            if self.task == "detection" and not os.path.isfile(weights):
                raise FileNotFoundError(
                    "MMOT official YOLO11 weight is required for Det stage: {}".
                    format(weights))
            backbone = YOLO11BackboneAdapter(
                model_cfg=self.yolo11_cfg, weights=weights,
                freeze=self.freeze_yolo11_backbone, num_spectral=8)
            self.model = DiffusionNet(
                backbone, DiffusionHead(self.num_classes, self.width))
            self.model.apply(init_bn)
        return self.model

    def get_evaluator(self, batch_size, is_distributed, testdev=False):
        from yolox.evaluators import HSMOTRotatedDetectionEvaluator
        # Rotated detection AP is evaluated as one deterministic stream on
        # rank 0; the evaluator keeps the other DDP ranks synchronized.
        loader = self.get_eval_loader(1, False, testdev)
        return HSMOTRotatedDetectionEvaluator(
            dataloader=loader, num_classes=self.num_classes,
            confthre=0.001, detthre=0.001,
            nmsthre3d=self.nms_thresh3d, nmsthre2d=self.nms_thresh2d,
            amp=True)

    def get_optimizer(self, batch_size):
        if "optimizer" not in self.__dict__:
            self.optimizer = AdamW(
                [parameter for parameter in self.model.parameters()
                 if parameter.requires_grad],
                lr=2.5e-5, weight_decay=1e-4)
        return self.optimizer
