# encoding: utf-8
"""Stage-1 HSMOT detector: YOLO11L + MMOT ConvMSI + rotated diffusion."""

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
        # Avoid shrinking an entire aerial mosaic to nearly empty content.
        # Tracking configs inherit this range, and Pair sides share one draw.
        self.scale = (0.5, 1.5)
        # A five-parameter rotated box cannot represent the parallelogram
        # produced by shear. Preserve the source quadrilateral annotations
        # verbatim instead of rectifying them through minAreaRect.
        self.shear = 0.0
        self.max_epoch = 20
        self.print_interval = 20
        self.train_vis_interval = 500
        self.save_interval = 5
        self.eval_interval = 3
        # Keep Stage-1 validation/checkpoint cadence unchanged after mosaic is
        # disabled; the upstream trainer otherwise validates every epoch.
        self.no_aug_eval_interval = 3
        self.no_aug_epochs = 5
        self.basic_lr_per_img = 0.001 / 64.0
        # Match the reproduced DiffusionTrack rule: scale 0.001 / 64 by the
        # effective global batch. BS=2 with accumulation=8 peaks at 2.5e-4.
        self.scheduler = "yoloxwarmcos"
        self.min_lr_ratio = 0.05
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
        self.max_labels = 500
        # Global validation BS=6 is split across both DDP ranks (3 per GPU).
        # Backbone features are shared by prev->curr and curr->curr.
        self.val_batch_size = 6
        self.val_visualization_conf = 0.05
        self.val_visualization_max_dets = 30
        self.train_data_dir = os.environ.get(
            "HSMOT_TRAIN_ROOT", os.path.join(_PAIRMOT_ROOT, "data/hsmot/train"))
        self.val_data_dir = os.environ.get(
            "HSMOT_VAL_ROOT", os.path.join(_PAIRMOT_ROOT, "data/hsmot/test"))
        self.hsmot_img_subdir = os.environ.get("HSMOT_IMG_SUBDIR", "npy")
        self.hsmot_img_format = os.environ.get("HSMOT_IMG_FORMAT", "npy")
        self.hsmot_target_dtype = "float32"
        self.yolo11_cfg = "yolo11l-obb.yaml"
        self.yolo11_weights = os.environ.get(
            "YOLO11_WEIGHTS",
            "/data4/litianhao/PairMmot/pretrained_weights/mmot_official/"
            "yolo11L-8ch-3dstem.pt")
        self.freeze_yolo11_backbone = False
        self.yolo11_stem_type = "convmsi"
        self.yolo11_native_stem_weights = ""
        self.yolo11_align_convmsi_rng = True
        self.yolo11_backbone_load_mode = "direct"
        self.min_diffusion_side = 1.0
        self.diffusion_schedule_float64 = False
        self.lx_regression_init = False
        self.optimizer_group_mode = "split_stem"
        self.adamw_foreach = None
        self.lr_update_timing = "before"

    def _dataset(self, root, transform):
        from yolox.data import HSMOTDataset
        return HSMOTDataset(
            data_dir=root, img_size=self.input_size, preproc=transform,
            ann_subdir="mot", img_subdir=self.hsmot_img_subdir,
            img_format=self.hsmot_img_format,
            target_dtype=self.hsmot_target_dtype)

    def get_data_loader(self, batch_size, is_distributed, no_aug=False):
        from yolox.data import (DataLoader, InfiniteSampler, MosaicDetection,
                                TrainTransform, YoloBatchSampler)
        # Match the MMOT official Ultralytics checkpoint preprocessing:
        # uint8 -> float32 / 255, with no additional mean/std normalization.
        input_means = getattr(self, "input_means", None)
        input_stds = getattr(self, "input_stds", None)
        transform = TrainTransform(
            rgb_means=input_means, std=input_stds,
            max_labels=self.max_labels,
            hsmot_augment_mode=getattr(
                self, "hsmot_augment_mode", "none"))
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
        from yolox.data import DiffusionValTransform, HSMOTPairEvalDataset
        dataset = HSMOTPairEvalDataset(self._dataset(
            self.val_data_dir,
            DiffusionValTransform(
                rgb_means=getattr(self, "input_means", None),
                std=getattr(self, "input_stds", None),
                max_labels=self.max_labels)))
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
                freeze=self.freeze_yolo11_backbone, num_spectral=8,
                num_classes=self.num_classes,
                stem_type=self.yolo11_stem_type,
                native_stem_weights=self.yolo11_native_stem_weights,
                align_convmsi_rng=self.yolo11_align_convmsi_rng,
                load_mode=self.yolo11_backbone_load_mode)
            self.model = DiffusionNet(
                backbone, DiffusionHead(
                    self.num_classes, self.width,
                    min_diffusion_side=self.min_diffusion_side,
                    diffusion_schedule_float64=(
                        self.diffusion_schedule_float64),
                    lx_regression_init=self.lx_regression_init))
            self.model.apply(init_bn)
        return self.model

    def get_evaluator(self, batch_size, is_distributed, testdev=False):
        from yolox.evaluators import HSMOTRotatedDetectionEvaluator
        # Pair detection is stateless and is evaluated in real batches.  KL
        # tracking consumes the saved result cache instead of rerunning it.
        loader = self.get_eval_loader(
            self.val_batch_size, is_distributed, testdev)
        return HSMOTRotatedDetectionEvaluator(
            dataloader=loader, num_classes=self.num_classes,
            confthre=0.001, detthre=0.001,
            nmsthre3d=self.nms_thresh3d, nmsthre2d=self.nms_thresh2d,
            # Torch 2.0/mmcv 2.2 has no BF16 RotatedROIAlign kernel.
            amp=False,
            visualize=True,
            visualization_conf=self.val_visualization_conf,
            visualization_max_dets=self.val_visualization_max_dets,
            cache_root=os.path.join(
                self.output_dir, self.exp_name, "val_det"))

    def get_optimizer(self, batch_size):
        if "optimizer" not in self.__dict__:
            base_lr = getattr(self, "optimizer_base_lr", 2.5e-5)
            stem_lr_multiplier = getattr(
                self, "stem_lr_multiplier", 10.0)
            stem = self.model.backbone.task_model.model[0]
            stem_ids = {id(parameter) for parameter in stem.parameters()}
            stem_parameters = [
                parameter for parameter in stem.parameters()
                if parameter.requires_grad]
            other_parameters = [
                parameter for parameter in self.model.parameters()
                if parameter.requires_grad and id(parameter) not in stem_ids]
            if self.optimizer_group_mode == "single":
                if stem_lr_multiplier != 1.0:
                    raise ValueError(
                        "single optimizer group requires stem LR multiplier 1")
                parameters = [
                    parameter for parameter in self.model.parameters()
                    if parameter.requires_grad]
                optimizer_kwargs = {}
                if self.adamw_foreach is not None:
                    optimizer_kwargs["foreach"] = self.adamw_foreach
                self.optimizer = AdamW(
                    parameters, lr=base_lr, weight_decay=1e-4,
                    **optimizer_kwargs)
            elif self.optimizer_group_mode == "split_stem":
                optimizer_kwargs = {}
                if self.adamw_foreach is not None:
                    optimizer_kwargs["foreach"] = self.adamw_foreach
                self.optimizer = AdamW(
                    [
                        dict(
                            params=other_parameters, lr=base_lr,
                            lr_scale=1.0),
                        dict(
                            params=stem_parameters,
                            lr=base_lr * stem_lr_multiplier,
                            lr_scale=stem_lr_multiplier),
                    ],
                    lr=base_lr, weight_decay=1e-4,
                    **optimizer_kwargs)
            else:
                raise ValueError(
                    "optimizer_group_mode must be 'split_stem' or 'single'")
        return self.optimizer
