"""Single-image x20 overfit diagnostic for the HSMOT detector."""

import os

import torch.distributed as dist

from yolo11l_diffusion_det_hsmot import Exp as HSMOTExp


class Exp(HSMOTExp):
    def __init__(self):
        super().__init__()
        self.exp_name = os.path.splitext(os.path.basename(__file__))[0]
        root = os.environ.get(
            "HSMOT_OVERFIT_ROOT",
            "/data4/litianhao/PairMmot/overfit_data/"
            "diffusiontrack_one_image_x20")
        self.train_data_dir = root
        self.val_data_dir = root
        # Diagnostic schedule: one LR-warmup epoch, followed by forty epochs
        # with the auxiliary L1 loss enabled.  Spatial augmentation is
        # disabled for the complete run so that this is a strict memorisation
        # test of one fixed 900x1200 sample.
        self.max_epoch = 41
        self.warmup_epochs = 1
        self.scheduler_base_lr = 2.5e-5
        self.save_interval = 41
        self.eval_interval = 5
        self.no_aug_eval_interval = 5
        # Keep the regular warmup/cosine LR schedule across the diagnostic;
        # augmentation is already disabled by this data loader.  L1 starts at
        # epoch two through the explicit diagnostic-only override below.
        self.no_aug_epochs = 0
        self.l1_start_epoch = 2
        self.random_size = None
        self.enable_mixup = False
        self.data_num_workers = 0
        self.save_latest_each_epoch = False
        self.save_after_eval = False
        self.save_last_mosaic_checkpoint = False
        # Optional diagnostic-only specialization for the one-step inference
        # endpoint.  The formal HSMOT experiment keeps the original uniform
        # diffusion-timestep sampling because this attribute exists only in
        # the overfit experiment and defaults to disabled.
        self.fixed_training_t = None
        self.fixed_noise_seed = None

    def get_model(self):
        model = super().get_model()
        if self.fixed_training_t is None:
            if hasattr(model.head, "fixed_training_t"):
                delattr(model.head, "fixed_training_t")
        else:
            model.head.fixed_training_t = int(self.fixed_training_t)
        if self.fixed_noise_seed is None:
            if hasattr(model.head, "fixed_noise_seed"):
                delattr(model.head, "fixed_noise_seed")
        else:
            model.head.fixed_noise_seed = int(self.fixed_noise_seed)
        return model

    def get_data_loader(self, batch_size, is_distributed, no_aug=False):
        from yolox.data import (DataLoader, InfiniteSampler, MosaicDetection,
                                TrainTransform, YoloBatchSampler)
        transform = TrainTransform(
            rgb_means=None, std=None, max_labels=500)
        dataset = self._dataset(self.train_data_dir, transform)
        # Disable spatial composition so every sample is the exact same image.
        dataset = MosaicDetection(
            dataset, mosaic=False, img_size=self.input_size,
            preproc=transform, enable_mixup=False)
        self.dataset = dataset
        if is_distributed:
            batch_size //= dist.get_world_size()
        sampler = InfiniteSampler(len(dataset), seed=self.seed or 0)
        batch_sampler = YoloBatchSampler(
            sampler=sampler, batch_size=batch_size, drop_last=False,
            input_dimension=self.input_size, mosaic=False)
        return DataLoader(
            dataset, batch_sampler=batch_sampler,
            num_workers=self.data_num_workers, pin_memory=True)
