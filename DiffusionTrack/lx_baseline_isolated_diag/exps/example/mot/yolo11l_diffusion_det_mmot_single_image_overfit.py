"""Isolated single-image diagnostic using Linxu's baseline DiffusionTrack."""

from yolo11l_diffusion_det_mmot import Exp as LinxuBaselineExp


class Exp(LinxuBaselineExp):
    def __init__(self):
        super().__init__()
        self.exp_name = "yolo11l_diffusion_det_mmot_single_image_overfit"
        self.train_ann = "single_data43_2_000001_x20.json"
        self.val_ann = self.train_ann

        # Keep Linxu's model, diffusion, matcher/loss, optimizer and schedule
        # parameters.  Only remove stochastic image augmentation/multiscale so
        # this remains a true one-image memorization diagnostic.
        self.random_size = None
        self.enable_mixup = False
        self.max_epoch = 100
        self.no_aug_epochs = 5
        self.eval_interval = 5
        self.print_interval = 10
        self.data_num_workers = 2
        self.save_checkpoints = False

    def get_data_loader(self, batch_size, is_distributed, no_aug=False):
        return super().get_data_loader(
            batch_size=batch_size,
            is_distributed=is_distributed,
            no_aug=True,
        )
