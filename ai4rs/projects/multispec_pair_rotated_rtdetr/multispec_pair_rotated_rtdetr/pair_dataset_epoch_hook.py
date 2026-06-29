"""Epoch synchronization for random temporal-pair sampling."""

from mmengine.hooks import Hook
from mmengine.runner import Runner

from mmrotate.registry import HOOKS


@HOOKS.register_module()
class PairDatasetEpochHook(Hook):
    """Tell an epoch-aware pair dataset which temporal partners to use."""

    priority = 'ABOVE_NORMAL'

    def before_train_epoch(self, runner: Runner) -> None:
        dataset = runner.train_dataloader.dataset
        if hasattr(dataset, 'set_epoch'):
            dataset.set_epoch(runner.epoch)
