# Copyright (c) AI4RS. All rights reserved.
import os.path as osp
from mmengine.dist import get_dist_info
from mmengine.hooks import Hook
from mmengine.runner import Runner

from mmrotate.registry import HOOKS
from mmrotate.utils.plot_training_curves import plot_training_curves


@HOOKS.register_module()
class TrainingCurveHook(Hook):
    """Generate training curve plots after training finishes.

    Reads ``vis_data/scalars.json`` produced by LocalVisBackend and saves
    PNG plots under ``vis_data/curves/``.

    Args:
        scalars_name (str): scalars json filename. Defaults to 'scalars.json'.
        out_subdir (str): subdirectory under vis_data for plots.
            Defaults to 'curves'.
        dpi (int): figure DPI. Defaults to 150.
    """

    priority = 'VERY_LOW'

    def __init__(self,
                 scalars_name: str = 'scalars.json',
                 out_subdir: str = 'curves',
                 dpi: int = 150) -> None:
        self.scalars_name = scalars_name
        self.out_subdir = out_subdir
        self.dpi = dpi

    def _plot(self, runner: Runner, stage: str) -> None:
        rank, _ = get_dist_info()
        if rank != 0:
            return

        scalars_path = osp.join(runner.log_dir, 'vis_data', self.scalars_name)
        if not osp.isfile(scalars_path):
            runner.logger.warning(
                f'TrainingCurveHook: {scalars_path} not found, skip plotting.')
            return

        out_dir = osp.join(runner.log_dir, 'vis_data', self.out_subdir)
        try:
            saved = plot_training_curves(
                scalars_path, out_dir=out_dir, dpi=self.dpi)
        except Exception as exc:
            runner.logger.warning(
                f'TrainingCurveHook: failed to plot curves: {exc}')
            return

        runner.logger.info(
            f'TrainingCurveHook: saved {stage} training curves to '
            f'{out_dir}: {", ".join(saved.keys())}')

    def after_val_epoch(self, runner: Runner, metrics=None) -> None:
        self._plot(runner, 'validation')

    def after_train(self, runner: Runner) -> None:
        self._plot(runner, 'final')
