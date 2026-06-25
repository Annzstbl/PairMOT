# Copyright (c) AI4RS. All rights reserved.
from typing import Callable, Optional, Union

from mmengine.model import is_model_wrapper
from mmengine.registry import RUNNERS
from mmengine.runner import Runner
from mmengine.runner.checkpoint import (_load_checkpoint,
                                        _load_checkpoint_to_model)

from .logger import MultispecMMLogger


@RUNNERS.register_module()
class MultispecRunner(Runner):
    """Runner with ``MultispecMMLogger`` and checkpoint load file logging."""

    def build_logger(self,
                     log_level: Union[int, str] = 'INFO',
                     log_file: Optional[str] = None,
                     **kwargs) -> MultispecMMLogger:
        """Build logger; read ``console_log_level`` from config when set."""
        import os.path as osp

        if log_file is None:
            log_file = osp.join(self._log_dir, f'{self.timestamp}.log')

        log_cfg = dict(log_level=log_level, log_file=log_file, **kwargs)
        log_cfg.setdefault('name', self._experiment_name)
        log_cfg.setdefault('file_mode', 'a')

        cfg = getattr(self, 'cfg', None)
        if cfg is not None:
            if cfg.get('file_log_level') is not None:
                log_cfg['file_log_level'] = cfg.file_log_level
            if cfg.get('console_log_level') is not None:
                log_cfg['console_log_level'] = cfg.console_log_level
            patterns = cfg.get('console_suppress_patterns')
            if patterns:
                log_cfg['console_suppress_patterns'] = list(patterns)

        return MultispecMMLogger.get_instance(**log_cfg)  # type: ignore

    def load_checkpoint(
            self,
            filename: str,
            map_location: Union[str, Callable] = 'cpu',
            strict: bool = False,
            revise_keys: list = [(r'^module.', '')]) -> dict:
        checkpoint = _load_checkpoint(
            filename, map_location=map_location, logger=self.logger)

        self.call_hook('after_load_checkpoint', checkpoint=checkpoint)

        if is_model_wrapper(self.model):
            model = self.model.module
        else:
            model = self.model

        checkpoint = _load_checkpoint_to_model(
            model,
            checkpoint,
            strict,
            logger=self.logger,
            revise_keys=revise_keys)

        self._has_loaded = True
        self.logger.info(f'Load checkpoint from {filename}')
        return checkpoint
