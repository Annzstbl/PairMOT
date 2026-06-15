# Copyright (c) AI4RS. All rights reserved.
from typing import Callable, Union

from mmengine.model import is_model_wrapper
from mmengine.registry import RUNNERS
from mmengine.runner import Runner
from mmengine.runner.checkpoint import (_load_checkpoint,
                                        _load_checkpoint_to_model)


@RUNNERS.register_module()
class MultispecRunner(Runner):
    """Runner that writes checkpoint load details to the file logger."""

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
