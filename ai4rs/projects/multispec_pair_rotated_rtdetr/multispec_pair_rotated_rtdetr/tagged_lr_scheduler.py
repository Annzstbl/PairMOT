"""Learning-rate schedulers that act on explicitly tagged parameter groups."""

from typing import Any

from mmengine.optim.scheduler.lr_scheduler import MultiStepLR
from mmengine.registry import PARAM_SCHEDULERS


@PARAM_SCHEDULERS.register_module()
class TaggedMultiStepLR(MultiStepLR):
    """Apply milestone multipliers only to tagged optimizer groups.

    Untagged groups retain their current learning rate.  The tag is ordinary
    optimizer-group metadata, so the scheduler remains checkpointable and
    does not alter the model or inference graph.
    """

    def __init__(self,
                 optimizer: Any,
                 tag_key: str,
                 tag_value: Any = True,
                 *args: Any,
                 **kwargs: Any) -> None:
        if not tag_key:
            raise ValueError('tag_key must be a nonempty string')
        self.tag_key = tag_key
        self.tag_value = tag_value
        super().__init__(optimizer, *args, **kwargs)

    def _get_value(self) -> list[float]:
        milestone_count = self.milestones[self.last_step]
        values = []
        for group in self.optimizer.param_groups:
            value = group[self.param_name]
            if (milestone_count
                    and group.get(self.tag_key) == self.tag_value):
                value *= self.gamma**milestone_count
            values.append(value)
        return values
