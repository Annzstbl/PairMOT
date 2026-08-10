"""Standard One-Cycle learning rate with preserved optimizer-group ratios."""

from typing import Any

from mmengine.optim.scheduler import OneCycleLR
from mmengine.registry import PARAM_SCHEDULERS


@PARAM_SCHEDULERS.register_module()
class RatioPreservingOneCycleLR(OneCycleLR):
    """Expand one peak factor over the optimizer's existing group LRs.

    MMEngine's :class:`OneCycleLR` accepts a list-valued ``eta_max``.  This
    adapter derives that list from the learning rates produced by the
    optimizer's ``paramwise_cfg`` before scheduler construction.  It therefore
    keeps every existing ``lr_mult`` while delegating the complete schedule,
    phase boundaries, cosine annealing, and checkpoint state to the standard
    implementation.
    """

    def __init__(self,
                 optimizer: Any,
                 eta_max_factor: float,
                 *args: Any,
                 **kwargs: Any) -> None:
        eta_max_factor = float(eta_max_factor)
        if eta_max_factor <= 0:
            raise ValueError('eta_max_factor must be positive')
        group_lrs = [float(group['lr']) for group in optimizer.param_groups]
        if not group_lrs:
            raise ValueError('optimizer must contain at least one param group')
        eta_max = [lr * eta_max_factor for lr in group_lrs]
        self.eta_max_factor = eta_max_factor
        super().__init__(
            optimizer, eta_max=eta_max, *args, **kwargs)
