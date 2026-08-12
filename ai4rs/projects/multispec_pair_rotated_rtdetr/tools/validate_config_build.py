"""Load, deepcopy, and fully build one MMEngine config."""
import copy
import sys

from mmengine.config import Config
from mmengine.runner import Runner


cfg = Config.fromfile(sys.argv[1])
copy.deepcopy(cfg)
runner = Runner.from_cfg(cfg)
print(
    'READY',
    cfg.train_dataloader.batch_size,
    cfg.train_cfg.max_epochs,
    sum(param.numel() for param in runner.model.parameters()),
    cfg.optim_wrapper.optimizer.lr,
)
