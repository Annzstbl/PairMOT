"""Load, deepcopy, and fully build one MMEngine config."""
import copy
import sys

from mmengine.config import Config
from mmengine.runner import Runner
from mmengine.utils import import_modules_from_strings
from mmdet.utils import register_all_modules as register_all_modules_mmdet
from mmrotate.utils import register_all_modules


cfg = Config.fromfile(sys.argv[1])
copy.deepcopy(cfg)
register_all_modules_mmdet(init_default_scope=False)
register_all_modules(init_default_scope=False)
if cfg.get('custom_imports', None):
    import_modules_from_strings(**cfg.custom_imports)
runner = Runner.from_cfg(cfg)
print(
    'READY',
    cfg.train_dataloader.batch_size,
    cfg.train_cfg.max_epochs,
    sum(param.numel() for param in runner.model.parameters()),
    cfg.optim_wrapper.optimizer.lr,
)
