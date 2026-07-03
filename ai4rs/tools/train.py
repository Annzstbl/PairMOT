# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import logging
import os
import os.path as osp

try:
    import torchvision
    torchvision.disable_beta_transforms_warning()
except ImportError:
    pass

from mmdet.utils import register_all_modules as register_all_modules_mmdet
from mmengine.config import Config, DictAction
from mmengine.logging import print_log
from mmengine.registry import RUNNERS
from mmengine.runner import Runner
from mmengine.utils import import_modules_from_strings

from mmrotate.utils import register_all_modules


def _sync_pair_val_output_dirs(cfg, old_work_dir):
    """Keep PairMOT validation artifacts under the effective work_dir."""
    if old_work_dir is None or cfg.get('work_dir', None) is None:
        return
    old_work_dir = osp.abspath(str(old_work_dir))
    new_work_dir = osp.abspath(str(cfg.work_dir))
    if old_work_dir == new_work_dir:
        return

    def _sync_evaluator(evaluator):
        if evaluator is None:
            return
        metrics = evaluator.get('metrics', None)
        metric_list = metrics if isinstance(metrics, list) else [metrics]
        for metric in metric_list:
            if not isinstance(metric, dict):
                continue
            for key in ('track_eval_out_dir', 'val_det_out_dir'):
                out_dir = metric.get(key, None)
                if not out_dir:
                    continue
                out_dir_abs = osp.abspath(str(out_dir))
                if out_dir_abs == old_work_dir:
                    metric[key] = new_work_dir
                elif out_dir_abs.startswith(old_work_dir + osp.sep):
                    metric[key] = osp.join(
                        new_work_dir, osp.relpath(out_dir_abs, old_work_dir))

    _sync_evaluator(cfg.get('val_evaluator', None))
    _sync_evaluator(cfg.get('test_evaluator', None))


def parse_args():
    parser = argparse.ArgumentParser(description='Train a detector')
    parser.add_argument('config', help='train config file path')
    parser.add_argument('--work-dir', help='the dir to save logs and models')
    parser.add_argument(
        '--amp',
        action='store_true',
        default=False,
        help='enable automatic-mixed-precision training')
    parser.add_argument(
        '--auto-scale-lr',
        action='store_true',
        help='enable automatically scaling LR.')
    parser.add_argument(
        '--resume',
        nargs='?',
        const='auto',
        default=None,
        help='resume from the latest checkpoint in the work_dir automatically, '
        'or resume from the specified checkpoint path')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. If the value to '
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        'Note that the quotation marks are necessary and that no white space '
        'is allowed.')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    # When using PyTorch version >= 2.0.0, the `torch.distributed.launch`
    # will pass the `--local-rank` parameter to `tools/train.py` instead
    # of `--local_rank`.
    parser.add_argument('--local_rank', '--local-rank', type=int, default=0)
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    return args


def main():
    args = parse_args()

    # register all modules in mmdet into the registries
    # do not init the default scope here because it will be init in the runner
    register_all_modules_mmdet(init_default_scope=False)
    register_all_modules(init_default_scope=False)

    # load config
    cfg = Config.fromfile(args.config)
    cfg.launcher = args.launcher
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)
    old_work_dir = cfg.get('work_dir', None)

    # work_dir is determined in this priority: CLI > segment in file > filename
    if args.work_dir is not None:
        # update configs according to CLI args if args.work_dir is not None
        cfg.work_dir = args.work_dir
    elif cfg.get('work_dir', None) is None:
        # use config filename as default work_dir if cfg.work_dir is None
        cfg.work_dir = osp.join('./work_dirs',
                                osp.splitext(osp.basename(args.config))[0])
    _sync_pair_val_output_dirs(cfg, old_work_dir)

    # enable automatic-mixed-precision training
    if args.amp is True:
        optim_wrapper = cfg.optim_wrapper.type
        if optim_wrapper == 'AmpOptimWrapper':
            print_log(
                'AMP training is already enabled in your config.',
                logger='current',
                level=logging.WARNING)
        else:
            assert optim_wrapper == 'OptimWrapper', (
                '`--amp` is only supported when the optimizer wrapper type is '
                f'`OptimWrapper` but got {optim_wrapper}.')
            cfg.optim_wrapper.type = 'AmpOptimWrapper'
            cfg.optim_wrapper.loss_scale = 'dynamic'

    # enable automatically scaling LR
    if args.auto_scale_lr:
        if 'auto_scale_lr' in cfg and \
                'enable' in cfg.auto_scale_lr and \
                'base_batch_size' in cfg.auto_scale_lr:
            cfg.auto_scale_lr.enable = True
        else:
            raise RuntimeError('Can not find "auto_scale_lr" or '
                               '"auto_scale_lr.enable" or '
                               '"auto_scale_lr.base_batch_size" in your'
                               ' configuration file.')

    # Keep config-level resume settings unless the CLI explicitly requests
    # auto-resume.  Many experiment configs use ``load_from`` for pretraining;
    # mmengine treats ``resume=True`` plus ``load_from`` as "resume from
    # load_from", which would restore optimizer/EMA from the pretrain checkpoint
    # instead of the latest checkpoint in ``work_dir``.
    if args.resume is not None:
        cfg.resume = True
        if args.resume == 'auto':
            cfg.load_from = None
        else:
            cfg.load_from = args.resume

    # Lazy-import configs (with read_base) skip custom_imports in fromfile.
    if cfg.get('custom_imports', None):
        import_modules_from_strings(**cfg.custom_imports)

    # build the runner from config
    if 'runner_type' not in cfg:
        runner = Runner.from_cfg(cfg)
    else:
        runner = RUNNERS.build(cfg)

    # start training
    runner.train()


if __name__ == '__main__':
    main()
