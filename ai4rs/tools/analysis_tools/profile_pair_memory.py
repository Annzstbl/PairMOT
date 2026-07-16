"""Profile CUDA memory by stage for Pair RT-DETR training.

This tool runs the real MMEngine train step but instruments the model's
existing component timer, data preprocessor, backward, and optimizer step.
It is intended for short single-GPU diagnosis, not benchmark training.
"""

from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from statistics import mean

import torch
from mmengine.config import Config
from mmengine.optim import AmpOptimWrapper, OptimWrapper
from mmengine.runner import Runner
from mmengine.utils import import_modules_from_strings
from mmdet.utils import register_all_modules as register_all_modules_mmdet
from mmrotate.utils import register_all_modules


MIB = 1024**2


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('config')
    parser.add_argument('--warmup-iters', type=int, default=1)
    parser.add_argument('--profile-iters', type=int, default=3)
    parser.add_argument('--split-backbone-neck', action='store_true')
    parser.add_argument('--output', default=None)
    return parser.parse_args()


def cuda_snapshot():
    free, total = torch.cuda.mem_get_info()
    return dict(
        allocated=torch.cuda.memory_allocated() / MIB,
        reserved=torch.cuda.memory_reserved() / MIB,
        device_used=(total - free) / MIB,
    )


class MemoryRecorder:
    def __init__(self):
        self.iteration = -1
        self.records = []

    def begin_iteration(self):
        self.iteration += 1

    def record(self, stage, fn):
        torch.cuda.synchronize()
        before = cuda_snapshot()
        torch.cuda.reset_peak_memory_stats()
        result = fn()
        torch.cuda.synchronize()
        after = cuda_snapshot()
        self.records.append(dict(
            iteration=self.iteration,
            stage=stage,
            before_allocated=before['allocated'],
            after_allocated=after['allocated'],
            peak_allocated=torch.cuda.max_memory_allocated() / MIB,
            before_reserved=before['reserved'],
            after_reserved=after['reserved'],
            peak_reserved=torch.cuda.max_memory_reserved() / MIB,
            device_used=after['device_used'],
        ))
        return result


def summarize(records, warmup_iters):
    measured = [r for r in records if r['iteration'] >= warmup_iters]
    grouped = defaultdict(list)
    for record in measured:
        grouped[record['stage']].append(record)

    summary = {}
    keys = (
        'before_allocated', 'after_allocated', 'peak_allocated',
        'before_reserved', 'after_reserved', 'peak_reserved', 'device_used')
    for stage, values in grouped.items():
        summary[stage] = {
            key: mean(value[key] for value in values) for key in keys
        }
        summary[stage]['retained_delta'] = mean(
            value['after_allocated'] - value['before_allocated']
            for value in values)
        summary[stage]['temporary_peak_delta'] = mean(
            value['peak_allocated'] - max(
                value['before_allocated'], value['after_allocated'])
            for value in values)
    return summary


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is required')
    if torch.cuda.device_count() != 1:
        raise RuntimeError(
            'Expose exactly one GPU, e.g. CUDA_VISIBLE_DEVICES=3')

    register_all_modules_mmdet(init_default_scope=False)
    register_all_modules(init_default_scope=False)
    cfg = Config.fromfile(args.config)
    cfg.launcher = 'none'
    cfg.work_dir = '/tmp/pairmot_memory_profile'
    cfg.train_cfg = dict(
        type='IterBasedTrainLoop',
        max_iters=args.warmup_iters + args.profile_iters,
        val_interval=10**9)
    cfg.val_dataloader = None
    cfg.val_cfg = None
    cfg.val_evaluator = None
    cfg.test_dataloader = None
    cfg.test_cfg = None
    cfg.test_evaluator = None
    cfg.train_dataloader.num_workers = 0
    cfg.train_dataloader.persistent_workers = False
    cfg.default_hooks.checkpoint.interval = 10**9
    cfg.default_hooks.logger.interval = 1
    cfg.custom_hooks = [
        hook for hook in cfg.custom_hooks
        if 'EMAHook' in str(hook.get('type', ''))
    ]
    if cfg.get('custom_imports'):
        import_modules_from_strings(**cfg.custom_imports)

    runner = Runner.from_cfg(cfg)
    recorder = MemoryRecorder()

    from projects.multispec_pair_rotated_rtdetr.multispec_pair_rotated_rtdetr.component_timer import (  # noqa: E501
        CudaComponentTimer)

    original_timer_record = CudaComponentTimer.record

    def timer_record(_timer, name, fn):
        if args.split_backbone_neck and name == 'backbone_neck':
            return fn()
        return recorder.record(name, fn)

    CudaComponentTimer.record = timer_record

    model = runner.model
    preprocessor = model.data_preprocessor
    original_preprocessor_forward = preprocessor.forward

    def preprocessor_forward(*call_args, **call_kwargs):
        recorder.begin_iteration()
        return recorder.record(
            'data_preprocessor',
            lambda: original_preprocessor_forward(*call_args, **call_kwargs))

    preprocessor.forward = preprocessor_forward

    module_forwards = []
    if args.split_backbone_neck:
        for name, module in (('backbone', model.backbone),
                             ('neck', model.neck)):
            original_forward = module.forward

            def measured_forward(*call_args, _name=name,
                                 _forward=original_forward, **call_kwargs):
                return recorder.record(
                    _name, lambda: _forward(*call_args, **call_kwargs))

            module.forward = measured_forward
            module_forwards.append((module, original_forward))

    wrapper_cls = (AmpOptimWrapper
                   if cfg.optim_wrapper.type == 'AmpOptimWrapper'
                   else OptimWrapper)
    original_backward = wrapper_cls.backward
    original_step = wrapper_cls.step

    def backward(wrapper, *call_args, **call_kwargs):
        return recorder.record(
            'backward',
            lambda: original_backward(wrapper, *call_args, **call_kwargs))

    def step(wrapper, *call_args, **call_kwargs):
        return recorder.record(
            'optimizer_step',
            lambda: original_step(wrapper, *call_args, **call_kwargs))

    wrapper_cls.backward = backward
    wrapper_cls.step = step

    try:
        runner.train()
    finally:
        CudaComponentTimer.record = original_timer_record
        wrapper_cls.backward = original_backward
        wrapper_cls.step = original_step
        for module, original_forward in module_forwards:
            module.forward = original_forward

    result = dict(
        config=os.path.abspath(args.config),
        gpu=torch.cuda.get_device_name(),
        warmup_iters=args.warmup_iters,
        profile_iters=args.profile_iters,
        summary=summarize(recorder.records, args.warmup_iters),
        records=recorder.records,
    )
    output = json.dumps(result, indent=2, sort_keys=True)
    print('\nPAIR_MEMORY_PROFILE=' + output)
    if args.output:
        with open(args.output, 'w', encoding='utf-8') as file:
            file.write(output + '\n')


if __name__ == '__main__':
    main()
