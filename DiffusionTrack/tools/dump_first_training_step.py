#!/usr/bin/env python3
"""Dump one seeded training forward/backward for cross-repository parity."""

import argparse
import hashlib
import itertools
import json
from pathlib import Path
import random
import sys

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from yolox.exp import get_exp  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=8823)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument(
        "--amp-dtype", choices=("fp32", "bf16"), default="fp32")
    parser.add_argument(
        "--layer-output", default=None,
        help="Optional torch file containing per-RCNN-layer logits and boxes.")
    parser.add_argument(
        "--detectron2-pooler", action="store_true",
        help="Replace the MMCV adapter with Detectron2 ROIAlignRotated while "
             "converting internal radian angles to degrees.")
    parser.add_argument(
        "--skip-optimizer", action="store_true",
        help="Skip optimizer construction for legacy environments; AdamW "
             "construction has no tensor state or RNG effect before step 1.")
    return parser.parse_args()


def tensor_sha256(value):
    tensor = value.detach().cpu().contiguous()
    return hashlib.sha256(
        tensor.reshape(-1).view(torch.uint8).numpy().tobytes()).hexdigest()


def rng_summary():
    return {
        "torch_cpu": hashlib.sha256(
            torch.get_rng_state().numpy().tobytes()).hexdigest(),
        "torch_cuda": hashlib.sha256(
            torch.cuda.get_rng_state().cpu().numpy().tobytes()).hexdigest(),
        "python": hashlib.sha256(
            repr(random.getstate()).encode()).hexdigest(),
        "numpy": hashlib.sha256(
            repr(np.random.get_state()).encode()).hexdigest(),
    }


def flatten_tensor_hashes(value, prefix="batch"):
    if torch.is_tensor(value):
        return {
            prefix: {
                "shape": list(value.shape),
                "dtype": str(value.dtype),
                "sha256": tensor_sha256(value),
            }
        }
    if isinstance(value, (list, tuple)):
        result = {}
        for index, item in enumerate(value):
            result.update(flatten_tensor_hashes(item, f"{prefix}.{index}"))
        return result
    if isinstance(value, dict):
        result = {}
        for key, item in sorted(value.items()):
            result.update(flatten_tensor_hashes(item, f"{prefix}.{key}"))
        return result
    return {}


def detach_tensor_leaves(value, prefix):
    """Return an audit-only mapping of every tensor nested in ``value``."""
    if torch.is_tensor(value):
        return {prefix: value.detach().cpu()}
    if isinstance(value, (list, tuple)):
        result = {}
        for index, item in enumerate(value):
            result.update(detach_tensor_leaves(item, f"{prefix}_{index}"))
        return result
    if isinstance(value, dict):
        result = {}
        for key, item in sorted(value.items()):
            result.update(detach_tensor_leaves(item, f"{prefix}_{key}"))
        return result
    return {}


def gradient_summary(model):
    groups = {
        "backbone": [0.0, 0.0, 0],
        "projs": [0.0, 0.0, 0],
        "head": [0.0, 0.0, 0],
    }
    selected = {}
    selected_suffixes = (
        "head_series.0.bboxes_delta.weight",
        "head_series.0.bboxes_delta.bias",
        "head_series.5.bboxes_delta.weight",
        "head_series.5.bboxes_delta.bias",
        "head_series.0.reg_module.0.weight",
        "head_series.5.reg_module.0.weight",
        "head_series.0.cls_module.0.weight",
        "head_series.5.cls_module.0.weight",
    )
    for name, parameter in model.named_parameters():
        if parameter.grad is None:
            continue
        gradient = parameter.grad.detach().float()
        group = (
            "backbone" if name.startswith("backbone.")
            else "projs" if name.startswith("projs.")
            else "head")
        groups[group][0] += float(gradient.square().sum())
        groups[group][1] = max(groups[group][1], float(gradient.abs().max()))
        groups[group][2] += gradient.numel()
        if name.endswith(selected_suffixes):
            selected[name] = {
                "shape": list(gradient.shape),
                "l2": float(gradient.norm()),
                "absmax": float(gradient.abs().max()),
                "sha256": tensor_sha256(gradient),
            }
    return {
        "groups": {
            name: {
                "l2": squared_sum ** 0.5,
                "absmax": absolute_max,
                "elements": elements,
            }
            for name, (squared_sum, absolute_max, elements)
            in groups.items()
        },
        "selected": selected,
    }


def install_layer_hooks(model):
    captured = {}
    head_series = model.head.head.head_series

    def make_hook(index):
        def hook(_module, _inputs, output):
            logits, boxes = output[0], output[1]
            captured.setdefault(f"layer{index + 1}", {}).update({
                "ref_logits": logits[0].detach().cpu(),
                "cur_logits": logits[1].detach().cpu(),
                "ref_boxes": boxes[0].detach().cpu(),
                "cur_boxes": boxes[1].detach().cpu(),
            })
        return hook

    def make_pre_hook(index):
        def hook(_module, inputs):
            # ``DynamicHead`` receives the pair as a tuple of per-frame
            # tensors.  Retain each side explicitly: concatenating would
            # hide a potential frame-order discrepancy in the parity audit.
            input_boxes = inputs[1]
            forwarded_inputs = None
            if not torch.is_tensor(input_boxes) and not isinstance(
                    input_boxes, (tuple, list)):
                # Later refinement layers receive a one-shot generator.  A
                # direct ``tuple(generator)`` would consume it and alter the
                # model.  ``tee`` leaves one identical iterator for RCNN.
                snapshot, forwarded = itertools.tee(input_boxes)
                input_boxes = tuple(snapshot)
                forwarded_inputs = (
                    inputs[0], forwarded, *inputs[2:])
            if isinstance(input_boxes, (tuple, list)):
                input_box_values = {
                    "ref_input_boxes": input_boxes[0].detach().cpu(),
                    "cur_input_boxes": input_boxes[1].detach().cpu(),
                }
            else:
                input_box_values = {
                    "input_boxes": input_boxes.detach().cpu(),
                }
            captured.setdefault(f"layer{index + 1}", {}).update({
                **input_box_values,
                "time_embedding": inputs[4].detach().cpu(),
            })
            if index == 0:
                captured["layer1"].update(
                    detach_tensor_leaves(inputs[0], "feature"))
            return forwarded_inputs
        return hook

    handles = []
    for index, module in enumerate(head_series):
        handles.append(module.register_forward_pre_hook(make_pre_hook(index)))
        handles.append(module.register_forward_hook(make_hook(index)))
    return captured, handles


def install_pooler_hooks(model, captured):
    """Capture the six ROI tensors without modifying either pooler."""
    calls = []

    def hook(_module, _inputs, output):
        calls.append(output.detach().cpu())

    handle = model.head.head.box_pooler.register_forward_hook(hook)
    return calls, handle


def install_regression_hooks(model, captured):
    """Record the raw five residuals before any box-unit conversion."""
    handles = []
    for index, head in enumerate(model.head.head.head_series):
        def hook(_module, _inputs, output, layer=index):
            captured.setdefault(f"layer{layer + 1}", {})[
                "bbox_deltas"] = output.detach().cpu()
        handles.append(head.bboxes_delta.register_forward_hook(hook))
    return handles


def install_matcher_hook(model):
    captured = []

    def hook(_module, _inputs, output):
        indices, best_query_per_gt = output
        captured.append({
            "pairs": [
                {
                    "query_indices": torch.nonzero(
                        selected_query, as_tuple=False
                    ).squeeze(1).detach().cpu().tolist(),
                    "gt_indices": gt_indices.detach().cpu().tolist(),
                    "best_query_per_gt": best_query.detach().cpu().tolist(),
                }
                for (selected_query, gt_indices), best_query
                in zip(indices, best_query_per_gt)
            ]
        })

    handle = model.head.criterion.matcher.register_forward_hook(hook)
    return captured, handle


def install_detectron2_pooler(model):
    from detectron2.modeling.poolers import ROIPooler
    from detectron2.structures import RotatedBoxes

    source = model.head.head.box_pooler

    input_angles_are_degrees = bool(
        getattr(model.head, "box_angle_degrees", False))

    class RadianToDegreePooler(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.pooler = ROIPooler(
                output_size=source.output_size[0],
                scales=source.scales,
                sampling_ratio=source.sampling_ratio,
                pooler_type="ROIAlignRotated",
            )

        def forward(self, features, boxes_per_image):
            converted = []
            for boxes in boxes_per_image:
                angle = (
                    boxes[:, 4:5] if input_angles_are_degrees
                    else torch.rad2deg(boxes[:, 4:5]))
                degree_boxes = torch.cat([boxes[:, :4], angle], dim=1)
                converted.append(RotatedBoxes(degree_boxes))
            return self.pooler(features, converted)

    model.head.head.box_pooler = RadianToDegreePooler()


def main():
    args = parse_args()
    exp = get_exp(args.config, None)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    # A parity diagnostic must select the same convolution algorithms across
    # fresh processes. ``benchmark=True`` can choose different kernels even
    # with identical RNG states and makes the audit itself non-reproducible.
    torch.backends.cudnn.benchmark = False
    rng = {"seeded": rng_summary()}

    model = exp.get_model()
    if args.detectron2_pooler:
        install_detectron2_pooler(model)
    model = model.cuda().train()
    rng["after_model"] = rng_summary()
    # Match Trainer construction order. AdamW has no tensor state before its
    # first step, but constructing it is retained for exact orchestration.
    if not args.skip_optimizer:
        exp.get_optimizer(args.batch_size)
    loader = exp.get_data_loader(
        args.batch_size, is_distributed=False, no_aug=False)
    batch = next(iter(loader))
    rng["after_loader"] = rng_summary()

    images = batch[0].cuda(non_blocking=False)
    target_dim = getattr(exp, "target_dim", 9)
    targets = batch[1][:, :, :target_dim].float().cuda(non_blocking=False)
    data_dtype = (
        torch.bfloat16 if args.amp_dtype == "bf16" else torch.float32)
    images = images.to(data_dtype)
    captured, handles = install_layer_hooks(model)
    pooler_calls, pooler_handle = install_pooler_hooks(model, captured)
    regression_handles = install_regression_hooks(model, captured)
    assignments, matcher_handle = install_matcher_hook(model)
    rng["before_forward"] = rng_summary()
    with torch.cuda.amp.autocast(
            enabled=args.amp_dtype != "fp32", dtype=data_dtype):
        outputs = model(
            (images, None), (targets, None),
            getattr(exp, "random_flip", False), exp.input_size)
    for handle in handles:
        handle.remove()
    pooler_handle.remove()
    for handle in regression_handles:
        handle.remove()
    matcher_handle.remove()
    for index, output in enumerate(pooler_calls):
        captured[f"pooler{index + 1}"] = {"output": output}
    rng["after_forward"] = rng_summary()
    outputs["total_loss"].backward()
    rng["after_backward"] = rng_summary()

    result = {
        "config": args.config,
        "seed": args.seed,
        "batch_size": args.batch_size,
        "amp_dtype": args.amp_dtype,
        "detectron2_pooler": args.detectron2_pooler,
        "skip_optimizer": args.skip_optimizer,
        "batch": flatten_tensor_hashes(batch),
        "losses": {
            key: float(value.detach().float())
            for key, value in outputs.items()
            if torch.is_tensor(value) and value.numel() == 1
        },
        "gradients": gradient_summary(model),
        # Criterion calls matcher in final-layer, then auxiliary-layer order.
        "assignments": {
            (
                "layer6" if index == 0 else f"layer{index}"
            ): assignment
            for index, assignment in enumerate(assignments)
        },
        "rng": rng,
    }
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(rendered)
    if args.layer_output:
        layer_output = Path(args.layer_output)
        layer_output.parent.mkdir(parents=True, exist_ok=True)
        torch.save(captured, layer_output)
    print(rendered, end="")


if __name__ == "__main__":
    main()
