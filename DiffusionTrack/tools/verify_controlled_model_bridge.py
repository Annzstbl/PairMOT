#!/usr/bin/env python3
"""Verify that two experiment configs differ only in non-state switches."""

import argparse
import gc
import hashlib
import json
from pathlib import Path
import random
import re
import subprocess
import sys

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from yolox.exp import get_exp  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-a")
    parser.add_argument("--config-b")
    parser.add_argument("--single-config", help=argparse.SUPPRESS)
    parser.add_argument("--seed", type=int, default=8823)
    parser.add_argument("--model-attr", action="append", default=[])
    parser.add_argument("--output")
    parser.add_argument(
        "--require-identical-state", action="store_true",
        help="Exit nonzero unless complete model state hashes match.",
    )
    return parser.parse_args()


def tensor_hash_items(items):
    digest = hashlib.sha256()
    elements = 0
    tensors = 0
    per_tensor = {}
    for name, tensor in items:
        value = tensor.detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(str(value.dtype).encode())
        digest.update(str(tuple(value.shape)).encode())
        raw = value.reshape(-1).view(torch.uint8).numpy().tobytes()
        digest.update(raw)
        per_tensor[name] = hashlib.sha256(raw).hexdigest()
        elements += value.numel()
        tensors += 1
    return {
        "sha256": digest.hexdigest(),
        "elements": elements,
        "tensors": tensors,
        "tensor_sha256": per_tensor,
    }


def resolve_attr(root, path):
    value = root
    for field, index in re.findall(r"([^.\[\]]+)(?:\[([0-9]+)\])?", path):
        value = getattr(value, field)
        if index:
            value = value[int(index)]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, (tuple, list)):
        return list(value)
    return repr(value)


def json_value(value):
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, (tuple, list)):
        converted = [json_value(item) for item in value]
        return converted if all(item is not None for item in converted) else None
    return None


def build_summary(config, seed, model_attrs):
    # The launcher imports/constructs the Exp before the distributed worker
    # seeds training. Reproduce that order so Python module-cache activity is
    # not mistaken for a model-side RNG difference.
    exp = get_exp(config, None)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    model = exp.get_model()
    state = tensor_hash_items(model.state_dict().items())
    parameters = tensor_hash_items(model.named_parameters())
    attrs = {path: resolve_attr(model, path) for path in model_attrs}
    exp_values = {}
    for key, value in sorted(vars(exp).items()):
        converted = json_value(value)
        if converted is not None:
            exp_values[key] = converted
    rng = {
        "torch_sha256": hashlib.sha256(
            torch.get_rng_state().numpy().tobytes()).hexdigest(),
        "numpy_state_sha256": hashlib.sha256(
            repr(np.random.get_state()).encode()).hexdigest(),
        "python_state_sha256": hashlib.sha256(
            repr(random.getstate()).encode()).hexdigest(),
    }
    del model, exp
    gc.collect()
    return {
        "config": config,
        "state": state,
        "parameters": parameters,
        "model_attributes": attrs,
        "experiment_values": exp_values,
        "post_build_rng": rng,
    }


def main():
    args = parse_args()
    if args.single_config:
        rendered = json.dumps(
            build_summary(args.single_config, args.seed, args.model_attr),
            indent=2, sort_keys=True) + "\n"
        print(rendered, end="")
        if args.output:
            output = Path(args.output)
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(rendered)
        return
    if not args.config_a or not args.config_b:
        raise SystemExit("--config-a and --config-b are required")

    def isolated_summary(config):
        command = [
            sys.executable, str(Path(__file__).resolve()),
            "--single-config", config, "--seed", str(args.seed),
        ]
        for path in args.model_attr:
            command.extend(["--model-attr", path])
        completed = subprocess.run(
            command, check=True, text=True, stdout=subprocess.PIPE)
        # Some model loaders print transfer diagnostics before the pretty
        # multi-line JSON summary. Decode from the first JSON object.
        json_start = completed.stdout.find("{")
        if json_start < 0:
            raise RuntimeError(
                f"worker produced no JSON for {config}:\n"
                f"{completed.stdout[-2000:]}")
        return json.loads(completed.stdout[json_start:])

    summaries = [
        isolated_summary(args.config_a),
        isolated_summary(args.config_b),
    ]
    state_hashes = [summary["state"].pop("tensor_sha256")
                    for summary in summaries]
    parameter_hashes = [summary["parameters"].pop("tensor_sha256")
                        for summary in summaries]

    def changed_tensors(hashes):
        keys = set(hashes[0]) | set(hashes[1])
        return [
            key for key in sorted(keys)
            if hashes[0].get(key) != hashes[1].get(key)
        ]

    keys = set(summaries[0]["experiment_values"]) | set(
        summaries[1]["experiment_values"])
    exp_differences = {
        key: [
            summaries[0]["experiment_values"].get(key),
            summaries[1]["experiment_values"].get(key),
        ]
        for key in sorted(keys)
        if summaries[0]["experiment_values"].get(key)
        != summaries[1]["experiment_values"].get(key)
    }
    result = {
        "seed": args.seed,
        "runs": summaries,
        "state_identical": (
            summaries[0]["state"]["sha256"]
            == summaries[1]["state"]["sha256"]),
        "parameters_identical": (
            summaries[0]["parameters"]["sha256"]
            == summaries[1]["parameters"]["sha256"]),
        "post_build_rng_identical": (
            summaries[0]["post_build_rng"]
            == summaries[1]["post_build_rng"]),
        "state_changed_tensors": changed_tensors(state_hashes),
        "parameter_changed_tensors": changed_tensors(parameter_hashes),
        "experiment_value_differences": exp_differences,
    }
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    print(rendered, end="")
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered)
    if args.require_identical_state and not result["state_identical"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
