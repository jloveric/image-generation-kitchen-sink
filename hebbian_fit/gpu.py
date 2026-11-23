"""Bindings for the Mojo GPU model. NumPy owns initialization and checkpoints."""

import importlib.util
import os
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np

from .model import GATES, NumpyModel, validate_checkpoint


def load_extension():
    name = "_hebbian_gpu"
    if name in sys.modules:
        return sys.modules[name]
    source = Path(__file__).with_name(name + ".mojo")
    directory = Path(".cache") / "hebbian-fit"
    directory.mkdir(parents=True, exist_ok=True)
    target = os.environ.get("HEBBIAN_GPU_TARGET", "sm_120")
    if not target.replace("_", "").isalnum():
        raise ValueError("Invalid HEBBIAN_GPU_TARGET")
    binary = directory / f"{name}-{target}.so"
    if not binary.exists() or binary.stat().st_mtime < max(p.stat().st_mtime for p in source.parent.glob("*.mojo")):
        compiler = os.environ.get("MOJO_BIN") or shutil.which("mojo")
        if compiler is None:
            raise RuntimeError("Install the GPU extra or set MOJO_BIN to a Mojo compiler")
        temporary = binary.with_suffix(f".{os.getpid()}.so")
        try:
            subprocess.run([compiler, "build", "--emit", "shared-lib", "--target-accelerator",
                            target, "-I", str(source.parent), str(source), "-o", str(temporary)], check=True)
            temporary.replace(binary)
        finally:
            temporary.unlink(missing_ok=True)
    spec = importlib.util.spec_from_file_location(name, binary)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    sys.modules[name] = module
    return module


class MojoModel:
    def __init__(self, config, seed, capacity=256, encoder=False):
        self.config = config
        self.capacity = capacity
        self.network = load_extension().Network(
            config.inputs, config.outputs, config.width, config.depth, config.winners,
            config.readout == "sum", capacity, config.competition_groups,
            GATES.index(config.gate), config.gate_widths, config.readout_divisor, config.heads,
            config.unit_norm_updates, config.sine_omega or 0.0, config.basis_size,
            config.hidden_rule == "lion", config.hidden_rule == "activity-sign", config.output_bound or 0.0,
            (config.output_levels - 1) / 2 if config.output_levels else 0.0,
            config.edge_transitions or (0.,) * config.depth,
            config.clip_output, config.rate_multipliers, config.readout == "class-energy", encoder,
        )
        initial = NumpyModel(config, seed).state()
        self.shapes = [initial[f"weight_{i}"].shape for i in range(config.depth)]
        self.offsets = np.cumsum([0] + [int(np.prod(shape)) for shape in self.shapes])
        self.load(initial)

    def predict(self, x):
        x = batch_array(x, self.config.inputs)
        y = np.empty((len(x), self.config.outputs), np.float32)
        masks = [np.empty((len(x), self.config.units), np.float32)
                 for _ in range(self.config.depth)]
        self.network.infer(x, y, masks)
        return y, [mask.astype(bool) for mask in masks]

    def predict_values(self, x):
        x = batch_array(x, self.config.inputs)
        y = np.empty((len(x), self.config.outputs), np.float32)
        self.network.infer(x, y, [])
        return y

    def step(self, x, y, rate, *, return_signs=False):
        x = batch_array(x, self.config.inputs)
        y = batch_array(y, self.config.outputs, len(x))
        signs = np.empty_like(y) if return_signs else None
        self.network.train(x, y, float(rate), signs if return_signs else [])
        return signs

    def state(self):
        weights = np.empty(self.offsets[-1], np.float32)
        history = np.empty_like(weights) if self.config.hidden_rule == "lion" else None
        self.network.save(weights, history)
        state = {}
        for i, shape in enumerate(self.shapes):
            part = slice(self.offsets[i], self.offsets[i + 1])
            state[f"weight_{i}"] = weights[part].reshape(shape)
            if history is not None:
                state[f"history_{i}"] = history[part].reshape(shape)
        return state

    def load(self, state):
        validate_checkpoint(state, self.config)

        def pack(name):
            return np.concatenate([state_array(state[f"{name}_{i}"], shape).ravel()
                                   for i, shape in enumerate(self.shapes)], dtype=np.float32)

        weights = pack("weight")
        history = pack("history") if self.config.hidden_rule == "lion" else None
        self.network.load(weights, history)


def batch_array(array, columns, rows=None):
    result = np.ascontiguousarray(array, np.float32)
    if result.ndim != 2 or result.shape[1] != columns or (rows is not None and len(result) != rows):
        raise ValueError("Incorrect batch shape")
    return result


def state_array(array, shape):
    result = np.ascontiguousarray(array, np.float32)
    if result.shape != shape:
        raise ValueError(f"Checkpoint array must have shape {shape}")
    return result
