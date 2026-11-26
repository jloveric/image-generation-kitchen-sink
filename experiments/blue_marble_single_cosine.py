"""Test one uninterrupted cosine with the historical layer taper."""

import argparse
from dataclasses import asdict, replace
import json
from pathlib import Path
from types import SimpleNamespace

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from experiments.blue_marble_layer_steps import HISTORICAL
from experiments.blue_marble_quality import Trial, save_progress
from experiments.blue_marble_report import rgb_sse
from experiments.fixed_capacity import predict
from hebbian_fit.__main__ import training_epochs
from hebbian_fit.data import rgb_data
from hebbian_fit.model import ModelConfig, NumpyModel
from papers.lion import RESULTS, report


STUDY = "blue-marble-single-cosine-taper"
CHECK_EPOCH = 100


def progress_path(seed):
    return Path("runs/blue-marble-quality") / STUDY / str(seed) / "progress.npz"


def run(historical, until):
    settings = historical["settings"]
    initial, refinement = settings["phases"]
    _, rate, batch, switch, _ = initial
    duration, decay = refinement[3:]
    budget = switch + duration
    seed = historical["seed"]
    config = ModelConfig(**(settings["config"] | {
        "layer_gate_widths": tuple(settings["config"]["layer_gate_widths"]),
    }))
    model = NumpyModel(config, seed)
    data = rgb_data(split=False)
    checkpoint = progress_path(seed)
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    history = []
    best_error = rgb_sse(model, data, data.train)
    best_epoch = completed = 0
    best_state = model.state()
    if checkpoint.exists():
        with np.load(checkpoint, allow_pickle=False) as saved:
            metadata = json.loads(saved["metadata"].item())
            completed, best_epoch, best_error, history = (
                metadata[key] for key in ("epoch", "best_epoch", "best_error", "history")
            )
            model.load({key.removeprefix("current_"): saved[key] for key in saved.files
                        if key.startswith("current_")})
            best_state = {key.removeprefix("best_"): saved[key] for key in saved.files
                          if key.startswith("best_")}

    segments = ((0, switch, 1., seed), (switch, duration, decay, seed + 1000))
    for offset, segment_epochs, layer_decay, shuffle_seed in segments:
        stop = min(segment_epochs, until - offset)
        start = max(0, completed - offset)
        if stop <= start:
            continue
        # Keep the historical epoch permutations while indexing one global cosine.
        model.rate_multipliers = replace(config, layer_rate_decay=layer_decay).rate_multipliers
        args = SimpleNamespace(seed=shuffle_seed, batch=batch, rate=rate, epochs=budget, schedule="cosine")
        for local_epoch in training_epochs(model, data, args, data.train, stop,
                                           start_epoch=start, epoch_offset=offset):
            epoch = offset + local_epoch
            if epoch % 5:
                continue
            error = rgb_sse(model, data, data.train)
            if not np.isfinite(error):
                raise ValueError(f"Nonfinite predictions at epoch {epoch}")
            history.append(dict(epoch=epoch, squared_error=error))
            if error < best_error:
                best_error, best_epoch, best_state = error, epoch, model.state()
            if epoch % 25 == 0 or local_epoch == stop:
                save_progress(checkpoint, model, best_state,
                              dict(epoch=epoch, best_epoch=best_epoch, best_error=best_error, history=history))
                print(f"Epoch {epoch}/{budget}, summed squared RGB error {error:.2f}", flush=True)
    return dict(seed=seed, config=asdict(config), batch=batch, rate=rate, scheduled_epochs=budget,
                taper_after_epoch=switch, taper=decay, history=history,
                selected_epoch=best_epoch, squared_error=best_error)


def compare(run_result, historical, replay):
    switch = run_result["taper_after_epoch"]
    previous = [p for p in historical["history"] if p["epoch"] <= switch]
    previous += [dict(epoch=switch + p["epoch"], squared_error=p["squared_error"])
                 for p in replay["history"] if p["epoch"] > 0]
    fig, ax = plt.subplots(figsize=(7.1, 3.6), layout="constrained")
    for history, label, color in ((previous, "Two cosines, batch 128", "#565c66"),
                                   (run_result["history"], "One cosine, batch 128", "#00767b")):
        ax.plot([p["epoch"] for p in history], [p["squared_error"] for p in history], label=label, color=color)
    if run_result["history"][-1]["epoch"] > previous[-1]["epoch"]:
        archived = [p for p in historical["history"] if switch <= p["epoch"] <= run_result["history"][-1]["epoch"]]
        ax.plot([p["epoch"] for p in archived], [p["squared_error"] for p in archived],
                label="Historical refinement, batch 64", color="#a07612", linestyle="--")
    ax.axvline(switch, color="#a34d32", linestyle=":", label="Layer taper begins")
    ax.set(xlabel="Epoch", ylabel="Summed squared RGB error", yscale="log")
    ax.grid(axis="y", alpha=.25)
    ax.legend(frameon=False)
    fig.savefig(RESULTS / f"{STUDY}.png", dpi=160)
    plt.close(fig)
    return previous


def render_fits(result, historical):
    data = rgb_data(split=False)
    settings = historical["settings"]
    config = ModelConfig(**(result["config"] | {
        "layer_gate_widths": tuple(result["config"]["layer_gate_widths"]),
    }))
    seed = result["seed"]
    model = NumpyModel(config, seed)
    trial = Trial(f"{settings['label']}-all-pixels", config, tuple(map(tuple, settings["phases"])),
                  settings["backend"], settings["schedule"])
    fits = [("NASA target", data.y)]
    for title, checkpoint, error in (
            ("One cosine, batch 128", progress_path(seed), result["squared_error"]),
            ("Historical two cosines", trial.path(seed) / "checkpoint.npz", historical["squared_error"])):
        with np.load(checkpoint, allow_pickle=False) as saved:
            state = ({key.removeprefix("best_"): saved[key] for key in saved.files if key.startswith("best_")}
                     if "metadata" in saved else saved)
            model.load(state)
        fitted = predict(model, data, data.train)
        np.testing.assert_allclose(np.square((fitted - data.y) / 2, dtype=np.float64).sum(), error, rtol=1e-6)
        fits.append((f"{title}\nSquared error {error:.2f}", fitted))
    fig, axes = plt.subplots(1, 3, figsize=(8, 3), layout="constrained")
    for ax, (title, fitted) in zip(axes, fits, strict=True):
        ax.imshow((fitted.reshape(*data.shape, 3) + 1) / 2, interpolation="nearest")
        ax.set_title(title)
        ax.set_axis_off()
    fig.supxlabel("Source photograph AS17-148-22727 · NASA Johnson Space Center", fontsize=9)
    fig.savefig(RESULTS / f"{STUDY}-fits.png", dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--until", type=int, default=CHECK_EPOCH, help="Pause after this epoch on the original cosine")
    args = parser.parse_args()
    if not 5 <= args.until <= 560 or args.until % 5:
        parser.error("--until must be a multiple of 5 between 5 and 560")
    historical = report(HISTORICAL)
    reference = next(row for row in report("blue-marble-refinement-batch-size")["runs"] if row["batch"] == 128)
    result = run(historical, args.until)
    result["two_cosines_batch_128"] = compare(result, historical, reference)
    if result["history"][-1]["epoch"] == result["scheduled_epochs"]:
        render_fits(result, historical)
    (RESULTS / f"{STUDY}.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
