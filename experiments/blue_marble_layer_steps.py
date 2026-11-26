"""Compare shared and geometrically scaled Lion steps under one cosine."""

import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, replace
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from experiments.blue_marble_quality import train
from experiments.fixed_capacity import predict
from hebbian_fit.data import rgb_data
from papers.lion import MARBLE_SEED, RESULTS, marble_trial, report


STUDY = "blue-marble-layer-steps"
CHECK_EPOCH = 100
HISTORICAL = "blue-marble-output-depth-12-two-phase-all-lion-all-pixels"


def trial_for(decay, normalize=False):
    baseline = marble_trial("lion", "numpy", batch=64)
    name = "uniform" if decay == 1. else "scaled"
    config = replace(baseline.config, layer_rate_decay=decay, normalize_hidden_inputs=normalize)
    if config.normalize_hidden_inputs:
        name += f"-interlayer-{config.normalize_hidden_inputs}"
    return replace(baseline, label=f"layer-steps-{name}-all-pixels", config=config,
                   phases=(("lion", .0015, 64, 560, decay),))


def measure(decay, until_epoch, normalize=False):
    trial = trial_for(decay, normalize)
    data = rgb_data(split=False)
    row = train(trial, MARBLE_SEED, data, checkpoint_pixels=data.train, until_epoch=until_epoch)
    return dict(settings=asdict(trial), seed=MARBLE_SEED, selected_epoch=row["phase_epochs"][0],
                squared_error=row["validation"]["sse"],
                history=[dict(epoch=p["epoch"], squared_error=p["train_sse"]) for p in row["history"]])


def render_comparison(rows, historical, normalization=False):
    data = rgb_data(split=False)
    size = len(data.x)
    fig, ax = plt.subplots(figsize=(7.2, 3.7), layout="constrained")
    colors = ("#00767b", "#a34d32", "#565c66")
    label = {"rms": "RMS normalization", "max-abs": "Max-abs normalization", False: "Scaled step"}[normalization]
    labels = ("Uniform step", label, "Historical two-stage run")
    study = f"blue-marble-interlayer-{normalization}" if normalization else STUDY
    max_updates = max(row["history"][-1]["epoch"] for row in rows) * (size // 64)
    for row, color, label in zip((*rows, historical), colors, labels, strict=True):
        phases = row["settings"]["phases"]
        boundaries = np.cumsum([0] + [phase[3] for phase in phases[:-1]])
        epochs = np.array([p["epoch"] for p in row["history"]])
        errors = np.array([p["squared_error"] for p in row["history"]])
        updates = np.zeros_like(epochs)
        for (_, _, batch, duration, _), start in zip(phases, boundaries, strict=True):
            updates += np.clip(epochs - start, 0, duration) * (size // batch)
        inside = updates <= max_updates
        ax.plot(updates[inside], errors[inside], label=label, color=color)
    ax.set(xlabel="Minibatch updates", ylabel="Summed squared RGB error", yscale="log")
    ax.grid(axis="y", alpha=.25)
    ax.legend(frameon=False)
    fig.savefig(RESULTS / f"{study}-convergence.png", dpi=160)
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(8, 3), layout="constrained")
    shape = (*data.shape, 3)
    axes[0].imshow((data.y.reshape(shape) + 1) / 2)
    axes[0].set_title("NASA target")
    for ax, row, label in zip(axes[1:], rows, labels[:2], strict=True):
        cfg = row["settings"]["config"]
        trial = trial_for(cfg["layer_rate_decay"], cfg.get("normalize_hidden_inputs", False))
        directory = trial.path(row["seed"])
        model = trial.model(row["seed"])
        checkpoint = directory / "checkpoint.npz"
        if checkpoint.exists():
            with np.load(checkpoint, allow_pickle=False) as saved:
                model.load(saved)
        else:
            with np.load(directory / "phase-1-progress.npz", allow_pickle=False) as saved:
                model.load({key.removeprefix("best_"): saved[key] for key in saved.files
                            if key.startswith("best_")})
        fitted = predict(model, data, data.train)
        error = np.square((fitted - data.y) / 2, dtype=np.float64).sum()
        np.testing.assert_allclose(error, row["squared_error"], rtol=1e-6)
        ax.imshow((fitted.reshape(shape) + 1) / 2, interpolation="nearest")
        ax.set_title(f"{label} · epoch {row['selected_epoch']}\nSquared error {error:.2f}")
    for ax in axes:
        ax.set_axis_off()
    fig.supxlabel("Source photograph AS17-148-22727 · NASA Johnson Space Center", fontsize=9)
    fig.savefig(RESULTS / f"{study}-fits.png", dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--normalization", nargs="?", const="rms", default=False, choices=("rms", "max-abs"),
                        help="Compare uniform steps with and without interlayer normalization")
    parser.add_argument("--until", "--scaled-until", dest="until", type=int, default=CHECK_EPOCH,
                        help="Stop the experimental run at this epoch on the original cosine")
    args = parser.parse_args()
    if not 5 <= args.until <= 560 or args.until % 5:
        parser.error("--until must be a multiple of 5 between 5 and 560")
    until_epoch = None if args.until == 560 else args.until
    with ProcessPoolExecutor(max_workers=2) as pool:
        jobs = [pool.submit(measure, 1., CHECK_EPOCH),
                pool.submit(measure, 1. if args.normalization else .7, until_epoch, args.normalization)]
        rows = [job.result() for job in jobs]
    historical = report(HISTORICAL)
    render_comparison(rows, historical, args.normalization)
    study = f"blue-marble-interlayer-{args.normalization}" if args.normalization else STUDY
    (RESULTS / f"{study}.json").write_text(json.dumps({"runs": rows}, indent=2, allow_nan=False) + "\n")
