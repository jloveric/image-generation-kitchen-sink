"""Confirm the validation-selected Hebbian networks and publish the capacity report."""
import argparse
import json
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict

import numpy as np

from hebbian_fit.data import BAND_EDGES, EDGES, HEIGHTS, scalar_data
from .fixed_capacity import RESULTS, STAGES, Trial, predict, score, train


SEEDS = tuple(range(1, 7))


def select():
    groups = defaultdict(list)
    trials = dict.fromkeys(trial for stage in STAGES.values() for trial in stage())
    rows = [json.loads((trial.path(1) / "validation.json").read_text()) for trial in trials]
    for trial, row in zip(trials, rows):
        if row["validation"] is not None:
            groups[trial.task, trial.width, trial.depth].append((row["validation"]["rmse"], trial))
    winners = {f"{task}/{2 * width}x{depth}": min(group, key=lambda pair: pair[0])[1]
               for (task, width, depth), group in sorted(groups.items())}
    matched = {f"matched/{2 * width}x1": Trial(width=width) for width in (32, 128, 512)}
    depth = {f"matched-depth/64x{n}": Trial(depth=n, competition="layer", tau=.15,
                                             rate=.03, readout_layers="all") for n in (1, 2, 3)}
    sine = {f"matched-sine/{2 * width}x1": Trial(task="sine", width=width, competition="layer",
                                                tau=.15, rate=.001, batch=32, epochs=120)
            for width in (32, 128, 512)}
    two_layer = {f"matched-two-layer/{2 * width}x2": Trial(
        width=width, depth=2, competition="layer", tau=.15, rate=.01,
        readout_layers="all", batch=64) for width in (32, 64, 128)}
    return rows, {**winners, **matched, **depth, **sine, **two_layer}


def confirmed(trial, row, data):
    model = trial.model(row["seed"])
    path = trial.path(row["seed"]) / "checkpoint.npz"
    if not row["updated_layers"] or not all(row["updated_layers"]):
        raise RuntimeError(f"Expected every hidden layer to update for {path}")
    with np.load(path) as state:
        model.load(state)
    metrics = score(model, data, data.test, trial.task)
    return dict(**row, test=metrics)


def summarize(trial, rows):
    data = scalar_data(trial.task)
    runs = [confirmed(trial, row, data) for row in rows]
    medians = {key: float(np.median([run["test"][key] for run in runs])) for key in runs[0]["test"]}
    params = sum(weight.size for weight in trial.model(1).weights)
    return dict(trial=asdict(trial), parameters=params, medians=medians, runs=runs)


def curve(trial, seed, data):
    model = trial.model(seed)
    with np.load(trial.path(seed) / "checkpoint.npz") as state:
        model.load(state)
    return predict(model, data, data.test).ravel()


def plot_fits(picks):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    step_data = scalar_data("steps")
    x = step_data.x[step_data.test, 0]
    step_trials = [(picks["matched-depth/64x1"], "64 hidden neurons"),
                   (picks["matched-depth/64x2"], "64 × 64 hidden neurons"),
                   (picks["matched-two-layer/256x2"], "256 × 256 hidden neurons")]
    sine_trials = [(picks["matched-sine/64x1"], "64 hidden neurons"),
                   (picks["matched-sine/256x1"], "256 hidden neurons")]
    fig, axes = plt.subplots(3, 1, figsize=(10, 8.2), layout="constrained")
    for ax in axes[:2]:
        ax.stairs(HEIGHTS, EDGES, baseline=None, color="#253345", lw=1.1, label="Target")
    smooth = np.linspace(0, 1, 500)
    axes[0].plot(smooth, .75 * np.cos(2 * np.pi * smooth), color="#253345", lw=1.1)
    for (trial, label), color in zip(step_trials, ("#9e6a34", "#2d876c", "#2862a5")):
        estimates = curve(trial, 3, step_data)
        axes[0].plot(x, estimates, color=color, lw=.95, label=label)
        fine = (x >= BAND_EDGES[-2]) & (x < 0)
        axes[1].plot(x[fine], estimates[fine], color=color, lw=.95, label=label)
    axes[0].set(title="Staircase and smooth region", xlim=(-1, 1))
    axes[1].set(title="Staircase detail · narrowest 64 plateaus", xlim=(BAND_EDGES[-2], 0))
    sine = scalar_data("sine")
    sine_x = sine.x[sine.test, 0]
    axes[2].plot(sine_x, sine.y[sine.test, 0], label="Target", color="#253345", lw=1.1)
    for (trial, label), color in zip(sine_trials, ("#9e6a34", "#2862a5")):
        axes[2].plot(sine_x, curve(trial, 3, sine), color=color, label=label, lw=.95)
    axes[2].set(title="Sine", xlim=(-1, 1))
    for ax in axes:
        ax.legend(loc="upper right", ncols=2, fontsize=8)
        ax.set(ylabel="Output")
    axes[-1].set_xlabel("Input")
    fig.suptitle("Fixed-pool Hebbian networks · held-out coordinates, seed 3")
    fig.savefig(RESULTS / "fixed-capacity.png", dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    validation_rows, picks = select()
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        jobs = [(trial, seed) for trial in dict.fromkeys(picks.values()) for seed in SEEDS]
        trained = dict(zip(jobs, executor.map(train, jobs)))
    summary = {
        key: summarize(trial, [trained[trial, seed] for seed in SEEDS])
        for key, trial in picks.items()
    }
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "fixed-capacity.json").write_text(json.dumps(
        dict(validation_screen=validation_rows, confirmed=summary), indent=2, allow_nan=False) + "\n")
    plot_fits(picks)
    for key, entry in summary.items():
        print(key, entry["parameters"], entry["medians"], entry["trial"], flush=True)


if __name__ == "__main__":
    main()
