"""Compare staircase networks with matched training schedules."""

import argparse
import json
from dataclasses import asdict, replace

import numpy as np

from hebbian_fit.data import scalar_data
from hebbian_fit.model import NumpyModel

from .fixed_capacity import RESULTS, predict
from .scalar_quality import fit
from .staircase_smoothness import TEST_SEEDS, VALIDATION_SEEDS, candidates, median, score


def width_trials(backend):
    refined = next((trial, parent) for trial, parent in candidates(backend)
                   if trial.label == "steps-12x96-gaussian-wider-bandwidth-refined")
    yield refined
    reference, parent = refined
    for units in (128, 192):
        config = replace(reference.config, width=units // 2)
        initial = replace(parent, label=f"steps-width-{units}", config=config)
        yield replace(reference, label=f"{initial.label}-refined", config=config), initial


def dual_trial(reference, parent):
    config = replace(reference.config, gate="gaussian-dual")
    initial = replace(parent, label=f"steps-width-{config.units}-dual", config=config)
    return replace(reference, label=f"{initial.label}-refined", config=config), initial


def evaluate(trial, parent, data):
    groups = {}
    for name, seeds, indices in (
        ("validation", VALIDATION_SEEDS, data.validation), ("test", TEST_SEEDS, data.test),
    ):
        rows = []
        metrics = []
        for seed in seeds:
            checkpoint = fit(trial, data, seed, parent)
            row = score(trial, data, seed, indices)
            metrics.append(row)
            rows.append(dict(seed=seed, phase_epochs=checkpoint["phase_epochs"], **row))
            print(trial.label, name, seed, row, flush=True)
        groups[name] = dict(runs=rows, median=median(metrics))
    units = trial.config.units
    parameters = units * (trial.config.inputs + 1 + (trial.config.depth - 1) * (units + 1))
    return dict(settings=asdict(trial), parameters=parameters, **groups)


def render(data, trials, filename="staircase-width.png"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    seed = TEST_SEEDS[1]
    x = data.x[data.test, 0]
    target = data.y[data.test, 0]
    fig, axes = plt.subplots(2, 2, figsize=(11, 6.5), layout="constrained")
    for ax, (title, lo, hi) in zip(axes.flat, (
        ("Whole domain", -1, 1), ("Intermediate steps", -2 / 3, -1 / 3),
        ("Finest steps", -1 / 3, -.25), ("Smooth-region detail", .06, .30),
    ), strict=True):
        keep = (x >= lo) & (x <= hi)
        ax.plot(x[keep], target[keep], label="Target", color="#454f5e", lw=1.7)
        ax.set(title=title, xlabel="x", ylabel="y", xlim=(lo, hi))
        ax.grid(alpha=.15)
    compare_depth = len({trial.config.depth for trial in trials}) > 1
    for trial, color in zip(trials, ("#28629e", "#bc6a40", "#47816f", "#896796")):
        model = NumpyModel(trial.config, seed)
        with np.load(trial.path(seed) / "checkpoint.npz") as state:
            model.load(state)
        values = predict(model, data, data.test)[:, 0]
        label = f"{trial.config.depth} layers" if compare_depth else f"{trial.config.units} neurons"
        if trial.config.gate == "gaussian-dual":
            label += " · two widths"
        for ax in axes.flat:
            lo, hi = ax.get_xlim()
            keep = (x >= lo) & (x <= hi)
            ax.plot(x[keep], values[keep], label=label, color=color, lw=1)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=len(handles), frameon=False, fontsize=9)
    comparison = "depth" if compare_depth else "width"
    fig.suptitle(f"Staircase {comparison} comparison · seed {seed}")
    fig.savefig(RESULTS / filename, dpi=165)
    plt.close(fig)


def compare_24_layers(backend):
    reference, parent = next(width_trials(backend))
    gates = reference.config.gate_widths
    config = replace(reference.config, depth=24, layer_gate_widths=tuple(
        round(float(width), 3) for width in np.linspace(gates[0], gates[-1], 24)))
    initial = replace(parent, label="steps-depth-24", config=config)
    deeper = replace(reference, label="steps-depth-24-refined", config=config)
    data = scalar_data("steps", reference.samples)
    seed = TEST_SEEDS[1]
    rows = []
    for trial, start in ((reference, parent), (deeper, initial)):
        checkpoint = fit(trial, data, seed, start)
        metrics = score(trial, data, seed, data.test)
        rows.append(dict(settings=asdict(trial), seed=seed,
                         parameters=sum(w.size for w in NumpyModel(trial.config, seed).weights),
                         phase_epochs=checkpoint["phase_epochs"], test=metrics))
        print(trial.label, metrics, flush=True)
    (RESULTS / "staircase-depth-24.json").write_text(json.dumps(rows, indent=2, allow_nan=False) + "\n")
    render(data, [reference, deeper], "staircase-depth-24.png")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("numpy", "mojo"), default="numpy")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--include-dual", action="store_true", help="Compare a 96-neuron model with two Gaussian widths")
    mode.add_argument("--depth-24", action="store_true", help="Compare 12 and 24 layers on seed 8")
    args = parser.parse_args()
    if args.depth_24:
        compare_24_layers(args.backend)
        return
    trials = list(width_trials(args.backend))
    if args.include_dual:
        trials.append(dual_trial(*trials[0]))
    data = scalar_data("steps", trials[0][0].samples)
    report = [evaluate(trial, parent, data) for trial, parent in trials]
    (RESULTS / "staircase-width.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    render(data, [trial for trial, _ in trials])


if __name__ == "__main__":
    main()
