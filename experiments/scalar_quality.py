"""Measure dense Hebbian scalar regressors on high-resolution coordinate grids."""

import argparse
import json
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import numpy as np

from hebbian_fit.data import BAND_EDGES, scalar_data
from hebbian_fit.model import ModelConfig, NumpyModel

from .blue_marble_quality import Trial, train
from .fixed_capacity import RESULTS, predict, rmse


COUNTS = {"sine": 4096, "steps": 12288}


@dataclass(frozen=True)
class ScalarTrial(Trial):
    task: str = "sine"
    samples: int = 4096

    def path(self, seed):
        return Path("runs/scalar-quality") / super().path(seed).parent.name / str(seed)


def squared_error(model, data, indices):
    residual = predict(model, data, indices) - data.y[indices]
    return float(np.square(residual, dtype=np.float64).sum())


def trials(task, backend):
    widths = (.24, .08) if task == "steps" else (.32, .16)
    if task == "sine":
        settings = ((24, 1, "layer", "all"), (24, 2, "layer", "all"),
                    (32, 4, "layer", "all"), (32, 4, "layer", "last"),
                    (32, 4, "class", "all"), (48, 6, "layer", "all"))
        phases = (("signsgd", .003, 64, 80, 1.), ("lion", .0008, 64, 100, .75))
    else:
        settings = ((32, 2, "layer", "all"), (48, 4, "layer", "all"),
                    (48, 8, "layer", "all"), (72, 8, "layer", "all"),
                    (48, 8, "layer", "last"), (48, 8, "class", "all"),
                    (48, 12, "layer", "all"))
        phases = (("signsgd", .003, 128, 80, 1.), ("lion", .0015, 128, 120, .8))
    for width, depth, competition, heads in settings:
        gates = tuple(round(float(tau), 3) for tau in np.linspace(*widths, depth))
        config = ModelConfig(width=width, depth=depth, gate="gaussian", competition=competition,
                             layer_gate_widths=gates, readout="class-sum", readout_layers=heads)
        label = f"{task}-{depth}x{config.units}-{competition}-{heads}"
        yield ScalarTrial(label, config, phases, backend, task=task, samples=COUNTS[task])


def fit(trial, data, seed, parent=None):
    return train(trial, seed, data, parent=parent, squared_error=squared_error)


def staircase_followup(base):
    wider = base.config
    initial, refinement = base.phases
    for label, depth, width, final_width, batch, epochs in (
        ("long-refinement", 8, 72, .08, 128, 320),
        ("narrow-gaussians", 8, 72, .04, 128, 240),
        ("deeper", 12, 72, .08, 128, 240),
        ("smaller-batches", 8, 72, .08, 64, 180),
        ("16-layers", 16, 48, .06, 128, 180),
        ("20-layers", 20, 48, .06, 128, 180),
    ):
        config = replace(wider, depth=depth, width=width, layer_gate_widths=tuple(
            round(float(value), 3) for value in np.linspace(.24, final_width, depth)))
        phases = ((initial[0], initial[1], batch, initial[3], initial[4]),
                  (refinement[0], refinement[1], batch, epochs, refinement[4]))
        yield replace(base, label=f"steps-{label}", config=config, phases=phases)


def evaluate_trial(trial, data, seed, *, indices=None):
    model = trial.model(seed)
    with np.load(trial.path(seed) / "checkpoint.npz") as state:
        model.load(state)
    if indices is None:
        indices = data.test
    target = data.y[indices].ravel()
    values = predict(model, data, indices).ravel()
    residual = values - target
    x = data.x[indices].ravel()
    row = dict(seed=seed, rmse=rmse(residual))
    if trial.task == "steps":
        for i, (left, right) in enumerate(zip(BAND_EDGES[:-1], BAND_EDGES[1:]), start=1):
            row[f"band_{i}_rmse"] = rmse(residual[(x >= left) & (x < right)])
        smooth = x >= 0
        row["smooth_rmse"] = rmse(residual[smooth])
        residual, x = residual[smooth], x[smooth]
    row["slope_error_rmse"] = rmse(np.diff(residual.astype(np.float64)) / np.diff(x))
    return row, values


def render(data, task, winner, row, history, filename):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    x = data.x[data.test].ravel()
    target = data.y[data.test].ravel()
    panels = (("Whole domain", -1, 1), ("Fine steps", -1 / 3, -1 / 4)) if task == "steps" else (
        ("Whole domain", -1, 1), ("Positive peak", .15, .35))
    fig, axes = plt.subplots(2, 2, figsize=(9.4, 6), layout="constrained")
    for ax, (label, left, right) in zip(axes[0], panels):
        keep = (x >= left) & (x <= right)
        ax.plot(x[keep], target[keep], color="#6b7280", label="Target", lw=1.7)
        ax.plot(x[keep], winner[keep], color="#225f9e", label="Hebbian fit", lw=1.)
        ax.set(title=label, xlabel="x", ylabel="y")
    axes[0, 0].legend(frameon=False)
    error = winner - target
    axes[1, 0].plot(x, error, color="#225f9e", lw=.65)
    axes[1, 0].set(title="Fit minus target", xlabel="x", ylabel="Error")
    for key, label in (("train_sse", "Training"), ("validation_sse", "Validation")):
        axes[1, 1].plot([p["epoch"] for p in history], [p[key] for p in history], label=label)
    axes[1, 1].set(title="Convergence", xlabel="Scheduled epoch", ylabel="Sum of squared errors", yscale="log")
    axes[1, 1].legend(frameon=False)
    fig.suptitle(f"{'Sine' if task == 'sine' else 'Staircase'} · seed {row['seed']} · test RMSE {row['rmse']:.4f}")
    fig.savefig(RESULTS / filename, dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("task", choices=COUNTS)
    parser.add_argument("--backend", choices=("numpy", "mojo"), default="numpy")
    parser.add_argument("--followup", action="store_true", help="Also screen staircase refinements")
    parser.add_argument("--smooth-peak", action="store_true", help="Also test a smooth Gaussian center")
    parser.add_argument("--screen-only", action="store_true", help="Stop after scoring initialization seed 1")
    args = parser.parse_args()
    if args.followup and args.task != "steps":
        parser.error("--followup applies to the staircase")
    data = scalar_data(args.task, COUNTS[args.task])
    configs = list(trials(args.task, args.backend))
    if args.smooth_peak:
        sizes = {4, 6} if args.task == "sine" else {8, 12}
        configs.extend(replace(trial, label=f"{trial.label}-smooth-peak",
                               config=replace(trial.config, gate="gaussian-smooth"))
                       for trial in configs.copy()
                       if trial.config.depth in sizes and trial.config.competition == "layer"
                       and trial.config.readout_layers == "all")
    if args.followup:
        configs.extend(staircase_followup(next(t for t in configs if t.label == "steps-8x144-layer-all")))
        deep = next(t for t in configs if t.label == "steps-16-layers")
        configs.append(replace(deep, label=f"{deep.label}-smooth-peak",
                               config=replace(deep.config, gate="gaussian-smooth")))
    screen = {}
    for trial in configs:
        row = fit(trial, data, 1)
        screen[trial] = row
        print(trial.label, np.sqrt(row["validation"]["sse"] / trial.samples), flush=True)
    if args.screen_only:
        return
    finalists = sorted(configs, key=lambda t: screen[t]["validation"]["sse"])[:2]
    if args.followup:
        previous = next(t for t in configs if t.label == "steps-12x96-layer-all")
        if previous not in finalists:
            finalists.append(previous)
    parents = {}
    for trial in finalists.copy():
        phase = ("lion", .0004 if args.task == "sine" else .0008, trial.phases[-1][2], 160, .85)
        longer = replace(trial, label=f"{trial.label}-longer", phases=trial.phases + (phase,))
        parents[longer] = trial
        finalists.append(longer)
    validation = {}
    for trial in finalists:
        rows = [fit(trial, data, seed, parents.get(trial)) for seed in (1, 2, 3)]
        validation[trial] = rows
        print(trial.label, "validation", np.sqrt(np.median([r["validation"]["sse"] for r in rows]) / trial.samples),
              flush=True)
    best = min(finalists, key=lambda t: np.median([r["validation"]["sse"] for r in validation[t]]))
    measurements = []
    for seed in (7, 8, 9):
        fit_row = fit(best, data, seed, parents.get(best))
        row, values = evaluate_trial(best, data, seed)
        measurements.append((row, values, fit_row))
    ordered = sorted(measurements, key=lambda entry: entry[0]["rmse"])
    row, values, fit_row = ordered[1]
    runs = [entry[0] for entry in measurements]
    medians = {key: float(np.median([item[key] for item in runs])) for key in runs[0] if key != "seed"}
    params = sum(weight.size for weight in NumpyModel(best.config, 1).weights)
    report = dict(settings=asdict(best), parameters=params, train_samples=len(data.train),
                  validation_samples=len(data.validation), test_samples=len(data.test),
                  medians=medians, runs=runs, example_seed=row["seed"],
                  phase_epochs={entry[0]["seed"]: entry[2]["phase_epochs"] for entry in measurements},
                  screen=list(screen.values()), validation=[dict(settings=asdict(t), runs=r) for t, r in validation.items()])
    if args.task == "sine" and args.smooth_peak:
        labels = {"sine-4x64-layer-all", "sine-4x64-layer-all-smooth-peak"}
        comparison = []
        for trial in configs:
            if trial.label not in labels:
                continue
            rows = []
            for seed in (1, 2, 3):
                fit(trial, data, seed)
                rows.append(evaluate_trial(trial, data, seed, indices=data.validation)[0])
            comparison.append(dict(gate=trial.config.gate, runs=rows))
        report["peak_comparison"] = comparison
    filename = f"scalar-quality-{args.task}"
    (RESULTS / f"{filename}.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    render(data, args.task, values, row, fit_row["history"], f"{filename}.png")
    print(best.label, medians, flush=True)


if __name__ == "__main__":
    main()
