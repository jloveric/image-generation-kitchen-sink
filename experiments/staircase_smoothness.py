"""Compare staircase fit accuracy and within-region roughness."""

import argparse
import json
from dataclasses import asdict, replace

import numpy as np

from hebbian_fit.data import BAND_EDGES, EDGES, scalar_data
from hebbian_fit.model import NumpyModel

from .blue_marble_quality import layer_readouts
from .fixed_capacity import RESULTS, predict, rmse
from .scalar_quality import COUNTS, fit, render, trials


VALIDATION_SEEDS = (1, 2, 3)
TEST_SEEDS = (7, 8, 9)


def candidates(backend):
    presets = {t.label: t for t in trials("steps", backend)}
    original = presets["steps-12x96-layer-all"]
    baseline = replace(original, label=f"{original.label}-longer",
                       phases=original.phases + (("lion", .0008, 128, 160, .85),))
    yield baseline, original
    for label, gate, rate, batch in (
        ("polish", "gaussian", .00025, 256),
        ("smooth-center", "gaussian-smooth", .0005, 128),
    ):
        yield replace(baseline, label=f"steps-{label}", config=replace(baseline.config, gate=gate),
                      phases=baseline.phases + (("lion", rate, batch, 120, .9),)), baseline
    wider_late = replace(baseline.config, layer_gate_widths=tuple(
        max(.12, width) for width in baseline.config.gate_widths))
    yield replace(baseline, label="steps-wider-final-gates", config=wider_late,
                  phases=baseline.phases + (("lion", .0006, 128, 120, .9),)), baseline
    for depth in (8, 12):
        trial = presets["steps-8x144-layer-all"] if depth == 8 else original
        yield replace(trial, label=f"{trial.label}-smooth-peak",
                      config=replace(trial.config, gate="gaussian-smooth")), None
    for width, gate, end in ((48, "gaussian", .14), (48, "gaussian-smooth", .14),
                              (64, "gaussian", .12), (64, "gaussian-smooth", .12)):
        widths = tuple(round(float(value), 3) for value in np.linspace(.24, end, 12))
        config = replace(original.config, width=width, gate=gate, layer_gate_widths=widths)
        trial = replace(original, label=f"steps-12x{config.units}-{gate}-wider-bandwidth", config=config)
        yield trial, None
        if width == 48 and gate == "gaussian":
            yield replace(trial, label=f"{trial.label}-refined",
                          phases=trial.phases + (("lion", .0005, 128, 140, .85),)), trial


def measures(x, target, values):
    smooth = x >= 0
    error = values - target
    interiors = []
    edges = EDGES[EDGES <= BAND_EDGES[-2]]
    for left, right in zip(edges[:-1], edges[1:]):
        margin = (right - left) / 8
        part = values[(x >= left + margin) & (x <= right - margin)]
        interiors.append(part - part.mean())
    return dict(
        rmse=rmse(error),
        smooth_rmse=rmse(error[smooth]),
        finest_steps_rmse=rmse(error[(x >= BAND_EDGES[-2]) & (x < 0)]),
        plateau_ripple=rmse(np.concatenate(interiors)),
        smooth_slope_error=rmse(np.diff(error[smooth]) / np.diff(x[smooth])),
    )


def arrays(data, indices):
    return data.x[indices, 0], data.y[indices, 0]


def score(trial, data, seed, indices):
    model = NumpyModel(trial.config, seed)
    with np.load(trial.path(seed) / "checkpoint.npz") as state:
        model.load(state)
    x, target = arrays(data, indices)
    return measures(x, target, predict(model, data, indices)[:, 0])


def median(rows):
    return {key: float(np.median([r[key] for r in rows])) for key in rows[0]}


def quieter(summaries, baseline):
    reference = summaries[baseline]
    return sorted(
        (trial for trial, scores in summaries.items()
         if all(scores[key] < reference[key] for key in ("plateau_ripple", "smooth_slope_error"))),
        key=lambda trial: summaries[trial]["rmse"],
    )


def run_trial(trial, data, seed, parents):
    parent = parents[trial]
    if parent in parents:
        fit(parent, data, seed, parents[parent])
    return fit(trial, data, seed, parent)


def layer_predictions(trial, data, seed):
    model = NumpyModel(trial.config, seed)
    with np.load(trial.path(seed) / "checkpoint.npz") as state:
        model.load(state)
    terms, full = layer_readouts(model, data.x[data.test])
    individual = np.stack([model.format_output(term.copy())[:, 0] for term in terms])
    prefixes = np.stack([
        model.format_output(terms[:depth].mean(axis=0))[:, 0]
        for depth in range(1, len(terms) + 1)
    ])
    np.testing.assert_array_equal(prefixes[-1], full[:, 0])
    return individual, prefixes


def render_layers(x, target, curves, seed, cumulative):
    import matplotlib.pyplot as plt

    columns = 3
    rows = (len(curves) + columns - 1) // columns
    fig, axes = plt.subplots(rows, columns, figsize=(10.7, 2.05 * rows), squeeze=False,
                             sharex=True, sharey=True, layout="constrained")
    for index, (ax, curve) in enumerate(zip(axes.flat, curves), start=1):
        ax.plot(x, target, color="#828996", lw=.85)
        ax.plot(x, curve, color="#225f9e", lw=1)
        label = f"Layers 1–{index}" if cumulative and index > 1 else f"Layer {index}"
        ax.set_title(label, fontsize=10)
        ax.grid(alpha=.16)
    for ax in list(axes.flat)[len(curves):]:
        ax.set_axis_off()
    fig.supxlabel("Input x")
    fig.supylabel("Prediction")
    title = "Cumulative averages" if cumulative else "Individual readouts"
    fig.suptitle(f"Staircase · {title} · seed {seed}\nGray is the target")
    kind = "cumulative" if cumulative else "individual"
    fig.savefig(RESULTS / f"staircase-{kind}-layers.png", dpi=170)
    plt.close(fig)


def render_comparison(x, target, baseline, selected, depth_rows, seed, improved):
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(10.6, 6.2), layout="constrained")
    for ax, title, low, high in (
        (axes[0, 0], "Coarse plateaus", -1., BAND_EDGES[1]),
        (axes[0, 1], "Smooth-region detail", .06, .30),
    ):
        keep = (x >= low) & (x <= high)
        ax.plot(x[keep], target[keep], label="Target", lw=1.4, color="#616977")
        ax.plot(x[keep], baseline[keep], label="Previous fit", lw=1, color="#bb6b3f")
        if improved:
            ax.plot(x[keep], selected[keep], label="Selected fit", lw=1.15, color="#22639e")
        ax.set(title=title, xlabel="x", ylabel="y")
        ax.grid(alpha=.16)
    axes[0, 0].legend(frameon=False)
    for ax, key, title in (
        (axes[1, 0], "rmse", "Value error"),
        (axes[1, 1], "smooth_slope_error", "Slope error on the smooth half"),
    ):
        for label, color in (("individual", "#9467a0"), ("cumulative", "#22639e")):
            values = [row[key] for row in depth_rows[label]]
            ax.plot(np.arange(1, len(values) + 1), values, ".-", label=label.capitalize(), color=color)
        ax.set(title=title, xlabel="Layer", ylabel="Median test RMSE", yscale="log")
        ax.grid(alpha=.16)
    axes[1, 1].legend(frameon=False)
    fig.suptitle(f"Staircase smoothness · curve example from seed {seed}")
    fig.savefig(RESULTS / "staircase-smoothness.png", dpi=170)
    plt.close(fig)


def publish(data, baseline, best, validation, parents):
    import matplotlib
    matplotlib.use("Agg")

    tests = {}
    for trial in dict.fromkeys((baseline, best)):
        rows = []
        for seed in TEST_SEEDS:
            run_trial(trial, data, seed, parents)
            rows.append(dict(seed=seed, **score(trial, data, seed, data.test)))
        tests[trial] = rows
    example = sorted(tests[best], key=lambda row: row["rmse"])[1]
    seed = example["seed"]
    x, target = arrays(data, data.test)
    metrics = {"individual": [], "cumulative": []}
    for test_seed in TEST_SEEDS:
        individual, cumulative = layer_predictions(best, data, test_seed)
        for label, predictions in (("individual", individual), ("cumulative", cumulative)):
            metrics[label].append([measures(x, target, curve) for curve in predictions])
    layer_medians = {label: [median(rows) for rows in zip(*runs, strict=True)] for label, runs in metrics.items()}
    individual, cumulative = layer_predictions(best, data, seed)
    _, reference = layer_predictions(baseline, data, seed)
    fit_row = run_trial(best, data, seed, parents)
    render(data, "steps", cumulative[-1], example, fit_row["history"], "staircase-refined-fit.png")
    render_layers(x, target, individual, seed, cumulative=False)
    render_layers(x, target, cumulative, seed, cumulative=True)
    render_comparison(x, target, reference[-1], cumulative[-1], layer_medians, seed, best != baseline)
    report = dict(baseline=asdict(baseline), selected=asdict(best), samples_per_split=len(data.train),
                  example_seed=seed, validation_seeds=VALIDATION_SEEDS, test_seeds=TEST_SEEDS,
                  validation={t.label: median(rows) for t, rows in validation.items()},
                  test={t.label: rows for t, rows in tests.items()}, layer_medians=layer_medians)
    (RESULTS / "staircase-smoothness.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    for trial, rows in tests.items():
        print(trial.label, {k: v for k, v in median(rows).items() if k != "seed"}, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("numpy", "mojo"), default="numpy")
    parser.add_argument("--screen-only", action="store_true")
    args = parser.parse_args()
    data = scalar_data("steps", COUNTS["steps"])
    parents = dict(candidates(args.backend))
    baseline = next(iter(parents))
    screen = {}
    for trial in parents:
        run_trial(trial, data, 1, parents)
        screen[trial] = score(trial, data, 1, data.validation)
        print(trial.label, screen[trial], flush=True)
    if args.screen_only:
        return
    validation = {}
    for trial in (baseline, *quieter(screen, baseline)[:3]):
        rows = [screen[trial]]
        for seed in VALIDATION_SEEDS[1:]:
            run_trial(trial, data, seed, parents)
            rows.append(score(trial, data, seed, data.validation))
        validation[trial] = rows
        print(trial.label, "validation", median(rows), flush=True)
    winners = quieter({trial: median(rows) for trial, rows in validation.items()}, baseline)
    best = next(iter(winners), baseline)
    publish(data, baseline, best, validation, parents)


if __name__ == "__main__":
    main()
