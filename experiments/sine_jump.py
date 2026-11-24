"""Fit a sine wave with a single jump using the paper's dense Hebbian MLP."""

from dataclasses import asdict, replace
from itertools import count
import json

import numpy as np

from hebbian_fit.data import scalar_data

from .fixed_capacity import RESULTS, rmse
from .scalar_quality import fit
from .sine_paper import SAMPLES, TEST_SEEDS, TRAIN_DOMAIN, curve
from .sine_paper import TRIAL as SINE_TRIAL


JUMP_X = .25
JUMP_HEIGHT = .5
AMPLITUDE = .5
LOCAL_RADIUS = .10
AWAY_RADIUS = .05
DEVELOPMENT_SEEDS = (1, 2, 3)
TRIAL = replace(SINE_TRIAL, label="sine-crest-jump", task="sine-jump")
LONGER = replace(TRIAL, label="sine-crest-jump-longer",
                 phases=TRIAL.phases + (("lion", .00045, 64, 480, .90),))


def deeper(depth):
    widths = tuple(round(float(w), 3) for w in np.linspace(.32, .16, depth))
    config = replace(TRIAL.config, depth=depth, layer_gate_widths=widths)
    return replace(TRIAL, label=f"sine-crest-jump-{depth}-layers", config=config)


def wide_trials():
    for units in (512, 2048, 8192):
        for gate in (.16, .32):
            config = replace(TRIAL.config, depth=1, width=units // 2, layer_gate_widths=(gate,))
            scale = np.sqrt(64 / units)
            phases = tuple((rule, float(rate * scale), batch, epochs, decay)
                           for rule, rate, batch, epochs, decay in TRIAL.phases[:2])
            yield replace(TRIAL, label=f"sine-crest-jump-1x{units}-gauss{gate:.2f}",
                          config=config, phases=phases)
    for units, gate, rate in ((512, .08, .0015), (2048, .08, .0015), (2048, .12, .003),
                              (2048, .16, .003), (8192, .10, .0015)):
        config = replace(TRIAL.config, depth=1, width=units // 2, layer_gate_widths=(gate,))
        phases = (("signsgd", rate, 64, 80, 1.), ("lion", rate * (.0008 / .003), 64, 100, .75))
        yield replace(TRIAL, label=f"sine-crest-jump-1x{units}-gauss{gate:.2f}-step{rate:g}",
                      config=config, phases=phases)
    for units, gate, initialization in ((512, .16, "normal"), (2048, .16, "normal"), (8192, .16, "normal"),
                                         (512, .18, "unit-sphere"), (2048, .18, "unit-sphere"),
                                         (2048, .10, "normal")):
        config = replace(TRIAL.config, depth=1, width=units // 2, layer_gate_widths=(gate,),
                         initialization=initialization)
        phases = tuple((rule, rate * 64 / units, batch, epochs, decay)
                       for rule, rate, batch, epochs, decay in TRIAL.phases[:2])
        yield replace(TRIAL, label=f"sine-crest-jump-1x{units}-gauss{gate:.2f}-{initialization}-small-step",
                      config=config, phases=phases)


def refine_wide(trial):
    scale = np.sqrt(64 / trial.config.units)
    extra = tuple((rule, float(rate * scale), batch, epochs, decay)
                  for rule, rate, batch, epochs, decay in TRIAL.phases[2:])
    return replace(trial, label=f"{trial.label}-refined", phases=trial.phases + extra)


def validation_rmse(trial, data, seeds, parent=None):
    errors = [fit(trial, data, seed, parent=parent)["validation"]["sse"] for seed in seeds]
    return float(np.sqrt(np.median(errors) / len(data.validation)))


def scale_steps(trial, factor):
    phases = tuple((rule, rate * factor, batch, epochs, decay)
                   for rule, rate, batch, epochs, decay in trial.phases)
    return replace(trial, label=f"{trial.label}-step-x{factor:g}", phases=phases)


def search_lower_steps(trial, data):
    best = trial
    best_error = validation_rmse(trial, data, DEVELOPMENT_SEEDS)
    sweep = [dict(factor=1.0, settings=asdict(trial), median_validation_rmse=best_error)]
    for exponent in count(1):
        factor = round(.8 ** exponent, 12)
        candidate = scale_steps(trial, factor)
        error = validation_rmse(candidate, data, DEVELOPMENT_SEEDS)
        sweep.append(dict(factor=factor, settings=asdict(candidate), median_validation_rmse=error))
        print(f"24 layers at {factor:g}× step · validation RMSE {error:.6f}", flush=True)
        if error >= best_error:
            return best, sweep
        best, best_error = candidate, error


def trial_name(trial):
    if trial.config.depth == 1:
        return f"1 layer · {trial.config.units:,} neurons"
    return f"{trial.config.depth} layers"


def sine(x):
    return AMPLITUDE * np.sin(2 * np.pi * np.asarray(x, dtype=np.float64))


def target(x):
    return sine(x) + JUMP_HEIGHT * (x > JUMP_X)


def dataset():
    data = scalar_data("sine", SAMPLES)
    return replace(data, y=target(data.x).astype(np.float32))


def load_model(seed, trial=TRIAL):
    model = trial.model(seed)
    with np.load(trial.path(seed) / "checkpoint.npz", allow_pickle=False) as state:
        model.load(state)
    return model


def transition(x, prediction):
    # Subtract the smooth term for measurement only, to isolate the fitted jump.
    step = prediction - sine(x)
    crossings = []
    for fraction in (.1, .5, .9):
        level = fraction * JUMP_HEIGHT
        indices = np.flatnonzero((step[:-1] <= level) & (step[1:] > level))
        positions = x[indices] + (level - step[indices]) * (
            (x[indices + 1] - x[indices]) / (step[indices + 1] - step[indices]))
        crossings.append(positions)
    low, midpoint, high = crossings
    if not len(midpoint):
        return dict(x10=None, x50=None, x90=None, width_10_to_90=None)
    center = float(midpoint[np.argmin(np.abs(midpoint - JUMP_X))])
    lower = low[low <= center]
    upper = high[high >= center]
    return dict(x10=float(lower[-1]) if len(lower) else None, x50=center,
                x90=float(upper[0]) if len(upper) else None,
                width_10_to_90=float(upper[0] - lower[-1]) if len(lower) and len(upper) else None)


def evaluate(model, data):
    x = data.x[data.test, 0]
    error = curve(model, x) - data.y[data.test, 0]
    nearby = np.abs(x - JUMP_X) <= AWAY_RADIUS
    away_intervals = ~(nearby[:-1] | nearby[1:])
    error_slope = np.diff(error) / np.diff(x.astype(np.float64))
    local_x = np.linspace(JUMP_X - LOCAL_RADIUS, JUMP_X + LOCAL_RADIUS, 8001, dtype=np.float32)
    return dict(test_rmse=rmse(error), away_from_jump_rmse=rmse(error[~nearby]),
                away_from_jump_slope_rmse=rmse(error_slope[away_intervals]),
                near_jump_rmse=rmse(error[nearby]), test_max_abs_error=float(np.abs(error).max()),
                transition=transition(local_x, curve(model, local_x)))


def render(model, row, comparisons, baseline=None, *, label=None, filename="sine-jump-fit"):
    label = label or f"{model.config.depth} layers (selected)"
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (whole, zoom) = plt.subplots(1, 2, figsize=(9.1, 4.15), width_ratios=[1.35, 1],
                                      layout="constrained")
    whole.set_title("Full fit", loc="left", fontsize=10)
    zoom.set_title(f"Jump at the crest, x = {JUMP_X:.2f}", loc="left", fontsize=10)
    zoom.set_xticks(JUMP_X + np.array([-.08, -.04, 0, .04, .08]))
    whole.set_ylabel("Function value")
    spans = (TRAIN_DOMAIN, (JUMP_X - LOCAL_RADIUS, JUMP_X + LOCAL_RADIUS))
    for ax, (left, right) in zip((whole, zoom), spans):
        # Plot the two analytic branches up to the same coordinate.
        x_left = np.linspace(left, JUMP_X, 3001)
        x_right = np.linspace(JUMP_X, right, 3001)
        target_x = np.concatenate((x_left, x_right))
        target_y = np.concatenate((sine(x_left), sine(x_right) + JUMP_HEIGHT))
        x = np.linspace(left, right, 8001, dtype=np.float32)
        ax.plot(target_x, target_y, color="#545961", linestyle=(0, (4, 2)), lw=1.35, label="Target")
        if ax is zoom and baseline is not None:
            ax.plot(x, curve(baseline, x), color="#a06133", linestyle=":", lw=1.5,
                    label="4 layers")
        for (comparison, name), color in zip(comparisons, ("#805187", "#507c38", "#946328"), strict=False):
            ax.plot(x, curve(comparison, x), color=color, linestyle="-.", lw=1.4, label=name)
        ax.plot(x, curve(model, x), color="#176b8a", lw=1.65, label=label)
        ax.set(xlabel="Input x", xlim=(left, right))
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_axisbelow(True)
        ax.grid(axis="y", color="#e3e6e9", lw=.6)
    diagnostics = row["transition"]
    width = diagnostics["width_10_to_90"]
    if width is not None:
        zoom.axvspan(diagnostics["x10"], diagnostics["x90"], color="#d9edf1", alpha=.62, lw=0, zorder=0)
        zoom.text(.035, .94, f"10–90% width {width:.3f}", transform=zoom.transAxes,
                  va="top", fontsize=9)
    handles, labels = zoom.get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside upper center", ncol=len(labels), frameon=False)
    fig.savefig(RESULTS / f"{filename}.png", dpi=180)
    fig.savefig(RESULTS / f"{filename}.pdf")
    plt.close(fig)


def measure(trial, data, seeds, *, validation=False, parent=None):
    measured = replace(data, test=data.validation) if validation else data
    rows = []
    for seed in seeds:
        training = fit(trial, data, seed, parent=parent)
        row = dict(seed=seed, **evaluate(load_model(seed, trial), measured),
                   phase_epochs=training["phase_epochs"])
        rows.append(row)
    return rows


def summarize(rows):
    metrics = (key for key in rows[0] if key not in ("seed", "phase_epochs", "transition"))
    medians = {key: float(np.median([row[key] for row in rows])) for key in metrics}
    widths = [row["transition"]["width_10_to_90"] for row in rows
              if row["transition"]["width_10_to_90"] is not None]
    medians["transition_width_10_to_90"] = float(np.median(widths)) if widths else None
    medians["measurable_transitions"] = len(widths)
    return medians


def main():
    data = dataset()
    depth_trials = [deeper(depth) for depth in (6, 8, 12)]
    original_24 = deeper(24)
    depth_24, step_sweep = search_lower_steps(original_24, data)
    wide_screen = {trial: validation_rmse(trial, data, (DEVELOPMENT_SEEDS[0],)) for trial in wide_trials()}
    candidates = sorted(wide_screen, key=wide_screen.__getitem__)[:3]
    wide_validation = {}
    parents = {LONGER: TRIAL}
    for candidate in candidates:
        revised = refine_wide(candidate)
        parents[revised] = candidate
        wide_validation[candidate] = validation_rmse(candidate, data, DEVELOPMENT_SEEDS)
        wide_validation[revised] = validation_rmse(revised, data, DEVELOPMENT_SEEDS, parent=candidate)
    best_wide = min(wide_validation, key=wide_validation.__getitem__)
    trials = [TRIAL, LONGER, *depth_trials, depth_24, best_wide]
    validation = {
        trial: validation_rmse(trial, data, DEVELOPMENT_SEEDS, parent=parents.get(trial))
        for trial in trials
    }
    best_deep = min(depth_trials, key=validation.__getitem__)
    selected = min(trials, key=validation.__getitem__)
    plotted_trials = (best_deep, depth_24, best_wide)
    comparisons = {}
    for trial in (TRIAL, LONGER, *plotted_trials):
        rows = measure(trial, data, TEST_SEEDS, parent=parents.get(trial))
        model = load_model(TEST_SEEDS[0], trial)
        medians = summarize(rows)
        comparisons[trial.label] = dict(settings=asdict(trial), parameters=sum(w.size for w in model.weights),
                                         runs=rows, medians=medians)
        print(trial.label, json.dumps(medians), flush=True)
    winner = comparisons[selected.label]
    example = sorted(winner["runs"], key=lambda row: row["test_rmse"])[len(TEST_SEEDS) // 2]
    baseline = load_model(example["seed"]) if selected != TRIAL else None
    others = [(load_model(example["seed"], trial), trial_name(trial))
              for trial in plotted_trials if trial != selected]
    render(load_model(example["seed"], selected), example, others, baseline,
           label=f"{trial_name(selected)} (selected)")
    report = dict(
        target="0.5*sin(2*pi*x) + 0.5*(x > 0.25)", jump_x=JUMP_X, jump_height=JUMP_HEIGHT,
        split_sizes=dict(training=len(data.train), validation=len(data.validation), test=len(data.test)),
        train_domain=TRAIN_DOMAIN, jump_neighborhood_radius=AWAY_RADIUS, transition_search_radius=LOCAL_RADIUS,
        selection_metric="Median in-range validation RMSE", development_seeds=DEVELOPMENT_SEEDS,
        screening={trial.label: dict(settings=asdict(trial), median_validation_rmse=validation[trial])
                   for trial in trials},
        selected=selected.label, example_seed=example["seed"],
        plotted_trials=[trial.label for trial in plotted_trials], selected_wide=best_wide.label,
        wide_screen={trial.label: dict(settings=asdict(trial), validation_rmse=score)
                     for trial, score in wide_screen.items()},
        wide_finalists={trial.label: dict(settings=asdict(trial), median_validation_rmse=score,
                                         parent=parents[trial].label if trial in parents else None)
                        for trial, score in wide_validation.items()},
        step_sweep_24=dict(
            trials=step_sweep, selected=depth_24.label,
            rule="Multiply each phase step by 0.8 per trial and stop at the first non-improvement",
        ),
        held_out=comparisons,
    )
    if depth_24 != original_24:
        original_rows = measure(original_24, data, TEST_SEEDS)
        report["step_sweep_24"]["original_held_out"] = dict(
            runs=original_rows, medians=summarize(original_rows),
        )
    (RESULTS / "sine-jump-fit.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
