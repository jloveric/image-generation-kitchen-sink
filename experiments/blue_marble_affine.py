"""Select an affine Hebbian RGB fit by validation sum of squared errors."""

import argparse
import hashlib
import json
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, replace
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace

import numpy as np

from hebbian_fit.__main__ import training_epochs
from hebbian_fit.data import IMAGE, rgb_data

from .blue_marble_refinement import PixelTrial
from .blue_marble_refinement_report import TEST_SEEDS, summarize_grain
from .blue_marble_report import VALIDATION_SEEDS, flat_pairs, full_fit, render_fit, rgb_sse, summarize, train_all
from .fixed_capacity import RESULTS


REFERENCE = PixelTrial(width=96, depth=2, connection="affine", rule="signsgd", gate="gaussian",
                       competition="layer", readout_layers="all", layer_gates=(.16, .10),
                       rate=.002, epochs=80)


@dataclass(frozen=True)
class UnclippedPixelTrial(PixelTrial):
    clip_output: bool = False

    def model_config(self):
        return replace(super().model_config(), clip_output=self.clip_output)

    def checkpoint_score(self, metrics):
        return metrics["rmse"]


def trials():
    base = replace(REFERENCE, width=48, layer_gates=(.25, .15))
    yield replace(base, epochs=55)
    for width in (32, 48, 64, 96):
        for gates, rate, batch in (((.25, .15), .002, 128),
                                   ((.25, .15), .003, 64),
                                   ((.16, .10), .002, 128)):
            yield replace(base, width=width, layer_gates=gates, rate=rate, batch=batch)
    for width in (24, 48, 64):
        for gates in ((.25, .18, .12), (.30, .25, .20)):
            yield replace(base, width=width, depth=3, layer_gates=gates)
    for width in (64, 128):
        yield replace(base, width=width, depth=1, layer_gates=None, tau=.065,
                      readout_layers="last")
    for width in (32, 64):
        yield replace(base, width=width, competition="class")
        yield replace(base, width=width, readout_layers="last")


def tapered_gates(depth, first=.24):
    return tuple(round(float(value), 3) for value in np.linspace(first, .08, depth))


def depth_trials():
    # These widths keep the parameter count close to the two-layer reference.
    for depth, width in ((3, 68), (4, 55), (5, 48), (6, 43)):
        gates = (.16,) + (.10,) * (depth - 1)
        yield replace(REFERENCE, width=width, depth=depth, layer_gates=gates)
        yield replace(REFERENCE, width=width, depth=depth, layer_gates=tapered_gates(depth))
    for depth, width in ((3, 96), (4, 64), (5, 64), (6, 48)):
        yield replace(REFERENCE, width=width, depth=depth, layer_gates=tapered_gates(depth))
    for depth, width in ((4, 48), (5, 48), (6, 40)):
        yield replace(REFERENCE, width=width, depth=depth, layer_gates=tapered_gates(depth),
                      readout_layers="last")
    for depth, width in ((2, 96), (4, 55), (6, 43)):
        yield replace(REFERENCE, width=width, depth=depth, layer_gates=tapered_gates(depth, first=.16),
                      epochs=140)
    yield from initialization_trials()
    yield from extended_depth_trials()


def initialization_trials():
    yield replace(REFERENCE, epochs=140)
    for depth, width in ((2, 96), (3, 68), (4, 55), (6, 43)):
        gates = (.16,) + (.10,) * (depth - 1)
        yield replace(REFERENCE, width=width, depth=depth, layer_gates=gates,
                      initialization="unit-sphere", epochs=100)


def extended_depth_trials():
    for depth, width in ((8, 36), (10, 32), (8, 48)):
        yield replace(REFERENCE, width=width, depth=depth, layer_gates=tapered_gates(depth))


def step_trials(reference):
    for rate in (.00075, .00125, .003, .004, .006, .008):
        yield replace(reference, rate=rate)
    for rate in (.00125, reference.rate, .004):
        yield replace(reference, rate=rate, epochs=140)


def validation_sse(row, data):
    if row["validation"] is None:
        return np.inf
    # The cached RMSE uses RGB in [-1, 1]. Recover the sum for RGB in [0, 1].
    return (row["validation"]["rmse"] / 2) ** 2 * data.y[data.validation].size


def convergence_trace(job):
    trial, seed = job
    cache = trial.path(seed) / "convergence.json"
    if cache.exists():
        return json.loads(cache.read_text())
    data = trial.dataset()
    model = trial.model(seed)
    args = SimpleNamespace(seed=seed, rate=trial.rate, batch=trial.batch, epochs=trial.epochs)

    def snapshot(epoch):
        return {"epoch": epoch, "train_sse": rgb_sse(model, data, data.train),
                "validation_sse": rgb_sse(model, data, data.validation)}

    history = [snapshot(0)]
    for epoch in training_epochs(model, data, args, data.train, trial.epochs):
        if epoch % 5 == 0 or epoch == trial.epochs:
            history.append(snapshot(epoch))
    trace = {"seed": seed, "history": history}
    with TemporaryDirectory(dir=cache.parent) as directory:
        temporary = Path(directory) / cache.name
        temporary.write_text(json.dumps(trace, allow_nan=False) + "\n")
        temporary.replace(cache)
    return trace


def render_convergence(results, traces, filename, xlabel="Epoch", labels=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(9, 3.5), layout="constrained")
    for ax, metric, title in zip(axes, ("train_sse", "validation_sse"), ("Training pixels", "Validation pixels")):
        for (name, runs), color in zip(traces.items(), ("#2963a3", "#64748b")):
            epochs = [point["epoch"] for point in runs[0]["history"]]
            values = np.array([[point[metric] for point in run["history"]] for run in runs])
            trial = results[name]["trial"]
            if labels:
                label = labels[name]
            elif "clip_output" in results["selected"]["trial"]:
                label = "Clipping on" if trial.get("clip_output", True) else "Clipping off"
            elif "layer_rate_decay" in results["selected"]["trial"]:
                label = (f"Earlier steps × {trial['layer_rate_decay']:g} per layer"
                         if "layer_rate_decay" in trial else "Equal step sizes")
            else:
                label = f"{trial['depth']} layers · step {trial['rate']:g} · {trial['epochs']} epochs"
            ax.fill_between(epochs, values.min(axis=0), values.max(axis=0), color=color, alpha=.13)
            ax.plot(epochs, np.median(values, axis=0), color=color, marker=".", markersize=3, label=label)
        ax.set(title=title, xlabel=xlabel, ylabel="RGB sum of squared errors (log scale)", yscale="log")
        ax.grid(axis="y", alpha=.2)
        ax.legend(frameon=False)
    fig.suptitle("Blue Marble convergence · median and range over seeds 1–3")
    fig.savefig(RESULTS / filename, dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--study", choices=("affine", "depth", "step-size", "unclipped"), default="affine")
    args = parser.parse_args()
    data = rgb_data()
    comparison = args.study != "affine"
    reference = REFERENCE
    previous_trace = []
    if args.study in ("step-size", "unclipped"):
        prior_study = "depth" if args.study == "step-size" else "step-size"
        prior_report = RESULTS / f"blue-marble-{prior_study}.json"
        if not prior_report.exists():
            parser.error(f"Run --study {prior_study} before --study {args.study}")
        prior = json.loads(prior_report.read_text())
        settings = dict(prior["results"]["selected"]["trial"])
        settings["layer_gates"] = tuple(settings["layer_gates"])
        reference = PixelTrial(**settings)
        previous_trace = prior["convergence"]["selected"]
        candidates = list(step_trials(reference)) if args.study == "step-size" else [UnclippedPixelTrial(**settings)]
    elif args.study == "depth":
        candidates = list(depth_trials())
    else:
        candidates = list(trials())
    screen_seed = VALIDATION_SEEDS[0]

    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        screened_trials = [*candidates, reference] if comparison else candidates
        screen = train_all(executor, screened_trials, (screen_seed,))
        finalists = sorted(candidates, key=lambda t: validation_sse(screen[t, screen_seed], data))[:4]
        for trial in finalists:
            print("finalist", validation_sse(screen[trial, screen_seed], data), trial, flush=True)
        confirmation = list(dict.fromkeys([reference, *finalists])) if comparison else finalists
        validation = train_all(executor, confirmation, VALIDATION_SEEDS)
        selected = min(finalists, key=lambda t: np.median(
            [validation_sse(validation[t, seed], data) for seed in VALIDATION_SEEDS]))
        print("selected", selected, flush=True)
        picks = {"selected": selected}
        if comparison:
            picks["reference"] = reference
        trace_jobs = {name: [executor.submit(convergence_trace, (trial, seed)) for seed in VALIDATION_SEEDS]
                      for name, trial in picks.items() if comparison and (name == "selected" or not previous_trace)}
        trained = train_all(executor, picks.values(), TEST_SEEDS)
        list(executor.map(full_fit, [(trial, seed, trained[trial, seed]["epoch"])
                                    for trial in picks.values() for seed in TEST_SEEDS]))
        traces = {name: [job.result() for job in jobs] for name, jobs in trace_jobs.items()}
        if previous_trace:
            traces["reference"] = previous_trace

    results = {}
    for label, trial in picks.items():
        rows = [trained[trial, seed] for seed in TEST_SEEDS]
        if comparison:
            results[label] = summarize(trial, rows, squared_error_only=True)
        else:
            results[label] = summarize_grain(trial, rows, data)
    report = {"image_sha256": hashlib.sha256(IMAGE.read_bytes()).hexdigest(),
              "screen": list(screen.values()), "validation": list(validation.values()),
              "results": results}
    if comparison:
        scored = report["screen"] + report["validation"]
        scored.extend(run for result in results.values() for run in result["runs"])
        for row in scored:
            if row["validation"]:
                row["validation"] = {"sse": validation_sse(row, data)}
        report["convergence"] = traces
    else:
        report["grain_test_pair_count"] = len(flat_pairs(data, data.test)[0])
    name = f"blue-marble-{args.study}"
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / f"{name}.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    label = "affine Hebbian fit without clipping" if args.study == "unclipped" else "affine Hebbian fit"
    render_fit(selected, results["selected"], data, filename=f"{name}.png", label=label)
    if comparison:
        render_convergence(results, traces, f"{name}-convergence.png")
    for label, result in results.items():
        print(label, result["parameters"], result["medians"], flush=True)


if __name__ == "__main__":
    main()
