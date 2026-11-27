"""Compare generic dense Hebbian regressors on Blue Marble."""

import argparse
import hashlib
import json
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace

import numpy as np

from hebbian_fit.__main__ import training_epochs
from hebbian_fit.data import IMAGE, rgb_data
from hebbian_fit.gpu import MojoModel
from hebbian_fit.model import ADAPTIVE_LION_RULES, ModelConfig, NumpyModel

from .blue_marble_affine import TEST_SEEDS, VALIDATION_SEEDS, render_convergence
from .blue_marble_continuation import detail_pixels
from .blue_marble_report import render_fit, rgb_sse
from .fixed_capacity import RESULTS, predict


@dataclass(frozen=True)
class Trial:
    label: str
    config: ModelConfig
    phases: tuple[tuple[str, float, int, int, float], ...]
    backend: str
    schedule: str = field(default="hold-cosine", repr=False)

    def path(self, seed):
        signature = repr(self) if self.schedule == "hold-cosine" else repr((self, self.schedule))
        if self.config.normalize_hidden_inputs:
            signature += f"-interlayer-{self.config.normalize_hidden_inputs}"
        if self.config.skip_input_to_hidden:
            signature += "-skip-input"
        if self.config.hidden_rule == "lion-response":
            signature += repr(self.config.response_limit_ratio)
        fingerprint = hashlib.sha256(signature.encode()).hexdigest()[:12]
        return Path("runs/blue-marble-quality") / f"{self.label}-{self.backend}-{fingerprint}" / str(seed)

    def model(self, seed):
        if self.backend == "mojo":
            return MojoModel(self.config, seed, capacity=max(256, *(p[2] for p in self.phases)))
        return NumpyModel(self.config, seed)


def experiments(backend):
    base = ModelConfig(2, 3, 48, depth=8, competition="layer", gate="gaussian",
                       readout="class-sum", readout_layers="all",
                       layer_gate_widths=(.24, .217, .194, .171, .149, .126, .103, .08),
                       output_bound=1., output_levels=256)
    two_phase = (("lion", .004, 128, 80, 1.), ("lion", .0015, 64, 100, .7))
    uniform = (("lion", .004, 128, 80, 1.), ("lion", .0006, 64, 100, 1.))
    for group in ("layer", "class", "output"):
        config = replace(base, competition=group)
        yield Trial(f"{group}-matched", config, two_phase, backend)
        yield Trial(f"{group}-uniform", config, uniform, backend)
        for width in ((72,) if group == "output" else (72, 96)):
            yield Trial(f"{group}-{width}", replace(config, width=width), two_phase, backend)
    for depth in (4, 12):
        widths = tuple(float(w) for w in np.linspace(.24, .08, depth))
        yield Trial(f"layer-depth-{depth}", replace(base, depth=depth, layer_gate_widths=widths),
                    two_phase, backend)


def save_progress(path, model, state, metadata):
    arrays = {f"current_{key}": value for key, value in model.state().items()}
    arrays.update({f"best_{key}": value for key, value in state.items()})
    with TemporaryDirectory(dir=path.parent) as tmp:
        snapshot = Path(tmp) / path.name
        np.savez(snapshot, metadata=json.dumps(metadata, allow_nan=False), **arrays)
        snapshot.replace(path)


def train(trial, seed, data, full=False, parent=None, checkpoint_pixels=None, squared_error=rgb_sse,
          *, until_epoch=None):
    if until_epoch is not None and (
            full or parent is not None or len(trial.phases) != 1
            or not 0 < until_epoch < trial.phases[0][3] or until_epoch % 5):
        raise ValueError("A training pause must be at a checkpoint epoch within a single phase")
    directory = trial.path(seed)
    directory.mkdir(parents=True, exist_ok=True)
    checkpoint = directory / ("full-fit.npz" if full else "checkpoint.npz")
    summary = directory / "validation.json"
    if checkpoint.exists() and summary.exists():
        row = json.loads(summary.read_text())
        # Older cached histories used the selected first-phase epoch as the offset.
        for index, point in enumerate(row["history"], start=1):
            point["epoch"] = 5 * index
        return row
    phase_budgets = json.loads(summary.read_text())["phase_epochs"] if full else None
    pixels = np.arange(len(data.x)) if full else data.train
    score_pixels = data.validation if checkpoint_pixels is None else checkpoint_pixels
    history = []
    phase_epochs = []
    phase_start = 0
    state = {}
    if parent is not None:
        prior = train(parent, seed, data, full, checkpoint_pixels=checkpoint_pixels, squared_error=squared_error)
        history = prior["history"].copy()
        phase_epochs = prior["phase_epochs"].copy()
        phase_start = sum(p[3] for p in parent.phases)
        with np.load(parent.path(seed) / checkpoint.name) as saved:
            state = dict(saved)
    remaining = trial.phases[len(phase_epochs):]
    progress_paths = []
    for index, (rule, rate, batch, epochs, decay) in enumerate(remaining, start=len(phase_epochs)):
        phase_config = replace(trial.config, hidden_rule=rule, layer_rate_decay=decay)
        model = replace(trial, config=phase_config).model(seed)
        model.load({key: state.get(key, value) for key, value in model.state().items()})
        budget = phase_budgets[index] if full else epochs
        args = SimpleNamespace(seed=seed + 1000 * index, batch=batch, rate=rate, epochs=epochs,
                               schedule=trial.schedule)
        chosen = 0
        start = 0
        progress = directory / f"phase-{index + 1}-progress.npz"
        if not full:
            best = squared_error(model, data, score_pixels)
            state = model.state()
            progress_paths.append(progress)
            if progress.exists():
                with np.load(progress) as saved:
                    meta = json.loads(saved["metadata"].item())
                    start, best, chosen, history = (meta[k] for k in ("epoch", "best", "chosen", "history"))
                    model.load({key.removeprefix("current_"): saved[key] for key in saved.files
                                if key.startswith("current_")})
                    state = {key.removeprefix("best_"): saved[key] for key in saved.files if key.startswith("best_")}
        stop = budget if until_epoch is None else min(budget, until_epoch)
        for epoch in training_epochs(model, data, args, pixels, stop, start_epoch=start):
            if full or epoch % 5:
                continue
            error = squared_error(model, data, score_pixels)
            if not np.isfinite(error):
                raise ValueError(f"Nonfinite predictions for {trial.label}, seed {seed}, epoch {epoch}")
            train_error = squared_error(model, data, pixels) if checkpoint_pixels is None else error
            point = dict(epoch=phase_start + epoch, validation_sse=error, train_sse=train_error)
            if rule in ADAPTIVE_LION_RULES and isinstance(model, NumpyModel):
                point["mean_step_multipliers"] = model.take_mean_step_multipliers()
            if checkpoint_pixels is not None and epoch % 25 == 0:
                print(f"{trial.label}: epoch {epoch}/{budget}, squared RGB error {error:.2f}", flush=True)
            history.append(point)
            if error < best:
                best, chosen, state = error, epoch, model.state()
            if (epoch % 50 == 0 or epoch == until_epoch) and epoch < budget:
                save_progress(progress, model, state,
                              dict(epoch=epoch, best=best, chosen=chosen, history=history))
        if until_epoch is not None:
            # Keep the current trajectory for resumption and select only for reporting.
            return dict(seed=seed, phase_epochs=[chosen], validation={"sse": best}, history=history)
        if full:
            state = model.state()
        phase_epochs.append(chosen)
        phase_start += epochs
    with TemporaryDirectory(dir=directory) as tmp:
        weights = Path(tmp) / checkpoint.name
        np.savez_compressed(weights, **state)
        weights.replace(checkpoint)
        if not full:
            row = dict(seed=seed, phase_epochs=phase_epochs, validation={"sse": best}, history=history)
            path = Path(tmp) / summary.name
            path.write_text(json.dumps(row, allow_nan=False))
            path.replace(summary)
    for progress in progress_paths:
        progress.unlink(missing_ok=True)
    return json.loads(summary.read_text())


def score(trial, data, seed, parent=None):
    row = train(trial, seed, data, parent=parent)
    train(trial, seed, data, full=True, parent=parent)
    model = trial.model(seed)
    with np.load(trial.path(seed) / "checkpoint.npz") as state:
        model.load(state)
    held_out = rgb_sse(model, data, data.test)
    detail = rgb_sse(model, data, detail_pixels(data))
    with np.load(trial.path(seed) / "full-fit.npz") as state:
        model.load(state)
    pixels = np.arange(len(data.x))
    fitted = predict(model, data, pixels)
    delta = np.rint(np.abs(fitted - data.y) * 127.5)
    row["test"] = dict(test_sse=held_out, detail_test_sse=detail,
                       full_fit_sse=rgb_sse(model, data, pixels),
                       full_fit_exact_pixels=float(np.mean(np.all(delta == 0, axis=1))),
                       full_fit_within_one_code=float(np.mean(np.all(delta <= 1, axis=1))))
    return row


def pixel_cell_centers(grid):
    return (grid[:-1, :-1] + grid[:-1, 1:] + grid[1:, :-1] + grid[1:, 1:]) / 4


def layer_readouts(model, coordinates):
    chunks = []
    full = []
    for start in range(0, len(coordinates), 256):
        prediction, traces = model.forward(coordinates[start:start + 256])
        chunks.append([model.pool(activity) for _, activity, _ in traces])
        full.append(prediction)
    return np.concatenate(chunks, axis=1), np.concatenate(full)


def render_layer_views(config, state, data, subpixel_data, seed, filename):
    import matplotlib.pyplot as plt

    model = NumpyModel(config, seed)
    model.load(state)
    contributions, full = layer_readouts(model, data.x)
    cumulative = [contributions[:k].mean(axis=0) for k in range(1, config.depth + 1)]
    np.testing.assert_array_equal(model.format_output(cumulative[-1].copy()), full)
    descriptions = (
        ("individual-layers", "One layer at a time", contributions, lambda k: f"Layer {k}"),
        ("cumulative-layers", "Mean of the first layers", cumulative, lambda k: f"Layers 1–{k}" if k > 1 else "Layer 1"),
    )
    scores = {}
    columns = 4 if config.depth <= 16 else 5
    rows = (config.depth + columns - 1) // columns
    for suffix, title, predictions, tile_title in descriptions:
        fig, axes = plt.subplots(rows, columns, figsize=(2.35 * columns, 2.55 * rows),
                                 layout="constrained", squeeze=False)
        for ax in axes.flat:
            ax.set_axis_off()
        errors = []
        for index, (ax, values) in enumerate(zip(axes.flat, predictions), start=1):
            prediction = model.format_output(values.copy())
            rgb = (prediction.reshape(*data.shape, 3) + 1) / 2
            error = float(np.square((prediction - data.y) / 2, dtype=np.float64).sum())
            errors.append(error)
            ax.imshow(np.clip(rgb, 0, 1), interpolation="nearest")
            ax.set_title(f"{tile_title(index)}  ·  {error:.1f}", fontsize=9)
        fig.suptitle(f"Blue Marble · {title} · seed {seed}\nLabels show summed squared RGB error")
        fig.savefig(filename.with_name(f"{filename.name}-{suffix}.png"), dpi=160)
        plt.close(fig)
        scores[suffix.replace("-", "_")] = errors
    midpoint_readouts, _ = layer_readouts(model, subpixel_data.x)
    scores["cumulative_bilinear_centers"] = [
        float(np.square((model.format_output(midpoint_readouts[:k].mean(axis=0)) - subpixel_data.y) / 2,
                        dtype=np.float64).sum())
        for k in range(1, config.depth + 1)
    ]
    return scores


def fit_all_pixels(trial, data, seed):
    """Train and select checkpoints on the complete coordinate and RGB array."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    label = trial.label.removesuffix("-fine") + "-all-pixels"
    direct = replace(trial, label=label)
    pixels = data.train
    if len(pixels) != len(data.x) or len(data.validation) or len(data.test):
        raise ValueError("The all-pixel fit requires a dataset without a split")
    parent = None
    if not (direct.path(seed) / "validation.json").exists():
        # Save each completed phase so a long fit can resume from it.
        for phase_count in range(1, len(direct.phases)):
            prefix = replace(direct, phases=direct.phases[:phase_count])
            train(prefix, seed, data, parent=parent, checkpoint_pixels=pixels)
            parent = prefix
    row = train(direct, seed, data, parent=parent, checkpoint_pixels=pixels)
    model = direct.model(seed)
    with np.load(direct.path(seed) / "checkpoint.npz") as checkpoint:
        state = dict(checkpoint)
    model.load(state)
    target = (data.y.reshape(*data.shape, 3) + 1) / 2
    fit = (predict(model, data, pixels).reshape(target.shape) + 1) / 2
    error = row["validation"]["sse"]
    centers = pixel_cell_centers(data.x.reshape(*data.shape, 2)).reshape(-1, 2)
    bilinear = pixel_cell_centers(data.y.reshape(*data.shape, 3)).reshape(-1, 3)
    subpixel_data = replace(data, x=centers, y=bilinear)
    subpixel_error = rgb_sse(model, subpixel_data, np.arange(len(centers)))
    filename = RESULTS / f"blue-marble-{trial.label}-all-pixels"
    layer_errors = render_layer_views(trial.config, state, data, subpixel_data, seed, filename)
    report = dict(settings=asdict(trial), seed=seed, phase_epochs=row["phase_epochs"],
                  squared_error=error, pixel_count=len(pixels),
                  checkpoint_selection="training_squared_rgb_error", validation_pixel_count=0, test_pixel_count=0,
                  subpixel_bilinear_squared_error=subpixel_error, subpixel_count=len(centers),
                  layer_squared_errors=layer_errors,
                  history=[{key: value for key, value in point.items() if key not in ("train_sse", "validation_sse")}
                           | {"squared_error": point["train_sse"]} for point in row["history"]])
    filename.with_name(f"{filename.name}.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    fig, axes = plt.subplots(2, 2, figsize=(6.3, 6.2), layout="constrained")
    for column, (image, title) in enumerate(((target, "NASA target"), (fit, f"All-pixel fit · squared error {error:.1f}"))):
        axes[0, column].imshow(image, interpolation="nearest")
        axes[0, column].set_title(title)
        axes[1, column].imshow(image[42:86, 43:87], interpolation="nearest")
    for ax in axes.flat:
        ax.set_axis_off()
    fig.suptitle(f"Blue Marble · {trial.config.depth} layers × {trial.config.units} neurons · seed {seed}")
    fig.supxlabel("Source photograph AS17-148-22727 · NASA Johnson Space Center", fontsize=9)
    fig.savefig(filename.with_name(f"{filename.name}.png"), dpi=160)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(7.1, 3.), layout="constrained")
    history = report["history"]
    ax.plot([p["epoch"] for p in history], [p["squared_error"] for p in history], color="#2963a3")
    boundary = 0
    for _, _, _, epochs, _ in trial.phases[:-1]:
        boundary += epochs
        ax.axvline(boundary, color="#64748b", linestyle=":", linewidth=1)
    ax.set(title="All-pixel image fit", xlabel="Scheduled epoch", ylabel="Summed squared RGB error", yscale="log")
    ax.grid(axis="y", alpha=.2)
    fig.savefig(filename.with_name(f"{filename.name}-convergence.png"), dpi=160)
    plt.close(fig)
    print(label, row["phase_epochs"], error, "bilinear center error", subpixel_error, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("numpy", "mojo"), default="numpy")
    parser.add_argument("--screen-only", action="store_true")
    parser.add_argument("--candidate", action="append", help="Screen only this candidate. May be repeated.")
    parser.add_argument("--extend", action="store_true", help="Also compare third-phase schedules on the wide per-output model")
    parser.add_argument("--simple-cosine", action="store_true", help="Also compare single-phase cosine decay")
    parser.add_argument("--capacity", action="store_true", help="Also test wider and deeper per-output models")
    parser.add_argument("--fit-all-pixels", metavar="CANDIDATE",
                        help="Select checkpoints by training error on the complete image using seed 5")
    args = parser.parse_args()
    data = rgb_data(split=not args.fit_all_pixels)
    trials = {trial.label: trial for trial in experiments(args.backend)}
    if args.capacity:
        base = trials["output-72"]
        trials["output-96"] = replace(base, label="output-96", config=replace(base.config, width=96))
        for depth in (12, 20):
            label = f"output-depth-{depth}"
            gates = tuple(round(float(tau), 3) for tau in np.linspace(.24, .08, depth))
            deep = replace(base.config, depth=depth, layer_gate_widths=gates)
            trials[label] = replace(base, label=label, config=deep)
    if args.simple_cosine:
        base = trials["output-72"]
        phases = (("lion", .0025, 128, 340, 1.),)
        for label, schedule in (("single-cosine", "cosine"), ("single-hold-cosine", "hold-cosine")):
            trials[label] = replace(base, label=label, phases=phases, schedule=schedule)
    parents = {}
    if args.extend:
        parent = trials["output-72"]
        for label, rate in (("output-72-long", .001), ("output-72-fine", .0002)):
            longer = replace(parent, label=label, phases=parent.phases + (("lion", rate, 64, 160, .75),))
            trials[label] = longer
            parents[label] = parent
        if args.capacity:
            for name in ("output-depth-12", "output-depth-20", "output-96"):
                parent = trials[name]
                label = f"{name}-fine"
                trials[label] = replace(parent, label=label, phases=parent.phases + (("lion", .0002, 64, 160, .75),))
                parents[label] = parent
            base = trials["output-depth-12-fine"]
            rule, rate, batch, _, decay = base.phases[1]
            label = "output-depth-12-longer"
            trials[label] = replace(base, label=label,
                                    phases=(base.phases[0], (rule, rate, batch, 240, decay), base.phases[2]))
            for label, epochs in (("output-depth-12-two-phase", 480), ("output-depth-12-two-phase-800", 800)):
                trials[label] = replace(base, label=label,
                                        phases=(base.phases[0], (rule, rate, batch, epochs, decay)))
    if args.fit_all_pixels:
        if args.fit_all_pixels not in trials:
            parser.error(f"Unknown candidate {args.fit_all_pixels}")
        fit_all_pixels(trials[args.fit_all_pixels], data, seed=5)
        return
    if args.candidate:
        if not args.screen_only:
            parser.error("--candidate requires --screen-only")
        unknown = set(args.candidate) - trials.keys()
        if unknown:
            parser.error(f"Unknown candidates {', '.join(sorted(unknown))}")
        trials = {name: trials[name] for name in args.candidate}
    validation = {}
    for name, trial in trials.items():
        rows = [train(trial, seed, data, parent=parents.get(name)) for seed in VALIDATION_SEEDS]
        median = float(np.median([row["validation"]["sse"] for row in rows]))
        validation[name] = dict(settings=asdict(trial), runs=rows, median=median)
        print(name, median, flush=True)
    winner = min(trials, key=lambda name: validation[name]["median"])
    if args.screen_only:
        return
    best = trials[winner]
    rows = [score(best, data, seed, parent=parents.get(winner)) for seed in TEST_SEEDS]
    medians = {key: float(np.median([row["test"][key] for row in rows])) for key in rows[0]["test"]}
    weights = NumpyModel(best.config, 1).weights
    result = dict(trial=asdict(best), runs=rows, medians=medians, parameters=sum(w.size for w in weights))
    report = dict(image_sha256=hashlib.sha256(IMAGE.read_bytes()).hexdigest(),
                  selected=winner, validation=validation, result=result)
    file = f"blue-marble-quality-{args.backend}" + ("-extended" if args.extend else "")
    if args.simple_cosine:
        file += "-cosine"
    if args.capacity:
        file += "-capacity"
    (RESULTS / f"{file}.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    scopes = {"layer": "whole-layer", "output": "per-output", "class": "per-class"}
    fit_label = f"{best.config.units} neurons per layer · {scopes[best.config.competition]} Gaussian"
    render_fit(best, result, data, f"{file}.png", fit_label)
    control = "layer-matched" if winner != "layer-matched" else "class-matched"
    comparison = {"selected": winner, "reference": control}
    labels = {key: f"{scopes[trials[name].config.competition]} · {trials[name].config.units} neurons"
              for key, name in comparison.items()}
    render_convergence({k: {"trial": asdict(trials[name].config)} for k, name in comparison.items()},
                       {k: validation[name]["runs"] for k, name in comparison.items()},
                       f"{file}-convergence.png", labels=labels)
    print(winner, medians, flush=True)


if __name__ == "__main__":
    main()
