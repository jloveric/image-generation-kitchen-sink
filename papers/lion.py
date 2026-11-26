"""Train and compare both Lion updates on the manuscript examples."""

import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, replace
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from experiments import sine_jump
from experiments.blue_marble_quality import experiments as marble_experiments, fit_all_pixels
from experiments.celeba import figure, load_checkpoint, scale
from experiments.celeba_decoder_connections import contact_sheet
from experiments.celeba_diversity import model_for
from experiments.scalar_quality import fit
from hebbian_fit.autoencoder import Generator, rgb_bytes
from hebbian_fit.data import rgb_data
from hebbian_fit.model import NumpyModel

RESULTS = Path("assets/results")
MARBLE = "output-depth-12-cosine"
MARBLE_SEED = 5
SINE = "sine-jump-fit-cosine"
COORDINATE_RATE = .00075
MARBLE_WINNER_COUNTS = (8, 18, 36, 54)
PORTRAITS = "celeba-256"
PORTRAIT_RUNS = Path("runs/celeba-sequential/full")
PORTRAIT_BASELINE = PORTRAIT_RUNS / "code256-balanced-lite-long/full"
PORTRAIT_EXPLICIT = PORTRAIT_RUNS / "code256-balanced-lite-long-lion-explicit/full"


def report(name):
    return json.loads((RESULTS / f"{name}.json").read_text())


RULE_LABELS = {"lion": "Semi-implicit Lion", "lion-explicit": "Explicit Lion"}


def cosine_trial(trial, rule, batch, epochs):
    config = replace(trial.config, hidden_rule=rule, layer_rate_decay=1.)
    return replace(trial, label=f"{trial.label}-{rule}", config=config, schedule="cosine",
                   phases=((rule, COORDINATE_RATE, batch, epochs, 1.),))


def comparison_trial(trial, rule):
    return cosine_trial(trial, rule, batch=64, epochs=640)


def sine_trials():
    wide = next(trial for trial in sine_jump.wide_trials()
                if trial.label == "sine-crest-jump-1x512-gauss0.32")
    return (wide, sine_jump.TRIAL, sine_jump.deeper(12), sine_jump.deeper(24))


def measure_sine(trial, seed):
    data = sine_jump.dataset()
    training = fit(trial, data, seed)
    result = sine_jump.evaluate(sine_jump.load_model(seed, trial), data)
    print(f"{trial.label}, run {seed}, test RMSE {result['test_rmse']:.6f}", flush=True)
    return dict(seed=seed, **result, phase_epochs=training["phase_epochs"]), training["history"]


def run_sine(workers, rule, depth=None):
    trials = [trial for trial in sine_trials() if depth is None or trial.config.depth == depth]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        jobs = {}
        for original in trials:
            trial = comparison_trial(original, rule)
            jobs[original.label] = [pool.submit(measure_sine, trial, seed)
                                    for seed in sine_jump.TEST_SEEDS]
        comparisons = {}
        for original in trials:
            rows, histories = zip(*(future.result() for future in jobs[original.label]))
            medians = sine_jump.summarize(rows)
            comparisons[original.label] = dict(
                settings=asdict(comparison_trial(original, rule)),
                runs=[dict(row, history=history) for row, history in zip(rows, histories)],
                medians=medians,
            )
    suffix = f"all-{rule}" if depth is None else f"{rule.removeprefix('lion-')}-{depth}-layers"
    (RESULTS / f"{SINE}-{suffix}.json").write_text(json.dumps(comparisons, indent=2, allow_nan=False) + "\n")


def render_sine():
    trials = sine_trials()
    baseline = report(f"{SINE}-all-lion")
    fig, axes = plt.subplots(len(trials), 2, figsize=(8.8, 8.2), layout="constrained")
    for original, row in zip(trials, axes):
        # Use the same initialization on both curves in each row.
        runs = sorted(baseline[original.label]["runs"], key=lambda r: r["test_rmse"])
        example = runs[len(runs) // 2]
        seed = example["seed"]
        models = [sine_jump.load_model(seed, comparison_trial(original, rule)) for rule in RULE_LABELS]
        row[0].set_ylabel(f"{original.config.depth} × {original.config.units}")
        for ax, (left, right) in zip(row, ((-1, 1), (.15, .35))):
            jump = sine_jump.JUMP_X
            for bounds, height in (((left, jump), 0), ((jump, right), sine_jump.JUMP_HEIGHT)):
                x = np.linspace(*bounds, 2048)
                ax.plot(x, sine_jump.sine(x) + height, color="#424952", ls=":",
                        label="Target" if height == 0 else None)
            x = np.linspace(left, right, 4097, dtype=np.float32)
            for model, label, color in zip(models, RULE_LABELS.values(),
                                           ("#00767b", "#a34d32")):
                ax.plot(x, sine_jump.curve(model, x), label=label, color=color, lw=1.1)
            ax.set_xlim(left, right)
            ax.grid(axis="y", alpha=.25)
            ax.spines[["top", "right"]].set_visible(False)
    axes[0, 0].set_title("Full interval")
    axes[0, 1].set_title("Near the jump")
    for ax in axes[-1]:
        ax.set_xlabel("Input x")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, ncol=3, loc="outside lower center", frameon=False)
    fig.savefig(RESULTS / f"{SINE}-lion-comparison.png", dpi=160)
    plt.close(fig)


def marble_trial(rule, backend, top_k=None, batch=128):
    base = next(trial for trial in marble_experiments(backend) if trial.label == "output-72")
    widths = tuple(round(float(tau), 3) for tau in np.linspace(.24, .08, 12))
    config = replace(base.config, depth=12, layer_gate_widths=widths)
    label = MARBLE
    if top_k is not None:
        label = f"{MARBLE}-top-k" if top_k == 36 else f"{MARBLE}-top-k-{top_k}"
        config = replace(config, competition="class", gate="hard", top_k=top_k, layer_gate_widths=None)
    if batch != 128:
        label = f"{label}-batch-{batch}"
    trial = replace(base, label=label, config=config)
    return cosine_trial(trial, rule, batch=batch, epochs=560)


def run_marble(rule, backend, top_k, batch):
    trial = marble_trial(rule, backend, top_k, batch)
    fit_all_pixels(trial, rgb_data(split=False), MARBLE_SEED)
    if rule == "lion-agreement":
        compare_marble_agreement()
    elif rule == "lion" and batch != 128:
        compare_marble_batch(batch)


def compare_marble_batch(batch):
    trials = [marble_trial("lion", "numpy", batch=size) for size in (128, batch)]
    records = [report(f"blue-marble-{trial.label}-all-pixels") for trial in trials]
    labels = ("Batch size 128", f"Batch size {batch}")
    name = f"blue-marble-batch-{batch}"
    render_marble_convergence(records, labels, name=f"{name}-convergence")
    render_marble_fits(rgb_data(split=False), trials, records, labels, name=f"{name}-fits")


def compare_marble_agreement():
    rules = ("lion", "lion-agreement")
    trials = [marble_trial(rule, "numpy") for rule in rules]
    records = [report(f"blue-marble-{trial.label}-all-pixels") for trial in trials]
    labels = ("Semi-implicit Lion", "Lion with agreement")
    render_marble_convergence(records, labels, name="blue-marble-lion-agreement-convergence")
    render_marble_fits(rgb_data(split=False), trials, records, labels, name="blue-marble-lion-agreement-fits")

    render_mean_step_multipliers(records[1], "blue-marble-lion-agreement-timesteps")


def render_mean_step_multipliers(record, name):
    history = record["history"]
    epochs = [point["epoch"] for point in history]
    multipliers = np.array([point["mean_step_multipliers"] for point in history])
    fig, ax = plt.subplots(figsize=(7.1, 3.6), layout="constrained")
    colors = plt.colormaps["viridis"].resampled(multipliers.shape[1])
    for layer, means in enumerate(multipliers.T):
        ax.plot(epochs, means, color=colors(layer), label=str(layer + 1))
    ax.axhline(1., color="#64748b", linestyle=":", linewidth=1)
    ax.set(xlabel="Epoch", ylabel="Mean timestep multiplier")
    ax.legend(title="Layer", ncols=2, frameon=False, loc="center left", bbox_to_anchor=(1, .5))
    fig.savefig(RESULTS / f"{name}.png", dpi=160)
    plt.close(fig)


def compare_marble():
    report_names = [f"blue-marble-{MARBLE}-{rule}-all-pixels" for rule in RULE_LABELS]
    records = [report(name) for name in report_names]
    trials = [marble_trial(rule, record["settings"]["backend"])
              for rule, record in zip(RULE_LABELS, records)]
    if trials[0].backend != trials[1].backend:
        raise ValueError("Train both Blue Marble models with the same backend")
    render_marble_convergence(records, RULE_LABELS.values(), name="blue-marble-lion-convergence")
    render_marble_fits(rgb_data(split=False), trials, records, RULE_LABELS.values(),
                       name="blue-marble-lion-fits")


def compare_marble_competition(counts=(36,)):
    rule = "lion"
    settings = (None, *counts)
    names = [marble_trial(rule, "numpy", count).label for count in settings]
    records = [report(f"blue-marble-{name}-all-pixels") for name in names]
    trials = [marble_trial(rule, record["settings"]["backend"], count)
              for record, count in zip(records, settings, strict=True)]
    if len({trial.backend for trial in trials}) != 1:
        raise ValueError("Train all Blue Marble models with the same backend")
    labels = ["Gaussian", *(f"Top-K, K={count}" for count in counts)]
    render_marble_convergence(records, labels, name="blue-marble-competition-convergence")
    render_marble_fits(rgb_data(split=False), trials, records, labels, name="blue-marble-competition-fits")


def render_marble_fits(data, trials, records, labels, name, *, checkpoints=None):
    target = data.y.reshape(*data.shape, 3)
    images = [("NASA target", target)]
    if checkpoints is None:
        checkpoints = [marble_checkpoint(trial, record) for trial, record in zip(trials, records, strict=True)]
    for trial, record, label, checkpoint in zip(trials, records, labels, checkpoints, strict=True):
        model = NumpyModel(trial.config, record["seed"])
        with np.load(checkpoint, allow_pickle=False) as state:
            weights = ({key.removeprefix("best_"): state[key] for key in state.files if key.startswith("best_")}
                       if "metadata" in state else state)
            model.load(weights)
        prediction = np.concatenate([model.predict_values(batch) for batch in np.array_split(data.x, 64)])
        prediction = prediction.reshape(target.shape)
        error = np.square((prediction - target) / 2, dtype=np.float64).sum()
        np.testing.assert_allclose(error, record["squared_error"], rtol=1e-4)
        images.append((f"{label}\nSquared RGB error {record['squared_error']:.2f}", prediction))
    fig, axes = plt.subplots(2, len(images), figsize=(2.5 * len(images) + .7, 5.9), layout="constrained")
    for (title, image), (whole, detail) in zip(images, axes.T):
        rgb = (image + 1) / 2
        whole.imshow(rgb, interpolation="nearest")
        whole.set_title(title)
        detail.imshow(rgb[42:86, 43:87], interpolation="nearest")
    for ax in axes.flat:
        ax.set_axis_off()
    fig.supxlabel("Source photograph AS17-148-22727 · NASA Johnson Space Center", fontsize=9)
    fig.savefig(RESULTS / f"{name}.png", dpi=160)
    plt.close(fig)


def marble_checkpoint(trial, record):
    direct = replace(trial, label=f"{trial.label}-all-pixels")
    directory = direct.path(record["seed"])
    return directory / "checkpoint.npz"


def render_marble_convergence(records, labels, name):
    fig, ax = plt.subplots(figsize=(7.1, 3.6), layout="constrained")
    colors = ("#00767b", "#a34d32", "#7056a1", "#a07612", "#496082", "#5c7437", "#973a73", "#414141")
    for record, label, color in zip(records, labels, colors[:len(records)], strict=True):
        history = record["history"]
        ax.plot([point["epoch"] for point in history], [point["squared_error"] for point in history],
                label=label, color=color)
    ax.set(xlabel="Epoch", ylabel="Summed squared RGB error", yscale="log")
    ax.grid(axis="y", alpha=.25)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False)
    fig.savefig(RESULTS / f"{name}.png", dpi=160)
    plt.close(fig)


def portrait_errors(model, test):
    errors = np.empty(len(test), dtype=np.float64)
    batch = model.images_per_batch
    for start in range(0, len(test), batch):
        part = slice(start, start + batch)
        target = scale(test[part])
        code = model.encode(target.reshape(len(target), -1))
        difference = (model.decode(code) - target) / 2
        errors[part] = np.square(difference).reshape(len(target), -1).sum(axis=1, dtype=np.float64)
    return errors


def render_portraits(run):
    published = report(PORTRAITS)
    baseline = PORTRAIT_BASELINE
    width = published["fixed_weights"]["inference_vector_gate_width"]
    result = json.loads((run / "results.json").read_text())
    original = json.loads((baseline / "results.json").read_text())
    assert original["config"] | {"rule": "lion-explicit"} == result["config"]
    assert original["curve"][0] == result["curve"][0]
    assert original["parameters"] == result["parameters"] == published["parameters"]
    assert original["test"]["images"] == result["test"]["images"]
    test = np.load(run.parent / "test.npy", mmap_mode="r")
    indices = np.linspace(0, len(test) - 1, 12, dtype=int)
    images = test[indices]
    rows = [("Original portraits", "Official test split", images)]
    errors = []
    for path, label, record in ((baseline, "Semi-implicit Lion", original), (run, "Explicit Lion", result)):
        model = model_for(record["config"], path, vector_gate_width=width)
        load_checkpoint(path / "selected.npz", model)
        code = model.encode(scale(images).reshape(len(images), -1))
        errors.append(portrait_errors(model, test))
        rows.append((label, f"Test RGB error {errors[-1].mean():.2f}", rgb_bytes(model.decode(code))))
        del model
    np.testing.assert_allclose(errors[0].mean(), published["test_error"], rtol=1e-5)
    output = f"{PORTRAITS}-lion-explicit"
    contact_sheet(RESULTS / f"{output}-reconstructions.png", rows)
    generator = Generator(run / "decoder-mixture-selected.npz")
    figure(np.stack([generator.generate(i) for i in range(36)]),
           RESULTS / f"{output}-portraits.png")
    result["reconstruction_indices"] = indices.tolist()
    result["inference_vector_gate_width"] = width
    result["test_sse_per_portrait"] = float(errors[1].mean())
    result["lion_test_sse_per_portrait"] = float(errors[0].mean())
    result["test_portraits_with_lower_error_than_semi_implicit"] = int((errors[1] < errors[0]).sum())
    (RESULTS / f"{output}.json").write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("task", choices=("sine", "marble", "sine-figures", "marble-figures",
                                         "marble-competition", "portraits"))
    parser.add_argument("--workers", type=int, default=1, help="Parallel sine runs")
    parser.add_argument("--depth", type=int, choices=(1, 4, 12, 24), help="Run only this sine network depth")
    parser.add_argument("--run", type=Path, help="Full CelebA run and mixture export")
    parser.add_argument("--rule", choices=(*RULE_LABELS, "lion-agreement", "lion-adaptive-agreement"), default="lion")
    parser.add_argument("--backend", choices=("numpy", "mojo"), default="numpy", help="Blue Marble backend")
    parser.add_argument("--batch", type=int, help="Blue Marble minibatch size (default 128)")
    parser.add_argument("--top-k", nargs="?", const=36, type=int, metavar="K",
                        help="Keep K activities per class by magnitude for Blue Marble (36 if omitted)")
    parser.add_argument("--sweep", action="store_true", help="Include all four winner counts in the comparison")
    args = parser.parse_args()
    if args.batch is not None and (args.task != "marble" or args.batch < 1):
        parser.error("--batch requires the marble task and a positive count")
    if args.depth is not None and args.task != "sine":
        parser.error("--depth is for the sine task")
    if args.rule == "lion-adaptive-agreement" and args.task != "sine":
        parser.error("Use experiments.blue_marble_shared_step for adaptive-agreement Blue Marble runs")
    batch = args.batch or 128
    if args.rule == "lion-agreement" and (args.task != "marble" or args.backend != "numpy" or args.top_k is not None):
        parser.error("The agreement experiment uses Blue Marble with the NumPy backend and Gaussian competition")
    if batch != 128 and (args.rule != "lion" or args.backend != "numpy" or args.top_k is not None):
        parser.error("The batch comparison uses semi-implicit Lion with the NumPy backend and Gaussian competition")
    if args.top_k is not None and args.task != "marble":
        parser.error("--top-k is for the marble task")
    if args.top_k is not None and not 1 <= args.top_k <= 72:
        parser.error("Top-K must be between 1 and 72, the class width")
    if args.sweep and args.task != "marble-competition":
        parser.error("--sweep is for the marble-competition task")
    if args.task == "sine":
        run_sine(args.workers, args.rule, args.depth)
    elif args.task == "marble":
        run_marble(args.rule, args.backend, args.top_k, batch)
    elif args.task == "sine-figures":
        render_sine()
    elif args.task == "marble-figures":
        compare_marble()
    elif args.task == "marble-competition":
        compare_marble_competition(MARBLE_WINNER_COUNTS if args.sweep else (36,))
    else:
        render_portraits(args.run or PORTRAIT_EXPLICIT)


if __name__ == "__main__":
    main()
