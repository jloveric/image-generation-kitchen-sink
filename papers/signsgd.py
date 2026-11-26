"""Compare memoryless signed coactivity against the paper's Lion runs."""

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
from experiments.blue_marble_quality import Trial, fit_all_pixels
from experiments.celeba import figure, load_checkpoint, scale
from experiments.celeba_decoder_connections import contact_sheet
from experiments.celeba_diversity import model_for
from experiments.scalar_quality import fit
from hebbian_fit.autoencoder import Generator, rgb_bytes
from hebbian_fit.data import rgb_data
from hebbian_fit.model import ModelConfig

RESULTS = Path("assets/results")
MARBLE = "blue-marble-output-depth-12-two-phase-all-pixels"
SINE = "sine-jump-fit"
PORTRAITS = "celeba-target150"
PORTRAIT_RUNS = Path("runs/celeba-sequential/full")


def report(name):
    return json.loads((RESULTS / f"{name}.json").read_text())


def memoryless(trial):
    return replace(trial, label=f"{trial.label}-signsgd",
                   config=replace(trial.config, hidden_rule="signsgd"),
                   phases=tuple(("signsgd", *phase[1:]) for phase in trial.phases))


def sine_trials():
    wide = next(trial for trial in sine_jump.wide_trials()
                if trial.label == "sine-crest-jump-1x512-gauss0.32")
    return (sine_jump.refine_wide(wide), sine_jump.TRIAL, sine_jump.deeper(12),
            sine_jump.scale_steps(sine_jump.deeper(24), .64))


def measure_sine(trial, seed):
    data = sine_jump.dataset()
    training = fit(trial, data, seed)
    result = sine_jump.evaluate(sine_jump.load_model(seed, trial), data)
    print(f"{trial.label}, run {seed}, test RMSE {result['test_rmse']:.6f}", flush=True)
    return dict(seed=seed, **result, phase_epochs=training["phase_epochs"]), training["history"]


def run_sine(workers):
    baseline = report(SINE)["held_out"]
    trials = sine_trials()
    with ProcessPoolExecutor(max_workers=workers) as pool:
        jobs = {}
        for original in trials:
            trial = memoryless(original)
            jobs[original.label] = [pool.submit(measure_sine, trial, seed)
                                    for seed in sine_jump.TEST_SEEDS]
        comparisons = {}
        for original in trials:
            rows, histories = zip(*(future.result() for future in jobs[original.label]))
            medians = sine_jump.summarize(rows)
            comparisons[original.label] = dict(
                settings=asdict(memoryless(original)),
                runs=[dict(row, history=history) for row, history in zip(rows, histories)],
                medians=medians,
                lion_phases_medians=baseline[original.label]["medians"],
            )
    (RESULTS / f"{SINE}-signsgd.json").write_text(
        json.dumps(comparisons, indent=2, allow_nan=False) + "\n")
    render_sine(trials, baseline)


def render_sine(trials, baseline):
    data = sine_jump.dataset()
    fig, axes = plt.subplots(len(trials), 2, figsize=(8.8, 8.2), layout="constrained")
    for original, row in zip(trials, axes):
        # Use the same initialization on both curves in each row.
        runs = sorted(baseline[original.label]["runs"], key=lambda r: r["test_rmse"])
        example = runs[len(runs) // 2]
        seed = example["seed"]
        model = published_sine_model(original, data, example)
        models = (model, sine_jump.load_model(seed, memoryless(original)))
        row[0].set_ylabel(f"{original.config.depth} × {original.config.units}")
        for ax, (left, right) in zip(row, ((-1, 1), (.15, .35))):
            jump = sine_jump.JUMP_X
            for bounds, height in (((left, jump), 0), ((jump, right), sine_jump.JUMP_HEIGHT)):
                x = np.linspace(*bounds, 2048)
                ax.plot(x, sine_jump.sine(x) + height, color="#424952", ls=":",
                        label="Target" if height == 0 else None)
            x = np.linspace(left, right, 4097, dtype=np.float32)
            for model, label, color in zip(models, ("Lion phases", "SignSGD throughout"),
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
    fig.savefig(RESULTS / f"{SINE}-signsgd.png", dpi=160)
    plt.close(fig)


def published_sine_model(trial, data, example):
    seed = example["seed"]
    path = trial.path(seed)
    # Older runs have a different directory hash because ModelConfig gained fields.
    for checkpoint in path.parent.parent.glob(f"{trial.label}-numpy-*/{seed}/checkpoint.npz"):
        model = trial.model(seed)
        with np.load(checkpoint, allow_pickle=False) as state:
            model.load(state)
        if np.isclose(sine_jump.evaluate(model, data)["test_rmse"], example["test_rmse"], atol=1e-6):
            return model
    fit(trial, data, seed)
    model = sine_jump.load_model(seed, trial)
    np.testing.assert_allclose(sine_jump.evaluate(model, data)["test_rmse"], example["test_rmse"], atol=1e-6)
    return model


def run_marble():
    published = report(MARBLE)
    settings = published["settings"]
    config = ModelConfig(**(settings["config"] | {
        "layer_gate_widths": tuple(settings["config"]["layer_gate_widths"]),
    }))
    trial = Trial(settings["label"], config, tuple(map(tuple, settings["phases"])),
                  settings["backend"], settings["schedule"])
    comparison = memoryless(trial)
    fit_all_pixels(comparison, rgb_data(split=False), published["seed"])
    render_marble_convergence(published, report(f"blue-marble-{comparison.label}-all-pixels"))


def render_marble_convergence(published, signsgd):
    fig, ax = plt.subplots(figsize=(7.1, 3.6), layout="constrained")
    for record, label, color in ((published, "Lion in phase two", "#00767b"),
                                 (signsgd, "SignSGD throughout", "#a34d32")):
        history = record["history"]
        ax.plot([point["epoch"] for point in history], [point["squared_error"] for point in history],
                label=label, color=color)
    phase_one = published["settings"]["phases"][0][3]
    ax.axvline(phase_one, color="#64748b", lw=1, ls=":", label="End of shared first phase")
    ax.set(xlabel="Scheduled epoch", ylabel="Summed squared RGB error", yscale="log")
    ax.grid(axis="y", alpha=.25)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False)
    fig.savefig(RESULTS / f"{MARBLE}-signsgd-comparison.png", dpi=160)
    plt.close(fig)


def render_portraits(run):
    published = report(PORTRAITS)
    baseline = Path(published["export"]).parent
    result = json.loads((run / "results.json").read_text())
    original = json.loads((baseline / "results.json").read_text())
    config_defaults = dict(zero_init_bias=False, encoder_top_k=[0, 0], vector_top_k=0,
                           encoder_hidden_widths=[])
    assert config_defaults | original["config"] | {"rule": "signsgd"} == config_defaults | result["config"]
    assert original["curve"][0] == result["curve"][0]
    assert original["parameters"] == result["parameters"]
    assert original["test"]["images"] == result["test"]["images"]
    test = np.load(run.parent / "test.npy", mmap_mode="r")
    indices = published["reconstruction_indices"]
    images = test[indices]
    rows = [("Original portraits", "Official test split", images)]
    for path, label, record in ((baseline, "Lion", original), (run, "SignSGD", result)):
        model = model_for(record["config"], path)
        load_checkpoint(path / "selected.npz", model)
        code = model.encode(scale(images).reshape(len(images), -1))
        error = record["test"]["reconstruction_sse"] / len(test)
        rows.append((label, f"Test RGB error {error:.2f}", rgb_bytes(model.decode(code))))
        del model
    contact_sheet(RESULTS / f"{PORTRAITS}-signsgd-reconstructions.png", rows)
    generator = Generator(run / "decoder-mixture.npz")
    figure(np.stack([generator.generate(i) for i in range(36)]),
           RESULTS / f"{PORTRAITS}-signsgd-portraits.png")
    result["reconstruction_indices"] = indices
    result["test_sse_per_portrait"] = result["test"]["reconstruction_sse"] / len(test)
    result["lion_test_sse_per_portrait"] = published["test_sse"]
    (RESULTS / f"{PORTRAITS}-signsgd.json").write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("task", choices=("sine", "marble", "portraits"))
    parser.add_argument("--workers", type=int, default=1, help="Parallel sine runs")
    parser.add_argument("--run", type=Path,
                        default=PORTRAIT_RUNS / "target150-standard-full128-broad-signsgd/full",
                        help="Full CelebA run and mixture export for the portrait figures")
    args = parser.parse_args()
    if args.task == "sine":
        run_sine(args.workers)
    elif args.task == "marble":
        run_marble()
    else:
        render_portraits(args.run)


if __name__ == "__main__":
    main()
