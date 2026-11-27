"""Train and measure the signed-activity regressors used in the paper."""

import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, replace
import json

import numpy as np

from experiments import blue_marble_shared_step, sine_jump
from hebbian_fit.data import rgb_data
from papers import lion


RULE = "lion-adaptive-agreement"


def trial_for(task, rate=None, epochs=None, depth=None, width=None, batch=None, backend="numpy"):
    if task == "sine":
        trial = lion.comparison_trial(sine_jump.deeper(depth or 24), RULE)
    else:
        trial = blue_marble_shared_step.trial_for(RULE, .004)
    rule, starting_rate, size, budget, decay = trial.phases[0]
    config = replace(trial.config, readout="signed-activity", width=width or 2 * trial.config.width)
    if depth is not None and depth != config.depth:
        widths = tuple(round(float(w), 3) for w in np.linspace(
            config.gate_widths[0], config.gate_widths[-1], depth
        ))
        config = replace(config, depth=depth, layer_gate_widths=widths)
    phase = (rule, rate or starting_rate, batch or size, epochs or budget, decay)
    name = f"{task}-signed-activity-{config.depth}x{config.units}-{phase[1]:g}-b{phase[2]}-e{phase[3]}"
    if backend != "numpy":
        name += f"-{backend}"
    return replace(trial, label=name, config=config, phases=(phase,), backend=backend)


def sine(trial):
    with ProcessPoolExecutor(max_workers=len(sine_jump.TEST_SEEDS)) as pool:
        jobs = [pool.submit(lion.measure_sine, trial, seed) for seed in sine_jump.TEST_SEEDS]
        rows, histories = zip(*(job.result() for job in jobs))
    medians = sine_jump.summarize(rows)
    row = sorted(rows, key=lambda entry: entry["test_rmse"])[len(rows) // 2]
    model = sine_jump.load_model(row["seed"], trial)
    sine_jump.render(model, row, [], label="Signed activity", filename=trial.label)
    return dict(settings=asdict(trial), medians=medians, runs=[
        dict(row, history=history) for row, history in zip(rows, histories, strict=True)
    ])


def marble(trial, until):
    result = blue_marble_shared_step.run(trial, until)
    checkpoint = trial.path(lion.MARBLE_SEED) / (
        "checkpoint.npz" if until == trial.phases[0][3] else "phase-1-progress.npz"
    )
    labels = ("Signed activity",)
    lion.render_marble_fits(rgb_data(split=False), (trial,), (result,), labels,
                            f"{trial.label}-fits", checkpoints=(checkpoint,))
    lion.render_marble_convergence((result,), labels, trial.label)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("task", choices=("sine", "marble"))
    parser.add_argument("--rate", type=float)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--depth", type=int)
    parser.add_argument("--width-per-output", type=int)
    parser.add_argument("--batch", type=int)
    parser.add_argument("--backend", choices=("numpy", "mojo"), default="numpy")
    parser.add_argument("--until", type=int, help="Marble epoch to pause on the scheduled cosine")
    args = parser.parse_args()
    if (args.rate is not None and not 0 < args.rate < np.inf
            or any(value is not None and value < 1 for value in
                   (args.epochs, args.depth, args.width_per_output, args.batch))):
        parser.error("Rate, epoch budget, depth, width and batch size must be positive")
    if args.until is not None and args.task != "marble":
        parser.error("--until applies to Blue Marble")
    trial = trial_for(args.task, args.rate, args.epochs, args.depth, args.width_per_output,
                      args.batch, args.backend)
    if args.task == "sine":
        result = sine(trial)
        print(f"{trial.label}: median test RMSE {result['medians']['test_rmse']:.8f}", flush=True)
    else:
        result = marble(trial, args.until or trial.phases[0][3])
        print(f"{trial.label}: summed squared RGB error {result['squared_error']:.6f}", flush=True)
    (lion.RESULTS / f"{trial.label}.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
