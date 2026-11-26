"""Evaluate a shared cosine with uniform local updates on Blue Marble."""

import argparse
from dataclasses import asdict, replace
import json

from experiments.blue_marble_batch_switch import replay
from experiments.blue_marble_layer_steps import HISTORICAL
from experiments.blue_marble_quality import train
from hebbian_fit.data import rgb_data
from papers.lion import MARBLE_SEED, RESULTS, marble_trial, render_marble_convergence, render_marble_fits, report


BASELINE = "blue-marble-single-cosine-taper"
BUDGET = 560


def trial_for(rule, rate, *, skip_input=False, unit_norm=False, last_output=False,
              batch=128, gate_width=None, shallow_wide=False):
    trial = marble_trial(rule, "numpy", batch=batch)
    config = replace(trial.config, skip_input_to_hidden=skip_input)
    if unit_norm:
        config = replace(config, initialization="unit-affine", unit_norm_updates=True)
    if last_output:
        config = replace(config, readout_layers="last")
    suffix = (("-skip-input" if skip_input else "") + ("-unit-norm" if unit_norm else "")
              + ("-last-output" if last_output else ""))
    if batch != 128:
        suffix += f"-batch-{batch}"
    if shallow_wide:
        config = replace(config, depth=4, width=144, layer_gate_widths=None)
        gate_width = .16 if gate_width is None else gate_width
        suffix += "-4x864"
    if gate_width is not None:
        config = replace(config, gate_width=gate_width, layer_gate_widths=None)
        suffix += f"-gate-{gate_width:g}"
    return replace(trial, label=f"shared-step-{rule}-{rate:g}{suffix}-all-pixels", config=config,
                   phases=((rule, rate, batch, BUDGET * batch // 128, 1.),))


def run(trial, until):
    data = rgb_data(split=False)
    budget = trial.phases[0][3]
    result = train(trial, MARBLE_SEED, data, checkpoint_pixels=data.train,
                   until_epoch=None if until == budget else until)
    history = [{key: value for key, value in point.items() if key not in ("train_sse", "validation_sse")}
               | {"squared_error": point["train_sse"]} for point in result["history"]]
    return dict(settings=asdict(trial), seed=MARBLE_SEED, selected_epoch=result["phase_epochs"][0],
                squared_error=result["validation"]["sse"], history=history)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rule", default="lion-coherence")
    parser.add_argument("--rate", type=float, default=.004)
    parser.add_argument("--until", type=int, default=100)
    parser.add_argument("--batch", type=int, choices=(64, 128, 256, 512), default=128,
                        help="Batch size, with the cosine budget kept fixed in minibatch updates")
    parser.add_argument("--skip-input", action="store_true", help="Give each hidden layer the raw input")
    parser.add_argument("--unit-norm", action="store_true", help="Keep each incoming weight vector at unit norm")
    parser.add_argument("--last-output", action="store_true", help="Use only the final hidden layer for predictions")
    parser.add_argument("--gate-width", type=float, help="Use one Gaussian competition width for every layer")
    parser.add_argument("--shallow-wide", action="store_true", help="Use four layers of 864 neurons")
    parser.add_argument("--fits", action="store_true", help="Render selected predictions alongside the photograph")
    parser.add_argument("--refinement", "--participation-refinement", nargs="?", const="lion-participation",
                        choices=("lion", "lion-participation", "lion-participation-power", "lion-pool-participation"),
                        help="Test a shared step for 20 epochs from the historical checkpoint")
    args = parser.parse_args()
    budget = BUDGET * args.batch // 128
    if not 5 <= args.until <= budget or args.until % 5:
        parser.error(f"--until must be a multiple of 5 between 5 and {budget}")
    if not 0 < args.rate < float("inf"):
        parser.error("--rate must be positive and finite")
    if args.refinement:
        result = replay(report(HISTORICAL), 128, 20, shared_rate=args.rate, shared_rule=args.refinement)
        baseline = next(row for row in report("blue-marble-refinement-batch-size")["runs"] if row["batch"] == 128)
        mode = "uniform-lion" if args.refinement == "lion" else args.refinement.removeprefix("lion-")
        name = f"blue-marble-{mode}-refinement-{args.rate:g}"
        label = "Uniform Lion step" if args.refinement == "lion" else "Participation multiplier"
        render_marble_convergence((baseline, result), ("Historical taper", label), name)
        payload = {"history": result["history"], "run": result, "tapered_history": baseline["history"]}
        (RESULTS / f"{name}.json").write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
        raise SystemExit(0)
    trial = trial_for(args.rule, args.rate, skip_input=args.skip_input, unit_norm=args.unit_norm,
                      last_output=args.last_output, batch=args.batch, gate_width=args.gate_width,
                      shallow_wide=args.shallow_wide)
    result = run(trial, args.until)
    baseline = report(BASELINE)
    final_epoch = result["history"][-1]["epoch"]
    baseline["history"] = [p | {"epoch": p["epoch"] * args.batch / 128}
                           for p in baseline["history"] if p["epoch"] * args.batch / 128 <= final_epoch]
    name = f"blue-marble-{result['settings']['label'].removesuffix('-all-pixels')}"
    label = ("Lion with sign agreement" if args.rule == "lion-adaptive-agreement"
             else f"{args.rule}, shared step")
    if args.skip_input:
        label += ", input skips"
    if args.unit_norm:
        label += ", unit norm"
    if args.last_output:
        label += ", final layer output"
    if args.gate_width is not None:
        label += f", gate width {args.gate_width:g}"
    if args.shallow_wide:
        label += ", 4 × 864 neurons"
    reference_label = "Tapered baseline" if args.batch == 128 else "Tapered baseline at equal updates"
    render_marble_convergence((baseline, result), (reference_label, label), name)
    (RESULTS / f"{name}.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    if args.fits:
        checkpoint = trial.path(MARBLE_SEED) / "checkpoint.npz"
        if not checkpoint.exists():
            checkpoint = checkpoint.with_name("phase-1-progress.npz")
        caption = f"MLP fit, epoch {result['selected_epoch']}"
        render_marble_fits(rgb_data(split=False), (trial,), (result,), (caption,), f"{name}-fits",
                           checkpoints=(checkpoint,))
