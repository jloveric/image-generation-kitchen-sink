"""Run one reproducible coordinate-fitting experiment."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from .data import BAND_EDGES, EDGES, HEIGHTS, IMAGE, rgb_data, scalar_data, step_metrics
from .model import BASIS_SIZE, COMPETITIONS, GATES, HIDDEN_RULES, INITIALIZATIONS, READOUTS, ModelConfig, NumpyModel


def evaluate(model, data, indices, task):
    c = model.config
    missing = np.zeros((c.depth, c.classes), dtype=np.int64)
    predictions = []
    predict_values = getattr(model, "predict_values", None)
    for part in np.array_split(indices, max(1, (len(indices) + 255) // 256)):
        if c.gate == "hard" or predict_values is None:
            values, masks = model.predict(data.x[part])
            for layer, mask in enumerate(masks):
                active = mask.reshape(len(part), c.classes, -1).any(axis=-1)
                missing[layer] += (~active).sum(axis=0)
        else:
            # Every neuron participates in a scaled gate.
            values = predict_values(data.x[part])
        predictions.append(values)
    prediction = np.concatenate(predictions)
    error = prediction - data.y[indices]
    result = dict(
        rmse=float(np.sqrt(np.mean(error.astype(np.float64) ** 2))),
        mae=float(np.mean(np.abs(error))),
        no_winner_fraction=(missing / len(indices)).tolist(),
    )
    if task == "steps":
        result.update(step_metrics(data.x[indices], error))
    if task == "image":
        result["rgb_squared_error_sum"] = float(np.square(error / 2, dtype=np.float64).sum())
        error = (np.clip(prediction, -1, 1) - data.y[indices]) / 2
        mse = np.mean(error.astype(np.float64) ** 2)
        result.update(
            psnr_db=float(-10 * np.log10(mse)) if mse else None,
            channel_mae=np.mean(np.abs(error), axis=0).tolist(),
        )
    return result, prediction


def training_epochs(model, data, args, indices, epochs, *, start_epoch=0, epoch_offset=0):
    rng = np.random.default_rng(args.seed + 1)
    hold = 0.2 if getattr(args, "schedule", "hold-cosine") == "hold-cosine" else 0.
    for epoch in range(1, epochs + 1):
        order = rng.permutation(indices)
        if epoch <= start_epoch:
            continue
        progress = max(0, (epoch_offset + epoch - 1) / args.epochs - hold) / (1 - hold)
        schedule = (1 + np.cos(np.pi * progress)) / 2
        for start in range(0, len(order), args.batch):
            batch = order[start:start + args.batch]
            model.step(data.x[batch], data.y[batch], args.rate * schedule)
        yield epoch


def fit(model, data, args, indices, validation=None, track=None, epochs=None):
    epochs = args.epochs if epochs is None else epochs
    best = None
    for epoch in training_epochs(model, data, args, indices, epochs):
        if validation is None:
            continue
        metrics, _ = evaluate(model, data, validation, args.task)
        score = metrics["rmse"]
        if args.task == "steps":
            score = max(metrics["smooth_rmse"] / 0.05, metrics["max_plateau_mae"] / 0.15)
        if args.task == "image":
            score = metrics["rgb_squared_error_sum"]
        if not np.isfinite(metrics["rmse"]):
            raise RuntimeError(f"Nonfinite predictions at epoch {epoch}")
        track.append(dict(epoch=epoch, **metrics))
        if best is None or score < best[0]:
            best = score, epoch, model.state()
    if best is not None:
        model.load(best[2])
        return best[1]
    return epochs


def sine_variation(model):
    grid = np.linspace(-1, 1, 16385, dtype=np.float32)[:, None]
    prediction = np.concatenate([
        model.predict(part)[0] for part in np.array_split(grid, 65)
    ]).ravel().astype(np.float64)
    differences = np.diff(prediction)
    return dict(
        grid_points=len(grid),
        max_adjacent_change=float(np.max(np.abs(differences))),
        max_second_difference=float(np.max(np.abs(np.diff(differences)))),
    )


def plot(directory, data, prediction, task, credit):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if task == "image":
        fig, axes = plt.subplots(1, 3, figsize=(10, 3.7), layout="constrained")
        target = (data.y.reshape(*data.shape, 3) + 1) / 2
        recon = np.clip((prediction.reshape(*data.shape, 3) + 1) / 2, 0, 1)
        axes[0].imshow(target)
        axes[1].imshow(recon)
        heat = axes[2].imshow(np.abs(target - recon).mean(axis=-1), vmin=0, vmax=1, cmap="magma")
        fig.colorbar(heat, ax=axes[2], fraction=0.046)
        for ax, title in zip(axes, ["Source", "Model reconstruction", "Mean RGB error"]):
            ax.set_title(title)
            ax.axis("off")
        if credit:
            axes[0].text(0, -0.06, credit, transform=axes[0].transAxes, fontsize=8)
    else:
        limits = [(-1, 1)]
        if task == "steps":
            limits.append((BAND_EDGES[-2], BAND_EDGES[-1]))
        fig, axes = plt.subplots(len(limits), 1, squeeze=False, figsize=(10, 3 * len(limits)),
                                 layout="constrained")
        for ax, (lo, hi) in zip(axes[:, 0], limits):
            i = data.test
            if task == "steps":
                ax.stairs(HEIGHTS, EDGES, baseline=None, label="Target", linewidth=1.5,
                          color="C0")
                smooth = np.linspace(0, 1, 1001)
                ax.plot(smooth, 0.75 * np.cos(2 * np.pi * smooth), color="C0", linewidth=1.5)
            else:
                ax.plot(data.x[i], data.y[i], label="Target", linewidth=1.5)
            ax.plot(data.x[i], prediction[i], label="Hebbian fit", linewidth=1)
            ax.set(xlabel="x", ylabel="y", xlim=(lo, hi))
            ax.legend()
    fig.savefig(directory / "fit.png", dpi=160)
    plt.close(fig)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("task", choices=("sine", "steps", "image"))
    p.add_argument("--architecture", choices=("dense",), default="dense", help=argparse.SUPPRESS)
    p.add_argument("--backend", choices=("numpy", "mojo"), default="numpy")
    p.add_argument("--competition", choices=COMPETITIONS, default="class",
                   help="Compete within a class, an output's class pair, or the whole layer")
    p.add_argument("--gate", choices=GATES, default="hard")
    p.add_argument("--sine-omega", type=float,
                   help="Apply sin(omega times the affine response) before competition in every hidden layer")
    p.add_argument("--gate-width", type=float, default=0.5,
                   help="Gaussian width or tent radius as a fraction of the pool maximum")
    p.add_argument("--layer-gate-widths", type=float, nargs="+",
                   help="Override the shared gate width with one width per hidden layer")
    p.add_argument("--readout", choices=READOUTS, default="fixed")
    p.add_argument("--readout-layers", choices=("last", "all"), default="last",
                   help="Average predictions from the specified hidden layers")
    p.add_argument("--output-bound", type=float,
                   help="Set the nominal output range to ±BOUND and enable clipping unless overridden")
    p.add_argument("--output-levels", type=int,
                   help="Compare discrete output values. Image defaults to 256 levels. Use 0 for continuous values.")
    p.add_argument("--clip-output", action=argparse.BooleanOptionalAction, default=True,
                   help="Clamp predictions to the configured output range (default).")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--width", type=int, default=32)
    p.add_argument("--winners", type=int, default=8, help="Number of winners for the hard gate")
    p.add_argument("--connection", choices=tuple(BASIS_SIZE), default="affine")
    p.add_argument("--edge-transitions", type=float, nargs="+",
                   help="Smooth each two-piece layer over a band with the given half-width")
    p.add_argument("--depth", type=int, default=2)
    p.add_argument("--normalize-hidden-inputs", nargs="?", choices=("rms", "max-abs"), const="rms", default=False,
                   help="Normalize each example's gated features between layers (NumPy)")
    p.add_argument("--initialization", choices=INITIALIZATIONS, default="normal",
                   help="Optionally normalize each incoming weight vector with its bias at initialization")
    p.add_argument("--unit-norm-updates", action="store_true",
                   help="Renormalize each updated hidden neuron's weights and bias to unit length")
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--schedule", choices=("cosine", "hold-cosine"), default="cosine",
                   help="Cosine decay from the first epoch, or the earlier 20%% hold followed by decay")
    p.add_argument("--layer-rate-decay", type=float, default=1.,
                   help="Multiply the step by this factor for each layer back from the final hidden layer")
    p.add_argument("--rate", type=float, default=0.00075)
    p.add_argument("--hidden-rule", choices=HIDDEN_RULES, default="lion",
                   help="Signed coactivity update for every hidden layer")
    p.add_argument("--image", type=Path, default=IMAGE)
    p.add_argument("--size", type=int, default=128)
    p.add_argument("--samples", type=int,
                   help="Number of coordinates per training, validation and test grid for sine or steps")
    p.add_argument("--fit-all-pixels", action="store_true",
                   help="Train on the complete image and select the checkpoint with lowest pixel error")
    p.add_argument("--output", type=Path)
    args = p.parse_args()
    if args.fit_all_pixels and args.task != "image":
        p.error("--fit-all-pixels requires the image task")
    if args.samples is not None and (args.task == "image" or args.samples < 1):
        p.error("--samples requires a positive count for sine or steps")
    if args.output_levels is None and args.task == "image" and args.output_bound is None:
        args.output_levels = 256
    if args.output_levels:
        if not 2 <= args.output_levels <= 65536:
            p.error("Use 0 or 2 to 65536 output levels")
        if args.output_bound is None:
            args.output_bound = 1.0
    if args.layer_gate_widths is not None:
        args.layer_gate_widths = tuple(args.layer_gate_widths)
    if args.edge_transitions is not None:
        args.edge_transitions = tuple(args.edge_transitions)
    if min(args.epochs, args.batch) < 1 or args.size < 4 or args.seed < 0:
        p.error("Use positive epoch and batch counts, image size >= 4, and a nonnegative seed")
    if not np.isfinite(args.rate) or args.rate <= 0:
        p.error("Learning rates must be positive and finite")
    return args


def main():
    args = parse_args()
    data = (rgb_data(args.image, args.size, split=not args.fit_all_pixels)
            if args.task == "image" else scalar_data(args.task, args.samples))
    config = ModelConfig(data.x.shape[1], data.y.shape[1], args.width, args.winners,
                         args.depth, args.competition, args.readout, args.gate, args.gate_width,
                         args.layer_gate_widths, args.readout_layers, args.initialization,
                         args.unit_norm_updates, args.sine_omega, args.connection, args.hidden_rule,
                         output_bound=args.output_bound, output_levels=args.output_levels or None,
                         edge_transitions=args.edge_transitions, clip_output=args.clip_output,
                         layer_rate_decay=args.layer_rate_decay, normalize_hidden_inputs=args.normalize_hidden_inputs)
    if args.backend == "mojo":
        from .gpu import MojoModel
        backend = MojoModel
    else:
        backend = NumpyModel
    path = Path("runs", args.task, args.backend, args.readout, args.competition)
    if args.normalize_hidden_inputs:
        path /= f"interlayer-{args.normalize_hidden_inputs}"
    if args.schedule == "cosine":
        path /= "cosine"
    if args.fit_all_pixels:
        path /= "all-pixels"
    if args.samples is not None:
        path /= f"samples-{args.samples}"
    if args.output_levels:
        path /= f"levels-{args.output_levels}"
    if args.output_bound is not None:
        path /= f"bound-{args.output_bound}"
    if not args.clip_output and args.output_bound is not None:
        path /= "unclipped"
    if args.layer_rate_decay != 1.:
        path /= f"layer-rate-decay-{args.layer_rate_decay}"
    if args.hidden_rule != "lion":
        path /= args.hidden_rule
    if args.connection != "affine":
        path /= args.connection
    if args.edge_transitions is not None:
        path /= "transition-" + "-".join(map(str, args.edge_transitions))
    if args.sine_omega is not None:
        path /= f"sine-{args.sine_omega}"
    if args.gate == "none":
        path /= "ungated"
    elif args.gate != "hard":
        widths = "-".join(str(width) for width in args.layer_gate_widths or (args.gate_width,))
        path /= f"{args.gate}-{widths}"
    if args.readout_layers == "all":
        path /= "all-layers"
    if args.initialization != "normal":
        path /= args.initialization
    if args.unit_norm_updates:
        path /= "unit-updates"
    run = args.output or path / f"seed-{args.seed}"
    run.mkdir(parents=True, exist_ok=False)
    settings = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    if args.task == "image":
        settings["image_sha256"] = hashlib.sha256(args.image.read_bytes()).hexdigest()
    (run / "config.json").write_text(json.dumps(settings, indent=2) + "\n")
    model = (backend(config, args.seed, capacity=max(256, args.batch))
             if args.backend == "mojo" else backend(config, args.seed))
    if args.task == "image" and not args.fit_all_pixels:
        initial = model.state()
    full = np.arange(len(data.x))
    train_indices = full if args.fit_all_pixels else data.train
    score_indices = full if args.fit_all_pixels else data.validation
    track = []
    epoch = fit(model, data, args, train_indices, score_indices, track)
    np.savez_compressed(run / "checkpoint.npz", **model.state())
    if args.fit_all_pixels:
        result = dict(epoch=epoch, trajectory=track)
        result["reconstruction"], prediction = evaluate(model, data, full, args.task)
        summary = f"all-pixel squared RGB error {result['reconstruction']['rgb_squared_error_sum']:.5f}"
    else:
        validation = track[epoch - 1]
        test, _ = evaluate(model, data, data.test, args.task)
        result = dict(validation=validation, test=test, trajectory=track)
        if args.task == "sine":
            result["variation"] = sine_variation(model)
        if args.task == "image":
            model.load(initial)
            fit(model, data, args, full, epochs=epoch)
            result["reconstruction"], prediction = evaluate(model, data, full, args.task)
            np.savez_compressed(run / "reconstruction.npz", **model.state())
        else:
            _, prediction = evaluate(model, data, full, args.task)
        summary = f"validation RMSE {validation['rmse']:.5f}  test RMSE {test['rmse']:.5f}"
    credit = "Source photo · NASA Johnson Space Center" if args.image.resolve() == IMAGE else None
    plot(run, data, prediction, args.task, credit)
    (run / "metrics.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(f"{run}  {summary}", flush=True)


if __name__ == "__main__":
    main()
