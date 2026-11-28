"""Reproduce the final measurements and figures used in the manuscript, from the repository root."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
from PIL import Image

from experiments import signed_activity, sine_jump
from experiments.blue_marble_quality import Trial, layer_readouts
from experiments.celeba import encode, evaluate, figure, load_checkpoint
from experiments.celeba_diversity import model_for
from hebbian_fit.autoencoder import Generator, SIZE, rgb_bytes
from hebbian_fit.data import rgb_data
from hebbian_fit.model import ModelConfig, NumpyModel
from papers import lion

OUT = Path(__file__).resolve().parent / "figures"
FACE_RUN = OUT.parents[1] / "runs/celeba-single-output-tuning/independent-hidden-votes/full"
FACE_MIXTURE = "decoder-mixture-selected.npz"
INK, TEAL = "#293942", "#00767b"
plt.rcParams.update({"pdf.fonttype": 42, "font.size": 10, "axes.spines.top": False,
                     "axes.spines.right": False, "axes.labelcolor": INK, "text.color": INK})


SINE_SIZES = ((1, 512), (4, 64), (12, 64), (24, 64))


def median_sine_fit(depth, width, center=None):
    trial = signed_activity.trial_for("sine", depth=depth, width=width, sine_center=center)
    rows = sorted(lion.report(trial.label)["runs"], key=lambda r: r["test_rmse"])
    median = rows[len(rows) // 2]
    return sine_jump.load_model(median["seed"], trial), median["test_rmse"]


def sine_target(ax, left, right):
    x_left = np.linspace(left, sine_jump.JUMP_X, 1001)
    x_right = np.linspace(sine_jump.JUMP_X, right, 1001)
    ax.plot(x_left, sine_jump.sine(x_left), ls=(0, (5, 2)), lw=1.35, color=INK, label="Target")
    ax.plot(x_right, sine_jump.sine(x_right)+sine_jump.JUMP_HEIGHT, ls=(0, (5, 2)), lw=1.35, color=INK)


def sine():
    fits = []
    styles = (("#b76e3d", ":"), ("#77518c", "-."), ("#477299", "--"), (TEAL, "-"))
    for (depth, width), (color, style) in zip(SINE_SIZES, styles, strict=True):
        model, rmse = median_sine_fit(depth, width)
        fits.append((model, f"{depth} × {width}", color, style))
        print(f"{depth} × {width}: median test RMSE {rmse:.5f}")
    fig, axes = plt.subplots(1, 2, figsize=(7.5, 3.05), width_ratios=(1.12, 1),
                             layout="constrained")
    for ax, (left, right), title in zip(axes, ((-1, 1), (.15, .35)), ("Curve fit", "Zoom of discontinuity")):
        x = np.linspace(left, right, 8001, dtype=np.float32)
        for model, label, color, style in fits:
            ax.plot(x, sine_jump.curve(model, x), lw=1.55, color=color, ls=style, label=label)
        sine_target(ax, left, right)
        ax.set(xlabel="Input  $x$", xlim=(left, right))
        ax.set_title(title, loc="left", fontsize=11, weight="bold")
        ax.grid(axis="y", color="#dae1e3", lw=.6)
        ax.set_axisbelow(True)
    axes[0].set_ylabel("Output")
    handles, labels = axes[1].get_legend_handles_labels()
    fig.legend([handles[-1], *handles[:-1]], [labels[-1], *labels[:-1]],
               loc="outside lower center", frameon=False, ncol=5, fontsize=9)
    axes[1].set_xticks([.20, .25, .30])
    axes[1].plot([.25], [.5], "o", color=INK, ms=4)
    axes[1].plot([.25], [1.], "o", mec=INK, mfc="white", ms=4)
    left, right = axes[1].get_xlim()
    bottom, top = axes[1].get_ylim()
    axes[0].add_patch(Rectangle((left, bottom), right - left, top - bottom,
                                fill=False, edgecolor=INK, linewidth=0.9))
    fig.savefig(OUT / "sine.pdf")
    fig.savefig(OUT / "sine.png", dpi=180)
    plt.close(fig)


def sine_centers():
    fig, axes = plt.subplots(4, 2, figsize=(8.5, 8.2), layout="constrained", sharex="col")
    for (depth, width), panels in zip(SINE_SIZES, axes, strict=True):
        fits = [median_sine_fit(depth, width, center) for center in ("max-abs", "smooth")]
        for ax, (left, right) in zip(panels, ((-1, 1), (.15, .35)), strict=True):
            x = np.linspace(left, right, 8001, dtype=np.float32)
            for (model, _), label, color, style in zip(fits, ("Maximum", "Smooth center"),
                                                       (TEAL, "#bd7134"), ("-", "-."), strict=True):
                ax.plot(x, sine_jump.curve(model, x), label=label, color=color, ls=style, lw=1.6)
            sine_target(ax, left, right)
            ax.set_xlim(left, right)
            ax.grid(axis="y", alpha=.25)
        panels[0].set_ylabel(f"{depth} × {width}\nOutput")
    axes[0, 0].set_title("Curve fit")
    axes[0, 1].set_title("Zoom of discontinuity")
    for ax in axes[-1]:
        ax.set_xlabel("Input  $x$")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", frameon=False, ncol=3)
    fig.savefig(lion.RESULTS / "sine-gaussian-centers.png", dpi=160)
    plt.close(fig)


def marble_layers(model, data, readouts, full):
    accumulated = [readouts[:k].mean(axis=0) for k in range(1, len(readouts) + 1)]
    np.testing.assert_array_equal(model.format_output(accumulated[-1]), full)
    fig = plt.figure(figsize=(7.5, 5.8), layout="constrained")
    groups = (("Layer alone", readouts), ("Accumulated contribution through layer", accumulated))
    for panel, (title, estimates) in zip(fig.subfigures(2, 1), groups):
        panel.suptitle(title, fontsize=11, fontweight="bold")
        for index, (ax, raw) in enumerate(zip(panel.subplots(2, 6).flat, estimates), start=1):
            rgb = (model.format_output(raw).reshape(*data.shape, 3) + 1) / 2
            ax.imshow(rgb, interpolation="none")
            ax.set_axis_off()
            ax.set_title(f"Layer {index}", fontsize=9, pad=2)
    fig.savefig(OUT / "earth-layers.pdf")
    plt.close(fig)
    return accumulated


def marble(checkpoint=None):
    report = lion.report(signed_activity.trial_for("marble", backend="mojo").label)
    settings = report["settings"]
    config = ModelConfig(**{**settings["config"], "layer_gate_widths": tuple(settings["config"]["layer_gate_widths"])})
    trial = Trial(settings["label"], config, tuple(map(tuple, settings["phases"])),
                  settings["backend"], settings["schedule"])
    model = NumpyModel(config, report["seed"])
    with np.load(checkpoint or trial.path(report["seed"]) / "checkpoint.npz", allow_pickle=False) as state:
        model.load(state)
    data = rgb_data(split=False)
    readouts, predicted = layer_readouts(model, data.x)
    accumulated = marble_layers(model, data, readouts, predicted)
    error = np.sum(np.square((predicted - data.y) / 2, dtype=np.float64))
    if checkpoint is None:
        np.testing.assert_allclose(error, report["squared_error"], rtol=1e-4)
    print(f"RGB squared error, summed over {len(data.x):,} pixels: {error:.6f}")
    for label, values in (("Individual", readouts), ("Accumulated", accumulated)):
        errors = [float(np.square((model.format_output(raw) - data.y) / 2, dtype=np.float64).sum())
                  for raw in values]
        print(label, "layer errors", ", ".join(f"{score:.6f}" for score in errors))
    images = [(arr.reshape(128, 128, 3) + 1) / 2 for arr in (data.y, predicted)]
    fig, axes = plt.subplots(1, 4, figsize=(7.5, 1.96), layout="constrained")
    titles = ("Photograph", "Network", "Photograph detail", "Network detail")
    for i, (ax, title) in enumerate(zip(axes, titles)):
        pixels = images[i % 2]
        ax.imshow(pixels if i < 2 else pixels[42:86, 43:87], interpolation="nearest")
        if i < 2:
            ax.add_patch(Rectangle((42.5, 41.5), 44, 44, ec="white", fc="none", lw=.9))
        ax.set_title(title, fontsize=10)
        ax.set_axis_off()
    fig.savefig(OUT / "earth.pdf")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("task", choices=("sine", "sine-centers", "marble", "images", "portraits", "reconstruction"))
    parser.add_argument("--run", type=Path, help="CelebA full run directory")
    parser.add_argument("--mixture", type=Path, help="Use this fitted code distribution and decoder export")
    parser.add_argument("--checkpoint", type=Path, help="Use this saved Blue Marble checkpoint")
    parser.add_argument("--example", type=Path, help="Save the first test image beside its reconstruction")
    parser.add_argument("--code-top-k", type=int,
                        help="Evaluate a saved CelebA checkpoint with a different bottleneck competition count")
    parser.add_argument("--vector-gate-width", type=float,
                        help="Evaluate a saved CelebA checkpoint at a different RGB Gaussian width")
    args = parser.parse_args()
    if args.checkpoint is not None and args.task != "marble":
        parser.error("--checkpoint is for the marble task")
    if args.mixture is not None and args.task not in ("images", "portraits"):
        parser.error("--mixture requires an image generation task")
    if args.example is not None and args.task != "reconstruction":
        parser.error("--example requires the reconstruction task")
    if args.task != "reconstruction" and (args.code_top_k is not None or args.vector_gate_width is not None):
        parser.error("Competition overrides require the reconstruction task")
    OUT.mkdir(exist_ok=True)
    run = args.run or (lion.PORTRAIT_BASELINE if args.task == "portraits" else FACE_RUN)
    if args.task == "sine":
        sine()
    elif args.task == "sine-centers":
        sine_centers()
    elif args.task == "marble":
        marble(args.checkpoint)
    elif args.task in ("images", "portraits"):
        filename = FACE_MIXTURE
        if args.task == "portraits" and not (run / filename).exists():
            filename = "decoder-mixture.npz"
        generator = Generator(args.mixture or run / filename)
        name = "celeba-generated.png" if args.task == "images" else "portraits.png"
        figure(np.stack([generator.generate(i) for i in range(36)]), OUT / name)
    else:
        config = json.loads((run / "results.json").read_text())["config"]
        model = model_for(config, run, backend="mojo", code_top_k=args.code_top_k,
                          vector_gate_width=args.vector_gate_width)
        load_checkpoint(run / "selected.npz", model)
        test = np.load(run.parent / "test.npy", mmap_mode="r")
        center = np.load(run / "encoder-center.npy").reshape(test.shape[1:])
        error = evaluate(model, test, center, shuffle=False)["reconstruction_sse"]
        if args.example is not None:
            original = test[0].reshape(SIZE, SIZE, 3)
            prediction = rgb_bytes(model.decode(encode(model, test[:1]))).reshape(original.shape)
            Image.fromarray(np.concatenate((original, prediction), axis=1)).save(args.example)
        print(f"Squared RGB error per image on {len(test):,} test images: {error/len(test):.6f}")


if __name__ == "__main__":
    main()
