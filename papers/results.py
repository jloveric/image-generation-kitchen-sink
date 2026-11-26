"""Reproduce the final measurements and figures used in the manuscript, from the repository root."""
import argparse
import json
from dataclasses import replace
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np

from experiments import sine_jump
from experiments.blue_marble_quality import Trial, layer_readouts
from experiments.celeba import evaluate, figure, load_checkpoint
from experiments.celeba_diversity import model_for
from hebbian_fit.autoencoder import Generator
from hebbian_fit.data import rgb_data
from hebbian_fit.model import ModelConfig, NumpyModel

OUT = Path(__file__).resolve().parent / "figures"
REPORTS = Path("assets/results")
FACE_RUN = Path("runs/celeba-sequential/full/target150-standard-full128-broad-lion/full")
INK, TEAL = "#293942", "#00767b"
plt.rcParams.update({"pdf.fonttype": 42, "font.size": 10, "axes.spines.top": False,
                     "axes.spines.right": False, "axes.labelcolor": INK, "text.color": INK})


def sine():
    data = sine_jump.dataset()
    trial = sine_jump.deeper(12)
    wide_start = next(t for t in sine_jump.wide_trials() if t.label == "sine-crest-jump-1x512-gauss0.32")
    others = [(sine_jump.refine_wide(wide_start), wide_start, "1 × 512", "#846829", ":"),
              (sine_jump.TRIAL, None, "4 × 64", "#526e91", "-."),
              (sine_jump.scale_steps(sine_jump.deeper(24), .64), None, "24 × 64", "#99516b", "--")]
    rows = sine_jump.measure(trial, data, sine_jump.TEST_SEEDS)
    print(trial.label, json.dumps(sine_jump.summarize(rows), indent=2))
    row = sorted(rows, key=lambda r: r["test_rmse"])[len(rows) // 2]
    model = sine_jump.load_model(row["seed"], trial)
    fig, axes = plt.subplots(1, 2, figsize=(7.5, 3.05), width_ratios=(1.12, 1),
                             layout="constrained")
    for ax, (left, right), title in zip(axes, ((-1, 1), (.15, .35)), ("Full interval", "Near the crest")):
        x = np.linspace(left, right, 8001, dtype=np.float32)
        ax.plot(x, sine_jump.curve(model, x), lw=1.75, color=TEAL, label="12 × 64")
        if ax is axes[1]:
            for other, parent, name, color, style in others:
                measured = sine_jump.measure(other, data, sine_jump.TEST_SEEDS, parent=parent)
                print(other.label, json.dumps(sine_jump.summarize(measured), indent=2))
                other_model = sine_jump.load_model(row["seed"], other)
                ax.plot(x, sine_jump.curve(other_model, x), lw=1.25, color=color, ls=style, label=name)
        x_left = np.linspace(left, sine_jump.JUMP_X, 1001)
        x_right = np.linspace(sine_jump.JUMP_X, right, 1001)
        ax.plot(x_left, sine_jump.sine(x_left), ls=(0, (5, 2)), lw=1.35, color=INK, label="Target")
        ax.plot(x_right, sine_jump.sine(x_right)+sine_jump.JUMP_HEIGHT, ls=(0, (5, 2)), lw=1.35, color=INK)
        ax.set(xlabel="Input  $x$", xlim=(left, right))
        ax.set_title(title, loc="left", fontsize=11, weight="bold")
        ax.grid(axis="y", color="#dae1e3", lw=.6)
        ax.set_axisbelow(True)
    axes[0].set_ylabel("Output")
    handles, labels = axes[1].get_legend_handles_labels()
    order = [4, 1, 2, 0, 3]
    fig.legend([handles[i] for i in order], [labels[i] for i in order],
               loc="outside lower center", frameon=False, ncol=5, fontsize=9)
    axes[1].set_xticks([.20, .25, .30])
    t = row["transition"]
    axes[1].axvspan(t["x10"], t["x90"], color="#deeeea", lw=0, zorder=0)
    axes[1].plot([.25], [.5], "o", color=INK, ms=4)
    axes[1].plot([.25], [1.], "o", mec=INK, mfc="white", ms=4)
    fig.savefig(OUT / "sine.pdf")
    plt.close(fig)


def marble_layers(model, data, readouts, full):
    # Align each layer's own RGB estimate with the accumulated contribution through that layer.
    fig, axes = plt.subplots(4, 6, figsize=(7.5, 5.15))
    fig.subplots_adjust(left=.165, right=.995, bottom=.01, top=.95, wspace=.06, hspace=.17)
    for index, values in enumerate(readouts):
        block, column = divmod(index, 6)
        mean = readouts[:index + 1].mean(axis=0)
        if index + 1 == len(readouts):
            np.testing.assert_array_equal(model.format_output(mean), full)
        for row, (label, raw) in enumerate((("Layer\nalone", values), ("Accumulated\ncontribution\nthrough layer", mean))):
            ax = axes[2*block + row, column]
            rgb = (model.format_output(raw).reshape(*data.shape, 3) + 1) / 2
            ax.imshow(rgb, interpolation="none")
            ax.set_axis_off()
            if row == 0:
                ax.set_title(f"Layer {index + 1}", fontsize=9, pad=3)
            if column == 0:
                ax.text(-.12, .5, label, transform=ax.transAxes, ha="right", va="center", fontsize=9)
    fig.savefig(OUT / "earth-layers.pdf")
    plt.close(fig)


def marble(checkpoint=None):
    report = json.loads((REPORTS / "blue-marble-output-depth-12-two-phase-all-pixels.json").read_text())
    settings = report["settings"]
    config = ModelConfig(**{**settings["config"], "layer_gate_widths": tuple(settings["config"]["layer_gate_widths"])})
    trial = Trial(settings["label"], config, tuple(map(tuple, settings["phases"])), settings["backend"])
    trial = replace(trial, label=trial.label + "-all-pixels")
    model = NumpyModel(config, report["seed"])
    with np.load(checkpoint or trial.path(report["seed"]) / "checkpoint.npz", allow_pickle=False) as state:
        model.load(state)
    data = rgb_data(split=False)
    readouts, predicted = layer_readouts(model, data.x)
    marble_layers(model, data, readouts, predicted)
    error = np.sum(np.square((predicted - data.y) / 2, dtype=np.float64))
    print(f"RGB squared error, summed over {len(data.x):,} pixels: {error:.6f}")
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
    parser.add_argument("task", choices=("sine", "marble", "portraits", "reconstruction"))
    parser.add_argument("--run", type=Path, default=FACE_RUN, help="Final CelebA full run directory")
    parser.add_argument("--checkpoint", type=Path, help="Use this saved Blue Marble checkpoint")
    parser.add_argument("--code-top-k", type=int,
                        help="Evaluate a saved CelebA checkpoint with a different bottleneck competition count")
    parser.add_argument("--vector-gate-width", type=float,
                        help="Evaluate a saved CelebA checkpoint at a different RGB Gaussian width")
    args = parser.parse_args()
    if args.checkpoint is not None and args.task != "marble":
        parser.error("--checkpoint is for the marble task")
    if args.task != "reconstruction" and (args.code_top_k is not None or args.vector_gate_width is not None):
        parser.error("Competition overrides require the reconstruction task")
    OUT.mkdir(exist_ok=True)
    if args.task == "sine":
        sine()
    elif args.task == "marble":
        marble(args.checkpoint)
    elif args.task == "portraits":
        generator = Generator(args.run / "decoder-mixture.npz")
        figure(np.stack([generator.generate(i) for i in range(36)]), OUT / "portraits.png")
    else:
        config = json.loads((args.run / "results.json").read_text())["config"]
        model = model_for(config, args.run, backend="mojo", code_top_k=args.code_top_k,
                          vector_gate_width=args.vector_gate_width)
        load_checkpoint(args.run / "selected.npz", model)
        test = np.load(args.run.parent / "test.npy", mmap_mode="r")
        center = np.load(args.run / "encoder-center.npy").reshape(test.shape[1:])
        error = evaluate(model, test, center, shuffle=False)["reconstruction_sse"]
        print(f"Squared RGB error per portrait on {len(test):,} test portraits: {error/len(test):.6f}")


if __name__ == "__main__":
    main()
