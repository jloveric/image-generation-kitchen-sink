"""Reproduce the dense Hebbian sine fit and its held-out error."""

from dataclasses import asdict
import json

import numpy as np

from hebbian_fit.data import scalar_data
from hebbian_fit.model import ModelConfig, NumpyModel

from .fixed_capacity import RESULTS, predict, rmse
from .scalar_quality import ScalarTrial, fit


SAMPLES = 4096
TEST_SEEDS = (7, 8, 9)
TRAIN_DOMAIN = (-1, 1)

TRIAL = ScalarTrial(
    label="sine-paper-4x64-refined-twice",
    config=ModelConfig(
        width=32, depth=4, competition="layer", gate="gaussian-smooth",
        layer_gate_widths=(.320, .267, .213, .160), readout="class-sum", readout_layers="all",
    ),
    # Each tuple specifies the rule, initial final-layer step, batch, epochs, and layer step multiplier.
    phases=(
        ("signsgd", .003, 64, 80, 1.),
        ("lion", .0008, 64, 100, .75),
        ("lion", .00035, 64, 220, .90),
        ("lion", .00012, 64, 240, .95),
    ),
    backend="numpy",
    samples=SAMPLES,
)


def curve(model, x):
    return np.concatenate([model.predict_values(x[start:start + 256, None]).ravel()
                           for start in range(0, len(x), 256)]).astype(np.float64)


def evaluate(model, data):
    x = data.x[data.test, 0].astype(np.float64)
    residual = predict(model, data, data.test).ravel().astype(np.float64) - data.y[data.test, 0]
    return {
        "test_rmse": rmse(residual),
        "test_max_abs_error": float(np.abs(residual).max()),
        "test_slope_rmse": rmse(np.diff(residual) / np.diff(x)),
    }


def render(model, row):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    x = np.linspace(*TRAIN_DOMAIN, 8001, dtype=np.float32)
    prediction = curve(model, x)
    target = np.sin(2 * np.pi * x.astype(np.float64))
    fig, (fit_ax, error_ax) = plt.subplots(2, 1, sharex=True, figsize=(8.0, 5.0),
                                          height_ratios=[2.6, 1], layout="constrained")
    for ax in (fit_ax, error_ax):
        ax.set_xlim(*TRAIN_DOMAIN)
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", color="#e3e6e9", lw=.6)
        ax.set_axisbelow(True)
    fit_ax.plot(x, target, color="#545961", lw=1.35, linestyle=(0, (4, 2)), label="Sine target")
    fit_ax.plot(x, prediction, color="#176b8a", lw=1.65, label="Hebbian prediction")
    fit_ax.set_ylabel("Function value")
    fit_ax.legend(loc="lower center", bbox_to_anchor=(.5, 1.01), ncol=2, frameon=False)
    error_ax.axhline(0, lw=.7, color="#767e87")
    error_ax.plot(x, prediction - target, color="#176b8a", lw=1)
    error_ax.set(xlabel="Input x", ylabel="Fit error")
    error_ax.set_title(f"In-range residuals · held-out RMSE {row['test_rmse']:.4f} · seed {row['seed']}",
                       loc="left", fontsize=9, pad=6)
    fig.savefig(RESULTS / "sine-paper-fit.png", dpi=180)
    fig.savefig(RESULTS / "sine-paper-fit.pdf")
    plt.close(fig)


def main():
    data = scalar_data("sine", SAMPLES)
    rows = []
    for seed in TEST_SEEDS:
        trained = fit(TRIAL, data, seed)
        model = NumpyModel(TRIAL.config, seed)
        with np.load(TRIAL.path(seed) / "checkpoint.npz", allow_pickle=False) as state:
            model.load(state)
        row = dict(seed=seed, **evaluate(model, data), phase_epochs=trained["phase_epochs"])
        rows.append(row)
        print(json.dumps(row), flush=True)
    example = sorted(rows, key=lambda row: row["test_rmse"])[len(rows) // 2]
    model = NumpyModel(TRIAL.config, example["seed"])
    with np.load(TRIAL.path(example["seed"]) / "checkpoint.npz", allow_pickle=False) as state:
        model.load(state)
    render(model, example)
    metrics = [key for key in example if key not in ("seed", "phase_epochs")]
    report = dict(settings=asdict(TRIAL), parameters=sum(layer.size for layer in model.weights),
                  target="sin(2*pi*x)", development_seeds=[1, 2, 3],
                  train_domain=TRAIN_DOMAIN,
                  train_samples=len(data.train), validation_samples=len(data.validation),
                  test_samples=len(data.test), example_seed=example["seed"], runs=rows,
                  medians={key: float(np.median([row[key] for row in rows])) for key in metrics})
    (RESULTS / "sine-paper-fit.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
