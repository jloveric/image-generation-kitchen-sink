"""Reproduce the six-seed staircase comparison with parameter-free outputs."""

import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from hebbian_fit.__main__ import evaluate, plot, training_epochs
from hebbian_fit.data import BAND_EDGES, scalar_data
from hebbian_fit.model import PAIRED_READOUTS, ModelConfig, NumpyModel


def model_for(pool, seed):
    return NumpyModel(ModelConfig(width=32, depth=1, connection="discontinuous", gate="gaussian",
                                  gate_width=.1, readout=pool, hidden_rule="signsgd"), seed)


def train(job):
    pool, rate, seed = job
    model = model_for(pool, seed)
    data = scalar_data("steps")
    args = SimpleNamespace(seed=seed, rate=rate, batch=256, epochs=60)
    best = np.inf
    for epoch in training_epochs(model, data, args, data.train, args.epochs):
        if epoch % 5 == 0:
            metrics, _ = evaluate(model, data, data.validation, "steps")
            score = max(metrics["smooth_rmse"] / .05, metrics["max_plateau_mae"] / .15)
            if score < best:
                best = score
                selected = dict(pool=pool, rate=rate, seed=seed, epoch=epoch, validation_rmse=metrics["rmse"])
                state = model.state()
    return selected, state


def main():
    data = scalar_data("steps")
    out = Path("assets/results")
    out.mkdir(parents=True, exist_ok=True)
    rates = [.00001, .0001, .001, .003, .01, .03, .1, .3, 1.]
    with ProcessPoolExecutor(max_workers=6) as executor:
        screen = list(executor.map(train, [(pool, rate, 1) for pool in PAIRED_READOUTS for rate in rates]))
        chosen = [min((r for r in screen if r[0]["pool"] == pool),
                      key=lambda r: r[0]["validation_rmse"]) for pool in PAIRED_READOUTS]
        jobs = [(row["pool"], row["rate"], seed) for row, _ in chosen for seed in range(2, 7)]
        runs = chosen + list(executor.map(train, jobs))
    results = []
    x = data.x[data.test, 0]
    fine = (x >= BAND_EDGES[-2]) & (x < 0)
    for row, state in runs:
        model = model_for(row["pool"], row["seed"])
        model.load(state)
        metrics, pred = evaluate(model, data, data.test, "steps")
        error = pred[fine] - data.y[data.test][fine]
        results.append(dict(**row, test_rmse=metrics["rmse"],
                            fine_rmse=float(np.sqrt(np.mean(np.square(error, dtype=np.float64))))))
        if row["pool"] == "class-sum" and row["seed"] == 3:
            pred = model.predict_values(data.x)
            plot(out, data, pred, "steps", None)
            (out / "fit.png").replace(out / "fixed-staircase.png")
    (out / "fixed-staircase.json").write_text(json.dumps(
        dict(validation=[row for row, _ in screen], runs=results), indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
