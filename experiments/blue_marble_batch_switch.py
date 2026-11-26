"""Test the refinement batch size from the historical epoch-80 checkpoint."""

from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
import json
from types import SimpleNamespace

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from experiments.blue_marble_layer_steps import HISTORICAL
from experiments.blue_marble_quality import Trial
from experiments.blue_marble_report import rgb_sse
from hebbian_fit.__main__ import training_epochs
from hebbian_fit.data import rgb_data
from hebbian_fit.model import ModelConfig
from papers.lion import RESULTS, report


STUDY = "blue-marble-refinement-batch-size"


def replay(historical, batch, epochs, *, shared_rate=None, shared_rule="lion-participation"):
    settings = historical["settings"]
    initial, refinement = map(tuple, settings["phases"])
    seed = historical["seed"]
    config = ModelConfig(**(settings["config"] | {
        "layer_gate_widths": tuple(settings["config"]["layer_gate_widths"]),
    }))
    parent = Trial(f"{settings['label']}-all-pixels", config, (initial,),
                   settings["backend"], settings["schedule"])
    rule, rate, _, duration, decay = refinement
    if shared_rate is not None:
        if shared_rule not in ("lion", "lion-participation", "lion-participation-power", "lion-pool-participation"):
            raise ValueError("The replay requires a rule with the same checkpoint state as Lion")
        rule, rate, decay = shared_rule, shared_rate, 1.
    refined = replace(parent, config=replace(config, hidden_rule=rule, layer_rate_decay=decay))
    checkpoint = parent.path(seed) / "checkpoint.npz"
    model = refined.model(seed)
    with np.load(checkpoint, allow_pickle=False) as state:
        model.load(state)

    data = rgb_data(split=False)
    pixels = data.train
    start = rgb_sse(model, data, pixels)
    archived_curve = {point["epoch"]: point["squared_error"] for point in historical["history"]}
    np.testing.assert_allclose(start, archived_curve[initial[3]], rtol=1e-6)
    history = [dict(epoch=0, updates=0, squared_error=start)]
    args = SimpleNamespace(seed=seed + 1000, batch=batch, rate=rate,
                           epochs=duration, schedule=parent.schedule)
    for epoch in training_epochs(model, data, args, pixels, epochs):
        if epoch % 5 == 0:
            error = rgb_sse(model, data, pixels)
            history.append(dict(epoch=epoch, updates=epoch * (len(pixels) // batch), squared_error=error))
            if shared_rate is not None and shared_rule != "lion":
                history[-1]["mean_step_multipliers"] = model.take_mean_step_multipliers()
            print(f"Batch {batch}, refinement epoch {epoch}: {error:.2f}", flush=True)
    return dict(batch=batch, seed=seed, checkpoint=str(checkpoint),
                refinement=dict(rule=rule, rate=rate, layer_rate_decay=decay,
                                schedule=parent.schedule, scheduled_epochs=duration,
                                shuffle_seed=args.seed + 1),
                history=history)


def render(rows):
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.5), layout="constrained", sharey=True)
    for ax, key, xlabel in zip(axes, ("epoch", "updates"), ("Refinement epochs", "Minibatch updates"), strict=True):
        for row, color in zip(rows, ("#00767b", "#a34d32"), strict=True):
            history = row["history"]
            ax.plot([p[key] for p in history], [p["squared_error"] for p in history],
                    marker="o", color=color, label=f"Batch {row['batch']}")
        ax.set_xlabel(xlabel)
        ax.grid(axis="y", alpha=.25)
    axes[0].set_ylabel("Summed squared RGB error")
    axes[1].legend(frameon=False)
    fig.savefig(RESULTS / f"{STUDY}.png", dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    historical = report(HISTORICAL)
    with ProcessPoolExecutor(max_workers=2) as pool:
        jobs = [pool.submit(replay, historical, batch, epochs) for batch, epochs in ((64, 10), (128, 20))]
        rows = [job.result() for job in jobs]
    render(rows)
    (RESULTS / f"{STUDY}.json").write_text(json.dumps({"runs": rows}, indent=2, allow_nan=False) + "\n")
