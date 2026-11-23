"""Measure width and depth of dense, fixed-pool Hebbian regressors."""
import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace

import numpy as np

from hebbian_fit.__main__ import evaluate, training_epochs
from hebbian_fit.data import BAND_EDGES, scalar_data, step_metrics
from hebbian_fit.model import ModelConfig, NumpyModel


RUNS = Path("runs/fixed-capacity")
RESULTS = Path("assets/results")


@dataclass(frozen=True)
class Trial:
    task: str = "steps"
    width: int = 32  # per class
    depth: int = 1
    rate: float = .1
    tau: float = .1
    connection: str = "discontinuous"
    competition: str = "class"
    readout: str = "class-sum"
    readout_layers: str = "last"
    initialization: str = "normal"
    unit_norm_updates: bool = False
    gate: str = "gaussian"
    winners: int = 8
    sine_omega: float | None = None
    epochs: int = 60
    batch: int = 256

    def model(self, seed):
        return NumpyModel(self.model_config(), seed)

    def dataset(self):
        return scalar_data(self.task)

    def model_config(self):
        return ModelConfig(width=self.width, depth=self.depth, connection=self.connection,
                           competition=self.competition, gate=self.gate, winners=self.winners,
                           gate_width=self.tau, readout=self.readout, hidden_rule="signsgd",
                           readout_layers=self.readout_layers, initialization=self.initialization,
                           unit_norm_updates=self.unit_norm_updates, sine_omega=self.sine_omega)

    def checkpoint_score(self, metrics):
        if self.task == "steps":
            return max(metrics["smooth_rmse"] / .05, metrics["worst_plateau_mae"] / .15)
        return metrics["rmse"]

    def path(self, seed):
        name = "_".join(str(v) for v in asdict(self).values())
        return RUNS / name / str(seed)


def rmse(values):
    return float(np.sqrt(np.mean(np.square(values, dtype=np.float64))))


def predict(model, data, indices):
    return np.concatenate([model.predict_values(data.x[indices[start:start + 256]])
                           for start in range(0, len(indices), 256)])


def score(model, data, indices, task):
    if task == "image":
        return evaluate(model, data, indices, task)[0]
    error = predict(model, data, indices) - data.y[indices]
    result = {"rmse": rmse(error)}
    if task == "steps":
        x = data.x[indices, 0]
        detail = step_metrics(x, error)
        result.update(smooth_rmse=detail["smooth_rmse"], worst_plateau_mae=detail["max_plateau_mae"],
                      fine_rmse=rmse(error[(x >= BAND_EDGES[-2]) & (x < 0)]))
    return result


def train(job):
    trial, seed = job
    path = trial.path(seed)
    report = path / "validation.json"
    if report.exists():
        return json.loads(report.read_text())
    data = trial.dataset()
    model = trial.model(seed)
    initial = model.state()
    args = SimpleNamespace(seed=seed, rate=trial.rate, batch=trial.batch, epochs=trial.epochs)
    best = np.inf
    chosen = None
    for epoch in training_epochs(model, data, args, data.train, trial.epochs):
        if epoch % 5 != 0:
            continue
        metrics = score(model, data, data.validation, trial.task)
        if not np.isfinite(metrics["rmse"]):
            break
        criterion = trial.checkpoint_score(metrics)
        if criterion < best:
            best = criterion
            chosen = dict(trial=asdict(trial), seed=seed, epoch=epoch, validation=metrics)
            state = model.state()
    path.mkdir(parents=True, exist_ok=True)
    if chosen is None:
        chosen = dict(trial=asdict(trial), seed=seed, epoch=None, validation=None)
    else:
        chosen["updated_layers"] = [
            bool(np.any(state[k] != initial[k])) for k in initial if k.startswith("weight_")
        ]
    with TemporaryDirectory(dir=path) as temporary:
        work = Path(temporary)
        if chosen["validation"] is not None:
            checkpoint = work / "checkpoint.npz"
            np.savez_compressed(checkpoint, **state)
            checkpoint.replace(path / "checkpoint.npz")
        metadata = work / report.name
        metadata.write_text(json.dumps(chosen, allow_nan=False) + "\n")
        metadata.replace(report)
    return chosen


def width_trials():
    for width in (32, 128, 512):
        for rate in (.003, .01, .03, .1, .3):
            yield Trial(width=width, rate=rate)
        for tau in (.035, .25, .5):
            for pool in ("class", "layer"):
                for rate in (.01, .1):
                    yield Trial(width=width, tau=tau, competition=pool, rate=rate)


def depth_trials():
    for width, depth in ((32, 2), (128, 2), (32, 3), (64, 3)):
        for pool in ("class", "layer"):
            for heads in ("last", "all"):
                for rate in (.003, .03, .1):
                    yield Trial(width=width, depth=depth, competition=pool,
                                readout_layers=heads, rate=rate, tau=.15)


def variants():
    for width in (32, 128, 512):
        for rate in (.003, .03, .1):
            yield Trial(width=width, rate=rate, gate="hard", winners=4, readout="fixed")
            yield Trial(width=width, rate=rate, gate="none")
            for omega in (4., 16., 64.):
                yield Trial(width=width, rate=rate, sine_omega=omega, tau=.15,
                            connection="affine", initialization="unit-sphere")
    for width in (32, 128, 512):
        for rate in (.001, .01, .1):
            for omega in (None, 4.):
                yield Trial(task="sine", width=width, rate=rate, sine_omega=omega, batch=64,
                            tau=.15, connection="affine", readout="class-sum")
                yield Trial(task="sine", width=width, depth=2, rate=rate, sine_omega=omega, batch=64,
                            tau=.15, connection="affine", readout_layers="all")


def refine():
    for width in (32, 128, 512):
        for tau in (.10, .175, .25):
            for rate in (.003, .03, .1):
                yield Trial(width=width, tau=tau, rate=rate, competition="layer")
        for rate in (.01, .1, .3):
            yield Trial(width=width, rate=rate, competition="layer", tau=.10,
                        connection="affine")
            yield Trial(width=width, rate=rate, competition="layer", tau=.10,
                        connection="continuous")
        for rate in (.003, .03):
            for omega in (1., 2., 8.):
                yield Trial(task="sine", width=width, rate=rate, sine_omega=omega, batch=64,
                            tau=.15, connection="affine", competition="layer")


def schedule_trials():
    # Smaller batches give the local rule more updates per pass over the data.
    for width in (32, 128, 512):
        for rate in (.001, .01, .03):
            yield Trial(width=width, rate=rate, competition="layer", tau=.25, batch=64)
    for width in (32, 64, 128):
        for rate in (.001, .01, .03):
            yield Trial(width=width, depth=2, rate=rate, competition="layer", tau=.15,
                        readout_layers="all", batch=64)
    for width in (32, 128, 512):
        for rate in (.001, .01, .03):
            yield Trial(task="sine", width=width, rate=rate, competition="layer", tau=.15,
                        batch=32, epochs=120)
            yield Trial(task="sine", width=width, rate=rate, competition="layer", tau=.15,
                        batch=32, epochs=120, connection="affine")


def normalized_trials():
    for width in (32, 128, 512):
        for rate in (.003, .03):
            yield Trial(width=width, rate=rate, competition="layer", readout="fixed", tau=.5)
            yield Trial(width=width, rate=rate, competition="layer", readout="fixed",
                        gate="hard", winners=16)
    for width in (32, 128):
        for rate in (.003, .03):
            yield Trial(width=width, depth=2, competition="layer", rate=rate,
                        initialization="unit-sphere", tau=.25, readout_layers="all")


def screen(trials, workers):
    with ProcessPoolExecutor(max_workers=workers) as executor:
        for row in executor.map(train, [(t, 1) for t in trials]):
            metrics = row["validation"]
            trial = row["trial"]
            print(trial["task"], trial["width"], trial["depth"], trial["rate"],
                  trial["tau"], trial["gate"], trial["competition"], trial["sine_omega"],
                  round(metrics["rmse"], 5) if metrics else None, flush=True)


STAGES = {"width": width_trials, "depth": depth_trials, "variants": variants,
          "refine": refine, "schedule": schedule_trials, "normalized": normalized_trials}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=tuple(STAGES))
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    screen(STAGES[args.stage](), args.workers)


if __name__ == "__main__":
    main()
