"""Continue the clipped eight-layer fit and compare schedules on validation seeds."""
import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace

import numpy as np

from hebbian_fit.__main__ import training_epochs
from .blue_marble_affine import PixelTrial, TEST_SEEDS, VALIDATION_SEEDS, render_convergence, validation_sse
from .blue_marble_report import full_fit, render_fit, rgb_sse, summarize, train_all
from .fixed_capacity import RESULTS, predict


@dataclass(frozen=True)
class Continuation(PixelTrial):
    def path(self, seed):
        return Path("runs/blue-marble-continuation") / super().path(seed).parent.name / str(seed)


@dataclass(frozen=True)
class DepthScaledContinuation(Continuation):
    layer_rate_decay: float = .7

    def model_config(self):
        return replace(super().model_config(), layer_rate_decay=self.layer_rate_decay)


def starting_model(base, trial, seed, name):
    model = trial.model(seed)
    with np.load(base.path(seed) / name) as state:
        starting_state = model.state()
        if trial.width == 2 * base.width:
            # Two copies at half the activity preserve each class sum and Gaussian gate.
            # Opposite perturbations give the copies distinct local learning trajectories.
            rng = np.random.default_rng(seed + 2000)
            for layer in range(base.depth):
                key = f"weight_{layer}"
                original = state[key]
                if layer:
                    original = np.vstack((np.repeat(original[:-1], 2, axis=0), original[-1:]))
                half = original * .5
                variation = rng.normal(0, .025 * half.std(), half.shape).astype(half.dtype)
                starting_state[key] = np.stack((half + variation, half - variation), axis=2).reshape(
                    half.shape[0], 2 * half.shape[1])
        else:
            starting_state.update(state)
        model.load(starting_state)
    return model


def detail_pixels(data):
    """Quarter of the test pixels with the largest color change to a direct neighbor."""
    rgb = data.y.reshape(*data.shape, 3)
    contrast = np.zeros(data.shape)
    for image, scores in ((rgb, contrast), (rgb.swapaxes(0, 1), contrast.T)):
        distance = np.square(image[1:] - image[:-1], dtype=np.float64).sum(axis=-1)
        np.maximum(scores[:-1], distance, out=scores[:-1])
        np.maximum(scores[1:], distance, out=scores[1:])
    order = np.argsort(-contrast.ravel()[data.test], kind="stable")
    return data.test[order[:(len(order) + 3) // 4]]


def run(job):
    base, trial, seed, full = job
    directory = trial.path(seed)
    directory.mkdir(parents=True, exist_ok=True)
    summary = directory / "validation.json"
    name = "full-fit.npz" if full else "checkpoint.npz"
    if (directory / name).exists() and summary.exists():
        return json.loads(summary.read_text())
    data = base.dataset()
    model = starting_model(base, trial, seed, name)
    rng_seed = seed + 1000
    args = SimpleNamespace(seed=rng_seed, batch=trial.batch, rate=trial.rate, epochs=trial.epochs)
    best = rgb_sse(model, data, data.validation) if not full else None
    chosen = 0
    history = []
    state = model.state()
    budget = json.loads(summary.read_text())["epoch"] if full else trial.epochs
    pixels = np.arange(len(data.x)) if full else data.train
    if not full:
        history.append(dict(epoch=0, train_sse=rgb_sse(model, data, pixels), validation_sse=best))
    for epoch in training_epochs(model, data, args, pixels, budget):
        if not full and epoch % 5 == 0:
            validation = rgb_sse(model, data, data.validation)
            history.append(dict(epoch=epoch, train_sse=rgb_sse(model, data, pixels), validation_sse=validation))
            if validation < best:
                best, chosen, state = validation, epoch, model.state()
    with TemporaryDirectory(dir=directory) as tmp:
        checkpoint = Path(tmp) / name
        np.savez_compressed(checkpoint, **(model.state() if full else state))
        checkpoint.replace(directory / name)
        if not full:
            row = dict(seed=seed, epoch=chosen, validation={"sse": best}, history=history)
            tmp_summary = Path(tmp) / summary.name
            tmp_summary.write_text(json.dumps(row, allow_nan=False))
            tmp_summary.replace(summary)
    return json.loads(summary.read_text())


def median_validation(rows):
    return float(np.median([row["validation"]["sse"] for row in rows]))


def compare_lion(base, selected, reference, validation, workers):
    scaled = replace(selected, rule="lion")
    # The ordinary continuation already gives every layer the same rate.
    uniform = Continuation(**{k: v for k, v in asdict(scaled).items() if k != "layer_rate_decay"})
    scratch = replace(base, rule="lion")
    shared_gate = replace(scratch, layer_gates=None, tau=float(np.mean(base.layer_gates)))
    energy = replace(shared_gate, readout="class-energy")
    data = base.dataset()
    seeds = (*VALIDATION_SEEDS, *TEST_SEEDS)
    candidates = {"lion_uniform": uniform, "lion_scaled": scaled, "lion_from_start": scratch}
    metrics = {"memoryless_scaled": reference}
    traces = {"memoryless_scaled": validation}
    with ProcessPoolExecutor(max_workers=workers) as pool:
        # Control for the smaller average step as well as for the largest step.
        mean_rate = float(scaled.rate * np.mean(scaled.model_config().rate_multipliers))
        mean_matched = replace(uniform, rate=mean_rate)
        starts = {}
        for label, trial in (("uniform", uniform), ("mean_matched", mean_matched)):
            starts[label] = list(pool.map(run, [(base, trial, s, False) for s in VALIDATION_SEEDS]))
        if median_validation(starts["mean_matched"]) < median_validation(starts["uniform"]):
            candidates["lion_uniform_mean"] = mean_matched
        gate_scores = train_all(pool, [scratch, shared_gate, energy], VALIDATION_SEEDS)
        gate_screen = {label: [dict(seed=s, sse=validation_sse(gate_scores[trial, s], data))
                               for s in VALIDATION_SEEDS]
                       for label, trial in (("tapered", scratch), ("shared", shared_gate), ("shared_energy", energy))}
        gate_medians = {label: np.median([r["sse"] for r in rows]) for label, rows in gate_screen.items()}
        if gate_medians["shared"] < gate_medians["tapered"]:
            candidates["lion_shared_gate_from_start"] = shared_gate
        if gate_medians["shared_energy"] < gate_medians["shared"]:
            candidates["lion_energy_from_start"] = energy
        for label, trial in (("shared", shared_gate), ("shared_energy", energy)):
            for row in gate_screen[label]:
                model = trial.model(row["seed"])
                row["initial_validation_sse"] = rgb_sse(model, data, data.validation)
                with np.load(trial.path(row["seed"]) / "checkpoint.npz") as state:
                    model.load(state)
                rgb = (predict(model, data, data.validation) + 1) / 2
                row["validation_rgb_range"] = [float(rgb.min()), float(rgb.max())]
        for label, trial in candidates.items():
            if not isinstance(trial, Continuation):
                scores = train_all(pool, [trial], seeds)
                rows = [dict(scores[trial, s]) for s in seeds]
                for row in rows:
                    row["validation"] = {"sse": validation_sse(row, data)}
                list(pool.map(full_fit, [(trial, row["seed"], row["epoch"]) for row in rows[len(VALIDATION_SEEDS):]]))
            else:
                rows = list(pool.map(run, [(base, trial, s, False) for s in seeds]))
                list(pool.map(run, [(base, trial, s, True) for s in TEST_SEEDS]))
            traces[label] = rows[:len(VALIDATION_SEEDS)]
            metrics[label] = summarize(trial, rows[len(VALIDATION_SEEDS):], squared_error_only=True)
    best = min(candidates, key=lambda label: median_validation(traces[label]))
    chosen = candidates[best]
    detail = detail_pixels(data)
    for row in metrics[best]["runs"]:
        seed = row["seed"]
        initial = (starting_model(base, chosen, seed, "checkpoint.npz") if isinstance(chosen, Continuation)
                   else chosen.model(seed))
        trained = chosen.model(seed)
        with np.load(chosen.path(seed) / "checkpoint.npz") as state:
            trained.load(state)
        row["updated_layers"] = [bool(np.any(x != y)) for x, y in zip(initial.weights, trained.weights)]
        row["detail_test_sse"] = rgb_sse(trained, data, detail)
    report = {"phase_one": asdict(base), "selected_lion": best, "results": metrics, "validation": traces,
              "uniform_rate_screen": {"final_layer_rate": uniform.rate, "mean_layer_rate": mean_rate,
                                      "runs": starts},
              "scratch_gate_screen": {"shared_width": shared_gate.tau, "runs": gate_screen}}
    (RESULTS / "blue-marble-lion.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    plain = "lion_uniform_mean" if "lion_uniform_mean" in metrics else "lion_uniform"
    conditions = {"selected": plain, "reference": "lion_scaled"}
    render_convergence({k: metrics[v] for k, v in conditions.items()}, {k: traces[v] for k, v in conditions.items()},
                       "blue-marble-lion-convergence.png", xlabel="Additional training epochs",
                       labels={"selected": f"Lion · equal steps ({candidates[plain].rate:.5f})",
                               "reference": "Lion · smaller earlier steps"})
    render_fit(candidates[best], metrics[best], data, "blue-marble-lion.png", "Lion-style Hebbian fit")
    for label, result in metrics.items():
        print(label, result["medians"], flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=9)
    parser.add_argument("--compare-lion", action="store_true",
                        help="Repeat the selected second-phase schedule with the Lion-style Hebbian rule")
    args = parser.parse_args()
    prior = json.loads((RESULTS / "blue-marble-step-size.json").read_text())
    settings = dict(prior["results"]["selected"]["trial"])
    settings["layer_gates"] = tuple(settings["layer_gates"])
    base = PixelTrial(**settings)
    candidates = []
    for rate, batch, scale in ((.0008, 128, 1.), (.0015, 64, 1.), (.0015, 64, .85)):
        cfg = dict(settings, rate=rate, batch=batch, epochs=100,
                   layer_gates=tuple(round(w * scale, 5) for w in base.layer_gates))
        candidates.append(Continuation(**cfg))
    deeper = dict(settings, depth=12, rate=.0015, batch=64, epochs=100,
                  layer_gates=base.layer_gates + (.10,) * 4)
    candidates.append(Continuation(**deeper))
    candidates.append(Continuation(**dict(settings, width=2 * base.width, rate=.001, epochs=80)))
    for rate, batch in ((.0008, 128), (.0015, 64)):
        candidates.append(DepthScaledContinuation(**dict(settings, rate=rate, batch=batch, epochs=100)))
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        # A fresh checkout contains the published settings but no local checkpoints.
        train_all(pool, [base], VALIDATION_SEEDS)
        jobs = [(base, t, s, False) for t in candidates for s in VALIDATION_SEEDS]
        keys = [(trial, seed) for _, trial, seed, _ in jobs]
        rows = dict(zip(keys, pool.map(run, jobs), strict=True))

        def criterion(trial):
            return median_validation(rows[trial, seed] for seed in VALIDATION_SEEDS)

        best = min(candidates, key=criterion)
        print("Validation", [(t.depth, t.width, t.rate, criterion(t)) for t in candidates], flush=True)
        parents = train_all(pool, [base], TEST_SEEDS)
        list(pool.map(full_fit, [(base, s, parents[base, s]["epoch"]) for s in TEST_SEEDS]))
        tests = list(pool.map(run, [(base, best, seed, False) for seed in TEST_SEEDS]))
        list(pool.map(run, [(base, best, seed, True) for seed in TEST_SEEDS]))
    result = summarize(best, tests, squared_error_only=True)
    data = base.dataset()
    detail = detail_pixels(data)
    for row in result["runs"]:
        seed = row["seed"]
        parent = json.loads((base.path(seed) / "validation.json").read_text())
        row["parent_epoch"] = parent["epoch"]
        initial = starting_model(base, best, seed, "checkpoint.npz").weights
        with np.load(best.path(seed) / "checkpoint.npz") as trained:
            row["updated_layers"] = [
                bool(np.any(weight != trained[f"weight_{i}"])) for i, weight in enumerate(initial)
            ]
        for label, trial in (("reference", base), ("selected", best)):
            model = trial.model(seed)
            with np.load(trial.path(seed) / "checkpoint.npz") as trained:
                model.load(trained)
            row.setdefault("detail_test_sse", {})[label] = rgb_sse(model, data, detail)
    report = dict(image_sha256=prior["image_sha256"], detail_test_pixels=len(detail),
                  reference=prior["results"]["selected"], selected=result,
                  validation=[dict(settings=asdict(t), runs=[rows[t, s] for s in VALIDATION_SEEDS])
                              for t in candidates])
    (RESULTS / "blue-marble-continuation.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    render_fit(best, result, data, "blue-marble-continuation.png", "continued affine Hebbian fit")
    curves = {"selected": best}
    if isinstance(best, DepthScaledContinuation):
        curves["uniform"] = next(t for t in candidates if type(t) is Continuation
                                 and (t.rate, t.batch, t.layer_gates) == (best.rate, best.batch, best.layer_gates))
    render_convergence({name: {"trial": asdict(t)} for name, t in curves.items()},
                       {name: [rows[t, s] for s in VALIDATION_SEEDS] for name, t in curves.items()},
                       "blue-marble-continuation-convergence.png", xlabel="Additional training epochs")
    print(result["medians"], flush=True)
    if args.compare_lion:
        compare_lion(base, best, result, [rows[best, s] for s in VALIDATION_SEEDS], args.workers)


if __name__ == "__main__":
    main()
