"""Select Blue Marble settings on validation and publish paired image fits."""
import argparse
import hashlib
import json
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, replace
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace

import numpy as np

from hebbian_fit.__main__ import evaluate, fit
from hebbian_fit.data import IMAGE, rgb_data
from hebbian_fit.model import output_codes

from .blue_marble import REFERENCE, STAGES
from .fixed_capacity import RESULTS, predict, train


VALIDATION_SEEDS = (1, 2, 3)
TEST_SEEDS = (4, 5, 6)


def validation_screen():
    rows = {}
    finalists = {REFERENCE: None}
    for stage in STAGES.values():
        trials = list(dict.fromkeys(stage()))
        for trial in trials:
            rows[trial] = json.loads((trial.path(1) / "validation.json").read_text())
        ranked = sorted((trial for trial in trials if rows[trial]["validation"]),
                        key=lambda trial: rows[trial]["validation"]["psnr_db"], reverse=True)
        finalists.update(dict.fromkeys(ranked[:3]))
    return rows, list(finalists)


def train_all(executor, trials, seeds):
    jobs = [(trial, seed) for trial in trials for seed in seeds]
    return dict(zip(jobs, executor.map(train, jobs), strict=True))


def choose(executor, finalists):
    scores = train_all(executor, finalists, VALIDATION_SEEDS)

    def median_psnr(trial):
        return np.median([scores[trial, seed]["validation"]["psnr_db"] for seed in VALIDATION_SEEDS])

    valid = [trial for trial in finalists if all(scores[trial, seed]["validation"] for seed in VALIDATION_SEEDS)]
    best = max(valid, key=median_psnr)
    deep = max((trial for trial in valid if trial.depth > 1), key=median_psnr)
    return list(scores.values()), {"reference": REFERENCE, "selected": best, "deep": deep}


def bright_psnr(pred, target):
    bright = np.max((target + 1) / 2, axis=1) > .1
    error = (np.clip(pred[bright], -1, 1) - target[bright]) / 2
    return float(-10 * np.log10(np.mean(np.square(error, dtype=np.float64))))


def full_fit(job):
    trial, seed, epoch = job
    model = trial.model(seed)
    data = trial.dataset()
    path = trial.path(seed) / "full-fit.npz"
    if path.exists():
        with np.load(path) as state:
            model.load(state)
        return
    args = SimpleNamespace(task="image", epochs=trial.epochs, batch=trial.batch, rate=trial.rate, seed=seed)
    fit(model, data, args, np.arange(len(data.x)), epochs=epoch)
    with TemporaryDirectory(dir=path.parent) as directory:
        checkpoint = Path(directory) / path.name
        np.savez_compressed(checkpoint, **model.state())
        checkpoint.replace(path)


def rgb_sse(model, data, indices):
    residual = (predict(model, data, indices) - data.y[indices]) / 2
    return float(np.sum(np.square(residual, dtype=np.float64)))


def measure(trial, row, *, squared_error_only=False):
    seed = row["seed"]
    data = trial.dataset()
    model = trial.model(seed)
    initial = model.state()
    path = trial.path(seed)
    with np.load(path / "checkpoint.npz") as state:
        assert all(np.any(state[key] != initial[key]) for key in initial if key.startswith("weight_"))
        model.load(state)
    if squared_error_only:
        result = {"test_sse": rgb_sse(model, data, data.test)}
    else:
        metrics, pred = evaluate(model, data, data.test, "image")
        result = {"test_psnr_db": metrics["psnr_db"],
                  "test_rgb_mae": float(np.mean(metrics["channel_mae"])),
                  "bright_test_psnr_db": bright_psnr(pred, data.y[data.test])}
    full_path = path / "full-fit.npz"
    if full_path.exists():
        with np.load(full_path) as state:
            assert all(np.any(state[key] != initial[key]) for key in initial if key.startswith("weight_"))
            model.load(state)
        pixels = np.arange(len(data.x))
        if squared_error_only:
            result["full_fit_sse"] = rgb_sse(model, data, pixels)
        else:
            full, _ = evaluate(model, data, pixels, "image")
            result["full_fit_psnr_db"] = full["psnr_db"]
    return dict(row, test=result)


def summarize(trial, rows, *, squared_error_only=False):
    runs = [measure(trial, row, squared_error_only=squared_error_only) for row in rows]
    medians = {key: float(np.median([run["test"][key] for run in runs])) for key in runs[0]["test"]}
    return dict(trial=asdict(trial), parameters=sum(w.size for w in trial.model(1).weights),
                medians=medians, runs=runs)


def flat_pairs(data, subset):
    """Find bright, nearly uniform target color pairs within two pixels of each other."""
    indices = np.arange(len(data.x)).reshape(data.shape)
    rgb = output_codes(data.y, 1., 127.5)
    allowed = np.zeros(len(rgb), dtype=bool)
    allowed[subset] = True
    allowed &= rgb.max(axis=1) > 25
    height, width = data.shape
    left, right = [], []
    for dy in range(3):
        for dx in (range(1, 3) if dy == 0 else range(-2, 3)):
            a = indices[:height - dy, max(0, -dx):min(width, width - dx)].ravel()
            b = indices[dy:, max(0, dx):min(width, width + dx)].ravel()
            keep = allowed[a] & allowed[b] & (np.abs(rgb[a] - rgb[b]).max(axis=1) <= 8)
            left.append(a[keep])
            right.append(b[keep])
    return np.concatenate(left), np.concatenate(right)


def grain_score(trial, seed, data, pairs):
    a, b = pairs
    indices, inverse = np.unique(np.concatenate((a, b)), return_inverse=True)
    model = trial.model(seed)
    with np.load(trial.path(seed) / "checkpoint.npz") as weights:
        model.load(weights)
    error = 127.5 * (np.clip(predict(model, data, indices), -1, 1) - data.y[indices])
    differences = error[inverse[:len(a)]] - error[inverse[len(a):]]
    return float(np.sqrt(np.mean(np.square(differences, dtype=np.float64))))


def render_fit(trial, result, data, filename, label):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    run = result["runs"][1]
    seed = run["seed"]
    indices = np.arange(len(data.x))
    target = (data.y.reshape(*data.shape, 3) + 1) / 2
    model = trial.model(seed)
    images = [target]
    for checkpoint in ("checkpoint.npz", "full-fit.npz"):
        with np.load(trial.path(seed) / checkpoint) as state:
            model.load(state)
        images.append((np.clip(predict(model, data, indices), -1, 1).reshape(target.shape) + 1) / 2)
    scores = run["test"]
    if "test_sse" in scores:
        titles = ("NASA target", f"Split fit\nHeld-out squared error {scores['test_sse']:.1f}",
                  f"All-pixel fit\nIn-sample squared error {scores['full_fit_sse']:.1f}")
    else:
        titles = ("NASA target", f"Split fit · {scores['test_psnr_db']:.2f} dB held out",
                  f"All-pixel fit · {scores['full_fit_psnr_db']:.2f} dB in sample")
    fig, axes = plt.subplots(2, 3, figsize=(8.6, 5.8), layout="constrained")
    for col, (image, title) in enumerate(zip(images, titles)):
        axes[0, col].imshow(image, interpolation="nearest")
        axes[0, col].set_title(title, fontsize=10)
        axes[1, col].imshow(image[42:86, 43:87], interpolation="nearest")
        axes[1, col].set_title(("Target detail", "Split-fit detail", "All-pixel detail")[col], fontsize=9)
    for ax in axes.flat:
        ax.axis("off")
    fig.suptitle(f"Blue Marble · {label} · seed {seed}", fontsize=13)
    fig.supxlabel("Source photograph AS17-148-22727 · NASA Johnson Space Center", fontsize=9)
    fig.savefig(RESULTS / filename, dpi=160)
    plt.close(fig)


def render(picks, results, filename="blue-marble.png"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    seed = TEST_SEEDS[1]
    data = rgb_data()
    target = (data.y.reshape(128, 128, 3) + 1) / 2
    indices = np.arange(len(data.x))

    def image(trial, name):
        model = trial.model(seed)
        with np.load(trial.path(seed) / name) as state:
            model.load(state)
        return (np.clip(predict(model, data, indices), -1, 1).reshape(target.shape) + 1) / 2

    reference = image(picks["reference"], "checkpoint.npz")
    selected = image(picks["selected"], "checkpoint.npz")
    full = image(picks["selected"], "full-fit.npz")
    err = np.mean(np.abs(selected - target), axis=2)
    zoom = np.s_[42:86, 43:87]
    fig, axes = plt.subplots(2, 4, figsize=(10.4, 5.55), layout="constrained")
    best = results["selected"]["runs"][1]["test"]
    ref = results["reference"]["runs"][1]["test"]
    panels = [(target, "NASA target"),
              (reference, f"Reference · split fit\n{ref['test_psnr_db']:.2f} dB on test pixels"),
              (selected, f"Selected · split fit\n{best['test_psnr_db']:.2f} dB on test pixels"),
              (full, f"Selected · all-pixel fit\n{best['full_fit_psnr_db']:.2f} dB in sample")]
    for ax, (values, label) in zip(axes[0], panels):
        ax.imshow(values, interpolation="nearest")
        ax.set_title(label, fontsize=10)
        ax.axis("off")
    for ax, (values, label) in zip(axes[1, :3], [(target, "Target detail"),
                                                  (selected, "Split-fit detail"),
                                                  (full, "All-pixel fit detail")]):
        ax.imshow(values[zoom], interpolation="nearest")
        ax.set_title(label, fontsize=10)
        ax.axis("off")
    heat = axes[1, 3].imshow(err, cmap="magma", vmin=0, vmax=.4, interpolation="nearest")
    axes[1, 3].set_title("Split-fit mean RGB error", fontsize=10)
    axes[1, 3].axis("off")
    fig.colorbar(heat, ax=axes[1, 3], fraction=.045, pad=.045)
    fig.suptitle("Apollo 17 Blue Marble · fixed-sum Hebbian networks · seed 5", fontsize=13)
    fig.supxlabel("Source photograph AS17-148-22727 · NASA Johnson Space Center", fontsize=9)
    fig.savefig(RESULTS / filename, dpi=155)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    screen, finalists = validation_screen()
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        validation, picks = choose(executor, finalists)
        matched = {f"width/{6 * width}": replace(REFERENCE, width=width) for width in (8, 32, 64, 128)}
        picks.update(matched)
        for name, trial in picks.items():
            print(name, asdict(trial), flush=True)
        rows = train_all(executor, dict.fromkeys(picks.values()), TEST_SEEDS)
        full_models = dict.fromkeys((picks["reference"], picks["selected"]))
        list(executor.map(full_fit, [(trial, seed, rows[trial, seed]["epoch"])
                                    for trial in full_models for seed in TEST_SEEDS]))
    results = {name: summarize(trial, [rows[trial, seed] for seed in TEST_SEEDS])
               for name, trial in picks.items()}
    RESULTS.mkdir(parents=True, exist_ok=True)
    report = {"image_sha256": hashlib.sha256(IMAGE.read_bytes()).hexdigest(),
              "screen": list(screen.values()), "confirmation": validation, "results": results}
    (RESULTS / "blue-marble.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    render(picks, results)
    for name, result in results.items():
        print(name, result["parameters"], result["medians"], flush=True)


if __name__ == "__main__":
    main()
