"""Confirm pixel-level teaching on Blue Marble and compare matched controls."""
import argparse
import hashlib
import json
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict

import numpy as np

from hebbian_fit.data import IMAGE
from hebbian_fit.model import output_codes

from .blue_marble import ImageTrial
from .blue_marble_refinement import BASE, STAGES, BoundedImageTrial, PixelTrial
from .blue_marble_report import VALIDATION_SEEDS, flat_pairs, full_fit, grain_score, render, summarize, train_all
from .fixed_capacity import RESULTS, predict


TEST_SEEDS = tuple(range(4, 10))


def screen_rows():
    rows = {}
    shortlist = {BASE: None}
    for stage in STAGES.values():
        trials = list(dict.fromkeys(stage()))
        for trial in trials:
            path = trial.path(1) / "validation.json"
            rows[trial] = json.loads(path.read_text())
        ranked = sorted((t for t in trials if rows[t]["validation"]),
                        key=lambda t: rows[t]["validation"]["psnr_db"], reverse=True)
        shortlist.update(dict.fromkeys(ranked[:3]))
    return list(rows.values()), list(shortlist)


def choose(executor, shortlist):
    confirmation = train_all(executor, shortlist, VALIDATION_SEEDS)

    def median_psnr(trial):
        values = [confirmation[trial, seed]["validation"] for seed in VALIDATION_SEEDS]
        return float(np.median([v["psnr_db"] for v in values])) if all(values) else -np.inf

    selected = max((t for t in shortlist if isinstance(t, PixelTrial)), key=median_psnr)
    shared = asdict(selected)
    shared.pop("output_levels")
    clipped_control = BoundedImageTrial(**shared)
    shared.pop("output_bound")
    continuous_control = ImageTrial(**shared)
    picks = {"reference": BASE, "selected": selected, "clipped_control": clipped_control,
             "continuous_control": continuous_control}
    return list(confirmation.values()), picks


def summarize_pixels(trial, rows):
    result = summarize(trial, rows)
    data = trial.dataset()
    target_codes = output_codes(data.y[data.test], 1., 127.5)
    for run in result["runs"]:
        seed = run["seed"]
        model = trial.model(seed)
        with np.load(trial.path(seed) / "checkpoint.npz") as checkpoint:
            model.load(checkpoint)
        prediction_codes = output_codes(predict(model, data, data.test), 1., 127.5)
        matches = prediction_codes == target_codes
        run["test"]["exact_channel_fraction"] = float(matches.mean())
        run["test"]["exact_pixel_fraction"] = float(matches.all(axis=1).mean())
    for metric in ("exact_channel_fraction", "exact_pixel_fraction"):
        result["medians"][metric] = float(np.median([r["test"][metric] for r in result["runs"]]))
    return result


def summarize_grain(trial, rows, data):
    result = summarize_pixels(trial, rows)
    pairs = flat_pairs(data, data.test)
    for run in result["runs"]:
        run["test"]["grain_rgb_levels"] = grain_score(trial, run["seed"], data, pairs)
    result["medians"]["grain_rgb_levels"] = float(np.median(
        [run["test"]["grain_rgb_levels"] for run in result["runs"]]))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    screen, shortlist = screen_rows()
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        confirmation, picks = choose(executor, shortlist)
        for label, trial in picks.items():
            print(label, asdict(trial), flush=True)
        rows = train_all(executor, dict.fromkeys(picks.values()), TEST_SEEDS)
        full_models = (picks["reference"], picks["selected"])
        list(executor.map(full_fit, [(trial, seed, rows[trial, seed]["epoch"])
                                    for trial in full_models for seed in TEST_SEEDS]))
    result = {name: summarize_pixels(t, [rows[t, seed] for seed in TEST_SEEDS]) for name, t in picks.items()}
    report = {"source_sha256": hashlib.sha256(IMAGE.read_bytes()).hexdigest(),
              "screen": screen, "validation": confirmation, "results": result}
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "blue-marble-refinement.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    render(picks, result, filename="blue-marble-refinement.png")
    for name, row in result.items():
        print(name, row["parameters"], row["medians"], flush=True)


if __name__ == "__main__":
    main()
