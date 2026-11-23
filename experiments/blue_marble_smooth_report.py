"""Select a smooth Blue Marble fit on validation and score six paired seeds."""
import argparse
import hashlib
import json
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict

import numpy as np

from hebbian_fit.data import IMAGE, rgb_data

from .blue_marble_refinement_report import TEST_SEEDS, summarize_grain
from .blue_marble_report import VALIDATION_SEEDS, bright_psnr, flat_pairs, full_fit, grain_score, render_fit, train_all
from .blue_marble_smooth import REFERENCE, STAGES
from .fixed_capacity import RESULTS, predict


def screen_and_shortlist(data):
    rows = {}
    shortlist = {REFERENCE: None}
    pairs = flat_pairs(data, data.validation)
    for stage in STAGES.values():
        trials = list(dict.fromkeys(stage()))
        for trial in trials:
            row = json.loads((trial.path(1) / "validation.json").read_text())
            if row["validation"] is not None:
                row["validation"]["grain_rgb_levels"] = grain_score(trial, 1, data, pairs)
            rows[trial] = row
        valid = [t for t in trials if rows[t]["validation"]]
        quality = sorted(valid, key=lambda t: rows[t]["validation"]["psnr_db"], reverse=True)
        shortlist.update(dict.fromkeys(quality[:2]))
        eligible = [t for t in quality if rows[t]["validation"]["psnr_db"] >= 19.7]
        smooth = sorted(eligible, key=lambda t: rows[t]["validation"]["grain_rgb_levels"])
        shortlist.update(dict.fromkeys(smooth[:2]))
    return list(rows.values()), list(shortlist)


def select(executor, shortlist, data):
    rows = train_all(executor, shortlist, VALIDATION_SEEDS)
    pairs = flat_pairs(data, data.validation)
    for (trial, seed), row in rows.items():
        row = dict(row)
        if row["validation"]:
            model = trial.model(seed)
            with np.load(trial.path(seed) / "checkpoint.npz") as weights:
                model.load(weights)
            prediction = predict(model, data, data.validation)
            row["validation"] = dict(row["validation"],
                                     grain_rgb_levels=grain_score(trial, seed, data, pairs),
                                     bright_psnr_db=bright_psnr(prediction, data.y[data.validation]))
        rows[trial, seed] = row

    def median(trial, metric):
        return float(np.median([rows[trial, seed]["validation"][metric] for seed in VALIDATION_SEEDS]))

    base_psnr = median(REFERENCE, "psnr_db")
    base_bright = median(REFERENCE, "bright_psnr_db")
    base_grain = median(REFERENCE, "grain_rgb_levels")
    smooth = [t for t in shortlist if t != REFERENCE and all(rows[t, s]["validation"] for s in VALIDATION_SEEDS)]
    similar = [t for t in smooth if median(t, "psnr_db") >= base_psnr - .1
               and median(t, "bright_psnr_db") >= base_bright - .2]
    lower_grain = [t for t in similar if median(t, "grain_rgb_levels") < .92 * base_grain]
    if lower_grain:
        selected = max(lower_grain, key=lambda t: median(t, "psnr_db"))
    elif similar:
        selected = min(similar, key=lambda t: median(t, "grain_rgb_levels"))
    else:
        selected = max(smooth, key=lambda t: median(t, "psnr_db"))
    return list(rows.values()), selected


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    data = rgb_data()
    screen, shortlist = screen_and_shortlist(data)
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        validation, selected = select(executor, shortlist, data)
        print("selected", asdict(selected), flush=True)
        picks = {"reference": REFERENCE, "selected": selected}
        trained = train_all(executor, picks.values(), TEST_SEEDS)
        list(executor.map(full_fit, [(trial, seed, trained[trial, seed]["epoch"])
                                    for trial in picks.values() for seed in TEST_SEEDS]))
    result = {key: summarize_grain(t, [trained[t, seed] for seed in TEST_SEEDS], data) for key, t in picks.items()}
    report = {"image_sha256": hashlib.sha256(IMAGE.read_bytes()).hexdigest(), "screen": screen,
              "validation": validation, "results": result,
              "pair_counts": {name: len(flat_pairs(data, subset)[0])
                              for name, subset in (("validation", data.validation), ("test", data.test))}}
    (RESULTS / "blue-marble-smooth.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    render_fit(selected, result["selected"], data, "blue-marble-smooth.png", "smoothed Hebbian fit")
    for name, row in result.items():
        print(name, row["parameters"], row["medians"], flush=True)


if __name__ == "__main__":
    main()
