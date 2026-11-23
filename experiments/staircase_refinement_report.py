"""Select on three validation seeds and evaluate on six fresh paired seeds."""
import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, replace

import numpy as np

from hebbian_fit.data import BAND_EDGES, EDGES, HEIGHTS, scalar_data
from .fixed_capacity import RESULTS, train
from .fixed_capacity_report import curve, summarize
from .staircase_refinement import BASE, STAGES


VALIDATION_SEEDS = (1, 2, 3)
TEST_SEEDS = tuple(range(7, 13))
SINE_BASE = replace(BASE, task="sine", width=32, depth=1, batch=32, rate=.001, epochs=120)


def load_screen():
    rows = {}
    shortlist = set()
    for stage in STAGES.values():
        trials = list(dict.fromkeys(stage()))
        for trial in trials:
            path = trial.path(1) / "validation.json"
            rows[trial] = dict(json.loads(path.read_text()), trial=asdict(trial))
        steps = [trial for trial in trials if trial.task == "steps" and rows[trial]["validation"]]
        shortlist.update(sorted(steps, key=lambda trial: rows[trial]["validation"]["rmse"])[:2])
        candidates = [trial for trial in steps if rows[trial]["validation"]["rmse"] < .36]
        shortlist.update(sorted(candidates, key=lambda trial: rows[trial]["validation"]["fine_rmse"])[:2])
        sine = [trial for trial in trials if trial.task == "sine" and rows[trial]["validation"]]
        shortlist.update(sorted(sine, key=lambda trial: rows[trial]["validation"]["rmse"])[:2])
    shortlist.update((BASE, SINE_BASE))
    return rows, [trial for trial in (*rows, BASE) if trial in shortlist]


def train_all(executor, trials, seeds):
    jobs = [(trial, seed) for trial in trials for seed in seeds]
    return {job: dict(row, trial=asdict(job[0]))
            for job, row in zip(jobs, executor.map(train, jobs), strict=True)}


def median_validation(rows, trial, metric):
    scores = [rows[trial, seed]["validation"] for seed in VALIDATION_SEEDS]
    return float(np.median([row[metric] for row in scores])) if all(scores) else np.inf


def select(executor, shortlist):
    rows = train_all(executor, shortlist, VALIDATION_SEEDS)
    valid = [trial for trial in shortlist if all(rows[trial, seed]["validation"] for seed in VALIDATION_SEEDS)]
    steps = [trial for trial in valid if trial.task == "steps"]
    best = min(steps, key=lambda trial: median_validation(rows, trial, "rmse"))
    fine_candidates = [trial for trial in steps if median_validation(rows, trial, "rmse") < .36]
    fine = min(fine_candidates, key=lambda trial: median_validation(rows, trial, "fine_rmse"))
    activity = min((trial for trial in steps if trial.rule == "activity-sign"),
                   key=lambda trial: median_validation(rows, trial, "rmse"))
    sine = min((trial for trial in valid if trial.task == "sine"),
               key=lambda trial: median_validation(rows, trial, "rmse"))
    picks = dict(baseline=BASE, overall=best, fine=fine, activity=activity,
                 activity_control=replace(activity, rule="signsgd"), sine_baseline=SINE_BASE, sine=sine)
    return list(rows.values()), picks


def render(picks):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    steps = scalar_data("steps")
    x = steps.x[steps.test, 0]
    fig, axes = plt.subplots(3, 1, figsize=(10.2, 8.6), layout="constrained")
    for ax in axes[:2]:
        ax.stairs(HEIGHTS, EDGES, baseline=None, color="#273241", lw=1, label="Target")
    smooth = np.linspace(0, 1, 800)
    axes[0].plot(smooth, .75 * np.cos(2 * np.pi * smooth), color="#273241", lw=1)
    series = [("baseline", "Previous model", "#a26a3e"), ("overall", "Overall selection", "#2865a2")]
    if picks["fine"] != picks["overall"]:
        series.append(("fine", "Fine-step selection", "#38775e"))
    fine = (x >= BAND_EDGES[-2]) & (x < 0)
    for name, label, color in series:
        pred = curve(picks[name], TEST_SEEDS[2], steps)
        axes[0].plot(x, pred, color=color, lw=.95, label=label)
        axes[1].plot(x[fine], pred[fine], color=color, lw=.95, label=label)
    axes[0].set(xlim=(-1, 1), title="Held-out staircase · seed 9", ylabel="Output")
    axes[1].set(xlim=(BAND_EDGES[-2], 0), title="The 64 narrowest plateaus", ylabel="Output")
    sine = scalar_data("sine")
    sine_x = sine.x[sine.test, 0]
    axes[2].plot(sine_x, sine.y[sine.test, 0], color="#273241", lw=1, label="Target")
    for name, label, color in (("sine_baseline", "Previous model", "#a26a3e"),
                               ("sine", "Validation selection", "#2865a2")):
        axes[2].plot(sine_x, curve(picks[name], TEST_SEEDS[2], sine), color=color, lw=.95, label=label)
    axes[2].set(xlim=(-1, 1), title="Held-out sine · seed 9", xlabel="Input", ylabel="Output")
    for ax in axes:
        ax.legend(loc="upper right", ncols=2, fontsize=8)
    fig.savefig(RESULTS / "staircase-refinement.png", dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    screen, shortlist = load_screen()
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        validation, picks = select(executor, shortlist)
        for name, trial in picks.items():
            print(name, asdict(trial), flush=True)
        if args.validate_only:
            return
        trained = train_all(executor, dict.fromkeys(picks.values()), TEST_SEEDS)
    results = {name: summarize(trial, [trained[trial, seed] for seed in TEST_SEEDS])
               for name, trial in picks.items()}
    report = dict(screen=list(screen.values()), confirmation=validation, results=results)
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "staircase-refinement.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    render(picks)
    for name, result in results.items():
        print(name, result["parameters"], result["medians"], flush=True)


if __name__ == "__main__":
    main()
