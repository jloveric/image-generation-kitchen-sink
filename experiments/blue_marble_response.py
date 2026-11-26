"""Compare response-limited Lion settings on Blue Marble."""

import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, replace
import json

from experiments.blue_marble_quality import fit_all_pixels, train
from hebbian_fit.data import rgb_data
from papers.lion import (
    COORDINATE_RATE, MARBLE_SEED, RESULTS, marble_trial, render_marble_convergence,
    render_marble_fits, render_mean_step_multipliers, report,
)


PILOT_EPOCHS = 200
FULL_EPOCHS = 560
RATES = (COORDINATE_RATE, .003)
LIMITS = (.000375, COORDINATE_RATE, .003)
STUDY = "blue-marble-lion-response"


def response_trial(rate, limit, epochs):
    rule = "lion" if limit is None else "lion-response"
    baseline = marble_trial(rule, "numpy")
    label = f"response-{epochs}e-rate-{rate:g}-limit-{limit}" if limit else f"response-{epochs}e-lion-control"
    if limit is None and rate != COORDINATE_RATE:
        label += f"-{rate:g}"
    config = baseline.config if limit is None else replace(baseline.config, response_limit_ratio=limit / rate)
    return replace(baseline, label=label, config=config, phases=((rule, rate, 128, epochs, 1.),))


def pilot(rate, limit):
    trial = response_trial(rate, limit, PILOT_EPOCHS)
    direct = replace(trial, label=f"{trial.label}-all-pixels")
    data = rgb_data(split=False)
    result = train(direct, MARBLE_SEED, data, checkpoint_pixels=data.train)
    return dict(settings=asdict(trial), rate=rate, limit=limit, seed=MARBLE_SEED,
                squared_error=result["validation"]["sse"], epoch=result["phase_epochs"][0],
                history=[{"epoch": point["epoch"], "squared_error": point["train_sse"]}
                         | ({"mean_step_multipliers": point["mean_step_multipliers"]}
                            if limit is not None else {}) for point in result["history"]])


def run_study(workers, full=False):
    settings = [(COORDINATE_RATE, None), *((rate, limit) for rate in RATES for limit in LIMITS),
                (RATES[-1], None)]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        jobs = [pool.submit(pilot, rate, limit) for rate, limit in settings]
        rows = [job.result() for job in jobs]
    selected = min((row for row in rows if row["limit"] is not None), key=lambda row: row["squared_error"])
    summary = dict(pilot_epochs=PILOT_EPOCHS, pilots=rows, selected_settings={
        "rate": selected["rate"], "limit": selected["limit"]})
    summary_path = RESULTS / f"{STUDY}.json"
    summary_path.write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    labels = [f"Step {rate:g}, limit {limit:g}" if limit else f"Step {rate:g}, original Lion"
              for rate, limit in settings]
    render_marble_convergence(rows, labels, name=f"{STUDY}-pilot")
    render_mean_step_multipliers(selected, f"{STUDY}-pilot-timesteps")
    if not full:
        return

    baseline = marble_trial("lion", "numpy")
    finalist = response_trial(selected["rate"], selected["limit"], FULL_EPOCHS)
    high_rate_control = response_trial(RATES[-1], None, FULL_EPOCHS)
    print(f"Selected step {selected['rate']:g}, limit {selected['limit']:g} for the full run", flush=True)
    data = rgb_data(split=False)
    fit_all_pixels(finalist, data, MARBLE_SEED)
    fit_all_pixels(high_rate_control, data, MARBLE_SEED)
    trials = (baseline, finalist, high_rate_control)
    records = [report(f"blue-marble-{trial.label}-all-pixels") for trial in trials]
    summary["full_run"] = records[1]
    summary["full_control"] = records[2]
    summary_path.write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    labels = ("Lion, step 0.00075", "Response-limited Lion", "Lion, step 0.003")
    render_marble_convergence(records, labels, name=f"{STUDY}-convergence")
    render_marble_fits(data, trials, records, labels, name=f"{STUDY}-fits")
    render_mean_step_multipliers(records[1], f"{STUDY}-timesteps")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=1, help="Concurrent pilot fits")
    parser.add_argument("--full", action="store_true", help="Continue with the 560-epoch comparisons")
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be positive")
    run_study(args.workers, args.full)
