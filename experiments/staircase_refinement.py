"""Refine competition, minibatches, and depth in the fixed-output Hebbian staircase fit."""

import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass, replace
from pathlib import Path

from .fixed_capacity import Trial, train


@dataclass(frozen=True)
class Refinement(Trial):
    layer_gates: tuple[float, ...] | None = None
    rule: str = "signsgd"

    checkpoint: str = "balanced"

    def model_config(self):
        return replace(super().model_config(), layer_gate_widths=self.layer_gates, hidden_rule=self.rule)

    def checkpoint_score(self, metrics):
        if self.checkpoint == "rmse":
            return metrics["rmse"]
        return super().checkpoint_score(metrics)

    def path(self, seed):
        name = super().path(seed).parent.name
        if self.checkpoint == "balanced":
            name = name.removesuffix("_balanced")
        return Path("runs/staircase-refinement") / name / str(seed)


BASE = Refinement(width=128, depth=2, competition="layer", tau=.15, rate=.01,
                  readout_layers="all", batch=64)


def schedules():
    for batch, epochs in ((16, 60), (32, 90), (64, 180), (128, 120)):
        for rate in (.0025, .0075, .02):
            for pool_layers in ("last", "all"):
                yield replace(BASE, batch=batch, epochs=epochs, rate=rate, readout_layers=pool_layers)
    for rule in ("lion", "signsgd"):
        for rate in (.003, .01, .03):
            yield replace(BASE, rule=rule, rate=rate, batch=32, epochs=60)


def gates():
    for first in (.12, .25, .5, 1.):
        for last in (.09, .15, .25):
            for pool_layers in ("last", "all"):
                yield replace(BASE, layer_gates=(first, last), readout_layers=pool_layers,
                              batch=64, rate=.0075)
    for first in (.12, .25):
        for pool in ("class", "layer"):
            for rate in (.003, .01):
                yield replace(BASE, layer_gates=(first, .1), competition=pool,
                              rate=rate, batch=32)


def architecture():
    for width, depth in ((64, 3), (128, 3), (192, 2), (256, 2)):
        for tau in (.1, .17):
            for rate in (.004, .01):
                yield replace(BASE, width=width, depth=depth, tau=tau, rate=rate, batch=64)
    for connection in ("affine", "continuous", "discontinuous"):
        for keep in (4, 12, 32):
            for pool_layers in ("last", "all"):
                yield replace(BASE, gate="hard", winners=keep, readout="fixed", rate=.0075,
                              connection=connection, readout_layers=pool_layers)
    for initialization in ("affine-start", "unit-sphere"):
        for project in (False, True):
            for width in (64, 128):
                yield replace(BASE, width=width, initialization=initialization,
                              unit_norm_updates=project)


def activity():
    for width in (64, 128):
        for batch in (32, 64):
            for rate in (.0075, .02, .06):
                for gates in (None, (.25, .15)):
                    yield replace(BASE, width=width, rule="activity-sign", rate=rate,
                                  batch=batch, layer_gates=gates, checkpoint="rmse")
    for rate in (.003, .01, .03):
        yield replace(BASE, rate=rate, checkpoint="rmse")


def small_batches():
    for width in (32, 64):
        for batch in (8, 16):
            for rate in (.0008, .0025, .0075):
                for pool_layers in ("last", "all"):
                    yield replace(BASE, width=width, batch=batch, rate=rate, epochs=45,
                                  readout_layers=pool_layers, checkpoint="rmse")
    for batch in (16, 32):
        for rate in (.0025, .0075):
            yield replace(BASE, batch=batch, rate=rate, epochs=90, readout_layers="last",
                          checkpoint="rmse")
    for width in (32, 64):
        for depth in (1, 2):
            for rate in (.0008, .0025):
                yield replace(BASE, task="sine", width=width, depth=depth, batch=16,
                              tau=.15, rule="activity-sign", rate=rate, epochs=120,
                              checkpoint="rmse")
    yield replace(BASE, task="sine", width=32, depth=1, batch=32, rate=.001, epochs=120)


def final_layer():
    for batch in (16, 32, 64):
        for rate in (.0025, .0075, .02):
            for rule in ("signsgd", "activity-sign"):
                yield replace(BASE, readout_layers="last", batch=batch, rate=rate,
                              rule=rule, checkpoint="rmse")
    for rate in (.0075, .02):
        for gate_widths in ((.25, .15), (.15, .10), (.15, .25)):
            yield replace(BASE, batch=32, rule="activity-sign", layer_gates=gate_widths,
                          rate=rate, checkpoint="rmse")


def sine_settings():
    baseline = replace(BASE, task="sine", width=32, depth=1, batch=32, rate=.001, epochs=120)
    for batch in (8, 16, 32):
        for rate in (.0005, .0015, .004):
            for tau in (.12, .2):
                yield replace(baseline, batch=batch, rate=rate, tau=tau)
    for width in (16, 32):
        for tau in (.12, .18, .25):
            for connection in ("affine", "continuous"):
                yield replace(baseline, width=width, tau=tau, connection=connection, rate=.002)
    for tau in (.12, .18, .25):
        for rule in ("signsgd", "activity-sign"):
            yield replace(baseline, tau=tau, rate=.0025, epochs=240, rule=rule)


STAGES = {"schedule": schedules, "gates": gates, "architecture": architecture, "activity": activity, "small-batch": small_batches, "final-layer": final_layer, "sine": sine_settings}


def screen(stage, workers):
    with ProcessPoolExecutor(max_workers=workers) as executor:
        jobs = [(trial, 1) for trial in stage()]
        for row in executor.map(train, jobs):
            print(json.dumps(row), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=tuple(STAGES))
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args()
    screen(STAGES[args.stage], args.workers)


if __name__ == "__main__":
    main()
