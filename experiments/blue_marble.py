"""Fit the bundled NASA image from raw coordinates using fixed class sums."""

import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, replace
from pathlib import Path

from hebbian_fit.data import rgb_data

from .fixed_capacity import Trial, train


@dataclass(frozen=True)
class ImageTrial(Trial):
    task: str = "image"
    connection: str = "affine"
    competition: str = "class"
    tau: float = .15
    rate: float = .003
    batch: int = 128
    epochs: int = 40
    rule: str = "signsgd"
    layer_gates: tuple[float, ...] | None = None

    def dataset(self):
        return rgb_data()

    def model_config(self):
        return replace(super().model_config(), inputs=2, outputs=3, hidden_rule=self.rule,
                       layer_gate_widths=self.layer_gates)

    def checkpoint_score(self, metrics):
        psnr = metrics["psnr_db"]
        return -psnr if psnr is not None else float("-inf")

    def path(self, seed):
        return Path("runs/blue-marble") / super().path(seed).parent.name / str(seed)


REFERENCE = ImageTrial(width=16)


def shallow():
    for width in (8, 16, 32, 64, 128):
        for rate in (.001, .003, .01):
            for tau in (.1, .2, .35):
                yield replace(REFERENCE, width=width, rate=rate, tau=tau)
    yield REFERENCE


def deep():
    for width in (8, 16, 32):
        for rate in (.001, .004, .012):
            for pool in ("last", "all"):
                yield replace(REFERENCE, width=width, depth=2, rate=rate, readout_layers=pool,
                              tau=.2, layer_gates=(.30, .20))
    for width in (8, 16, 32):
        for rate in (.002, .008):
            yield replace(REFERENCE, width=width, depth=3, rate=rate, readout_layers="all",
                          tau=.2)


def competition():
    for width in (16, 32, 64):
        for rate in (.001, .006):
            for tau in (.12, .3):
                yield replace(REFERENCE, width=width, competition="layer", tau=tau, rate=rate)
                yield replace(REFERENCE, width=width, tau=tau, rate=rate,
                              connection="discontinuous")
        for winners in (2, 6):
            yield replace(REFERENCE, width=width, gate="hard", winners=winners,
                          readout="fixed", rate=.006)


def refine():
    for width in (16, 32, 64, 128):
        for tau in (.065, .095, .15):
            for rate in (.002, .006):
                yield replace(REFERENCE, width=width, competition="layer", tau=tau,
                              rate=rate, epochs=60)
    for width in (16, 32, 64):
        for rate in (.004, .012):
            yield replace(REFERENCE, width=width, rule="activity-sign", competition="layer",
                          tau=.12, rate=rate, batch=64, epochs=60)
    for width in (8, 16, 32):
        for connection in ("affine", "discontinuous"):
            yield replace(REFERENCE, width=width, depth=2, connection=connection,
                          competition="layer", tau=.15, layer_gates=(.25, .15),
                          rate=.003, readout_layers="all")
            yield replace(REFERENCE, width=width, depth=2, connection=connection,
                          gate="none", rate=.0006, readout_layers="last")
    for width in (8, 32, 64, 128):
        yield replace(REFERENCE, width=width)


def two_layer():
    best_shape = replace(REFERENCE, depth=2, connection="discontinuous", competition="layer",
                         layer_gates=(.25, .15), readout_layers="all")
    for width in (24, 48, 64):
        yield replace(best_shape, width=width)
    for width in (24, 32, 48):
        yield replace(best_shape, width=width, batch=64, rate=.0015, epochs=60)
        yield replace(best_shape, width=width, rule="activity-sign", rate=.006, epochs=60)
    for layer_gates in ((.17, .10), (.25, .10), (.35, .20)):
        yield replace(best_shape, width=32, layer_gates=layer_gates)
    yield replace(best_shape, width=32, epochs=80)


STAGES = {"shallow": shallow, "deep": deep, "competition": competition, "refine": refine, "two-layer": two_layer}


def screen(stage, workers):
    with ProcessPoolExecutor(max_workers=workers) as executor:
        for row in executor.map(train, [(trial, 1) for trial in STAGES[stage]()]):
            print(json.dumps(row), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=tuple(STAGES))
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args()
    screen(args.stage, args.workers)


if __name__ == "__main__":
    main()
