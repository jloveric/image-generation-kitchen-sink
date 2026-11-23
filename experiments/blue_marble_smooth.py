"""Reduce grain with smooth two-piece connections inside the shared Hebbian MLP."""
import argparse
from dataclasses import asdict, dataclass, replace

from .blue_marble_refinement import BASE, PixelTrial
from .fixed_capacity import screen


@dataclass(frozen=True)
class SmoothTrial(PixelTrial):
    edge_transitions: tuple[float, ...] = (.04, .04)

    def model_config(self):
        return replace(super().model_config(), edge_transitions=self.edge_transitions)


REFERENCE = PixelTrial(**asdict(replace(BASE, rule="lion", epochs=105)))
SMOOTH = SmoothTrial(**asdict(replace(REFERENCE, epochs=65)))


def transitions():
    for width in (24, 32, 48):
        for ramp in ((.04, .015), (.06, .05), (.08, .10), (.12, .20), (.0, .06), (.03, .0)):
            yield replace(SMOOTH, width=width, edge_transitions=ramp)


def schedule():
    for width in (32, 48):
        for ramp in ((.04, .025), (.08, .075)):
            for rate, epochs, batch in ((.003, 105, 128), (.0015, 85, 64), (.004, 60, 256)):
                yield replace(SMOOTH, width=width, edge_transitions=ramp, rate=rate,
                              epochs=epochs, batch=batch)


def deep():
    for width in (24, 32):
        for ramp in ((.05, .035, .035), (.08, .065, .065), (.1, .12, .1)):
            yield replace(SMOOTH, width=width, depth=3, layer_gates=(.30, .22, .16),
                          edge_transitions=ramp, epochs=70)
    for width in (32, 48, 64):
        yield replace(SMOOTH, width=width, edge_transitions=(.05, .045),
                      layer_gates=(.30, .22), rate=.002, epochs=75)


def second_layer():
    for width in (32, 48):
        for ramp in ((0., .04), (0., .09), (0., .14), (.012, .045), (.012, .08), (.025, .06)):
            yield replace(SMOOTH, width=width, edge_transitions=ramp)
    for ramp in ((0., .06), (.012, .06)):
        yield replace(SMOOTH, edge_transitions=ramp, epochs=100)


STAGES = {"transitions": transitions, "schedule": schedule, "deep": deep, "second-layer": second_layer}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=STAGES)
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args()
    screen(STAGES[args.stage](), args.workers)


if __name__ == "__main__":
    main()
