"""Refine the Blue Marble fit with dense Hebbian models and fixed RGB outputs."""
import argparse
from dataclasses import asdict, dataclass, replace

from .blue_marble import ImageTrial
from .fixed_capacity import screen


BASE = ImageTrial(width=48, depth=2, connection="discontinuous", competition="layer",
                  layer_gates=(.25, .15), readout_layers="all")


@dataclass(frozen=True)
class BoundedImageTrial(ImageTrial):
    output_bound: float = 1.

    def model_config(self):
        return replace(super().model_config(), output_bound=self.output_bound)


@dataclass(frozen=True)
class PixelTrial(BoundedImageTrial):
    output_levels: int = 256

    def model_config(self):
        return replace(super().model_config(), output_levels=self.output_levels)


def pixel_values():
    base = PixelTrial(**asdict(BASE))
    for width in (24, 32, 48, 64):
        for rate, epochs in ((.0015, 100), (.003, 60), (.003, 110)):
            yield replace(base, width=width, rate=rate, epochs=epochs)
    for width in (32, 48):
        for gates in ((.25, .22), (.35, .25)):
            yield replace(base, width=width, layer_gates=gates, epochs=80)
        yield replace(base, width=width, connection="continuous", epochs=80)


def pixel_updates():
    base = PixelTrial(**asdict(BASE))
    for width in (24, 32, 48):
        for rule, rate in (("lion", .003), ("activity-sign", .009)):
            yield replace(base, width=width, rule=rule, rate=rate, epochs=60)
        yield replace(base, width=width, rule="lion", rate=.003, epochs=105)
    yield base


STAGES = {"pixel-values": pixel_values, "pixel-updates": pixel_updates}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=STAGES)
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args()
    screen(STAGES[args.stage](), args.workers)


if __name__ == "__main__":
    main()
