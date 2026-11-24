"""Report sequential CelebA tuning and compare the selected generator with prior runs."""

import argparse
from hashlib import file_digest
import json
from pathlib import Path

import numpy as np

from experiments.celeba import VALIDATION, load_checkpoint, scale
from experiments.celeba_decoder_connections import DRAWS, contact_sheet, measure_samples
from experiments.celeba_diversity import model_for
from hebbian_fit.autoencoder import Generator, rgb_bytes


CAPACITY = Path("runs/celeba-capacity")
REFERENCE = Path("assets/results/celeba-decoder-connections.json")


def results(path):
    result = json.loads(path.read_text())
    config = result["config"]
    if config["decoder"] != "vector" or config.get("vector_decoder_skip"):
        raise ValueError(f"Expected a sequential image-vector model in {path}")
    architecture = result["architecture"]
    widths = []
    for layer in architecture.values():
        units = 2 * layer["outputs"] * layer["width"]
        if not widths:
            widths.append(layer["inputs"])
        if layer["inputs"] != widths[-1]:
            raise ValueError(f"Expected adjacent-layer connections in {path}")
        widths.append(units)
    return dict(path=str(path.parent), config=config, widths=widths, parameters=result["parameters"],
                validation_sse=result["selected"]["reconstruction_sse"] / VALIDATION,
                selected_epoch=result["selected"]["epoch"], curve=result["curve"],
                **({"test_sse": result["test"]["reconstruction_sse"] / result["test"]["images"]}
                   if "test" in result else {}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("runs/celeba-sequential"))
    parser.add_argument("--output", type=Path, default=Path("assets/results"))
    args = parser.parse_args()
    pilot = {p.parent.parent.name: results(p) for p in sorted(args.root.glob("screen/*/pilot/results.json"))}
    full = {p.parent.parent.name: results(p) for p in sorted(args.root.glob("full/*/full/results.json"))}
    best = min(full, key=lambda name: full[name]["validation_sse"])
    chosen = Path(full[best]["path"])
    reference = json.loads(REFERENCE.read_text())["runs"]["decoder-neck512"]

    models = [
        ("Prior three-layer decoder", "512-value code", CAPACITY / "decoder-neck512-pass2/full",
         "latest.npz", "end-decoder-mixture.npz"),
        ("Prior direct decoder", "2,048-value code", CAPACITY / "fixed768x2048/full",
         "selected.npz", "decoder-mixture.npz"),
        ("Tuned compact model", "512-value neck", args.root / "full/direct-768-512-paired/full",
         "selected.npz", "decoder-mixture.npz"),
        ("Tuned sequential model", f"{full[best]['config']['latent_width']:,}-value code",
         chosen, "selected.npz", "decoder-mixture.npz"),
    ]
    test = np.load(chosen.parent / "test.npy", mmap_mode="r")
    indices = np.linspace(0, len(test) - 1, len(DRAWS), dtype=int)
    images = test[indices]
    samples = []
    reconstructions = [("Original portraits", "Official test split", images)]
    sampling = {}
    for label, detail, run, checkpoint, export in models:
        path = run / export
        generator = Generator(path)
        if generator.decoder_skip or generator.mixture is None:
            raise ValueError(f"Expected a sequential mixture decoder in {path}")
        components, code_width, rank = generator.mixture[2].shape
        record, draws = measure_samples(generator)
        del generator
        with path.open("rb") as stream:
            digest = file_digest(stream, "sha256").hexdigest()
        sampling[label] = dict(export=str(path), sha256=digest, components=components,
                               code_width=code_width, rank=rank, **record)
        samples.append((label, detail, draws))

        config = json.loads((run / "results.json").read_text())["config"]
        model = model_for(config, run)
        load_checkpoint(run / checkpoint, model)
        reconstruction = model.decode(model.encode(scale(images).reshape(len(images), -1)))
        reconstructions.append((label, detail, rgb_bytes(reconstruction)))
        del model

    args.output.mkdir(parents=True, exist_ok=True)
    contact_sheet(args.output / "celeba-sequential.png", samples)
    contact_sheet(args.root / "reconstruction.png", reconstructions)
    report = dict(pilot=pilot, full=full, selected=best, test_portraits=len(test),
                  baseline_three_layer_test_sse=reference["second_pass_test_sse"],
                  baseline_direct_test_sse=results(CAPACITY / "fixed768x2048/full/results.json")["test_sse"],
                  sampling=sampling, displayed_draws=DRAWS, reconstruction_indices=indices.tolist())
    (args.output / "celeba-sequential.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(best, full[best]["validation_sse"], full[best]["test_sse"], flush=True)


if __name__ == "__main__":
    main()
