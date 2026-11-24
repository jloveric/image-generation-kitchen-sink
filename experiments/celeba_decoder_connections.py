"""Compare CelebA autoencoders at the end of their matched second passes."""

import argparse
from hashlib import file_digest
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from experiments.celeba import evaluate, load_checkpoint, scale
from experiments.celeba_diversity import high_frequency_chroma, model_for, portrait_edges, spread
from hebbian_fit.autoencoder import Generator, SIZE, rgb_bytes


RUNS = (
    ("decoder-skip768x3", "Concatenated", 2048, True),
    ("decoder-last768x3", "Sequential", 2048, False),
    ("decoder-neck512", "Sequential, narrow neck", 512, False),
)
BLOCKS = (0, 216, 432, 648)
BLOCK_SIZE = 216
DISPLAY_PER_BLOCK = 3
DRAWS = tuple(seed for start in BLOCKS for seed in range(start, start + DISPLAY_PER_BLOCK))


def measure_samples(generator):
    scores = []
    selected = []
    for start in BLOCKS:
        images = np.stack([generator.generate(seed) for seed in range(start, start + BLOCK_SIZE)])
        scores.append(dict(start=start, count=len(images), edge_spread=spread(portrait_edges(images)),
                           chroma=high_frequency_chroma(images)))
        selected.append(images[:DISPLAY_PER_BLOCK])
    averages = {key: float(np.mean([score[key] for score in scores])) for key in ("edge_spread", "chroma")}
    return dict(blocks=scores, **averages), np.concatenate(selected)


def contact_sheet(path, rows):
    tile, gap, label_width = 88, 4, 204
    height = (tile + gap) * len(rows) + gap
    width = label_width + (tile + gap) * len(rows[0][2]) + gap
    sheet = Image.new("RGB", (width, height), "#f4f5f5")
    draw = ImageDraw.Draw(sheet)
    label_font = ImageFont.truetype("DejaVuSans-Bold.ttf", 13)
    detail_font = ImageFont.truetype("DejaVuSans.ttf", 12)
    for row, (label, detail, images) in enumerate(rows):
        y = gap + row * (tile + gap)
        draw.text((12, y + 27), label, fill="#20333b", font=label_font)
        draw.text((12, y + 47), detail, fill="#41535c", font=detail_font)
        for column, image in enumerate(images.reshape(-1, SIZE, SIZE, 3)):
            pixels = Image.fromarray(image).resize((tile, tile), Image.Resampling.NEAREST)
            sheet.paste(pixels, (label_width + column * (tile + gap), y))
    sheet.save(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, default=Path("runs/celeba-capacity"))
    parser.add_argument("--output", type=Path, default=Path("assets/results"))
    parser.add_argument("--backend", choices=("mojo", "numpy"), default="mojo")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    local = args.runs / "decoder-connections"
    local.mkdir(exist_ok=True)

    test = np.load(args.runs / "decoder-skip768x3-pass2/test.npy", mmap_mode="r")
    indices = np.linspace(0, len(test) - 1, len(DRAWS), dtype=int)
    examples = test[indices]
    sample_rows = []
    recon_rows = [("Original portraits", "Official test split", examples)]
    report = dict(test_images=len(test), generated_draws=len(BLOCKS) * BLOCK_SIZE,
                  displayed_draws=DRAWS, reconstruction_indices=indices.tolist(),
                  sampler=dict(components=16, rank=64, scale=.85), runs={})
    config_defaults = {"vector_gate_width": .24, "zero_init_bias": False}
    reference_configs = {}
    for name, label, code_width, skip in RUNS:
        first = args.runs / name / "full"
        end = args.runs / f"{name}-pass2" / "full"
        first_report = json.loads((first / "results.json").read_text())
        end_report = json.loads((end / "results.json").read_text())
        for phase, record in enumerate((first_report, end_report), 1):
            config = record["config"]
            if (config["latent_width"], config["vector_decoder_skip"]) != (code_width, skip):
                raise ValueError(f"Unexpected decoder connections or code width for {name}")
            matched = {k: v for k, v in {**config_defaults, **config}.items()
                       if k not in ("latent_width", "vector_decoder_skip", "initialize_from")}
            if reference_configs.setdefault(phase, matched) != matched:
                raise ValueError(f"Training settings differ for {name}, pass {phase}")

        model = model_for(end_report["config"], end, backend=args.backend)
        progress = load_checkpoint(end / "latest.npz", model)
        if (progress["epoch"] != 1 or
                {**config_defaults, **progress["config"]} != {**config_defaults, **end_report["config"]}):
            raise ValueError(f"Expected a completed second pass for {name}")
        if model.decoder.config.inputs != (4352 if skip else 768):
            raise ValueError(f"Unexpected output fan-in for {name}")
        center = np.load(end / "encoder-center.npy").reshape(test.shape[1:])
        test_scores = evaluate(model, test, center, shuffle=True)
        codes = model.encode(scale(examples).reshape(len(examples), -1))
        reconstruction = rgb_bytes(model.decode(codes))
        del model

        path = end / "end-decoder-mixture.npz"
        generator = Generator(path)
        if (generator.decoder_skip != skip or generator.mixture[2].shape != (16, code_width, 64)):
            raise ValueError(f"Expected an endpoint export with the matched sampler for {name}")
        sampling, samples = measure_samples(generator)
        del generator
        detail = f"{code_width:,}-value code"
        sample_rows.append((label, detail, samples))
        recon_rows.append((label, detail, reconstruction))
        with path.open("rb") as stream:
            digest = file_digest(stream, "sha256").hexdigest()
        record = dict(label=label, architecture=end_report["architecture"],
                      parameters=end_report["parameters"], first_pass_checkpoint=first_report["selected"]["epoch"],
                      first_pass_test_sse=first_report["test"]["reconstruction_sse"] / len(test),
                      second_pass_test_sse=test_scores["reconstruction_sse"] / len(test),
                      second_pass_shuffled_test_sse=test_scores["shuffled_sse"] / len(test),
                      mean_image_test_sse=test_scores["mean_sse"] / len(test),
                      training_panel_second_pass=progress["curve"], sampling=sampling,
                      config=end_report["config"], export=str(path), export_sha256=digest)
        report["runs"][name] = record
        print(name, f"test RGB error {record['second_pass_test_sse']:.3f} "
                    f"(shuffled {record['second_pass_shuffled_test_sse']:.3f}), "
                    f"edge spread {sampling['edge_spread']:.3f}, color grain {sampling['chroma']:.4f}",
              flush=True)

    contact_sheet(args.output / "celeba-decoder-connections.png", sample_rows)
    contact_sheet(local / "reconstruction.png", recon_rows)
    (args.output / "celeba-decoder-connections.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
