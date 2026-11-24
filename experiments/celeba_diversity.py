"""Compare CelebA runs on a fixed panel and the same 36 decoder seeds."""

import argparse
import csv
import io
import json
from pathlib import Path
from zipfile import ZipFile

import numpy as np
from PIL import Image, ImageFilter

from experiments.celeba import IMAGES_PER_BATCH, SEED, encode, figure, load_checkpoint, manifest, subset
from hebbian_fit.autoencoder import Autoencoder, ENCODER_WIDTH, Generator, LATENT, SIZE, rgb_bytes

GRID_SEEDS = tuple(range(36))
MEASUREMENT_SEEDS = tuple(range(216))
ATTRIBUTES = ("Male", "Eyeglasses", "Mouth_Slightly_Open", "Smiling")


def portrait_edges(images):
    """Measure low-frequency luminance edges within the central face crop."""
    vectors = []
    for image in images.reshape(-1, SIZE, SIZE, 3):
        gray = Image.fromarray(image).convert("L").filter(ImageFilter.GaussianBlur(1.5))
        crop = np.asarray(gray, dtype=np.float32)[16:48:2, 16:48:2]
        dx = np.diff(crop, axis=1).ravel()
        dy = np.diff(crop, axis=0).ravel()
        vector = np.concatenate((dx, dy))
        norm = np.linalg.norm(vector)
        vectors.append(vector / norm if norm else vector)
    return np.stack(vectors)


def high_frequency_chroma(images):
    rgb = images.reshape(-1, SIZE, SIZE, 3).astype(np.float32)[:, 16:48, 16:48] / 255
    chroma = rgb[..., (0, 2)] - rgb[..., 1:2]
    horizontal = np.diff(chroma, n=2, axis=2)
    vertical = np.diff(chroma, n=2, axis=1)
    return float(np.sqrt((np.square(horizontal).mean() + np.square(vertical).mean()) / 2))


def face_lighting(images):
    crop = images.reshape(-1, SIZE, SIZE, 3)[:, 16:48, 16:48].astype(np.float32) / 255
    luminance = crop @ np.array([.2126, .7152, .0722], dtype=np.float32)
    return luminance.mean(axis=(1, 2)), luminance.std(axis=(1, 2))


def spread(vectors):
    # Mean squared distance over distinct pairs.
    return float(2 * np.var(vectors, axis=0).sum() * len(vectors) / (len(vectors) - 1))


def edge_error(actual, expected):
    return float(np.square(actual - expected).sum(axis=1).mean())


def validation_attributes(path, count):
    with ZipFile(path) as archive:
        _, names, _ = manifest(archive)
        selected = subset(names, count, SEED)
        wanted = set(selected)
        member = next(name for name in archive.namelist() if Path(name).name == "list_attr_celeba.csv")
        with archive.open(member) as data:
            rows = csv.DictReader(io.TextIOWrapper(data, encoding="utf-8-sig"))
            lookup = {row["image_id"]: row for row in rows if row["image_id"] in wanted}
    return {name: np.array([int(lookup[file][name]) > 0 for file in selected]) for name in ATTRIBUTES}


def attribute_contrast(reconstructed, targets, labels):
    ratios = {}
    for name, positives in labels.items():
        target_gap = targets[positives].mean(axis=0) - targets[~positives].mean(axis=0)
        recon_gap = reconstructed[positives].mean(axis=0) - reconstructed[~positives].mean(axis=0)
        ratios[name] = float((recon_gap @ target_gap) / (target_gap @ target_gap))
    return ratios


def model_for(config, path, backend="numpy"):
    center = np.load(path / "encoder-center.npy") if config.get("center_encoder") else None
    return Autoencoder(config["seed"], decoder_kind="vector", backend=backend,
                       images_per_batch=config.get("images_per_batch", IMAGES_PER_BATCH),
                       encoder_gate_widths=config.get("encoder_gate_widths", (.24, .18)),
                       projected_latent_votes=config.get("projected_latent_votes", False),
                       bottleneck_pool_size=config.get("bottleneck_pool_size"),
                       projected_encoder_votes=config.get("projected_encoder_votes", False),
                       center_each_input=config.get("center_each_input", False),
                       normalize_each_input=config.get("normalize_each_input", False),
                       encoder_votes_per_pair=config.get("encoder_votes_per_pair"),
                       encoder_width=config.get("encoder_width", ENCODER_WIDTH),
                       latent_width=config.get("latent_width", LATENT),
                       vector_output_class_width=config.get("vector_output_class_width", 1),
                       vector_decoder_skip=config.get("vector_decoder_skip", False),
                       projected_decoder_votes=config.get("projected_decoder_votes", False),
                       decoder_rate_decay=config.get("decoder_rate_decay", 1.),
                       input_center=center, rule=config["rule"],
                       vector_hidden_width=config.get("vector_hidden_width", 0),
                       vector_hidden_depth=config.get("vector_hidden_depth", 1),
                       vector_gate_width=config.get("vector_gate_width", .24))


def compare(path, validation, target_edges, labels, decoder_path=None, seed_start=0):
    result = json.loads((path / "results.json").read_text())
    config = result["config"]
    model = model_for(config, path)
    load_checkpoint(path / "selected.npz", model)
    reconstruction = rgb_bytes(model.decode(encode(model, validation)))
    recon_edges = portrait_edges(reconstruction)
    generator = Generator(decoder_path or path / "decoder.npz")
    generated = np.stack([generator.generate(seed_start + offset) for offset in MEASUREMENT_SEEDS])
    sample_edges = portrait_edges(generated)
    distances = ((sample_edges**2).sum(axis=1)[:, None] + (target_edges**2).sum(axis=1)
                 - 2 * (sample_edges @ target_edges.T))
    variances = np.square(np.linalg.svd(generator.root, compute_uv=False))
    variance_shares = variances / variances.sum()
    target_brightness, target_contrast = face_lighting(validation)
    recon_brightness, recon_contrast = face_lighting(reconstruction)
    sample_brightness, sample_contrast = face_lighting(generated)
    quantiles = np.linspace(.05, .95, 19)

    selected = result.get("validation", result["selected"])
    scores = dict(
        config=config,
        generator_prior="mixture" if generator.mixture is not None else "gaussian",
        selected_epoch=result["selected"]["epoch"],
        validation_sse=selected["reconstruction_sse"] / len(validation),
        top_code_variance=float(variance_shares[0]),
        effective_code_rank=float(1 / np.square(variance_shares).sum()),
        generated_edge_spread=spread(sample_edges),
        generated_high_frequency_chroma=high_frequency_chroma(generated),
        generated_nearest_validation_edge_error=float(np.maximum(distances.min(axis=1), 0).mean()),
        generated_brightness_quantile_error=float(np.abs(np.quantile(sample_brightness, quantiles)
                                                          - np.quantile(target_brightness, quantiles)).mean()),
        generated_contrast_quantile_error=float(np.abs(np.quantile(sample_contrast, quantiles)
                                                        - np.quantile(target_contrast, quantiles)).mean()),
        recon_edge_spread=spread(recon_edges),
        recon_edge_error=edge_error(recon_edges, target_edges),
        shuffled_edge_error=edge_error(np.roll(recon_edges, len(recon_edges) // 2, axis=0), target_edges),
        recon_brightness_mae=float(np.abs(recon_brightness - target_brightness).mean()),
        recon_contrast_mae=float(np.abs(recon_contrast - target_contrast).mean()),
    )
    if labels:
        scores["attribute_contrast"] = attribute_contrast(recon_edges, target_edges, labels)
    if "test" in result:
        scores["test_sse"] = result["test"]["reconstruction_sse"] / result["test"].get("images", 19962)
    if "training" in result:
        scores["training_sse"] = result["training"]["reconstruction_sse"] / result["training"]["images"]
    return scores, generated[:len(GRID_SEEDS)], reconstruction[:len(GRID_SEEDS)].reshape(-1, SIZE, SIZE, 3)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validation", type=Path, default=Path("runs/celeba/validation.npy"))
    parser.add_argument("--archive", type=Path, help="Aligned CelebA archive with the attribute CSV")
    parser.add_argument("--output", type=Path, default=Path("runs/celeba-diversity/comparison"))
    parser.add_argument("--seed-start", type=int, default=0, help="First of 216 consecutive generator seeds")
    parser.add_argument("runs", nargs="+",
                        help="Results directories or decoder exports as label=path")
    args = parser.parse_args()
    if args.seed_start < 0:
        parser.error("--seed-start must be nonnegative")
    args.output.mkdir(parents=True, exist_ok=True)
    validation = np.load(args.validation, mmap_mode="r")
    target_edges = portrait_edges(validation)
    labels = validation_attributes(args.archive, len(validation)) if args.archive else {}
    report = {"validation_portraits": len(validation), "measurement_seeds": [args.seed_start + i for i in MEASUREMENT_SEEDS],
              "grid_seeds": [args.seed_start + i for i in GRID_SEEDS],
              "validation_edge_spread": spread(target_edges),
              "validation_high_frequency_chroma": high_frequency_chroma(validation), "runs": {}}
    for spec in args.runs:
        label, value = spec.split("=", 1)
        path = Path(value)
        decoder_path = path if path.is_file() else None
        run = path.parent if decoder_path else path
        score, generated, reconstruction = compare(run, validation, target_edges, labels, decoder_path, args.seed_start)
        report["runs"][label] = score
        figure(generated, args.output / f"{label}-generated.png")
        figure(reconstruction, args.output / f"{label}-reconstruction.png")
        print(label, {key: round(value, 4) for key, value in score.items()
                      if isinstance(value, (float, int))}, flush=True)
    (args.output / "diversity.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
