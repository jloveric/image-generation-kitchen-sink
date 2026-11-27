"""Run the CelebA pilot and conditionally train on the full split."""

import argparse
import csv
from dataclasses import asdict
import io
import json
from pathlib import Path
import zipfile

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from hebbian_fit.autoencoder import (
    Autoencoder, COORDINATES, ENCODER_WIDTH, Generator, LATENT, SIZE, export_generator,
)

PILOT = 4096
VALIDATION = 512
IMAGES_PER_BATCH = 32
PIXELS_PER_IMAGE = 32
SEED = 7


def manifest(archive, partitions=None):
    if partitions is None:
        name = next((name for name in archive.namelist()
                     if Path(name).name in ("list_eval_partition.txt", "list_eval_partition.csv")), None)
        if name is None:
            raise ValueError("Supply CelebA's partition list with --partitions")
        contents = archive.read(name).decode("utf-8-sig")
    else:
        name = partitions.name
        contents = partitions.read_text(encoding="utf-8-sig")
    rows = contents.splitlines()
    if name.endswith(".csv"):
        rows = list(csv.reader(rows))[1:]
    else:
        rows = [row.split() for row in rows if row.strip()]
    splits = [[], [], []]
    for name, split in rows:
        splits[int(split)].append(name)
    counts = tuple(map(len, splits))
    if counts != (162770, 19867, 19962):
        raise ValueError(f"Expected the official CelebA partition sizes, got {counts}")
    return [np.array(sorted(split)) for split in splits]


def subset(names, count, seed):
    return np.sort(np.random.default_rng(seed).choice(names, count, replace=False))


def load_images(archive, names, path):
    if path.exists():
        cached = np.load(path, mmap_mode="r")
        if cached.shape == (len(names), SIZE * SIZE, 3) and cached.dtype == np.uint8:
            return cached
        raise ValueError(f"Mismatched image cache at {path}")
    members = {Path(name).name: name for name in archive.namelist() if name.lower().endswith(".jpg")}
    temporary = path.with_suffix(".tmp.npy")
    images = np.lib.format.open_memmap(temporary, mode="w+", dtype=np.uint8,
                                      shape=(len(names), SIZE * SIZE, 3))
    for i, name in enumerate(names):
        with Image.open(io.BytesIO(archive.read(members[name]))) as source:
            image = source.convert("RGB")
            w, h = image.size
            side = min(w, h)
            left, top = (w - side) // 2, (h - side) // 2
            pixels = image.crop((left, top, left + side, top + side)).resize(
                (SIZE, SIZE), Image.Resampling.LANCZOS)
            images[i] = np.asarray(pixels).reshape(-1, 3)
    images.flush()
    del images
    temporary.replace(path)
    return np.load(path, mmap_mode="r")


def scale(images):
    return images.astype(np.float32) / np.float32(127.5) - 1


def encode(model, images):
    return np.concatenate([
        model.encode(scale(images[i:i + IMAGES_PER_BATCH]).reshape(-1, SIZE * SIZE * 3))
        for i in range(0, len(images), IMAGES_PER_BATCH)
    ])


def rgb_sse(left, right):
    return float(np.sum(np.square((left - right) * .5), dtype=np.float64))


def evaluate(model, images, mean_image, shuffle=True, codes=None):
    if codes is None:
        codes = encode(model, images)
    sums = dict(reconstruction_sse=0., shuffled_sse=0., mean_sse=0.)
    batch = model.images_per_batch if model.decoder_kind == "vector" else 8
    for i in range(0, len(images), batch):
        part = slice(i, i + batch)
        target = scale(images[part])
        predicted = model.decode(codes[part])
        sums["reconstruction_sse"] += rgb_sse(predicted, target)
        if shuffle:
            keys = (np.arange(i, i + len(target)) + len(images) // 2) % len(images)
            sums["shuffled_sse"] += rgb_sse(model.decode(codes[keys]), target)
        sums["mean_sse"] += rgb_sse(mean_image, target)
    if not shuffle:
        sums.pop("shuffled_sse")
    return sums


def save_checkpoint(path, model, metadata):
    temporary = path.with_suffix(".tmp.npz")
    np.savez_compressed(temporary, metadata=json.dumps(metadata), **model.state())
    temporary.replace(path)


def load_checkpoint(path, model):
    with np.load(path, allow_pickle=False) as state:
        model.load({key: state[key] for key in state.files if key != "metadata"})
        return json.loads(state["metadata"].item())


def train(train_images, validation, path, args):
    path.mkdir(parents=True, exist_ok=True)
    evaluation = validation
    if args.select_on_training:
        indices = np.random.default_rng(SEED).choice(len(train_images), min(VALIDATION, len(train_images)),
                                                     replace=False)
        evaluation = train_images[indices]
    mean_image = (train_images.mean(axis=0) / 127.5 - 1).astype(np.float32)
    input_center = mean_image.ravel() if args.center_encoder else None
    if input_center is not None:
        np.save(path / "encoder-center.npy", input_center)
    model = Autoencoder(args.seed, backend=args.backend, rule=args.rule, depth=args.depth,
                        gate_widths=(args.gate_width,) * args.depth if args.gate_width else None,
                        coordinate_scale=args.coordinate_scale, decoder_rate_decay=args.decoder_rate_decay,
                        encoder_gate_widths=args.encoder_gate_widths, decoder_class_width=args.decoder_class_width,
                        images_per_batch=args.images_per_batch,
                        queries_per_batch=max(1024, args.images_per_batch * args.pixels_per_image),
                        input_center=input_center, decoder_kind=args.decoder,
                        decoder_init_scale=args.decoder_init_scale,
                        projected_latent_votes=args.projected_latent_votes,
                        bottleneck_pool_size=args.bottleneck_pool_size,
                        projected_encoder_votes=args.projected_encoder_votes,
                        center_each_input=args.center_each_input, vector_hidden_width=args.vector_hidden_width,
                        vector_hidden_depth=args.vector_hidden_depth,
                        normalize_each_input=args.normalize_each_input, paired_decoder_init=args.paired_decoder_init,
                        encoder_votes_per_pair=args.encoder_votes_per_pair,
                        encoder_width=args.encoder_width, latent_width=args.latent_width,
                        vector_output_class_width=args.vector_output_class_width,
                        vector_decoder_skip=args.vector_decoder_skip, projected_decoder_votes=args.projected_decoder_votes,
                        vector_gate_width=args.vector_gate_width, zero_init_bias=args.zero_init_bias,
                        encoder_top_k=args.encoder_top_k, vector_top_k=args.vector_top_k,
                        encoder_hidden_widths=args.encoder_hidden_widths, single_output_neuron=args.single_output_neuron)
    rng = np.random.default_rng(args.seed)
    latest, best = path / "latest.npz", path / "selected.npz"
    tuning = dict(coordinate_scale=1., decoder_rate_decay=1., encoder_gate_widths=[.24, .18],
                  decoder_class_width=32, shared_coordinates=False, center_encoder=False,
                  images_per_batch=IMAGES_PER_BATCH, pixels_per_image=PIXELS_PER_IMAGE, decoder="coordinate", decoder_init_scale=1.,
                  projected_latent_votes=False, bottleneck_pool_size=None,
                  projected_encoder_votes=False, center_each_input=False, vector_hidden_width=0,
                  normalize_each_input=False, paired_decoder_init=False, encoder_votes_per_pair=None,
                  encoder_width=ENCODER_WIDTH, latent_width=LATENT, select_on_training=False,
                  vector_output_class_width=1, vector_decoder_skip=False, projected_decoder_votes=False,
                  checks_per_epoch=1, vector_hidden_depth=1, vector_gate_width=.24,
                  zero_init_bias=False, encoder_top_k=[0, 0], vector_top_k=0, encoder_hidden_widths=[],
                  single_output_neuron=False)
    config = dict(epochs=args.epochs, seed=args.seed, lr=args.rate, encoder_rate=args.encoder_rate,
                  rule=args.rule, depth=args.depth, gate_width=args.gate_width, images=len(train_images),
                  validation_images=len(validation), **{key: getattr(args, key) for key in tuning})
    config["encoder_gate_widths"] = list(config["encoder_gate_widths"])
    if args.initialize_from:
        config["initialize_from"] = str(args.initialize_from)
    start_step, resume_order = 0, None
    if latest.exists():
        progress = load_checkpoint(latest, model)
        previous = {**tuning, "validation_images": VALIDATION, **progress["config"]}
        if previous != config:
            raise ValueError("Checkpoint settings differ. Use a new output directory.")
        rng.bit_generator.state = progress["rng"]
        curve = progress["curve"]
        selected = progress["best"]
        start = progress["epoch"]
        start_step = progress.get("step", 0)
        resume_order = progress.get("order")
    else:
        if args.initialize_from:
            state = model.state()
            with np.load(args.initialize_from, allow_pickle=False) as initial:
                model.load({key: initial[key] if "_weight_" in key else value
                            for key, value in state.items()})
        curve = [dict(epoch=0, **evaluate(model, evaluation, mean_image))]
        start, selected = 0, curve[0]
        save_checkpoint(best, model, selected)
        print(curve[0], flush=True)
    images_per_batch, pixels_per_image = args.images_per_batch, args.pixels_per_image
    steps_per_epoch = (len(train_images) + images_per_batch - 1) // images_per_batch
    total_steps = steps_per_epoch * args.epochs
    evaluate_every = 1 if args.select_on_training and args.epochs <= 12 else 5
    checks = min(args.checks_per_epoch, steps_per_epoch)
    check_steps = {(steps_per_epoch * check + checks - 1) // checks for check in range(1, checks + 1)}
    for epoch in range(start, args.epochs):
        first_step = start_step if epoch == start else 0
        order = np.asarray(resume_order, dtype=np.intp) if first_step else rng.permutation(len(train_images))
        check_epoch = checks > 1 or (epoch + 1) % evaluate_every == 0 or epoch + 1 == args.epochs
        for step in range(first_step, steps_per_epoch):
            first = step * images_per_batch
            images = scale(train_images[order[first:first + images_per_batch]])
            coordinates = None
            targets = images
            if args.decoder == "coordinate":
                if args.shared_coordinates:
                    chosen = rng.choice(len(COORDINATES), pixels_per_image, replace=False)
                    chosen = np.broadcast_to(chosen, (len(images), pixels_per_image))
                else:
                    chosen = np.stack([rng.choice(len(COORDINATES), pixels_per_image, replace=False)
                                       for _ in images])
                coordinates = COORDINATES[chosen]
                targets = images[np.arange(len(images))[:, None], chosen]
            phase = (epoch * steps_per_epoch + step) / max(1, total_steps - 1)
            rate = args.rate * .5 * (1 + np.cos(np.pi * max(0, (phase - .2) / .8)))
            model.step(images.reshape(len(images), -1), coordinates, targets, rate, args.encoder_rate)
            completed = step + 1
            end_of_epoch = completed == steps_per_epoch
            if end_of_epoch:
                print(f"Epoch {epoch + 1}/{args.epochs} complete", flush=True)
            if not check_epoch or completed not in check_steps:
                continue
            position = epoch + 1 if end_of_epoch else epoch + completed / steps_per_epoch
            row = dict(epoch=position, **evaluate(model, evaluation, mean_image))
            curve.append(row)
            print(row, flush=True)
            if row["reconstruction_sse"] < selected["reconstruction_sse"]:
                selected = row
                save_checkpoint(best, model, selected)
            progress = dict(config=config, rng=rng.bit_generator.state, curve=curve, best=selected,
                            epoch=epoch + 1 if end_of_epoch else epoch)
            if not end_of_epoch:
                progress.update(step=completed, order=order.tolist())
            save_checkpoint(latest, model, progress)
    load_checkpoint(best, model)
    return model, dict(config=config, curve=curve, selected=selected), mean_image


def nearest(training, generated):
    targets = generated.reshape(len(generated), -1).astype(np.float32) / 255
    target_power = (targets**2).sum(axis=1)
    nearest_index = np.zeros(len(generated), dtype=int)
    distances = np.full(len(generated), np.inf)
    for i in range(0, len(training), 512):
        block = training[i:i + 512].reshape(-1, targets.shape[1]).astype(np.float32) / 255
        squared = ((block**2).sum(axis=1)[:, None] + target_power
                   - 2 * (block @ targets.T))
        index = squared.argmin(axis=0)
        values = np.maximum(squared[index, np.arange(len(targets))], 0)
        improved = values < distances
        distances[improved] = values[improved]
        nearest_index[improved] = i + index[improved]
    return nearest_index, distances


def figure(images, output):
    Image.fromarray(np.concatenate([np.concatenate(row, axis=1)
                                    for row in np.split(images, 6)], axis=0)).save(output)


def report(path, model, stats, training, validation, test=None, mean_image=None, *, reconstruction_only=False):
    path.mkdir(parents=True, exist_ok=True)
    stats["architecture"] = {name: asdict(component.config) for name, component in model.components().items()}
    stats["parameters"] = sum(array.size for key, array in model.state().items() if "_weight_" in key)
    codes = None if reconstruction_only else encode(model, training)
    if codes is not None:
        export_generator(path / "decoder.npz", model.decoder, codes, model.coordinate_scale, model.decoder_hidden,
                         skip=model.decoder_skip)
        generator = Generator(path / "decoder.npz")
        generated = np.stack([generator.generate(seed) for seed in range(36)])
        figure(generated, path / "faces.png")
        closest, distances = nearest(training, generated)
        examples = training[closest].reshape(generated.shape)
        figure(np.concatenate((generated, examples), axis=2), path / "nearest-local.png")
        stats["nearest_training_sse"] = distances.tolist()
        stats["latent_std"] = codes.std(axis=0).tolist()
        stats["generated_mean_pixel_std"] = float(generated.astype(np.float32).std(axis=0).mean() / 255)
    if stats["config"]["select_on_training"]:
        stats["training"] = dict(images=len(training), **evaluate(model, training, mean_image, shuffle=False,
                                                                   codes=codes))
        stats["validation"] = dict(images=len(validation), **evaluate(model, validation, mean_image))
    if test is not None:
        print(f"Evaluating {len(test):,} official test portraits", flush=True)
        stats["test"] = dict(images=len(test), **evaluate(model, test, mean_image, shuffle=False))
    (path / "results.json").write_text(json.dumps(stats, indent=2, allow_nan=False) + "\n")
    epochs = [row["epoch"] for row in stats["curve"]]
    count = min(VALIDATION, len(training)) if stats["config"]["select_on_training"] else len(validation)
    fig, ax = plt.subplots(figsize=(7, 4))
    for key, label in (("reconstruction_sse", "Assigned code"), ("shuffled_sse", "Shuffled code"),
                       ("mean_sse", "Mean training portrait")):
        ax.plot(epochs, [row[key] / count for row in stats["curve"]], "o-", label=label)
    ax.set(xlabel="Epoch", ylabel="Summed squared RGB error per image")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path / "convergence.png")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path, help="ZIP containing aligned JPEGs and partition list")
    parser.add_argument("--partitions", type=Path, help="Separate official TXT or CSV partition list")
    parser.add_argument("--output", type=Path, default=Path("runs/celeba"))
    parser.add_argument("--backend", choices=("numpy", "mojo"), default="mojo")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--pilot-only", action="store_true")
    mode.add_argument("--full-only", action="store_true")
    parser.add_argument("--select-on-training", action="store_true",
                        help="Select checkpoints by RGB reconstruction on a fixed training panel")
    parser.add_argument("--initialize-from", type=Path,
                        help="Start a pilot-only or full-only run from another run's selected weights")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--pilot-images", type=int, default=PILOT,
                        help="Number of portraits drawn from the official training split for a pilot")
    parser.add_argument("--full-epochs", type=int,
                        help="Override the epoch budget for the full split")
    parser.add_argument("--checks-per-epoch", type=int, default=1,
                        help="Evaluate at this many evenly spaced checkpoints per epoch when greater than one")
    parser.add_argument("--reconstruction-only", action="store_true",
                        help="Save checkpoints and error scores without fitting a sampler")
    parser.add_argument("--rate", type=float, default=.004)
    parser.add_argument("--encoder-rate", type=float, default=1.)
    parser.add_argument("--rule", choices=("signsgd", "lion", "lion-explicit", "lion-adaptive-agreement"), default="lion")
    parser.add_argument("--bottleneck-pool-size", type=int,
                        help="Size of each equal bottleneck pool, in neurons")
    parser.add_argument("--projected-encoder-votes", action="store_true",
                        help="Assign both encoder layers balanced signed subsets of output votes")
    parser.add_argument("--encoder-votes-per-pair", type=int,
                        help="Use this many signed votes per pair with balanced encoder feedback")
    parser.add_argument("--center-each-input", action="store_true",
                        help="Subtract the mean of each flattened encoder input")
    parser.add_argument("--normalize-each-input", action="store_true",
                        help="Center each encoder input and normalize its RMS")
    parser.add_argument("--projected-latent-votes", action="store_true",
                        help="Give bottleneck pairs fixed signed subsets of output votes")
    parser.add_argument("--paired-decoder-init", action="store_true",
                        help="Initialize each RGB output pair with opposite affine coefficients")
    parser.add_argument("--zero-init-bias", action="store_true",
                        help="Initialize every dense bias to zero")
    parser.add_argument("--decoder-init-scale", type=float, default=1.)
    parser.add_argument("--decoder", choices=("coordinate", "vector"), default="coordinate")
    parser.add_argument("--vector-hidden-width", type=int, default=0,
                        help="Add a shared decoder hidden layer with this width per RGB class")
    parser.add_argument("--vector-hidden-depth", type=int, default=1,
                        help="Number of sequential hidden layers when --vector-hidden-width is positive")
    parser.add_argument("--vector-output-class-width", type=int, default=1,
                        help="Neurons per increase or decrease class for each vector output scalar")
    parser.add_argument("--single-output-neuron", action="store_true",
                        help="Predict each RGB channel from one neuron's clamped magnitude")
    parser.add_argument("--vector-gate-width", type=float, default=.24,
                        help="Gaussian competition width in the image-vector output layer")
    parser.add_argument("--vector-top-k", type=int, default=0,
                        help="Keep this many neurons per RGB output pool. Zero uses Gaussian competition.")
    parser.add_argument("--vector-decoder-skip", action="store_true",
                        help="Give the output layer both the code and the decoder hidden activations")
    parser.add_argument("--projected-decoder-votes", action="store_true",
                        help="Assign hidden decoder pairs balanced signed subsets of output votes")
    parser.add_argument("--depth", type=int, default=6)
    parser.add_argument("--images-per-batch", type=int, default=IMAGES_PER_BATCH)
    parser.add_argument("--pixels-per-image", type=int, default=PIXELS_PER_IMAGE)
    parser.add_argument("--decoder-class-width", type=int, default=32)
    parser.add_argument("--shared-coordinates", action="store_true")
    parser.add_argument("--center-encoder", action="store_true",
                        help="Subtract the training feature mean from encoder inputs")
    parser.add_argument("--gate-width", type=float)
    parser.add_argument("--coordinate-scale", type=float, default=1.)
    parser.add_argument("--encoder-gate-widths", nargs=2, type=float, default=(.24, .18))
    parser.add_argument("--encoder-top-k", nargs=2, type=int, default=[0, 0], metavar=("FIRST", "CODE"),
                        help="Keep this many neurons per pool in each encoder layer. Zero uses Gaussian competition.")
    parser.add_argument("--encoder-width", type=int, default=ENCODER_WIDTH,
                        help="Total number of neurons in the first encoder layer, a multiple of six")
    parser.add_argument("--encoder-hidden-widths", nargs="+", type=int, default=[], metavar="WIDTH",
                        help="Widths of extra dense encoder layers before the bottleneck")
    parser.add_argument("--latent-width", type=int, default=LATENT,
                        help="Number of neurons in the bottleneck, an even integer")
    parser.add_argument("--decoder-rate-decay", type=float, default=1.,
                        help="Rate multiplier for earlier decoder layers")
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()
    if args.initialize_from and not (args.pilot_only or args.full_only):
        parser.error("--initialize-from requires --pilot-only or --full-only")
    if (args.epochs < 1 or (args.full_epochs is not None and args.full_epochs < 1)
            or min(args.rate, args.encoder_rate, args.coordinate_scale) <= 0):
        parser.error("Epoch count, rates and coordinate scale must be positive")
    if args.images_per_batch < 1 or not 1 <= args.pixels_per_image <= len(COORDINATES):
        parser.error("Invalid minibatch dimensions")
    if args.checks_per_epoch < 1:
        parser.error("--checks-per-epoch must be positive")
    args.output.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(args.archive) as archive:
        train_names, val_names, test_names = manifest(archive, args.partitions)
        validation = load_images(archive, subset(val_names, VALIDATION, SEED),
                                 args.output / "validation.npy")
        eligible = True
        if not args.full_only:
            if not 1 <= args.pilot_images <= len(train_names):
                parser.error("--pilot-images must be in the official training split size")
            training = load_images(archive, subset(train_names, args.pilot_images, SEED),
                                   args.output / "pilot-train.npy")
            path = args.output / "pilot"
            model, stats, mean_image = train(training, validation, path, args)
            best = stats["selected"]
            eligible = (best["reconstruction_sse"] < best["mean_sse"]
                        and best["shuffled_sse"] >= 1.1 * best["reconstruction_sse"])
            stats["passes_scale_criterion"] = eligible
            report(path, model, stats, training, validation, mean_image=mean_image,
                   reconstruction_only=args.reconstruction_only)
            print(f"Pilot qualifies for full run: {eligible}", flush=True)
        if eligible and not args.pilot_only:
            training = load_images(archive, train_names, args.output / "full-train.npy")
            path = args.output / "full"
            full_args = argparse.Namespace(**{**vars(args), "epochs": args.full_epochs or args.epochs})
            model, stats, mean_image = train(training, validation, path, full_args)
            test = load_images(archive, test_names, args.output / "test.npy")
            report(path, model, stats, training, validation, test, mean_image,
                   reconstruction_only=args.reconstruction_only)


if __name__ == "__main__":
    main()
