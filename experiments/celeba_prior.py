"""Attach a mixture of empirical latent Gaussians to a trained CelebA decoder."""

import argparse
import json
from pathlib import Path

import numpy as np

from experiments.celeba import encode, load_checkpoint
from experiments.celeba_diversity import model_for


def fit_components(codes, components, local_scale, local_rank=None):
    # Group codes in their eight leading standardized directions. Statistics
    # within each group retain all code dimensions for sampling.
    values, axes = np.linalg.eigh(np.cov(codes, rowvar=False))
    embedded = (codes - codes.mean(axis=0)) @ axes[:, -8:][:, ::-1]
    embedded /= np.sqrt(np.maximum(values[-8:][::-1], 1e-8))
    rng = np.random.default_rng(0)
    centers = embedded[rng.choice(len(codes), components, replace=False)]
    membership = None
    for _ in range(30):
        distances = (centers**2).sum(axis=1) - 2 * embedded @ centers.T
        assigned = distances.argmin(axis=1)
        if np.array_equal(membership, assigned):
            break
        membership = assigned
        for i in range(components):
            members = embedded[membership == i]
            if len(members):
                centers[i] = members.mean(axis=0)

    weights, means, roots = [], [], []
    for i in range(components):
        members = codes[membership == i]
        if len(members) < codes.shape[1]:
            continue
        values, axes = np.linalg.eigh(np.cov(members, rowvar=False))
        if local_rank is not None:
            values, axes = values[-local_rank:], axes[:, -local_rank:]
        weights.append(len(members))
        means.append(members.mean(axis=0, dtype=np.float64))
        roots.append(local_scale * axes * np.sqrt(np.maximum(values, 0)))
    if not weights:
        raise ValueError("Each retained group needs at least one observation per code dimension")
    counts = np.asarray(weights, dtype=np.float64)
    return dict(mixture_weights=counts / counts.sum(), mixture_means=np.stack(means),
                mixture_roots=np.stack(roots))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path, help="Pilot or full results directory")
    parser.add_argument("--components", type=int, default=16)
    parser.add_argument("--local-scale", type=float, default=.6)
    parser.add_argument("--local-rank", type=int, help="Keep this many leading directions in each component")
    parser.add_argument("--output", type=Path, required=True, help="Mixture decoder export")
    args = parser.parse_args()
    if args.components < 2 or not np.isfinite(args.local_scale) or args.local_scale <= 0:
        parser.error("Choose at least two components and a positive finite local scale")
    if args.local_rank is not None and args.local_rank < 1:
        parser.error("Local rank must be positive")
    source = args.run / "decoder.npz"
    if source.resolve() == args.output.resolve():
        parser.error("Choose an output path distinct from the original decoder")
    config = json.loads((args.run / "results.json").read_text())["config"]
    model = model_for(config, args.run)
    if args.local_rank is not None and args.local_rank > model.latent_width:
        parser.error("Local rank cannot exceed code width")
    load_checkpoint(args.run / "selected.npz", model)
    data = args.run.parent / ("full-train.npy" if args.run.name == "full" else "pilot-train.npy")
    codes = encode(model, np.load(data, mmap_mode="r")).astype(np.float64)
    mixture = fit_components(codes, args.components, args.local_scale, args.local_rank)
    with np.load(source, allow_pickle=False) as saved:
        export = {**dict(saved), **mixture}
    temporary = args.output.with_suffix(".tmp.npz")
    np.savez_compressed(temporary, **export)
    temporary.replace(args.output)
    print(f"Exported {len(mixture['mixture_weights'])} Gaussian components to {args.output}")


if __name__ == "__main__":
    main()
