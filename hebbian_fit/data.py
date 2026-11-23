"""Targets, deterministic splits, and the discontinuity benchmark."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

IMAGE = Path(__file__).resolve().parents[1] / "assets/blue-marble-128.png"
BAND_EDGES = np.linspace(-1, 0, 4)
PLATEAU_COUNTS = (4, 16, 64)
EDGES = np.concatenate([
    *(np.linspace(lo, hi, n, endpoint=False)
      for lo, hi, n in zip(BAND_EDGES[:-1], BAND_EDGES[1:], PLATEAU_COUNTS)),
    [0],
])
HEIGHTS = np.tile(np.array([-0.75, -0.25, 0.25, 0.75]), sum(PLATEAU_COUNTS) // 4)


@dataclass
class Dataset:
    x: np.ndarray
    y: np.ndarray
    train: np.ndarray
    validation: np.ndarray
    test: np.ndarray
    shape: tuple | None = None


def scalar_data(task, count=None):
    if count is None:
        count = 512 if task == "sine" else 3072
    if not isinstance(count, (int, np.integer)) or count < 1:
        raise ValueError("The sample count per split must be positive")
    # Phases of one regular grid. None falls exactly on a jump.
    grid = -1 + (np.arange(3 * count) + 0.5) * (2 / (3 * count))
    if task == "sine":
        y = np.sin(2 * np.pi * grid)
    else:
        plateau = np.clip(np.searchsorted(EDGES, grid, side="right") - 1, 0, len(HEIGHTS) - 1)
        y = np.where(grid < 0, HEIGHTS[plateau], 0.75 * np.cos(2 * np.pi * grid))
    return Dataset(
        grid.astype(np.float32)[:, None], y.astype(np.float32)[:, None],
        np.arange(0, len(grid), 3), np.arange(1, len(grid), 3), np.arange(2, len(grid), 3),
    )


def rgb_data(image=IMAGE, size=128, *, split=True):
    with Image.open(image) as opened:
        rgb = ImageOps.exif_transpose(opened).convert("RGB")
        rgb = ImageOps.fit(rgb, (size, size), method=Image.Resampling.LANCZOS)
        y = np.asarray(rgb, dtype=np.float32).reshape(-1, 3) / np.float32(127.5) - 1
    axis = np.linspace(-1, 1, size, dtype=np.float32)
    x = np.stack(np.meshgrid(axis, axis), axis=-1).reshape(-1, 2)
    if not split:
        return Dataset(x, y, np.arange(len(x)), np.empty(0, dtype=int), np.empty(0, dtype=int), (size, size))
    permutation = np.random.default_rng(2026).permutation(len(x))
    n = len(x) // 10
    return Dataset(x, y, permutation[2 * n:], permutation[n:2 * n], permutation[:n], (size, size))


def step_metrics(x, error):
    x = x.ravel()
    error = error.ravel()
    smooth = x >= 0
    plateau_mae = []
    jump_neighborhood = np.zeros(len(x), dtype=bool)
    for i, (lo, hi) in enumerate(zip(EDGES[:-1], EDGES[1:])):
        inset = (hi - lo) / 4
        center = (x >= lo + inset) & (x < hi - inset)
        plateau_mae.append(float(np.mean(np.abs(error[center]))))
        if i + 2 < len(EDGES):
            radius = min(inset, (EDGES[i + 2] - hi) / 4)
            jump_neighborhood |= (x >= hi - radius) & (x < hi + radius)
    return dict(
        smooth_rmse=float(np.sqrt(np.mean(np.square(error[smooth], dtype=np.float64)))),
        max_plateau_mae=max(plateau_mae),
        plateau_mae=plateau_mae,
        jump_mae=float(np.mean(np.abs(error[jump_neighborhood]))),
    )
