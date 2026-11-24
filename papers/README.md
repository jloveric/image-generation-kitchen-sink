# Supervised Hebbian Class Networks for Continuous Outputs

[Read the paper (PDF)](main.pdf) · [LaTeX source](main.tex) · [Bibliography](references.bib)

The manuscript reports four network sizes on one discontinuous sine target, the all-pixel Blue Marble reconstruction, and the sequential CelebA autoencoder. The portrait panel and reconstruction score use the checkpoint chosen on the official validation split.

## Build the paper

The figures are included. From this directory, run `tectonic main.tex` or `latexmk -pdf main.tex`. Either produces `main.pdf` without training a model or obtaining CelebA. The manuscript uses the journal's anonymous submission format. The bundled style and bibliography files come from the official template at revision `7bf90efe`. [TMLR template](https://github.com/JmlrOrg/tmlr-style-file/tree/7bf90efe3a0debbba703c05c43f3ff7e4d4a2992) · [Template license](TMLR-LICENSE)

## Reproduce the figures and measurements

Run the following commands in Bash **from the repository root**. The scalar fits and the diagram scripts use NumPy and Matplotlib. The published Earth and portrait training runs used the Mojo backend on an NVIDIA RTX 5070 Laptop GPU. Use Python 3.12 or later and the project dependencies. Set `HEBBIAN_GPU_TARGET` to the target of your GPU if it differs from the default `sm_120`.

```bash
uv sync --extra gpu
source .venv/bin/activate
export OPENBLAS_NUM_THREADS=1
python -m papers.diagrams
```

For just the diagrams and the sine fits, `uv sync` installs the CPU dependencies. The two diagrams are editable vector figures. [Diagram source](diagrams.py) · [Figure and measurement code](results.py)

### Discontinuous sine

```bash
python -m papers.results sine
```

This trains the four reported configurations from the analytic target or loads the same completed runs from `runs/scalar-quality`. It prints the median errors and rise width for each configuration and writes the paper figure. Each split contains 4,096 distinct points in the training interval. Median test RMSE is 0.09044 for 1 × 512, 0.03168 for 4 × 64, 0.02634 for 12 × 64 and 0.01968 for the tuned 24 × 64 network. To reproduce the original validation search as well, use `python -m experiments.sine_jump`. [Target and experiment](../experiments/sine_jump.py) · [Recorded measurements](../assets/results/sine-jump-fit.json) · [Paper figure](figures/sine.pdf)

### Blue Marble

```bash
python -m experiments.blue_marble_quality --backend mojo --capacity --extend \
  --fit-all-pixels output-depth-12-two-phase
python -m papers.results marble
```

The first command trains the two phases on all 16,384 pixel centers and records the complete-image error. The second reads the final checkpoint, evaluates it on the same backend, and writes the target, fit and matching detail crops. The expected sum of squared RGB differences on the 0–1 scale is **47.174596**. The 128 × 128 target is bundled with the repository. [Input image](../assets/blue-marble-128.png) · [NASA source](https://science.nasa.gov/resource/the-blue-marble/) · [Run record](../assets/results/blue-marble-output-depth-12-two-phase-all-pixels.json) · [Paper figure](figures/earth.pdf)

### CelebA

Obtain the aligned CelebA JPEG archive and official evaluation partition list from the dataset authors. The loader accepts a ZIP of aligned JPEGs and their original filenames. If the list is already inside the ZIP, omit the `--partitions` argument. The loader center-crops each photograph to a square and resizes it to 64 × 64 RGB. [Dataset, partitions and terms of use](https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html) · [Training and preprocessing](../experiments/celeba.py)

```bash
face_archive=/path/to/img_align_celeba.zip
face_partitions=/path/to/list_eval_partition.txt
run=runs/celeba-sequential/full/target150-standard-full128-broad-lion

OPENBLAS_NUM_THREADS=2 python -m experiments.celeba "$face_archive" --partitions "$face_partitions" \
  --backend mojo --full-only --full-epochs 1 --checks-per-epoch 4 --reconstruction-only --output "$run" \
  --decoder vector --encoder-width 3072 --latent-width 2048 --encoder-gate-widths .85 .85 \
  --projected-encoder-votes --center-encoder --rate .004 --encoder-rate .15 --images-per-batch 128 --decoder-init-scale .12 --vector-gate-width .50 --rule lion

python -m papers.results reconstruction --run "$run/full"
OPENBLAS_NUM_THREADS=4 python -m experiments.celeba_prior "$run/full" \
  --components 16 --local-rank 64 --local-scale .85 --output "$run/full/decoder-mixture.npz"
python -m papers.results portraits --run "$run/full"
```

The training command makes one pass over all 162,770 training portraits. It selects the checkpoint with the lowest error on 512 official validation portraits. The paper uses the selected checkpoint for both reconstruction and generation. Its expected sum of squared RGB differences per portrait is **144.210077** on the 19,962 official test portraits. The sampler fits the selected encoder's codes for all training portraits. An existing mixture export is sufficient to draw the 36 portraits. [Sampler](../experiments/celeba_prior.py) · [Tuning results](../assets/results/celeba-target150.json) · [Paper portrait panel](figures/portraits.png)
