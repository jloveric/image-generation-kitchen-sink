# Supervised Hebbian Class Networks for Continuous Outputs

[Read the paper (PDF)](main.pdf) · [LaTeX source](main.tex) · [Bibliography](references.bib)

The manuscript reports four network sizes on one discontinuous sine target, the all-pixel Blue Marble reconstruction, and the sequential CelebA autoencoder. The portrait panel and reconstruction score use the checkpoint chosen on the official validation split.

## Build the paper

The figures are included. From this directory, run `tectonic main.tex` or `latexmk -pdf main.tex`. Either produces `main.pdf` without training a model or obtaining CelebA. The manuscript uses the TMLR template's preprint format. The bundled style and bibliography files come from the official template at revision `7bf90efe`. [TMLR template](https://github.com/JmlrOrg/tmlr-style-file/tree/7bf90efe3a0debbba703c05c43f3ff7e4d4a2992) · [Template license](TMLR-LICENSE)

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

The first command trains the two phases on all 16,384 pixel centers and records the complete-image error. The second reads the final checkpoint and uses NumPy to render both paper figures. For each of the twelve layers it shows the layer’s own RGB estimate and the accumulated contribution through that layer, computed by averaging the estimates up to and including it. It checks that the accumulated contribution through layer 12, after clipping and rounding, equals the full NumPy prediction at every pixel. The summed squared RGB difference on the 0–1 scale is **47.17** (47.174596 in the GPU training record and 47.173950 in the NumPy replay). To replay a checkpoint stored elsewhere, pass its path with `--checkpoint`. The 128 × 128 target is bundled with the repository. [Input image](../assets/blue-marble-128.png) · [NASA source](https://science.nasa.gov/resource/the-blue-marble/) · [Run record](../assets/results/blue-marble-output-depth-12-two-phase-all-pixels.json) · [Photograph and fit](figures/earth.pdf) · [Individual layers and accumulated contributions](figures/earth-layers.pdf)

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

### SignSGD comparison

SignSGD uses the sign of the current minibatch's signed Hebbian coactivity for each connection. Lion combines that coactivity with its history before taking the sign. For sine and Blue Marble, the paper's first phase already uses SignSGD. The comparison uses it in every phase, preserving the original step sizes, schedules, minibatches and checkpoint selection. CelebA uses the same training and evaluation splits and reports reconstruction error on the official test split.

| Experiment | Error measured | Paper schedule with Lion | SignSGD throughout |
| --- | --- | ---: | ---: |
| Sine, 1 × 512 | Median test RMSE | 0.09044 | 0.09168 |
| Sine, 4 × 64 | Median test RMSE | 0.03168 | 0.03227 |
| Sine, 12 × 64 | Median test RMSE | 0.02634 | 0.02854 |
| Sine, 24 × 64 | Median test RMSE | 0.01968 | 0.02346 |
| Blue Marble | Summed squared RGB error at 16,384 pixel centers | 47.17 | 167.96 |
| CelebA | Test summed squared RGB error per portrait | 144.21 | 146.71 |

At 24 sine layers, the median 10–90% rise width is 0.0223 with Lion and 0.0411 with SignSGD. [Fit comparison](../assets/results/sine-jump-fit-signsgd.png) · [Sine measurements](../assets/results/sine-jump-fit-signsgd.json)

Blue Marble used the same saved first phase in both runs. During the second phase, SignSGD reached its best error at epoch 250 of 480. Its error at epoch 480 was 169.26. Lion reached its best error at epoch 480. At points halfway between pixels, the squared error against bilinear interpolation was 87.28 for SignSGD and 97.62 for Lion. [Convergence comparison](../assets/results/blue-marble-output-depth-12-two-phase-all-pixels-signsgd-comparison.png) · [SignSGD image](../assets/results/blue-marble-output-depth-12-two-phase-signsgd-all-pixels.png) · [Blue Marble measurements](../assets/results/blue-marble-output-depth-12-two-phase-signsgd-all-pixels.json)

The CelebA reconstruction error differs by 1.7%. Both runs selected the final checkpoint on the 512-portrait validation panel. The SignSGD checkpoint comes from the earlier run `target150-standard-full128-small-broad-output`, whose configuration matches the Lion run except for the rule. A NumPy replay on all 19,962 test portraits reproduced its recorded error to five decimal places. [Matched reconstructions](../assets/results/celeba-target150-signsgd-reconstructions.png) · [SignSGD decoder samples](../assets/results/celeba-target150-signsgd-portraits.png) · [CelebA measurements](../assets/results/celeba-target150-signsgd.json)

Reproduce the coordinate comparisons from the repository root.

```bash
OPENBLAS_NUM_THREADS=1 python -m papers.signsgd sine --workers 4
python -m papers.signsgd marble
```

For CelebA, use the training command above with the output directory set to `runs/celeba-sequential/full/target150-standard-full128-broad-signsgd` and the rule set to `--rule signsgd`. With `run` set to that same directory, fit the code distribution and render the comparison.

```bash
OPENBLAS_NUM_THREADS=4 python -m experiments.celeba_prior "$run/full" \
  --components 16 --local-rank 64 --local-scale .85 --output "$run/full/decoder-mixture.npz"
python -m papers.signsgd portraits --run "$run/full"
```
