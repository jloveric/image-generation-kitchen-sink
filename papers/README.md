# Supervised Hebbian Class Networks: Regression Without Backpropagation

[Read the paper (PDF)](main.pdf) · [LaTeX source](main.tex) · [Bibliography](references.bib)

This paper extends John Loverich’s supervised Hebbian classifier to regression. All layers learn together from local Hebbian updates. The curve and Blue Marble networks sum signed activities. In the CelebA autoencoder each hidden neuron learns from its own group of output error signs, and each output neuron predicts one intensity. [Original paper, *Supervised Hebbian Class Networks: Learning Without Backpropagation*](https://arxiv.org/abs/2611.24314)

The manuscript reports four network sizes on one discontinuous sine target, the all-pixel Blue Marble reconstruction, and the sequential CelebA autoencoder with a 256-value bottleneck. The image panel and reconstruction score use the checkpoint chosen on the official validation split.

## Build the paper

The figures are included. From this directory, run `tectonic main.tex` or `latexmk -pdf main.tex`. Either produces `main.pdf` without training a model or obtaining CelebA. The manuscript uses the TMLR template's preprint format. The bundled style and bibliography files come from the official template at revision `7bf90efe`. [TMLR template](https://github.com/JmlrOrg/tmlr-style-file/tree/7bf90efe3a0debbba703c05c43f3ff7e4d4a2992) · [Template license](TMLR-LICENSE)

## Reproduce the figures and measurements

Run the following commands in Bash **from the repository root**. The sine fits use NumPy on the CPU. The signed-activity Blue Marble fit and the CelebA runs use the Mojo backend on an NVIDIA RTX 5070 Laptop GPU. Use Python 3.12 or later and the project dependencies. Set `HEBBIAN_GPU_TARGET` to the target of your GPU if it differs from the default `sm_120`.

```bash
uv sync --extra gpu
source .venv/bin/activate
export OPENBLAS_NUM_THREADS=1
python -m papers.diagrams
```

For just the diagrams and the sine fits, `uv sync` installs the CPU dependencies. The diagrams are editable vector figures. [Blue Marble network diagram](figures/implicit-image.pdf) · [CelebA autoencoder diagram](figures/autoencoder.pdf) · [Feedback to one hidden neuron](figures/pixel-votes.pdf) · [Diagram source](diagrams.py) · [Figure and measurement code](results.py)

### Discontinuous sine

The dataset is generated from $0.5\sin(2\pi x)+0.5\,\mathbf{1}_{x>0.25}$ on $[-1,1]$. There is no external download. [Target definition](../experiments/sine_jump.py) · [Grid and splits](../hebbian_fit/data.py)

```bash
python -m experiments.signed_activity sine --depth 1 --width-per-output 512
for depth in 4 12 24; do
  python -m experiments.signed_activity sine --depth "$depth" --width-per-output 64
done
python -m papers.results sine
```

For each size, train three independent fits for 640 epochs in batches of 64, using one cosine starting at 0.00075. Each split contains 4,096 distinct points. For each fit, select weights by validation error. The paper reports the median test RMSE at each size and plots the checkpoint that attained that value. [Curve fits and zoom](figures/sine.pdf)

| Hidden layers × width | Parameters | Max-abs test RMSE (paper) | Smooth-center test RMSE |
| --- | ---: | ---: | ---: |
| 1 × 512 | 1,024 | [0.26120](../assets/results/sine-signed-activity-1x512-0.00075-b64-e640-max-abs.json) | [0.37958](../assets/results/sine-signed-activity-1x512-0.00075-b64-e640.json) |
| 4 × 64 | 12,608 | [0.03095](../assets/results/sine-signed-activity-4x64-0.00075-b64-e640-max-abs.json) | [0.02856](../assets/results/sine-signed-activity-4x64-0.00075-b64-e640.json) |
| 12 × 64 | 45,888 | [0.01592](../assets/results/sine-signed-activity-12x64-0.00075-b64-e640-max-abs.json) | [0.01331](../assets/results/sine-signed-activity-12x64-0.00075-b64-e640.json) |
| 24 × 64 | 95,808 | [0.01141](../assets/results/sine-signed-activity-24x64-0.00075-b64-e640-max-abs.json) | [0.01131](../assets/results/sine-signed-activity-24x64-0.00075-b64-e640.json) |

The paper uses the maximum absolute preactivation in each pool as the Gaussian center. To reproduce the smooth-center column, repeat the training commands above with `--sine-center smooth`. After both sets of runs, `python -m papers.results sine-centers` plots the median fit for each configuration. [Curve fits with both centers](../assets/results/sine-gaussian-centers.png)

### Blue Marble

The exact input is the bundled 128 × 128 RGB crop of Apollo 17 frame AS17-148-22727. The loader scales its pixel coordinates and channel values to $[-1,1]$. Each hidden layer has 144 neurons per color. The red, green and blue pools receive their own channel’s error sign on each training example. All connections between adjacent layers are dense. Use the bundled PNG to match the paper. [Input image](../assets/blue-marble-128.png) · [NASA source photograph](https://science.nasa.gov/resource/the-blue-marble/)

```bash
OPENBLAS_NUM_THREADS=1 python -m experiments.signed_activity marble --backend mojo
OPENBLAS_NUM_THREADS=2 python -m papers.results marble
```

The signed-activity model trains on all 16,384 pixel centers for 560 epochs with batches of 128 and one cosine starting at 0.004. It reaches **9.22** summed squared RGB error on the $[0,1]$ scale. The second command uses the selected weights to render the individual and accumulated layer contributions. [Photograph and fit](figures/earth.pdf) · [Layer contributions](figures/earth-layers.pdf) · [Run settings and measurements](../assets/results/marble-signed-activity-12x432-0.004-b128-e560-mojo.json)

### CelebA

Obtain the aligned CelebA JPEG archive and official evaluation partition list from the dataset authors. The loader accepts a ZIP of aligned JPEGs and their original filenames. If the list is already inside the ZIP, omit the `--partitions` argument. The loader center-crops each photograph to a square and resizes it to 64 × 64 RGB. [Dataset, partitions and terms of use](https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html) · [Training and preprocessing](../experiments/celeba.py)

```bash
face_archive=/path/to/img_align_celeba.zip
face_partitions=/path/to/list_eval_partition.txt
run=runs/celeba-single-output-tuning/independent-hidden-votes

OPENBLAS_NUM_THREADS=2 python -m experiments.celeba "$face_archive" --partitions "$face_partitions" \
  --output "$run" --backend mojo --full-only --full-epochs 3 --checks-per-epoch 2 \
  --reconstruction-only --decoder vector --encoder-width 3072 --encoder-hidden-widths 1536 \
  --latent-width 256 --encoder-top-k 768 86 --encoder-gate-widths .85 .85 --center-encoder \
  --projected-encoder-votes --projected-decoder-votes --vector-hidden-depth 1 --vector-hidden-width 340 \
  --single-output-neuron --independent-hidden-votes --decoder-init-scale .12 --images-per-batch 128 \
  --rule lion --rate .0008 --encoder-rate .10 --decoder-rate-decay .10

python -m papers.results reconstruction --run "$run/full" --example papers/figures/celeba-reconstruction.png
python -m papers.diagrams
OPENBLAS_NUM_THREADS=2 python -m experiments.celeba_prior "$run/full" --backend mojo --checkpoint selected \
  --components 16 --local-rank 64 --local-scale .85 --output "$run/full/decoder-mixture-selected.npz"
python -m papers.results images --mixture "$run/full/decoder-mixture-selected.npz"
```

The layer widths are **12,288 → 3,072 → 1,536 → 256 → 2,040 → 12,288**, with 68,469,496 learned weights and biases. The width argument of 340 gives 680 neurons in each of three decoder hidden pools. Each final neuron predicts one 8-bit intensity by clipping its activity magnitude at one and rounding. [Network and local training rule](../hebbian_fit/autoencoder.py) · [Output rule](../hebbian_fit/model.py) · [Run settings, checkpoint selection and test measurements](../assets/results/celeba-independent-hidden-votes-full.json)

Training uses semi-implicit Hebbian Lion throughout three passes over all 162,770 training images. The RGB output starts at 0.0008 and the four hidden layers at 0.00008. All five layers hold their starting steps for the first fifth of training and follow one half cosine to zero. The lowest error on the 512 validation images occurs at 2.5 passes. The selected weights are in `$run/full/selected.npz`. On 19,962 official test images the sum of squared RGB differences on the [0, 1] scale is 3,428,247.35, or **171.74 per image**.

The reconstruction command scores the selected weights and saves the first test image beside its reconstruction for the network diagram. The sampler command encodes the training split with those same weights and fits a 16-component distribution in the embedding space. The image panel uses the first 36 draws, indexed 0 through 35, from that export. [Network diagram](figures/autoencoder.pdf) · [Test image and reconstruction](figures/celeba-reconstruction.png) · [Sampler source](../experiments/celeba_prior.py)

![36 images generated by the CelebA decoder used in the manuscript](figures/celeba-generated.png)

#### Independent feedback for hidden neurons

The curve, Blue Marble and CelebA experiments have 1, 3 and 12,288 output classes, respectively. In CelebA, each output neuron predicts the intensity of one class. Figure 7 shows two adjacent hidden layers and the output classes assigned to individual neurons in each. Each hidden neuron receives a fixed group of class error signs. Half of the group's error signs enter the sum with multiplier $+1$ and half with $-1$. Each covered channel belongs to two groups in that layer. The groups are formed by partitioning the same set of channels twice and shuffling the resulting groups across the neurons. The hidden layers receive 8, 16, 96 and 12 votes per neuron, respectively. All 256 bottleneck neurons have distinct groups. The sum for each neuron multiplies its signed activity in the Hebbian coactivity calculation.
