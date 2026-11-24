"""Dense image autoencoder trained with signed local coactivity."""

import argparse
from dataclasses import asdict
import json
from pathlib import Path

import numpy as np
from PIL import Image

from .gpu import MojoModel, batch_array
from .model import ModelConfig, NumpyModel, select, top_k_activity, with_bias

SIZE = 64
LATENT = 64
ENCODER_WIDTH = 192
AXIS = np.linspace(-1, 1, SIZE, dtype=np.float32)
COORDINATES = np.stack(np.meshgrid(AXIS, AXIS), axis=-1).reshape(-1, 2)


def decoder_config(rule="signsgd", depth=6, gate_widths=None, rate_decay=1., class_width=32,
                   latent_width=LATENT):
    return ModelConfig(inputs=latent_width + 2, outputs=3, width=class_width, depth=depth,
                       competition="output", gate="gaussian",
                       layer_gate_widths=gate_widths or tuple(np.linspace(.24, .14, depth)),
                       readout="class-sum", readout_layers="all", hidden_rule=rule,
                       output_bound=1, output_levels=256, layer_rate_decay=rate_decay)


def vector_decoder_config(rule, inputs=LATENT, class_width=1, gate_width=.24, top_k=0):
    # The model sees a code as one input vector and a portrait as one output vector.
    return ModelConfig(inputs=inputs, outputs=SIZE * SIZE * 3, width=class_width, depth=1,
                       competition="output", gate="hard" if top_k else "gaussian",
                       gate_width=gate_width, top_k=top_k or None,
                       readout="class-sum" if class_width == 1 else "fixed",
                       hidden_rule=rule, output_bound=1, output_levels=256)


def encoder_configs(inputs, rule, gate_widths=(.24, .18), encoder_width=ENCODER_WIDTH, latent_width=LATENT,
                    top_k=(0, 0), hidden_widths=()):
    widths = (encoder_width, *hidden_widths)
    return (
        ModelConfig(inputs=inputs, outputs=3, width=encoder_width // 6, depth=1,
                    competition="output", gate="hard" if top_k[0] else "gaussian", gate_width=gate_widths[0],
                    top_k=top_k[0] or None,
                    hidden_rule=rule),
        *(ModelConfig(inputs=previous, outputs=3, width=width // 6, depth=1,
                      competition="output", gate="gaussian", gate_width=gate_widths[0], hidden_rule=rule)
          for previous, width in zip(widths, hidden_widths)),
        ModelConfig(inputs=widths[-1], outputs=1, width=latent_width // 2, depth=1,
                    competition="output", gate="hard" if top_k[1] else "gaussian", gate_width=gate_widths[1],
                    top_k=top_k[1] or None,
                    hidden_rule=rule),
    )


def signed_totals(signs):
    # Count positive and negative votes independently. Opposite votes cancel in
    # the coactivity numerator, but both still contribute to normalization.
    positive = np.count_nonzero(signs > 0, axis=1)
    negative = np.count_nonzero(signs < 0, axis=1)
    return (positive - negative).astype(np.float32), (positive + negative).astype(np.float32)


def vote_assignment(pairs, seed, balanced, votes_per_pair=None):
    rng = np.random.default_rng(seed)
    outputs = SIZE * SIZE * 3
    votes = outputs // pairs
    if balanced:
        votes -= votes % 2
    if votes < 1 or (votes_per_pair is not None and not 1 <= votes_per_pair <= votes):
        raise ValueError("Too many encoder pairs or votes per pair for disjoint output feedback")
    groups = rng.permutation(outputs)[:pairs * votes].reshape(pairs, votes)
    if votes_per_pair is not None:
        groups = groups[:, :votes_per_pair].copy()
    polarity = (np.where(np.arange(groups.shape[1]) % 2, -1., 1.).astype(np.float32) if balanced
                else rng.choice(np.array([-1, 1], np.float32), size=groups.shape))
    return groups, polarity


def projected_votes(teaching, projection, pair_index):
    groups, polarity = projection
    votes = teaching[:, groups] * polarity
    return (np.ascontiguousarray(votes.sum(axis=2)[:, pair_index]),
            np.ascontiguousarray(np.count_nonzero(votes, axis=2)[:, pair_index], dtype=np.float32))


def class_pair_indices(width):
    class_width = width // 6
    return np.broadcast_to(np.arange(width // 2).reshape(3, 1, class_width), (3, 2, class_width)).ravel()


class EncoderLayer:
    def __init__(self, config, seed, capacity, backend, bottleneck, pool_size=None):
        self.config = config
        self.bottleneck = bottleneck
        if bottleneck:
            self.channel_width = 2 * ((config.width + 2) // 3)
            self.pool_size = pool_size if pool_size is not None else self.channel_width
            if config.gate == "hard" and config.keep_per_pool > self.pool_size:
                raise ValueError("Top-K count exceeds the bottleneck competition pool")
            self.groups = tuple(slice(i, i + self.pool_size) for i in range(0, config.units, self.pool_size))
            self.channels = np.arange(config.units) // self.channel_width
            self.polarity = (1 - 2 * (np.arange(config.units) % 2)).astype(np.float32)
        else:
            self.channel_width = self.pool_size = 0
            nodes = np.arange(config.units)
            self.channels = nodes // (2 * config.width)
            self.polarity = (1 - 2 * (nodes // config.width % 2)).astype(np.float32)
        self.model = (MojoModel(config, seed, capacity, encoder=True)
                      if backend == "mojo" else NumpyModel(config, seed))

    def forward(self, x):
        x = batch_array(x, self.config.inputs)
        if isinstance(self.model, MojoModel):
            out = np.empty((len(x), self.config.units), np.float32)
            self.model.network.encode(x, out, self.pool_size)
            return out, None
        incoming = with_bias(x)
        z = incoming @ self.model.weights[0]
        if self.bottleneck:
            out = np.empty_like(z)
            for group in self.groups:
                values = z[:, group]
                if self.config.gate == "hard":
                    out[:, group], _ = top_k_activity(values, self.config.keep_per_pool)
                else:
                    magnitudes = np.abs(values)
                    peak = magnitudes.max(axis=1, keepdims=True)
                    distance = (peak - magnitudes) / np.where(peak > 0, peak * self.config.gate_width, 1)
                    out[:, group] = values * np.exp(-.5 * np.minimum(distance, 13) ** 2)
        else:
            out, _ = select(z, self.config)
        return out, (incoming, out)

    def update(self, traces, signs, counts, rate):
        if isinstance(self.model, MojoModel):
            self.model.network.update_encoder(signs, counts, float(rate), self.channel_width)
            return
        incoming, activity = traces
        channels = np.arange(len(self.channels)) if signs.shape[1] == len(self.channels) else self.channels
        contributors = counts[:, channels] * (activity != 0)
        total = contributors.sum(axis=0)
        active = total > 0
        normalizer = total[active]
        activity = activity[:, active]
        directions = signs[:, channels[active]] * self.polarity[active]
        coactivity = incoming.T @ (activity * directions) / normalizer
        squared = (contributors[:, active] * activity**2).sum(axis=0) / normalizer
        if self.config.hidden_rule == "lion":
            history = self.model.history[0]
            signal = .125 * history[:, active] + .875 * coactivity
            history[:, active] = .99 * history[:, active] + .01 * coactivity
        else:
            signal = coactivity
        direction = np.sign(signal) / np.float32(np.sqrt(incoming.shape[1]))
        weights = self.model.weights[0]
        weights[:, active] = (weights[:, active] + rate * direction) / (1 + rate * squared)


def decode(decoder, codes, coordinates=COORDINATES, batch=1024, coordinate_scale=1.):
    # Each query specifies which image to draw and the point to sample.
    shape = (len(codes), len(coordinates), 3)
    result = np.empty((shape[0] * shape[1], 3), np.float32)
    for start in range(0, len(result), batch):
        indices = np.arange(start, min(start + batch, len(result)))
        x = np.concatenate((codes[indices // len(coordinates)],
                            coordinates[indices % len(coordinates)] * coordinate_scale), axis=1)
        result[indices] = decoder.predict_values(x)
    return result.reshape(shape)


def vector_decoder_features(codes, hidden, skip):
    # Every hidden layer sees the preceding stage. Only the output gets skip inputs.
    features = codes
    stages = [codes]
    traces = []
    for layer in hidden_layers(hidden):
        features, trace = layer.forward(features)
        stages.append(features)
        traces.append(trace)
    if skip and len(stages) > 1:
        features = np.concatenate(stages, axis=1)
    return features, traces


def hidden_layers(hidden):
    if hidden is None:
        return ()
    if isinstance(hidden, EncoderLayer):
        return (hidden,)
    return hidden


def hidden_name(index):
    # The first layer keeps its original checkpoint and decoder export keys.
    return "hidden" if index == 0 else f"hidden{index + 1}"


def decode_vector(decoder, codes, batch, hidden=None, skip=False):
    predicted = []
    for i in range(0, len(codes), batch):
        incoming, _ = vector_decoder_features(codes[i:i + batch], hidden, skip)
        predicted.append(decoder.predict_values(incoming))
    return np.concatenate(predicted).reshape(-1, SIZE * SIZE, 3)


class Autoencoder:
    def __init__(self, seed, backend="numpy", images_per_batch=32, queries_per_batch=1024,
                 inputs=SIZE * SIZE * 3, rule="signsgd", depth=6, gate_widths=None,
                 coordinate_scale=1., decoder_rate_decay=1., encoder_gate_widths=(.24, .18),
                 decoder_class_width=32, input_center=None, decoder_kind="coordinate",
                 decoder_init_scale=1., projected_latent_votes=False, bottleneck_pool_size=None,
                 projected_encoder_votes=False, center_each_input=False, vector_hidden_width=0,
                 normalize_each_input=False, paired_decoder_init=False, encoder_votes_per_pair=None,
                 encoder_width=ENCODER_WIDTH, latent_width=LATENT, vector_output_class_width=1,
                 vector_decoder_skip=False, projected_decoder_votes=False, vector_hidden_depth=1,
                 vector_gate_width=.24, zero_init_bias=False, encoder_top_k=(0, 0), vector_top_k=0,
                 encoder_hidden_widths=()):
        if decoder_kind not in ("coordinate", "vector"):
            raise ValueError("Unknown decoder kind")
        if vector_hidden_depth < 1 or (vector_hidden_depth > 1 and not vector_hidden_width):
            raise ValueError("Stacking decoder layers requires a positive vector hidden width and depth")
        if vector_hidden_width < 0 or (vector_hidden_width and decoder_kind != "vector"):
            raise ValueError("Extra decoder width requires the image-vector decoder")
        if (vector_decoder_skip or projected_decoder_votes) and not vector_hidden_width:
            raise ValueError("Decoder skip and projected votes require a vector hidden layer")
        if vector_hidden_width and not np.finfo(np.float32).tiny <= decoder_rate_decay <= 1:
            raise ValueError("Vector hidden layer rate multiplier must be in (0, 1]")
        if vector_output_class_width < 1 or (vector_output_class_width != 1 and decoder_kind != "vector"):
            raise ValueError("Extra output class width requires the image-vector decoder")
        if len(encoder_top_k) != 2 or min(*encoder_top_k, vector_top_k) < 0:
            raise ValueError("Supply two nonnegative encoder Top-K counts and a nonnegative output count")
        if vector_top_k and decoder_kind != "vector":
            raise ValueError("Output Top-K requires the image-vector decoder")
        if encoder_width < 6 or encoder_width % 6 or latent_width < 6 or latent_width % 2:
            raise ValueError("Encoder width must be a positive multiple of six and code width an even integer of at least six")
        encoder_hidden_widths = tuple(encoder_hidden_widths)
        if any(width < 6 or width % 6 for width in encoder_hidden_widths):
            raise ValueError("Extra encoder widths must be positive multiples of six")
        self.decoder_kind = decoder_kind
        self.latent_width = latent_width
        if bottleneck_pool_size is not None and (
                bottleneck_pool_size < 2 or bottleneck_pool_size % 2 or latent_width % bottleneck_pool_size):
            raise ValueError("Bottleneck pool size must evenly divide the code and contain complete pairs")
        if (projected_latent_votes or projected_encoder_votes) and decoder_kind != "vector":
            raise ValueError("Projected encoder votes require the image-vector decoder")
        if encoder_votes_per_pair is not None and (
                not projected_encoder_votes or encoder_votes_per_pair < 2 or encoder_votes_per_pair % 2):
            raise ValueError("Sparse balanced feedback needs a positive even vote count")
        encoder_widths = (encoder_width, *encoder_hidden_widths)
        vote_seeds = (1, *range(2 + vector_hidden_depth, 2 + vector_hidden_depth + len(encoder_hidden_widths)))
        self.encoder_projections = [
            (vote_assignment(width // 2, vote_seed, True, encoder_votes_per_pair), class_pair_indices(width))
            if projected_encoder_votes else (None, None)
            for width, vote_seed in zip(encoder_widths, vote_seeds, strict=True)
        ]
        self.encoder_projections.append((
            vote_assignment(latent_width // 2, 0, projected_encoder_votes, encoder_votes_per_pair)
            if projected_latent_votes or projected_encoder_votes else None,
            np.arange(latent_width) // 2,
        ))
        self.center_each_input = center_each_input or normalize_each_input
        self.normalize_each_input = normalize_each_input
        self.coordinate_scale = coordinate_scale
        self.input_center = None if input_center is None else np.asarray(input_center, np.float32)
        if self.input_center is not None and self.input_center.shape != (inputs,):
            raise ValueError("Encoder input center must match the flattened input")
        self.images_per_batch = images_per_batch
        self.queries_per_batch = queries_per_batch
        encoder_settings = encoder_configs(inputs, rule, encoder_gate_widths, encoder_width, latent_width,
                                           encoder_top_k, encoder_hidden_widths)
        code_index = len(encoder_settings) - 1
        encoder_seeds = (seed, *range(seed + 3, seed + 3 + len(encoder_hidden_widths)), seed + 1)
        self.encoder = [
            EncoderLayer(config, layer_seed, images_per_batch, backend, bottleneck=i == code_index,
                         pool_size=bottleneck_pool_size if i == code_index else None)
            for i, (config, layer_seed) in enumerate(zip(encoder_settings, encoder_seeds, strict=True))
        ]
        self.decoder_skip = vector_decoder_skip
        self.decoder_hidden_rate = decoder_rate_decay
        self.decoder_projections = []
        if projected_decoder_votes:
            pairs = 3 * vector_hidden_width
            self.decoder_pairs = class_pair_indices(2 * pairs)
        layers = []
        previous_width = latent_width
        for index in range(vector_hidden_depth if vector_hidden_width else 0):
            hidden_config = ModelConfig(inputs=previous_width, outputs=3, width=vector_hidden_width, depth=1,
                                        competition="output", gate="gaussian", gate_width=.5, hidden_rule=rule)
            layers.append(EncoderLayer(hidden_config, seed + 3 + len(encoder_hidden_widths) + index,
                                       images_per_batch, backend, False))
            previous_width = hidden_config.units
            if projected_decoder_votes:
                self.decoder_projections.append(vote_assignment(pairs, 2 + index, True))
        self.decoder_hidden = layers[0] if len(layers) == 1 else tuple(layers) or None
        if decoder_kind == "vector":
            decoder_inputs = previous_width
            if self.decoder_skip:
                decoder_inputs = latent_width + sum(layer.config.units for layer in layers)
            config = vector_decoder_config(rule, decoder_inputs, vector_output_class_width, vector_gate_width, vector_top_k)
            capacity = images_per_batch
        else:
            config = decoder_config(rule, depth, gate_widths, decoder_rate_decay, decoder_class_width, latent_width)
            capacity = queries_per_batch
        self.decoder = (MojoModel(config, seed + 2, capacity)
                        if backend == "mojo" else NumpyModel(config, seed + 2))
        if not np.isfinite(decoder_init_scale) or decoder_init_scale <= 0:
            raise ValueError("Decoder initialization scale must be finite and positive")
        if paired_decoder_init and decoder_kind != "vector":
            raise ValueError("Paired initialization requires the image-vector decoder")
        if decoder_init_scale != 1 or paired_decoder_init:
            state = self.decoder.state()
            for key, value in state.items():
                if key.startswith("weight_"):
                    value *= decoder_init_scale
            if paired_decoder_init:
                # The two activities start with equal magnitude and opposite sign.
                weight = state["weight_0"].reshape(-1, config.outputs, 2, config.width)
                weight[:, :, 1] = -weight[:, :, 0]
            self.decoder.load(state)
        if zero_init_bias:
            for component in self.components().values():
                state = component.state()
                for key, values in state.items():
                    if key.startswith("weight_"):
                        values[-1].fill(0)
                component.load(state)

    def encoder_inputs(self, images):
        if self.input_center is not None:
            images = images - self.input_center
        if self.center_each_input:
            images = images - images.mean(axis=1, keepdims=True)
        if self.normalize_each_input:
            rms = np.sqrt(np.mean(images**2, axis=1, keepdims=True))
            images = images * (.5 / np.maximum(rms, .1))
        return images

    def encode(self, images):
        codes = []
        for start in range(0, len(images), self.images_per_batch):
            code = self.encoder_inputs(images[start:start + self.images_per_batch])
            for layer in self.encoder:
                code, _ = layer.forward(code)
            codes.append(code)
        return np.concatenate(codes)

    def decode(self, codes, coordinates=None):
        if self.decoder_kind == "vector":
            if coordinates is not None:
                raise ValueError("The vector decoder renders the entire portrait")
            return decode_vector(self.decoder, codes, self.images_per_batch, self.decoder_hidden, self.decoder_skip)
        points = COORDINATES if coordinates is None else coordinates
        return decode(self.decoder, codes, points, self.queries_per_batch, self.coordinate_scale)

    def step(self, images, coordinates, targets, rate, encoder_rate_scale=1.):
        if not 0 < len(images) <= self.images_per_batch:
            raise ValueError("Invalid image training batch size")
        if self.decoder_kind == "vector":
            if coordinates is not None or targets.shape != (len(images), SIZE * SIZE, 3):
                raise ValueError("The vector decoder requires entire images as targets")
        elif (coordinates is None or coordinates.ndim != 3
              or coordinates.shape != (*targets.shape[:2], 2)
              or targets.shape != (*coordinates.shape[:2], 3)
              or len(coordinates) != len(images) or coordinates.shape[1] == 0
              or targets.size // 3 > self.queries_per_batch):
            raise ValueError("Invalid image training batch")
        code = self.encoder_inputs(images)
        traces = []
        for layer in self.encoder:
            code, trace = layer.forward(code)
            traces.append(trace)
        decoder_trace = None
        if self.decoder_kind == "vector":
            queries, decoder_trace = vector_decoder_features(code, self.decoder_hidden, self.decoder_skip)
        else:
            repeated = np.broadcast_to(code[:, None], (*coordinates.shape[:2], self.latent_width))
            queries = np.concatenate((repeated, coordinates * self.coordinate_scale), axis=2)
            queries = queries.reshape(-1, self.latent_width + 2)
        targets_flat = targets.reshape(len(queries), self.decoder.config.outputs)
        teaching = self.decoder.step(queries, targets_flat, rate, return_signs=True)
        signs, counts = signed_totals(teaching.reshape(targets.shape))
        for index, layer in enumerate(hidden_layers(self.decoder_hidden)):
            hidden_signs, hidden_counts = signs, counts
            if self.decoder_projections:
                hidden_signs, hidden_counts = projected_votes(teaching, self.decoder_projections[index], self.decoder_pairs)
            layer.update(decoder_trace[index], hidden_signs, hidden_counts, rate * self.decoder_hidden_rate)
        encoder_rate = rate * encoder_rate_scale
        for layer, trace, (projection, pairs) in zip(self.encoder, traces, self.encoder_projections, strict=True):
            layer_signs, layer_counts = ((signs, counts) if projection is None else
                                         projected_votes(teaching, projection, pairs))
            layer.update(trace, layer_signs, layer_counts, encoder_rate)

    def components(self):
        models = {f"encoder_{i}": layer.model for i, layer in enumerate(self.encoder)}
        for index, layer in enumerate(hidden_layers(self.decoder_hidden)):
            models[hidden_name(index)] = layer.model
        models["decoder"] = self.decoder
        return models

    def state(self):
        return {f"{name}_{key}": value for name, model in self.components().items()
                for key, value in model.state().items()}

    def load(self, state):
        for name, model in self.components().items():
            prefix = f"{name}_"
            model.load({key[len(prefix):]: state[key] for key in state if key.startswith(prefix)})


def rgb_bytes(values):
    return np.rint(np.clip((values + 1) * 127.5, 0, 255)).astype(np.uint8)


def export_generator(path, model, codes, coordinate_scale=1., hidden=None, skip=False, *, mixture=None):
    covariance = np.cov(codes, rowvar=False)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    root = eigenvectors * np.sqrt(np.maximum(eigenvalues, 0))
    weights = {key: value for key, value in model.state().items() if key.startswith("weight_")}
    config = {**asdict(model.config), "hidden_rule": "signsgd"}
    for index, layer in enumerate(hidden_layers(hidden)):
        name = hidden_name(index)
        hidden_config = {**asdict(layer.config), "hidden_rule": "signsgd"}
        weights[f"{name}_config"] = json.dumps(hidden_config)
        weights[f"{name}_weight"] = layer.model.state()["weight_0"]
    if skip:
        weights["decoder_skip"] = True
    path = Path(path)
    temporary = path.with_suffix(".tmp.npz")
    np.savez_compressed(temporary, config=json.dumps(config), mean=codes.mean(axis=0, dtype=np.float64),
                        root=root, coordinate_scale=coordinate_scale, **weights, **(mixture or {}))
    temporary.replace(path)


class Generator:
    def __init__(self, path):
        with np.load(path, allow_pickle=False) as data:
            self.decoder = NumpyModel(ModelConfig(**json.loads(data["config"].item())), 0)
            self.decoder.load({key: data[key] for key in data.files if key.startswith("weight_")})
            self.decoder_skip = bool(data["decoder_skip"].item()) if "decoder_skip" in data else False
            layers = []
            while f"{hidden_name(len(layers))}_config" in data:
                name = hidden_name(len(layers))
                config = ModelConfig(**json.loads(data[f"{name}_config"].item()))
                layer = EncoderLayer(config, 0, 1, "numpy", False)
                layer.model.load({"weight_0": data[f"{name}_weight"]})
                layers.append(layer)
            self.decoder_hidden = layers[0] if len(layers) == 1 else tuple(layers) or None
            self.mean = data["mean"]
            self.root = data["root"]
            self.mixture = None
            if "mixture_weights" in data:
                self.mixture = (data["mixture_weights"], data["mixture_means"], data["mixture_roots"])
            self.coordinate_scale = float(data["coordinate_scale"]) if "coordinate_scale" in data else 1.

    def generate(self, seed):
        rng = np.random.default_rng(seed)
        mean, root = self.mean, self.root
        if self.mixture is not None:
            weights, means, roots = self.mixture
            group = rng.choice(len(weights), p=weights)
            mean, root = means[group], roots[group]
        code = mean + root @ rng.standard_normal(root.shape[1])
        codes = code[None].astype(np.float32)
        pixels = (decode_vector(self.decoder, codes, 1, self.decoder_hidden, self.decoder_skip)
                  if self.decoder.config.outputs == SIZE * SIZE * 3
                  else decode(self.decoder, codes, coordinate_scale=self.coordinate_scale))
        return rgb_bytes(pixels).reshape(SIZE, SIZE, 3)


def main():
    parser = argparse.ArgumentParser(description="Render a synthetic face from a decoder export")
    parser.add_argument("decoder", type=Path)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    Image.fromarray(Generator(args.decoder).generate(args.seed)).save(args.output)


if __name__ == "__main__":
    main()
