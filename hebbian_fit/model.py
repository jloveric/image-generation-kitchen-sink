"""NumPy reference for supervised Hebbian class regression."""

from dataclasses import dataclass, replace

import numpy as np


GATES = ("hard", "gaussian", "tent", "none", "gaussian-smooth", "gaussian-dual")
INITIALIZATIONS = ("normal", "unit-affine", "unit-sphere", "affine-start")
BASIS_SIZE = {"affine": 1, "continuous": 3, "discontinuous": 4}
HIDDEN_RULES = ("lion", "signsgd", "activity-sign")
READOUTS = ("fixed", "class-sum", "sum", "class-energy")
COMPETITIONS = ("class", "output", "layer")
TENT_FLOOR = np.float32(0.01)


@dataclass(frozen=True)
class ModelConfig:
    inputs: int = 1
    outputs: int = 1
    width: int = 32  # per class
    winners: int = 8  # per class, or this times the number of classes per layer
    depth: int = 2
    competition: str = "class"
    readout: str = "fixed"
    gate: str = "hard"
    gate_width: float = 0.5
    layer_gate_widths: tuple[float, ...] | None = None
    readout_layers: str = "last"
    initialization: str = "normal"
    unit_norm_updates: bool = False
    sine_omega: float | None = None
    connection: str = "affine"
    hidden_rule: str = "lion"
    output_bound: float | None = None
    output_levels: int | None = None
    edge_transitions: tuple[float, ...] | None = None
    clip_output: bool = True
    layer_rate_decay: float = 1.

    def __post_init__(self):
        if self.hidden_rule not in HIDDEN_RULES:
            raise ValueError("Unknown hidden update rule")
        if self.connection not in BASIS_SIZE:
            raise ValueError("Unknown connection type")
        if min(self.inputs, self.outputs, self.width, self.depth) < 1:
            raise ValueError("Input size, output size, width and depth must be positive")
        if self.gate == "hard" and not 1 <= self.winners <= self.width:
            raise ValueError("Winner count must be between 1 and the class width")
        if self.competition not in COMPETITIONS or self.readout not in READOUTS:
            raise ValueError("Unknown competition or readout")
        if self.gate not in GATES:
            raise ValueError("Unknown competition gate")
        if self.gate == "gaussian-dual" and self.width % 2:
            raise ValueError("Two Gaussian widths require an even number of neurons per class")
        if self.layer_gate_widths is not None:
            if self.gate not in ("gaussian", "gaussian-smooth", "gaussian-dual", "tent") or len(self.layer_gate_widths) != self.depth:
                raise ValueError("Supply one scaled-gate width per hidden layer")
        limits = np.finfo(np.float32)
        if not 0 < self.layer_rate_decay <= 1 or self.rate_multipliers[0] < limits.tiny:
            raise ValueError("Layer rates must be positive and representable in float32")
        if self.edge_transitions is not None:
            if self.connection != "discontinuous" or len(self.edge_transitions) != self.depth:
                raise ValueError("Supply one transition width per two-piece layer")
            if any(value != 0 and not limits.tiny <= value <= limits.max for value in self.edge_transitions):
                raise ValueError("Transition widths must be zero or positive and finite in float32")
        if self.output_bound is not None and not limits.tiny <= self.output_bound <= limits.max:
            raise ValueError("Output bound must be positive and finite in float32")
        if self.output_levels is not None:
            if (self.output_bound is None or not isinstance(self.output_levels, (int, np.integer))
                    or not 2 <= self.output_levels <= 65536):
                raise ValueError("Output quantization needs a bound and 2 to 65536 levels")
        if self.sine_omega is not None and not limits.tiny <= self.sine_omega <= limits.max:
            raise ValueError("Sine frequency scale must be positive and finite in float32")
        if any(not limits.tiny <= value <= limits.max for value in self.gate_widths):
            raise ValueError("Gate width must be positive and finite in float32")
        if self.readout_layers not in ("last", "all"):
            raise ValueError("Unknown readout layers")
        if self.initialization not in INITIALIZATIONS:
            raise ValueError("Unknown initialization")

    @property
    def classes(self):
        return 2 * self.outputs

    @property
    def units(self):
        return self.classes * self.width

    @property
    def competition_groups(self):
        return {"class": self.classes, "output": self.outputs, "layer": 1}[self.competition]

    @property
    def heads(self):
        return self.depth if self.readout_layers == "all" else 1

    @property
    def basis_size(self):
        return BASIS_SIZE[self.connection]

    @property
    def readout_divisor(self):
        if self.readout == "class-sum":
            return 1
        return self.winners if self.gate == "hard" else self.width

    @property
    def rate_multipliers(self):
        return tuple(self.layer_rate_decay ** i for i in reversed(range(self.depth)))

    @property
    def gate_widths(self):
        if self.layer_gate_widths is None:
            return (self.gate_width,) * self.depth
        return self.layer_gate_widths


def output_codes(values, bound, half_span, clip=True):
    normalized = values / bound
    if clip:
        normalized = np.clip(normalized, -1, 1)
    return np.floor((normalized + 1) * half_span + .5)


def with_bias(x):
    return np.pad(x, ((0, 0), (0, 1)), constant_values=1)


def connection_inputs(x, connection, transition=0.):
    """Interpolation coefficients for each scalar edge, followed by one neuron bias."""
    if connection == "affine":
        return with_bias(x)
    n, inputs = x.shape
    k = BASIS_SIZE[connection]
    features = np.empty((n, inputs * k + 1), dtype=x.dtype)
    values = features[:, :-1].reshape(n, inputs, k)
    if transition > 0:
        # Blend the two extended lines with a cubic smoothstep near their join.
        t = .5 + .5 * np.clip(x / np.float32(transition), -1, 1)
        right = t * t * (3 - 2 * t)
        left = 1 - right
        values[:, :, 0] = -x * left
        values[:, :, 1] = (1 + x) * left
        values[:, :, 2] = (1 - x) * right
        values[:, :, 3] = x * right
    else:
        values[:, :, 0] = np.maximum(-x, 0)
        values[:, :, -1] = np.maximum(x, 0)
        if connection == "continuous":
            values[:, :, 1] = 1 - np.abs(x)
        else:
            left = x < 0
            values[:, :, 1] = np.where(left, 1 + x, 0)
            values[:, :, 2] = np.where(left, 0, 1 - x)
    features[:, -1] = 1
    return features


def select(z, config):
    if config.sine_omega is not None:
        z = np.sin(np.float32(config.sine_omega) * z)
    if config.gate == "none":
        return z, np.ones(z.shape, dtype=bool)
    groups = config.competition_groups
    grouped = z.reshape(len(z), groups, -1)
    magnitude = np.abs(grouped)
    if config.gate != "hard":
        peak = magnitude.max(axis=-1, keepdims=True)
        if config.gate == "gaussian-smooth":
            # The max only stabilizes the ratio of even moments.
            relative = magnitude / np.where(peak > 0, peak, np.float32(1))
            weights = relative ** 14
            denominator = weights.sum(axis=-1, keepdims=True)
            numerator = (weights * relative ** 2).sum(axis=-1, keepdims=True)
            peak = peak * np.sqrt(numerator / np.where(denominator > 0, denominator, np.float32(1)))
        gap = (peak - magnitude) / np.where(peak > 0, peak, np.float32(1))
        scaled_gap = gap / np.float32(config.gate_width)
        if config.gate == "gaussian-dual":
            scaled_gap[..., ::2] /= np.float32(.75)
            scaled_gap[..., 1::2] /= np.float32(1.25)
        if config.gate in ("gaussian", "gaussian-smooth", "gaussian-dual"):
            # Keep the multiplier positive even at extreme widths in float32.
            scaled_gap = np.minimum(np.abs(scaled_gap), np.float32(13))
            gate = np.exp(np.float32(-0.5) * scaled_gap * scaled_gap)
        else:
            gate = np.maximum(TENT_FLOOR, np.float32(1) - scaled_gap)
        return (grouped * gate).reshape(z.shape), np.ones(z.shape, dtype=bool)
    count = config.winners * config.classes // groups
    # Stable sorting makes ties go to the smaller neuron index.
    indices = np.argsort(-magnitude, axis=-1, kind="stable")[..., :count]
    mask = np.zeros(grouped.shape, dtype=bool)
    np.put_along_axis(mask, indices, True, axis=-1)
    mask = mask.reshape(z.shape)
    return np.where(mask, z, np.float32(0)), mask


class NumpyModel:
    def __init__(self, config, seed):
        self.config = config
        self.rate_multipliers = config.rate_multipliers
        self.output_half_span = np.float32((config.output_levels - 1) / 2) if config.output_levels else None
        self.edge_transitions = config.edge_transitions or (0.,) * config.depth
        self.layer_configs = (config,) * config.depth
        if config.layer_gate_widths is not None:
            self.layer_configs = tuple(
                replace(config, gate_width=width, layer_gate_widths=None)
                for width in config.gate_widths
            )
        rng = np.random.default_rng(seed)
        self.weights = []
        for fan_in in (config.inputs,) + (config.units,) * (config.depth - 1):
            k = config.basis_size
            affine_start = config.initialization == "affine-start"
            rows = fan_in if affine_start else fan_in * k
            weight = rng.normal(0, 1 / np.sqrt(fan_in), (rows + 1, config.units))
            if config.initialization not in ("unit-sphere", "affine-start"):
                weight[-1] *= 0.35 * np.sqrt(fan_in)
            weight = weight.astype(np.float32)
            if config.initialization != "normal":
                weight /= np.linalg.norm(weight, axis=0, keepdims=True)
            if affine_start and k > 1:
                # Begin with exactly the same random affine map for all edge types.
                endpoints = np.zeros((fan_in, k, config.units), dtype=np.float32)
                endpoints[:, 0] = -weight[:-1]
                endpoints[:, -1] = weight[:-1]
                weight = np.vstack((endpoints.reshape(fan_in * k, config.units), weight[-1]))
            self.weights.append(weight)
        self.history = [np.zeros_like(w) for w in self.weights] if config.hidden_rule == "lion" else []
        self.channels = np.arange(config.units) // (2 * config.width)
        self.polarity = (1 - 2 * (np.arange(config.units) // config.width % 2)).astype(np.float32)

    def forward(self, x):
        config = self.config
        activity = np.asarray(x, np.float32)
        traces = []
        for weight, layer_config, transition in zip(self.weights, self.layer_configs, self.edge_transitions):
            incoming = connection_inputs(activity, config.connection, transition)
            activity, mask = select(incoming @ weight, layer_config)
            traces.append((incoming, activity, mask))
        if config.heads == 1:
            prediction = self.pool(activity)
        else:
            contributions = [self.pool(post) for _, post, _ in traces]
            if config.readout == "class-energy":
                prediction = np.sum(contributions, axis=0)
            else:
                prediction = np.mean(contributions, axis=0)
        return self.format_output(prediction), traces

    def format_output(self, prediction):
        config = self.config
        if self.output_half_span is not None:
            bound = np.float32(config.output_bound)
            codes = output_codes(prediction, bound, self.output_half_span, config.clip_output)
            prediction = (codes / self.output_half_span - 1) * bound
        elif config.clip_output and config.output_bound is not None:
            np.clip(prediction, -config.output_bound, config.output_bound, out=prediction)
        return prediction

    def pool(self, activity):
        config = self.config
        grouped = activity.reshape(len(activity), config.outputs, 2, config.width)
        if config.readout == "sum":
            return grouped.sum(axis=(2, 3))
        values = np.square(grouped) if config.readout == "class-energy" else np.abs(grouped)
        totals = values.sum(axis=-1)
        return (totals[..., 0] - totals[..., 1]) / config.readout_divisor

    def predict(self, x):
        prediction, traces = self.forward(x)
        masks = [mask for _, _, mask in traces]
        return prediction, masks

    def predict_values(self, x):
        return self.forward(x)[0]

    def step(self, x, y, rate, *, return_signs=False):
        prediction, traces = self.forward(x)
        if self.output_half_span is None:
            residual = y - prediction
        else:
            bound = np.float32(self.config.output_bound)
            residual = (output_codes(y, bound, self.output_half_span, self.config.clip_output)
                        - output_codes(prediction, bound, self.output_half_span, self.config.clip_output))
        teaching = np.sign(residual)
        signs = teaching[:, self.channels] * self.polarity
        for layer, (weight, (incoming, activity, _)) in enumerate(zip(self.weights, traces)):
            contributes = (signs != 0) & (activity != 0)
            counts = contributes.sum(axis=0)
            normalizer = np.maximum(counts, 1).astype(np.float32)
            products = (incoming.T @ (signs * activity)) / normalizer
            squared = np.sum(np.where(contributes, activity * activity, 0), axis=0) / normalizer
            active = counts > 0
            if active.all():
                active = slice(None)
            observation = products[:, active]
            if self.config.hidden_rule == "lion":
                history = self.history[layer]
                signal = 0.125 * history[:, active] + 0.875 * observation
                history[:, active] = 0.99 * history[:, active] + 0.01 * observation
            else:
                signal = observation
            direction = np.sign(signal)
            direction /= np.float32(np.sqrt(weight.shape[0]))
            if self.config.hidden_rule == "activity-sign":
                power = squared[active]
                direction *= np.sqrt(power / (1 + power))
            layer_rate = rate * self.rate_multipliers[layer]
            weight[:, active] = (weight[:, active] + layer_rate * direction) / (1 + layer_rate * squared[active])
            if self.config.unit_norm_updates:
                updated = weight[:, active]
                norm = np.linalg.norm(updated, axis=0)
                weight[:, active] = updated / np.where(norm > 0, norm, np.float32(1))

        if return_signs:
            return teaching

    def state(self):
        return {
            **{f"weight_{i}": weight.copy() for i, weight in enumerate(self.weights)},
            **{f"history_{i}": history.copy() for i, history in enumerate(self.history)},
        }

    def load(self, state):
        validate_checkpoint(state, self.config)
        for i, weight in enumerate(self.weights):
            weight[:] = state[f"weight_{i}"]
        for i, history in enumerate(self.history):
            history[:] = state[f"history_{i}"]


def validate_checkpoint(state, config):
    names = {f"weight_{i}" for i in range(config.depth)}
    if config.hidden_rule == "lion":
        names.update(f"history_{i}" for i in range(config.depth))
    # Earlier fixed-pool checkpoints included an unused zero-filled output array.
    if "output" in state and not np.any(state["output"]):
        names.add("output")
    if set(state) != names:
        raise ValueError("Checkpoint must contain only state for the fixed-pool architecture")
