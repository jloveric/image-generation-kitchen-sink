"""NumPy reference for supervised Hebbian class regression."""

from dataclasses import dataclass, field, replace

import numpy as np


GATES = ("hard", "gaussian", "tent", "none", "gaussian-smooth", "gaussian-dual")
INITIALIZATIONS = ("normal", "unit-affine", "unit-sphere", "affine-start")
BASIS_SIZE = {"affine": 1, "continuous": 3, "discontinuous": 4}
COHERENCE_RULES = ("lion-coherence", "lion-coherence-squared", "lion-coherence-shrunk",
                   "lion-neuron-coherence", "lion-bounded-coherence", "lion-selective-coherence",
                   "lion-coherent-direction", "hebbian-coherence")
RESPONSE_RULES = ("lion-response", "lion-relative-response")
WEIGHT_POWER_RULES = ("lion-weight-power", "lion-bounded-weight-power", "lion-bounded-weight-fourth")
SMOOTH_PARTICIPATION_RULES = (
    "lion-smoothed-participation", "lion-slow-participation", "lion-slow-participation-power",
)
PARTICIPATION_RULES = ("lion-participation", "lion-participation-power",
                       "lion-pool-participation", *SMOOTH_PARTICIPATION_RULES)
ADAPTIVE_LION_RULES = ("lion-agreement", "lion-adaptive-agreement", "lion-input-scale",
                       *PARTICIPATION_RULES, *WEIGHT_POWER_RULES,
                       *RESPONSE_RULES, *COHERENCE_RULES)
COACTIVITY_DIRECTION_RULES = ("lion-coactivity-direction", "lion-cosine-coactivity", "lion-centered",
                             "lion-uniform-sign-step", "lion-coverage-decay", "lion-sqrt-coverage-decay",
                             "lion-checked-step")
NUMPY_ONLY_RULES = tuple(rule for rule in (*ADAPTIVE_LION_RULES, *COACTIVITY_DIRECTION_RULES)
                         if rule != "lion-adaptive-agreement")
HIDDEN_RULES = ("lion", "lion-explicit", "lion-adaptive-agreement", *NUMPY_ONLY_RULES, "signsgd", "activity-sign")
PAIRED_READOUTS = ("fixed", "class-sum", "sum", "class-energy")
READOUTS = (*PAIRED_READOUTS, "magnitude", "signed-activity")
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
    top_k: int | None = None  # exact number kept per competition pool
    response_limit_ratio: float = field(default=1., repr=False)
    normalize_hidden_inputs: str | bool = field(default=False, repr=False)
    skip_input_to_hidden: bool = field(default=False, repr=False)

    def __post_init__(self):
        if self.normalize_hidden_inputs is True:  # Original RMS checkpoints stored a boolean.
            object.__setattr__(self, "normalize_hidden_inputs", "rms")
        if self.normalize_hidden_inputs not in (False, "rms", "max-abs"):
            raise ValueError("Unknown interlayer normalization")
        if self.hidden_rule not in HIDDEN_RULES:
            raise ValueError("Unknown hidden update rule")
        if not np.isfinite(self.response_limit_ratio) or self.response_limit_ratio <= 0:
            raise ValueError("Response limit ratio must be positive and finite")
        if self.hidden_rule in RESPONSE_RULES and self.unit_norm_updates:
            raise ValueError("Response-limited Lion requires updates without weight renormalization")
        if self.connection not in BASIS_SIZE:
            raise ValueError("Unknown connection type")
        if min(self.inputs, self.outputs, self.width, self.depth) < 1:
            raise ValueError("Input size, output size, width and depth must be positive")
        if self.competition not in COMPETITIONS or self.readout not in READOUTS:
            raise ValueError("Unknown competition or readout")
        if self.readout == "magnitude" and self.width != 1:
            raise ValueError("Magnitude readout uses one neuron per output")
        if self.gate not in GATES:
            raise ValueError("Unknown competition gate")
        if self.top_k is None:
            if self.gate == "hard" and not 1 <= self.winners <= self.width:
                raise ValueError("Winner count must be between 1 and the class width")
        elif (self.gate != "hard" or not isinstance(self.top_k, (int, np.integer))
              or not 1 <= self.top_k <= self.units // self.competition_groups):
            raise ValueError("Top-K requires a hard gate and a count within the competition pool")
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
    def uses_history(self):
        return self.hidden_rule in ("lion", "lion-explicit", "lion-adaptive-agreement", *NUMPY_ONLY_RULES)

    def update_weights(self, weights, rate, direction, squared):
        if self.hidden_rule == "lion-explicit":
            return (1 - rate * squared) * weights + rate * direction
        return (weights + rate * direction) / (1 + rate * squared)

    @property
    def classes_per_output(self):
        return 1 if self.readout in ("magnitude", "signed-activity") else 2

    @property
    def classes(self):
        return self.classes_per_output * self.outputs

    @property
    def units(self):
        return self.classes * self.width

    @property
    def competition_groups(self):
        return {"class": self.classes, "output": self.outputs, "layer": 1}[self.competition]

    @property
    def keep_per_pool(self):
        return self.top_k if self.top_k is not None else self.winners * self.classes // self.competition_groups

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
        if self.top_k is not None:
            return max(1, self.top_k * self.competition_groups / self.classes)
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
    if config.gate == "hard":
        activity, mask = top_k_activity(grouped, config.keep_per_pool)
        return activity.reshape(z.shape), mask.reshape(z.shape)
    magnitude = np.abs(grouped)
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


def top_k_activity(grouped, count):
    if count >= grouped.shape[-1]:
        return grouped.copy(), np.ones(grouped.shape, dtype=bool)
    scores = -np.abs(grouped)
    if count > 0 and not np.isnan(scores).any():
        cutoff = np.partition(scores, count - 1, axis=-1)[..., count - 1:count]
        larger = scores < cutoff
        tied = scores == cutoff
        remaining = count - larger.sum(axis=-1, keepdims=True)
        # At the cutoff, keep the tied neurons with the smallest indices.
        mask = larger | (tied & (np.cumsum(tied, axis=-1, dtype=np.int32) <= remaining))
    else:
        indices = np.argsort(scores, axis=-1, kind="stable")[..., :count]
        mask = np.zeros(grouped.shape, dtype=bool)
        np.put_along_axis(mask, indices, True, axis=-1)
    return np.where(mask, grouped, np.float32(0)), mask


def inverse_participation(activity):
    """Inverse effective number of examples represented by a neuron's activity."""
    amplitudes = np.abs(activity)
    peak = amplitudes.max(axis=0)
    amplitudes /= np.where(peak > 0, peak, np.float32(1))
    total = np.square(amplitudes.sum(axis=0))
    energy = np.sum(amplitudes * amplitudes, axis=0)
    return np.divide(energy, total, out=np.zeros_like(total), where=total > 0)


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
        hidden_fan_in = config.units + (config.inputs if config.skip_input_to_hidden else 0)
        for fan_in in (config.inputs,) + (hidden_fan_in,) * (config.depth - 1):
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
        self.max_fan_in = max(weight.shape[0] for weight in self.weights)
        self.history = [np.zeros_like(w) for w in self.weights] if config.uses_history else []
        self.magnitude_history = [np.zeros_like(w) for w in self.weights] if config.hidden_rule in COHERENCE_RULES else []
        self.participation_history = (
            [np.ones(w.shape[1], dtype=w.dtype) for w in self.weights]
            if config.hidden_rule in SMOOTH_PARTICIPATION_RULES else []
        )
        self.agreement_history = (
            [np.ones(w.shape[1], dtype=w.dtype) for w in self.weights]
            if config.hidden_rule == "lion-adaptive-agreement" else []
        )
        if config.hidden_rule in ADAPTIVE_LION_RULES:
            self.multiplier_sums = np.zeros(config.depth, dtype=np.float64)
            self.multiplier_counts = np.zeros(config.depth, dtype=np.int64)
        nodes = np.arange(config.units)
        self.channels = nodes // (config.classes_per_output * config.width)
        self.polarity = (1 - 2 * (nodes // config.width % config.classes_per_output)).astype(np.float32)

    def forward(self, x, *, response_scales=None):
        config = self.config
        network_input = activity = np.asarray(x, np.float32)
        traces = []
        for weight, layer_config, transition in zip(self.weights, self.layer_configs, self.edge_transitions):
            if config.normalize_hidden_inputs and traces:
                # Normalize the gated features sent to the next layer, before appending its bias.
                if config.normalize_hidden_inputs == "max-abs":
                    peak = np.max(np.abs(activity), axis=1, keepdims=True)
                    activity = activity / np.where(peak > 0, peak, 1.)
                else:
                    activity = activity / np.sqrt(np.mean(activity * activity, axis=1, keepdims=True) + 1e-6)
            features = (np.concatenate((activity, network_input), axis=1)
                        if config.skip_input_to_hidden and traces else activity)
            incoming = connection_inputs(features, config.connection, transition)
            response = incoming @ weight
            if response_scales is not None:
                response_scales.append(np.linalg.norm(response, axis=0))
            activity, mask = select(response, layer_config)
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
        if config.readout == "magnitude":
            # Map image intensities in [0, 1] to the model's [-1, 1] convention.
            return 2 * np.abs(activity) - 1
        if config.readout == "signed-activity":
            return activity.reshape(len(activity), config.outputs, config.width).sum(axis=-1)
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
        rule = self.config.hidden_rule
        response_scales = [] if rule in (*RESPONSE_RULES, "lion-input-scale") else None
        prediction, traces = self.forward(x, response_scales=response_scales)
        teaching = self.teaching_signs(prediction, y)
        if rule == "lion-checked-step" and np.any(teaching):
            saved = self.state()
            self.apply_update(traces, teaching, rate, response_scales)
            after = self.teaching_signs(self.predict_values(x), y)
            reversed_sign = teaching * after < 0
            if np.any(reversed_sign):
                self.load(saved)
                self.apply_update(traces, np.where(reversed_sign, 0, teaching), rate, response_scales)
        else:
            self.apply_update(traces, teaching, rate, response_scales)
        if return_signs:
            return teaching

    def teaching_signs(self, prediction, y):
        if self.output_half_span is None:
            residual = y - prediction
        else:
            bound = np.float32(self.config.output_bound)
            residual = (output_codes(y, bound, self.output_half_span, self.config.clip_output)
                        - output_codes(prediction, bound, self.output_half_span, self.config.clip_output))
        return np.sign(residual)

    def apply_update(self, traces, teaching, rate, response_scales):
        rule = self.config.hidden_rule
        signs = teaching[:, self.channels] * self.polarity
        for layer, (weight, (incoming, activity, _)) in enumerate(zip(self.weights, traces)):
            direction_scale = 1.
            contributes = (signs != 0) & (activity != 0)
            counts = contributes.sum(axis=0)
            normalizer = np.maximum(counts, 1).astype(np.float32)
            # Signed pooling makes the activity's own sign take the role of
            # the fixed class polarity in the teaching signal.
            post = np.abs(activity) if self.config.readout == "signed-activity" else activity
            products = (incoming.T @ (signs * post)) / normalizer
            if rule == "lion-centered":
                # Center the incoming features while treating the constant bias separately.
                input_mean = incoming[:, :-1].mean(axis=0)
                products[:-1] -= input_mean[:, None] * products[-1]
            squared = np.sum(np.where(contributes, activity * activity, 0), axis=0) / normalizer
            if rule in ("lion-coverage-decay", "lion-sqrt-coverage-decay"):
                # The ratio of the squared absolute sum to the squared sum is effective coverage.
                absolute_sum = np.sum(np.where(contributes, np.abs(activity), 0), axis=0)
                coverage_decay = absolute_sum**2 / normalizer
                squared = (coverage_decay if rule == "lion-coverage-decay"
                           else np.sqrt(squared * coverage_decay))
            active = counts > 0
            if active.all():
                active = slice(None)
            current = weight[:, active]
            observation = products[:, active]
            if rule == "lion-cosine-coactivity":
                # Normalize the correlation between the presynaptic activity and signed response.
                input_size = np.sqrt(np.sum(incoming * incoming, axis=0))[:, None]
                response_size = np.sqrt(squared[active] * counts[active].astype(np.float32))
                denominator = input_size * response_size
                observation = np.divide(observation * counts[active].astype(np.float32), denominator,
                                        out=np.zeros_like(observation), where=denominator > 0)
            layer_rate = rate * self.rate_multipliers[layer]
            if self.config.uses_history:
                history = self.history[layer]
                past = history[:, active]
                signal = 0.125 * past + 0.875 * observation
                if rule == "lion-agreement":
                    multipliers = np.sign(past)
                    multipliers *= np.sign(observation)
                    multipliers *= .5
                    multipliers += 1.
                    self.multiplier_sums[layer] += multipliers.sum()
                    self.multiplier_counts[layer] += multipliers.size
                    layer_rate = np.float32(layer_rate) * multipliers
                elif rule == "lion-adaptive-agreement":
                    multipliers = self.adapt_step(layer, active, past, observation)
                    self.multiplier_sums[layer] += multipliers.sum()
                    self.multiplier_counts[layer] += multipliers.size
                    layer_rate = np.float32(layer_rate) * multipliers
                updated_history = 0.99 * past + 0.01 * observation
                history[:, active] = updated_history
                if self.magnitude_history:
                    magnitude_history = self.magnitude_history[layer]
                    magnitude = 0.99 * magnitude_history[:, active] + 0.01 * np.abs(observation)
                    magnitude_history[:, active] = magnitude
                    # Opposing observations cancel in the signed average.
                    multipliers = np.divide(np.abs(updated_history), magnitude,
                                            out=np.zeros_like(magnitude), where=magnitude > 0)
                    np.minimum(multipliers, 1., out=multipliers)
                    if rule == "lion-coherence-squared":
                        np.square(multipliers, out=multipliers)
                    elif rule in ("lion-coherence-shrunk", "lion-neuron-coherence"):
                        # The 0.99 EMA has a noise scale of about 0.07 for independent observations.
                        multipliers = multipliers**2 / (multipliers + np.float32(.07))
                        if rule == "lion-neuron-coherence":
                            multipliers = multipliers.mean(axis=0)
                    elif rule == "lion-bounded-coherence":
                        multipliers *= np.minimum(1., np.sum(current * current, axis=0))
                    elif rule == "lion-selective-coherence":
                        multipliers *= np.sqrt(inverse_participation(activity[:, active]))
                    self.multiplier_sums[layer] += multipliers.sum()
                    self.multiplier_counts[layer] += multipliers.size
                    if rule == "lion-coherent-direction":
                        direction_scale = multipliers
                    else:
                        layer_rate = np.float32(layer_rate) * multipliers
                    if rule == "hebbian-coherence":
                        signal = updated_history
            else:
                signal = observation
            if rule == "lion-coactivity-direction":
                length = np.linalg.norm(signal, axis=0)
                direction = np.divide(signal, length, out=np.zeros_like(signal), where=length > 0)
            else:
                direction = signal if rule == "lion-cosine-coactivity" else np.sign(signal)
                fan_in = self.max_fan_in if rule == "lion-uniform-sign-step" else weight.shape[0]
                direction /= np.float32(np.sqrt(fan_in))
            if rule == "activity-sign":
                power = squared[active]
                direction *= np.sqrt(power / (1 + power))
            if rule == "lion-coherent-direction":
                direction *= direction_scale
            if rule == "lion-input-scale":
                # Use the relative energy of the input and the receiving neuron's response.
                response_energy = np.square(response_scales[layer][active])
                input_energy = np.sum(incoming * incoming, axis=0)[:, None]
                total = input_energy + response_energy
                multipliers = np.divide(response_energy, total, out=np.zeros_like(total), where=total > 0)
                self.multiplier_sums[layer] += multipliers.sum()
                self.multiplier_counts[layer] += multipliers.size
                layer_rate = np.float32(layer_rate) * multipliers
            if rule in PARTICIPATION_RULES:
                if rule == "lion-pool-participation":
                    gains = inverse_participation(activity).reshape(self.config.competition_groups, -1)
                    mean = gains.sum(axis=1, keepdims=True) / np.maximum(1, (gains > 0).sum(axis=1, keepdims=True))
                    multipliers = np.broadcast_to(mean, gains.shape).reshape(-1)[active].astype(np.float32)
                else:
                    multipliers = inverse_participation(activity[:, active])
                if rule in ("lion-participation-power", "lion-pool-participation", "lion-slow-participation-power"):
                    multipliers *= np.sqrt(multipliers)
                if self.participation_history:
                    local_gain = self.participation_history[layer]
                    observation_weight = .01 if rule == "lion-smoothed-participation" else .0005
                    multipliers = (1 - observation_weight) * local_gain[active] + observation_weight * multipliers
                    local_gain[active] = multipliers
                self.multiplier_sums[layer] += multipliers.sum()
                self.multiplier_counts[layer] += multipliers.size
                layer_rate = np.float32(layer_rate) * multipliers
            if rule in WEIGHT_POWER_RULES:
                multipliers = np.sum(current * current, axis=0)
                if rule != "lion-weight-power":
                    np.minimum(multipliers, 1., out=multipliers)
                if rule == "lion-bounded-weight-fourth":
                    np.square(multipliers, out=multipliers)
                self.multiplier_sums[layer] += multipliers.sum()
                self.multiplier_counts[layer] += multipliers.size
                layer_rate = np.float32(layer_rate) * multipliers
            if rule in RESPONSE_RULES:
                # Subtract the current weights algebraically to avoid cancellation.
                power = squared[active]
                delta = layer_rate / (1 + layer_rate * power) * (direction - power * current)
                change = np.linalg.norm(incoming @ delta, axis=0)
                scale = response_scales[layer][active]
                if rule == "lion-relative-response":
                    budget = rate * scale
                    denominator = change + .25 * budget
                    multipliers = np.divide(budget, denominator, out=np.ones_like(change), where=denominator > 0)
                else:
                    budget = (rate * self.config.response_limit_ratio) * scale
                    multipliers = np.divide(budget, change, out=np.ones_like(change), where=change > budget)
                self.multiplier_sums[layer] += multipliers.sum()
                self.multiplier_counts[layer] += multipliers.size
                weight[:, active] = current + multipliers * delta
            else:
                updated = self.config.update_weights(current, layer_rate, direction, squared[active])
                if rule == "lion-centered":
                    # Convert the updated bias back to the original feature coordinates.
                    effective_rate = layer_rate / (1 + layer_rate * squared[active])
                    updated[-1] -= effective_rate * (input_mean @ direction[:-1])
                weight[:, active] = updated
            if self.config.unit_norm_updates:
                updated = weight[:, active]
                norm = np.linalg.norm(updated, axis=0)
                weight[:, active] = updated / np.where(norm > 0, norm, np.float32(1))

    def adapt_step(self, layer, active, history, coactivity):
        alignment = (np.sign(history) * np.sign(coactivity)).mean(axis=0)
        gain = self.agreement_history[layer]
        # Repeated agreement restores the scheduled step. Reversals shrink it.
        gain[active] = np.clip(gain[active] * np.power(np.float32(.99), -alignment), .01, 1.)
        return gain[active]

    def take_mean_step_multipliers(self):
        """Average the factors applied to participating updates since the previous call."""
        mean = self.multiplier_sums / np.maximum(1, self.multiplier_counts)
        self.multiplier_sums.fill(0)
        self.multiplier_counts.fill(0)
        return mean.tolist()

    def state(self):
        return {
            **{f"weight_{i}": weight.copy() for i, weight in enumerate(self.weights)},
            **{f"history_{i}": history.copy() for i, history in enumerate(self.history)},
            **{f"magnitude_{i}": magnitude.copy() for i, magnitude in enumerate(self.magnitude_history)},
            **{f"participation_{i}": gain.copy() for i, gain in enumerate(self.participation_history)},
            **{f"agreement_{i}": gain.copy() for i, gain in enumerate(self.agreement_history)},
        }

    def load(self, state):
        validate_checkpoint(state, self.config)
        for i, weight in enumerate(self.weights):
            weight[:] = state[f"weight_{i}"]
        for i, history in enumerate(self.history):
            history[:] = state[f"history_{i}"]
        for i, magnitude in enumerate(self.magnitude_history):
            magnitude[:] = state[f"magnitude_{i}"]
        for i, gain in enumerate(self.participation_history):
            gain[:] = state[f"participation_{i}"]
        for i, gain in enumerate(self.agreement_history):
            gain[:] = state[f"agreement_{i}"]


def validate_checkpoint(state, config):
    names = {f"weight_{i}" for i in range(config.depth)}
    if config.uses_history:
        names.update(f"history_{i}" for i in range(config.depth))
    if config.hidden_rule in COHERENCE_RULES:
        names.update(f"magnitude_{i}" for i in range(config.depth))
    if config.hidden_rule in SMOOTH_PARTICIPATION_RULES:
        names.update(f"participation_{i}" for i in range(config.depth))
    if config.hidden_rule == "lion-adaptive-agreement":
        names.update(f"agreement_{i}" for i in range(config.depth))
    # Earlier fixed-pool checkpoints included an unused zero-filled output array.
    if "output" in state and not np.any(state["output"]):
        names.add("output")
    if set(state) != names:
        raise ValueError("Checkpoint must contain only state for the fixed-pool architecture")
