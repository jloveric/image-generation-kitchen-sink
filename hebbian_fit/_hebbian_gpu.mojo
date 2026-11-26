"""GPU implementation of the dense competitive Hebbian model."""
from std.math import abs, ceildiv, exp, floor, log, max, min, sin, sqrt
from std.os import abort
from std.python import PythonObject
from std.python.bindings import PythonModuleBuilder
from max.gpu import global_idx, lane_id, WARP_SIZE
from max.gpu.host import DeviceContext
from max.gpu.primitives import warp

from _gpu_buffers import Block, Buffer, Read, Write, download, read, upload, write


def expand_edges(x: Read, features: Write, count: Int32, basis_size: Int32, transition: Float32):
    var i = global_idx.x
    if i >= Int(count):
        return
    var u = x[i]
    var k = Int(basis_size)
    var start = i * k
    if transition > 0:
        var t = 0.5 + 0.5 * max(Float32(-1), min(Float32(1), u / transition))
        var right = t * t * (3 - 2 * t)
        var left = 1 - right
        features[start] = -u * left
        features[start + 1] = (1 + u) * left
        features[start + 2] = (1 - u) * right
        features[start + 3] = u * right
    else:
        features[start] = max(-u, Float32(0))
        features[start + k - 1] = max(u, Float32(0))
        if k == 3:
            features[start + 1] = 1 - abs(u)
        else:
            features[start + 1] = 1 + u if u < 0 else 0
            features[start + 2] = 0 if u < 0 else 1 - u


def dense(x: Read, weights: Read, z: Write, batch_arg: Int32,
          inputs_arg: Int32, units_arg: Int32, offset_arg: Int32, omega: Float32):
    var batch = Int(batch_arg)
    var inputs = Int(inputs_arg)
    var units = Int(units_arg)
    var offset = Int(offset_arg)
    var i = global_idx.x
    if i >= batch * units:
        return
    var sample = i // units
    var node = i % units
    var value: Float32 = 0
    for other in range(inputs):
        value += x[sample * inputs + other] * weights[offset + other * units + node]
    value += weights[offset + inputs * units + node]
    if omega > 0:
        value = sin(omega * value)
    z[i] = value


def pass_activity(z: Read, activity: Write, mask: Write, size_arg: Int32):
    var i = global_idx.x
    if i < Int(size_arg):
        activity[i] = z[i]
        mask[i] = 1


def compete(z: Read, activity: Write, mask: Write, batch_arg: Int32, units_arg: Int32,
            group_size_arg: Int32, keep_arg: Int32):
    comptime assert z.flat_rank == 1 and activity.flat_rank == 1 and mask.flat_rank == 1
    var batch = Int(batch_arg)
    var units = Int(units_arg)
    var group_size = Int(group_size_arg)
    var keep = Int(keep_arg)
    var i = global_idx.x
    if i >= batch * units:
        return
    var group_start = (i % units // group_size) * group_size
    var first = (i // units) * units + group_start
    var end = first + min(group_size, units - group_start)
    var score = abs(z[i])
    var rank = 0
    for other in range(first, end):
        var candidate = abs(z[other])
        if candidate > score or (candidate == score and other < i):
            rank += 1
            if rank == keep:
                break
    if rank < keep:
        activity[i] = z[i]
        mask[i] = 1
    else:
        activity[i] = 0
        mask[i] = 0


def top_k_key(value: Float32, index: Int, count: Int) -> UInt64:
    # Positive float bits sort by magnitude. The low bits break ties by index.
    var magnitude = abs(value)
    var bits = Pointer(to=magnitude).unsafe_bitcast[UInt32]()[]
    return (UInt64(bits) << 32) | UInt64(count - 1 - index)


def compete_wide(z: Read, activity: Write, mask: Write, batch_arg: Int32, units_arg: Int32,
                 group_size_arg: Int32, keep: Int32):
    comptime assert z.flat_rank == 1 and activity.flat_rank == 1 and mask.flat_rank == 1
    var batch = Int(batch_arg)
    var units = Int(units_arg)
    var group_size = Int(group_size_arg)
    var groups = ceildiv(units, group_size)
    var group = global_idx.x // WARP_SIZE
    if group >= batch * groups:
        return
    var group_start = (group % groups) * group_size
    var first = (group // groups) * units + group_start
    var count = min(group_size, units - group_start)
    var lane = lane_id()
    var cutoff: UInt64 = 0
    if Int(keep) < count:
        # Find the Kth largest key without sorting or storing the complete pool.
        for bit in range(62, -1, -1):
            var digit = UInt64(1) << UInt64(bit)
            if bit < 32 and digit >= UInt64(count):
                continue
            var candidate = cutoff | digit
            var above: Int32 = 0
            for j in range(lane, count, WARP_SIZE):
                above += Int32(top_k_key(z[first + j], j, count) >= candidate)
            if warp.broadcast(warp.sum(above)) >= keep:
                cutoff = candidate
    for j in range(lane, count, WARP_SIZE):
        var value = z[first + j]
        var selected = top_k_key(value, j, count) >= cutoff
        activity[first + j] = value if selected else 0
        mask[first + j] = 1 if selected else 0


def pool_maxima(z: Read, peaks: Write, pool_count_arg: Int32, group_size_arg: Int32):
    comptime assert z.flat_rank == 1 and peaks.flat_rank == 1
    var pool = global_idx.x
    if pool >= Int(pool_count_arg):
        return
    var group_size = Int(group_size_arg)
    var peak: Float32 = 0
    var first = pool * group_size
    for j in range(first, first + group_size):
        peak = max(peak, abs(z[j]))
    peaks[pool] = peak


def wide_pool_maxima(z: Read, peaks: Write, pool_count_arg: Int32, group_size_arg: Int32):
    var pool = global_idx.x // WARP_SIZE
    if pool >= Int(pool_count_arg):
        return
    var lane = lane_id()
    var group_size = Int(group_size_arg)
    var first = pool * group_size
    var peak: Float32 = 0
    for j in range(lane, group_size, WARP_SIZE):
        peak = max(peak, abs(z[first + j]))
    peak = warp.max(peak)
    if lane == 0:
        peaks[pool] = peak


def smooth_pool_centers(z: Read, peaks: Write, pool_count_arg: Int32, group_size_arg: Int32):
    var pool = global_idx.x // WARP_SIZE
    if pool >= Int(pool_count_arg):
        return
    var lane = lane_id()
    var group_size = Int(group_size_arg)
    var first = pool * group_size
    var peak: Float32 = 0
    for index in range(lane, group_size, WARP_SIZE):
        peak = max(peak, abs(z[first + index]))
    peak = warp.broadcast(warp.max(peak))
    var numerator: Float32 = 0
    var denominator: Float32 = 0
    if peak > 0:
        for index in range(lane, group_size, WARP_SIZE):
            var relative = abs(z[first + index]) / peak
            var square = relative * relative
            var fourth = square * square
            var eighth = fourth * fourth
            var weight = eighth * fourth * square
            denominator += weight
            numerator += weight * square
    numerator = warp.sum(numerator)
    denominator = warp.sum(denominator)
    if lane == 0:
        peaks[pool] = peak * sqrt(numerator / denominator) if denominator > 0 else 0


def scale_activity(z: Read, peaks: Read, activity: Write, mask: Write,
                   size_arg: Int32, group_size_arg: Int32, gate_arg: Int32, width: Float32):
    comptime assert z.flat_rank == 1 and peaks.flat_rank == 1
    comptime assert activity.flat_rank == 1 and mask.flat_rank == 1
    var i = global_idx.x
    if i >= Int(size_arg):
        return
    var peak = peaks[i // Int(group_size_arg)]
    var scaled_gap: Float32 = 0
    if peak > 0:
        scaled_gap = ((peak - abs(z[i])) / peak) / width
        if gate_arg == 5:
            scaled_gap /= Float32(0.75 if i % 2 == 0 else 1.25)
    var weight: Float32
    if gate_arg == 1 or gate_arg == 4 or gate_arg == 5:
        var distance = min(abs(scaled_gap), Float32(13))
        weight = exp(-0.5 * distance * distance)
    else:
        weight = max(Float32(0.01), 1 - scaled_gap)
    activity[i] = z[i] * weight
    mask[i] = 1


def output_code(value: Float32, bound: Float32, half_span: Float32, clip: Int32) -> Float32:
    var normalized = value / bound
    if clip != 0:
        normalized = max(Float32(-1), min(Float32(1), normalized))
    return floor((normalized + 1) * half_span + 0.5)


def predict(activity: Read, prediction: Write, batch_arg: Int32,
            outputs_arg: Int32, width_arg: Int32, divisor: Float32, sum_arg: Int32,
            head_arg: Int32, heads_arg: Int32, bound: Float32, half_span: Float32, clip: Int32,
            class_energy: Int32):
    var batch = Int(batch_arg)
    var outputs = Int(outputs_arg)
    var width = Int(width_arg)
    var i = global_idx.x
    if i >= batch * outputs:
        return
    var sample = i // outputs
    var channel = i % outputs
    var pair = (sample * outputs + channel) * 2 * width
    var value: Float32 = 0
    if sum_arg != 0:
        for j in range(2 * width):
            value += activity[pair + j]
    else:
        var up: Float32 = 0
        var down: Float32 = 0
        for j in range(width):
            var increase = activity[pair + j]
            var decrease = activity[pair + width + j]
            if class_energy != 0:
                up += increase * increase
                down += decrease * decrease
            else:
                up += abs(increase)
                down += abs(decrease)
        value = (up - down) / divisor
    if head_arg > 0:
        value += prediction[i]
    if head_arg + 1 == heads_arg:
        if class_energy == 0:
            value /= Float32(heads_arg)
        if half_span > 0:
            value = (output_code(value, bound, half_span, clip) / half_span - 1) * bound
        elif clip != 0 and bound > 0:
            value = max(-bound, min(bound, value))
    prediction[i] = value


def sign(value: Float32) -> Float32:
    if value > 0:
        return 1
    if value < 0:
        return -1
    return 0


def teaching_signs(target: Read, prediction: Read, teaching: Write, size: Int32,
                   bound: Float32, half_span: Float32, clip: Int32):
    var i = global_idx.x
    if i < Int(size):
        var residual = target[i] - prediction[i]
        if half_span > 0:
            residual = output_code(target[i], bound, half_span, clip) - output_code(
                prediction[i], bound, half_span, clip
            )
        teaching[i] = sign(residual)


def updated_weight(weight: Float32, rate: Float32, direction: Float32,
                   squared: Float32, explicit_decay: Int32) -> Float32:
    if explicit_decay != 0:
        return (1 - rate * squared) * weight + rate * direction
    return (weight + rate * direction) / (1 + rate * squared)


def hebbian[observe: Bool = False](x: Read, activity: Read, teaching: Read,
            weights: Write, history: Write, observations: Write, stats: Write,
            batch_arg: Int32, inputs_arg: Int32, outputs_arg: Int32, width_arg: Int32,
            offset_arg: Int32, rate: Float32, lion: Int32, scale_direction: Int32, explicit_decay: Int32):
    var batch = Int(batch_arg)
    var inputs = Int(inputs_arg)
    var outputs = Int(outputs_arg)
    var width = Int(width_arg)
    var offset = Int(offset_arg)
    var i = global_idx.x
    var units = 2 * width * outputs
    if i >= (inputs + 1) * units:
        return
    var incoming = i // units
    var node = i % units
    var channel = node // (2 * width)
    var polarity = Float32(1 - 2 * (node // width % 2))
    var product: Float32 = 0
    var squared: Float32 = 0
    var count = 0
    for sample in range(batch):
        var index = sample * outputs + channel
        var direction = teaching[index] * polarity
        var post = activity[sample * units + node]
        if direction != 0 and post != 0:
            var pre: Float32 = 1
            if incoming < inputs:
                pre = x[sample * inputs + incoming]
            product += pre * (direction * post)
            squared += post * post
            count += 1
    comptime if observe:
        if incoming == 0:
            stats[node] = Float32(count)
            stats[units + node] = squared / Float32(max(count, 1))
        if count > 0:
            observations[i] = product / Float32(count)
    else:
        if count > 0:
            product /= Float32(count)
            squared /= Float32(count)
            var signal = product
            if lion != 0:
                var old = history[offset + i]
                signal = 0.125 * old + 0.875 * product
                history[offset + i] = 0.99 * old + 0.01 * product
            var direction = sign(signal) / sqrt(Float32(inputs + 1))
            if scale_direction != 0:
                direction *= sqrt(squared / (1 + squared))
            weights[offset + i] = updated_weight(weights[offset + i], rate, direction, squared, explicit_decay)


def bottleneck_gate(z: Read, activity: Write, batch_arg: Int32, units_arg: Int32,
                    width: Float32, pool_arg: Int32):
    var batch = Int(batch_arg)
    var units = Int(units_arg)
    var i = global_idx.x
    if i >= batch * units:
        return
    var pool = Int(pool_arg)
    var node = i % units
    var group_start = (node // pool) * pool
    var start = (i // units) * units + group_start
    var peak: Float32 = 0
    for j in range(min(pool, units - group_start)):
        peak = max(peak, abs(z[start + j]))
    var value = z[i]
    if peak > 0:
        var distance = min((peak - abs(value)) / (peak * width), Float32(13))
        value *= exp(-0.5 * distance * distance)
    activity[i] = value


def encoder_coactivity(activity: Read, signs: Read, counts: Read, signed_post: Write,
                       stats: Write, batch_arg: Int32, units_arg: Int32, class_width_arg: Int32,
                       bottleneck_channel_width: Int32, teaching_width_arg: Int32):
    var batch = Int(batch_arg)
    var units = Int(units_arg)
    var class_width = Int(class_width_arg)
    var teaching_width = Int(teaching_width_arg)
    var node = global_idx.x
    if node >= units:
        return
    var channel = node // (2 * class_width)
    var polarity = 1 - 2 * (node // class_width % 2)
    if bottleneck_channel_width > 0:
        channel = node // Int(bottleneck_channel_width)
        polarity = 1 - 2 * (node % 2)
    if teaching_width == units:
        channel = node
    var squared: Float32 = 0
    var count: Float32 = 0
    for n in range(batch):
        var post = activity[n * units + node]
        signed_post[n * units + node] = signs[n * teaching_width + channel] * Float32(polarity) * post
        if post != 0:
            var selected = counts[n * teaching_width + channel]
            squared += selected * post * post
            count += selected
    stats[node] = count
    stats[units + node] = squared / max(count, Float32(1))


def encoder_update[observe: Bool = False](x: Read, signed_post: Read, stats: Read,
                   weights: Write, history: Write, observations: Write,
                   batch_arg: Int32, inputs_arg: Int32, units_arg: Int32, rate: Float32, lion: Int32, explicit_decay: Int32):
    var batch = Int(batch_arg)
    var inputs = Int(inputs_arg)
    var units = Int(units_arg)
    var i = global_idx.x
    if i >= (inputs + 1) * units:
        return
    var node = i % units
    var count = stats[node]
    if count == 0:
        return
    var incoming = i // units
    var product: Float32 = 0
    for n in range(batch):
        var pre = x[n * inputs + incoming] if incoming < inputs else Float32(1)
        product += pre * signed_post[n * units + node]
    product /= count
    comptime if observe:
        observations[i] = product
    else:
        var signal = product
        if lion != 0:
            var old = history[i]
            signal = 0.125 * old + 0.875 * product
            history[i] = 0.99 * old + 0.01 * product
        var direction = sign(signal) / sqrt(Float32(inputs + 1))
        weights[i] = updated_weight(weights[i], rate, direction, stats[units + node], explicit_decay)


def adapt_gain(history: Read, observations: Read, stats: Read, gains: Write,
               rows_arg: Int32, units_arg: Int32, offset: Int32, gain_offset: Int32):
    var rows = Int(rows_arg)
    var units = Int(units_arg)
    var node = global_idx.x
    if node >= units or stats[node] == 0:
        return
    var alignment: Float32 = 0
    for row in range(rows):
        var i = row * units + node
        alignment += sign(history[Int(offset) + i]) * sign(observations[i])
    var factor = exp(-log(Float32(0.99)) * (alignment / Float32(rows)))
    var index = Int(gain_offset) + node
    gains[index] = max(Float32(0.01), min(Float32(1), gains[index] * factor))


def apply_adapted_lion(observations: Read, stats: Read, gains: Read, weights: Write, history: Write,
                       rows_arg: Int32, units_arg: Int32, offset: Int32, gain_offset: Int32, rate: Float32):
    var rows = Int(rows_arg)
    var units = Int(units_arg)
    var i = global_idx.x
    if i >= rows * units:
        return
    var node = i % units
    if stats[node] == 0:
        return
    var product = observations[i]
    var index = Int(offset) + i
    var old = history[index]
    var direction = sign(0.125 * old + 0.875 * product) / sqrt(Float32(rows))
    history[index] = 0.99 * old + 0.01 * product
    weights[index] = updated_weight(weights[index], rate * gains[Int(gain_offset) + node], direction, stats[units + node], 0)


def normalize_updated(weights: Write, activity: Read, teaching: Read,
                      batch_arg: Int32, rows_arg: Int32, units_arg: Int32, width_arg: Int32,
                      outputs_arg: Int32, offset_arg: Int32):
    var node = global_idx.x
    var units = Int(units_arg)
    if node >= units:
        return
    var channel = node // (2 * Int(width_arg))
    for sample in range(Int(batch_arg)):
        var index = sample * Int(outputs_arg) + channel
        if teaching[index] != 0 and activity[sample * units + node] != 0:
            var first = Int(offset_arg) + node
            var squared: Float32 = 0
            for row in range(Int(rows_arg)):
                var value = weights[first + row * units]
                squared += value * value
            var norm = sqrt(squared)
            if norm > 0:
                for row in range(Int(rows_arg)):
                    weights[first + row * units] /= norm
            return


struct Network(Movable, Writable):
    var ctx: DeviceContext
    var weights: Buffer
    var explicit_decay: Int32
    var lion: Bool
    var scale_direction: Bool
    var output_bound: Float32
    var output_half_span: Float32
    var clip_output: Int32
    var teaching: Buffer
    var history: Buffer
    var adaptive_agreement: Bool
    var agreement: Buffer
    var observations: Buffer
    var coordinates: Buffer
    var target: Buffer
    var prediction: Buffer
    var logits: Buffer
    var activity: List[Buffer]
    var features: List[Buffer]
    var basis_size: Int
    var edge_transitions: List[Float32]
    var masks: List[Buffer]
    var peaks: Buffer
    var inputs: Int
    var outputs: Int
    var depth: Int
    var width: Int
    var units: Int
    var group_size: Int
    var keep: Int
    var gate: Int32
    var gate_widths: List[Float32]
    var rate_multipliers: List[Float32]
    var readout_divisor: Float32
    var class_energy: Int32
    var heads: Int
    var unit_norm_updates: Bool
    var omega: Float32
    var raw_sum: Bool
    var capacity: Int
    var encoded_batch: Int
    var encoder: Bool
    var encoder_signs: Buffer
    var encoder_counts: Buffer
    var signed_post: Buffer
    var encoder_stats: Buffer

    def write_to(self, mut writer: Some[Writer]):
        writer.write("Hebbian GPU network")
    def write_repr_to(self, mut writer: Some[Writer]):
        self.write_to(writer)

    @staticmethod
    def py_init(out self: Network, args: PythonObject, kwargs: PythonObject) raises:
        self = Self(args)

    def __init__(out self, args: PythonObject) raises:
        self.inputs = Int(py=args[0])
        self.outputs = Int(py=args[1])
        self.width = Int(py=args[2])
        self.depth = Int(py=args[3])
        self.keep = Int(py=args[4])
        self.raw_sum = Bool(py=args[5])
        self.capacity = Int(py=args[6])
        self.units = 2 * self.outputs * self.width
        var groups = Int(py=args[7])
        self.group_size = self.units // groups
        self.gate = Int32(Int(py=args[8]))
        self.gate_widths = List[Float32]()
        self.rate_multipliers = List[Float32]()
        for i in range(self.depth):
            self.gate_widths.append(Float32(Float64(py=args[9][i])))
            self.rate_multipliers.append(Float32(Float64(py=args[21][i])))
        self.readout_divisor = Float32(Float64(py=args[10]))
        self.class_energy = Int32(Bool(py=args[22]))
        self.heads = Int(py=args[11])
        self.unit_norm_updates = Bool(py=args[12])
        self.omega = Float32(Float64(py=args[13]))
        self.basis_size = Int(py=args[14])
        self.lion = Bool(py=args[15])
        self.explicit_decay = Int32(Bool(py=args[24]))
        self.adaptive_agreement = Bool(py=args[25])
        self.scale_direction = Bool(py=args[16])
        self.output_bound = Float32(Float64(py=args[17]))
        self.output_half_span = Float32(Float64(py=args[18]))
        self.edge_transitions = List[Float32]()
        for layer in range(self.depth):
            self.edge_transitions.append(Float32(Float64(py=args[19][layer])))
        self.clip_output = Int32(Bool(py=args[20]))
        self.ctx = DeviceContext()
        var size = (self.inputs * self.basis_size + 1 + (self.depth - 1) * (self.units * self.basis_size + 1)) * self.units
        self.weights = self.ctx.enqueue_create_buffer[DType.float32](size)
        self.history = self.ctx.enqueue_create_buffer[DType.float32](size if self.lion else 1)
        self.agreement = self.ctx.enqueue_create_buffer[DType.float32](
            self.units * self.depth if self.adaptive_agreement else 1)
        var largest_input = self.inputs if self.depth == 1 else max(self.inputs, self.units)
        self.observations = self.ctx.enqueue_create_buffer[DType.float32](
            (largest_input * self.basis_size + 1) * self.units if self.adaptive_agreement else 1)
        self.coordinates = self.ctx.enqueue_create_buffer[DType.float32](self.capacity * self.inputs)
        self.target = self.ctx.enqueue_create_buffer[DType.float32](self.capacity * self.outputs)
        self.prediction = self.ctx.enqueue_create_buffer[DType.float32](self.capacity * self.outputs)
        self.teaching = self.ctx.enqueue_create_buffer[DType.float32](self.capacity * self.outputs)
        self.logits = self.ctx.enqueue_create_buffer[DType.float32](self.capacity * self.units)
        self.peaks = self.ctx.enqueue_create_buffer[DType.float32](len(self.logits) // self.group_size)
        self.encoded_batch = 0
        self.encoder = Bool(py=args[23])
        self.encoder_signs = self.ctx.enqueue_create_buffer[DType.float32](self.capacity * self.units if self.encoder else 1)
        self.encoder_counts = self.ctx.enqueue_create_buffer[DType.float32](len(self.encoder_signs))
        self.signed_post = self.ctx.enqueue_create_buffer[DType.float32](len(self.logits) if self.encoder else 1)
        self.encoder_stats = self.ctx.enqueue_create_buffer[DType.float32](
            2 * self.units if self.encoder or self.adaptive_agreement else 1)
        self.features = List[Buffer]()
        if self.basis_size > 1:
            for layer in range(self.depth):
                var fan_in = self.inputs if layer == 0 else self.units
                self.features.append(self.ctx.enqueue_create_buffer[DType.float32](
                    self.capacity * fan_in * self.basis_size))
        self.activity = List[Buffer]()
        self.masks = List[Buffer]()
        for _ in range(self.depth):
            self.activity.append(self.ctx.enqueue_create_buffer[DType.float32](len(self.logits)))
            self.masks.append(self.ctx.enqueue_create_buffer[DType.float32](len(self.logits)))

    @staticmethod
    def load(self: MutPointer[Self, MutAnyOrigin], weights: PythonObject,
             history: PythonObject, agreement: PythonObject) raises:
        upload(self[].weights, weights, len(self[].weights))
        if self[].lion:
            upload(self[].history, history, len(self[].history))
        if self[].adaptive_agreement:
            upload(self[].agreement, agreement, len(self[].agreement))

    @staticmethod
    def save(self: MutPointer[Self, MutAnyOrigin], weights: PythonObject,
             history: PythonObject, agreement: PythonObject) raises:
        download(self[].weights, weights, len(self[].weights))
        if self[].lion:
            download(self[].history, history, len(self[].history))
        if self[].adaptive_agreement:
            download(self[].agreement, agreement, len(self[].agreement))

    def adapt_and_update(mut self, inputs: Int, offset: Int, layer: Int, rate: Float32) raises:
        var rows = inputs + 1
        var gain_offset = layer * self.units
        # Reduce signs using the old history before applying the weight and history updates.
        self.ctx.enqueue_function[adapt_gain](
            read(self.history), read(self.observations), read(self.encoder_stats), write(self.agreement),
            Int32(rows), Int32(self.units), Int32(offset), Int32(gain_offset),
            grid_dim=ceildiv(self.units, Block), block_dim=Block)
        self.ctx.enqueue_function[apply_adapted_lion](
            read(self.observations), read(self.encoder_stats), read(self.agreement), write(self.weights), write(self.history),
            Int32(rows), Int32(self.units), Int32(offset), Int32(gain_offset), rate,
            grid_dim=ceildiv(rows * self.units, Block), block_dim=Block)

    def expand(mut self, source: Read, layer: Int, count: Int) raises:
        self.ctx.enqueue_function[expand_edges](
            source, write(self.features[layer]), Int32(count), Int32(self.basis_size), self.edge_transitions[layer],
            grid_dim=ceildiv(count, Block), block_dim=Block)

    def forward(mut self, batch: Int, bottleneck_pool: Int = 0) raises:
        var offset = 0
        var incoming = read(self.coordinates)
        var inputs = self.inputs
        var blocks = ceildiv(batch * self.units, Block)
        for layer in range(self.depth):
            if self.basis_size > 1:
                self.expand(incoming, layer, batch * inputs)
                incoming = read(self.features[layer])
                inputs *= self.basis_size
            self.ctx.enqueue_function[dense](
                incoming, read(self.weights), write(self.logits), Int32(batch), Int32(inputs), Int32(self.units),
                Int32(offset), self.omega, grid_dim=blocks, block_dim=Block)
            if self.gate == 0:
                var pool = bottleneck_pool if bottleneck_pool != 0 else self.group_size
                if self.keep >= pool:
                    self.ctx.enqueue_function[pass_activity](
                        read(self.logits), write(self.activity[layer]), write(self.masks[layer]),
                        Int32(batch * self.units), grid_dim=blocks, block_dim=Block)
                elif pool >= Block:
                    var pools = batch * ceildiv(self.units, pool)
                    self.ctx.enqueue_function[compete_wide](
                        read(self.logits), write(self.activity[layer]), write(self.masks[layer]),
                        Int32(batch), Int32(self.units), Int32(pool), Int32(self.keep),
                        grid_dim=ceildiv(pools * WARP_SIZE, Block), block_dim=Block)
                else:
                    self.ctx.enqueue_function[compete](
                        read(self.logits), write(self.activity[layer]), write(self.masks[layer]),
                        Int32(batch), Int32(self.units), Int32(pool), Int32(self.keep),
                        grid_dim=blocks, block_dim=Block)
            elif bottleneck_pool != 0:
                self.ctx.enqueue_function[bottleneck_gate](
                    read(self.logits), write(self.activity[layer]), Int32(batch), Int32(self.units),
                    self.gate_widths[layer], Int32(bottleneck_pool), grid_dim=blocks, block_dim=Block)
            elif self.gate == 3:
                self.ctx.enqueue_function[pass_activity](
                    read(self.logits), write(self.activity[layer]), write(self.masks[layer]),
                    Int32(batch * self.units), grid_dim=blocks, block_dim=Block)
            else:
                var pool_count = batch * self.units // self.group_size
                if self.gate == 4:
                    self.ctx.enqueue_function[smooth_pool_centers](
                        read(self.logits), write(self.peaks), Int32(pool_count), Int32(self.group_size),
                        grid_dim=ceildiv(pool_count * WARP_SIZE, Block), block_dim=Block)
                elif self.group_size >= Block:
                    self.ctx.enqueue_function[wide_pool_maxima](
                        read(self.logits), write(self.peaks), Int32(pool_count), Int32(self.group_size),
                        grid_dim=ceildiv(pool_count * WARP_SIZE, Block), block_dim=Block)
                else:
                    self.ctx.enqueue_function[pool_maxima](
                        read(self.logits), write(self.peaks), Int32(pool_count), Int32(self.group_size),
                        grid_dim=ceildiv(pool_count, Block), block_dim=Block)
                self.ctx.enqueue_function[scale_activity](
                    read(self.logits), read(self.peaks), write(self.activity[layer]),
                    write(self.masks[layer]), Int32(batch * self.units), Int32(self.group_size),
                    self.gate, self.gate_widths[layer], grid_dim=blocks, block_dim=Block)
            offset += (inputs + 1) * self.units
            inputs = self.units
            incoming = read(self.activity[layer])
        for head in range(self.heads):
            var layer = self.depth - self.heads + head
            self.ctx.enqueue_function[predict](
                read(self.activity[layer]), write(self.prediction),
                Int32(batch), Int32(self.outputs), Int32(self.width), self.readout_divisor,
                Int32(self.raw_sum), Int32(head), Int32(self.heads), self.output_bound, self.output_half_span,
                self.clip_output, self.class_energy,
                grid_dim=ceildiv(batch * self.outputs, Block), block_dim=Block)

    @staticmethod
    def infer(self: MutPointer[Self, MutAnyOrigin], x: PythonObject,
              y: PythonObject, masks: PythonObject) raises:
        var batch = len(x)
        if batch <= 0 or batch > self[].capacity:
            raise Error("Invalid GPU batch size")
        upload(self[].coordinates, x, batch * self[].inputs)
        self[].forward(batch)
        download(self[].prediction, y, batch * self[].outputs)
        for layer in range(len(masks)):
            download(self[].masks[layer], masks[layer], batch * self[].units)

    @staticmethod
    def encode(self: MutPointer[Self, MutAnyOrigin], x: PythonObject, y: PythonObject,
               bottleneck_pool: PythonObject) raises:
        var batch = len(x)
        if not self[].encoder or batch <= 0 or batch > self[].capacity:
            raise Error("Invalid GPU encoder batch")
        self[].encoded_batch = batch
        upload(self[].coordinates, x, batch * self[].inputs)
        self[].forward(batch, Int(py=bottleneck_pool))
        download(self[].activity[0], y, batch * self[].units)

    @staticmethod
    def update_encoder(self: MutPointer[Self, MutAnyOrigin], signed: PythonObject,
                       counts: PythonObject, rate: PythonObject, channel_width: PythonObject) raises:
        var batch = len(signed)
        if not self[].encoder or batch != self[].encoded_batch or batch <= 0:
            raise Error("Encoder update requires a matching forward batch")
        self[].encoded_batch = 0
        var teaching_width = len(signed[0])
        if teaching_width != 3 and teaching_width != self[].units:
            raise Error("Invalid encoder teaching width")
        upload(self[].encoder_signs, signed, batch * teaching_width)
        upload(self[].encoder_counts, counts, batch * teaching_width)
        self[].ctx.enqueue_function[encoder_coactivity](
            read(self[].activity[0]), read(self[].encoder_signs), read(self[].encoder_counts),
            write(self[].signed_post), write(self[].encoder_stats), Int32(batch), Int32(self[].units), Int32(self[].width),
            Int32(Int(py=channel_width)), Int32(teaching_width), grid_dim=ceildiv(self[].units, Block), block_dim=Block)
        var step = Float32(Float64(py=rate))
        if self[].adaptive_agreement:
            comptime observe = encoder_update[True]
            self[].ctx.enqueue_function[observe](
                read(self[].coordinates), read(self[].signed_post), read(self[].encoder_stats),
                write(self[].weights), write(self[].history), write(self[].observations),
                Int32(batch), Int32(self[].inputs), Int32(self[].units), step, Int32(self[].lion), self[].explicit_decay,
                grid_dim=ceildiv(len(self[].weights), Block), block_dim=Block)
            self[].adapt_and_update(self[].inputs, 0, 0, step)
        else:
            comptime update = encoder_update[False]
            self[].ctx.enqueue_function[update](
                read(self[].coordinates), read(self[].signed_post), read(self[].encoder_stats),
                write(self[].weights), write(self[].history), write(self[].observations),
                Int32(batch), Int32(self[].inputs), Int32(self[].units), step, Int32(self[].lion), self[].explicit_decay,
                grid_dim=ceildiv(len(self[].weights), Block), block_dim=Block)

    @staticmethod
    def train(self: MutPointer[Self, MutAnyOrigin], x: PythonObject,
              y: PythonObject, rate: PythonObject, signs: PythonObject) raises:
        var batch = len(x)
        if batch <= 0 or batch > self[].capacity:
            raise Error("Invalid GPU batch size")
        upload(self[].coordinates, x, batch * self[].inputs)
        upload(self[].target, y, batch * self[].outputs)
        self[].forward(batch)
        var output_count = batch * self[].outputs
        self[].ctx.enqueue_function[teaching_signs](
            read(self[].target), read(self[].prediction), write(self[].teaching), Int32(output_count),
            self[].output_bound, self[].output_half_span, self[].clip_output,
            grid_dim=ceildiv(output_count, Block), block_dim=Block)
        if len(signs) > 0:
            download(self[].teaching, signs, output_count)
        var inputs = self[].inputs
        var incoming = read(self[].coordinates)
        var offset = 0
        var hidden_rate = Float32(Float64(py=rate))
        for layer in range(self[].depth):
            if self[].basis_size > 1:
                incoming = read(self[].features[layer])
                inputs *= self[].basis_size
            var count = (inputs + 1) * self[].units
            var step = hidden_rate * self[].rate_multipliers[layer]
            if self[].adaptive_agreement:
                comptime observe = hebbian[True]
                self[].ctx.enqueue_function[observe](
                    incoming, read(self[].activity[layer]), read(self[].teaching),
                    write(self[].weights), write(self[].history), write(self[].observations), write(self[].encoder_stats),
                    Int32(batch), Int32(inputs), Int32(self[].outputs), Int32(self[].width),
                    Int32(offset), step, Int32(self[].lion), Int32(self[].scale_direction), self[].explicit_decay,
                    grid_dim=ceildiv(count, Block), block_dim=Block)
                self[].adapt_and_update(inputs, offset, layer, step)
            else:
                comptime update = hebbian[False]
                self[].ctx.enqueue_function[update](
                    incoming, read(self[].activity[layer]), read(self[].teaching),
                    write(self[].weights), write(self[].history), write(self[].observations), write(self[].encoder_stats),
                    Int32(batch), Int32(inputs), Int32(self[].outputs), Int32(self[].width),
                    Int32(offset), step, Int32(self[].lion), Int32(self[].scale_direction), self[].explicit_decay,
                    grid_dim=ceildiv(count, Block), block_dim=Block)
            if self[].unit_norm_updates:
                self[].ctx.enqueue_function[normalize_updated](
                    write(self[].weights), read(self[].activity[layer]), read(self[].teaching),
                    Int32(batch), Int32(inputs + 1), Int32(self[].units),
                    Int32(self[].width), Int32(self[].outputs), Int32(offset),
                    grid_dim=ceildiv(self[].units, Block), block_dim=Block)
            offset += count
            inputs = self[].units
            incoming = read(self[].activity[layer])


@export
def PyInit__hebbian_gpu() abi("C") -> PythonObject:
    try:
        var module = PythonModuleBuilder("_hebbian_gpu")
        _ = module.add_type[Network]("Network").def_py_init[Network.py_init]() \
            .def_method[Network.load]("load").def_method[Network.save]("save") \
            .def_method[Network.infer]("infer").def_method[Network.train]("train") \
            .def_method[Network.encode]("encode").def_method[Network.update_encoder]("update_encoder")
        return module.finalize()
    except error:
        abort(String(error))
