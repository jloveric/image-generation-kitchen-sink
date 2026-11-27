import os
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np

from hebbian_fit.autoencoder import (
    Autoencoder, EncoderLayer, Generator, encoder_configs, export_generator, rgb_bytes, signed_totals,
    vote_assignment, projected_votes,
)
from hebbian_fit.gpu import MojoModel
from hebbian_fit.model import ModelConfig, NumpyModel, select


class AutoencoderTest(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(13)
        self.images = rng.uniform(-1, 1, (4, 12)).astype(np.float32)
        self.points = rng.uniform(-1, 1, (4, 7, 2)).astype(np.float32)
        self.targets = np.where(rng.random((4, 7, 3)) > .5, .95, -.95).astype(np.float32)

    def model(self, backend="numpy", rule="signsgd"):
        return Autoencoder(3, backend, images_per_batch=4, queries_per_batch=32, inputs=12, rule=rule)

    def predictions(self, model):
        codes = model.encode(self.images)
        return np.concatenate([model.decode(code[None], points)
                               for code, points in zip(codes, self.points)])

    def test_encoder_reduction_matches_explicit_pixel_updates(self):
        signs = np.random.default_rng(2).integers(-1, 2, (4, 7, 3)).astype(np.float32)
        totals, counts = signed_totals(signs)
        for bottleneck, config in enumerate(encoder_configs(12, "signsgd")):
            layer = EncoderLayer(config, 1, 4, "numpy", bool(bottleneck))
            values = self.images if not bottleneck else np.tile(self.images, (1, 16))
            _, (pre, post) = layer.forward(values)
            teaching = signs.reshape(-1, 3)[:, layer.channels] * layer.polarity
            pre_repeated = np.repeat(pre, 7, axis=0)
            post_repeated = np.repeat(post, 7, axis=0)
            rate = .004
            expected = layer.model.weights[0].copy()
            for node, direction in enumerate(teaching.T):
                use = (direction != 0) & (post_repeated[:, node] != 0)
                values = post_repeated[use, node]
                local = np.mean(pre_repeated[use] * (direction[use] * values)[:, None], axis=0)
                expected[:, node] = (expected[:, node] + rate * np.sign(local) / np.sqrt(len(expected))) / (
                    1 + rate * np.mean(values**2))
            layer.update((pre, post), totals, counts, rate)
            np.testing.assert_allclose(layer.model.weights[0], expected, atol=1e-7)

    def test_only_final_sign_updates_every_layer(self):
        a, b = self.model(), self.model()
        before = a.state()
        prediction = self.predictions(a)
        # Same sign, two different magnitudes. Keep both far from quantization ties.
        target_a = np.where(prediction < 0, .9, -.9).astype(np.float32)
        target_b = target_a / 2
        a.step(self.images, self.points, target_a, .004)
        b.step(self.images, self.points, target_b, .004)
        other = b.state()
        for key, array in a.state().items():
            self.assertFalse(np.array_equal(array, before[key]), key)
            np.testing.assert_array_equal(array, other[key])

    def test_correct_prediction_does_not_update(self):
        model = self.model()
        before = model.state()
        model.step(self.images, self.points, self.predictions(model), .004)
        for key, array in model.state().items():
            np.testing.assert_array_equal(array, before[key])

    def test_deeper_encoder_updates_all_layers_from_the_same_residual_signs(self):
        settings = dict(images_per_batch=4, inputs=12, decoder_kind="vector", encoder_width=24,
                        encoder_hidden_widths=(18, 12), latent_width=8, projected_encoder_votes=True)
        a, b = Autoencoder(3, **settings), Autoencoder(3, **settings)
        self.assertEqual([(layer.config.inputs, layer.config.units) for layer in a.encoder],
                         [(12, 24), (24, 18), (18, 12), (12, 8)])
        initial = a.state()
        prediction = a.decode(a.encode(self.images))
        targets = np.where(prediction < 0, .9, -.9).astype(np.float32)
        a.step(self.images, None, targets, .004)
        b.step(self.images, None, targets / 2, .004)
        other = b.state()
        for key, array in a.state().items():
            self.assertFalse(np.array_equal(array, initial[key]), key)
            np.testing.assert_array_equal(array, other[key], err_msg=key)

    def test_top_k_autoencoder_uses_signed_updates_in_each_layer(self):
        model = Autoencoder(3, images_per_batch=4, inputs=12, decoder_kind="vector",
                            encoder_width=24, latent_width=26, encoder_top_k=(5, 6), vector_top_k=1)
        first, code = model.encoder
        activity, _ = first.forward(self.images)
        np.testing.assert_array_equal(np.count_nonzero(activity.reshape(4, 3, 8), axis=-1), 5)
        codes = model.encode(self.images)
        for pool in code.groups:
            np.testing.assert_array_equal(np.count_nonzero(codes[:, pool], axis=1), 6)
        prediction, traces = model.decoder.forward(codes)
        mask = traces[0][2]
        np.testing.assert_array_equal(mask.reshape(4, -1, 2).sum(axis=2), 1)
        initial = model.state()
        signed_targets = np.where(prediction < 0, 1., -1.).reshape(4, -1, 3)
        model.step(self.images, None, signed_targets, .004)
        for key, values in model.state().items():
            self.assertFalse(np.array_equal(values, initial[key]), key)

    def test_decoder_export(self):
        model = self.model()
        with TemporaryDirectory() as directory:
            path = Path(directory) / "decoder.npz"
            model.step(self.images, self.points, self.targets, .004)
            export_generator(path, model.decoder, model.encode(self.images))
            first = Generator(path).generate(0)
            np.testing.assert_array_equal(first, Generator(path).generate(0))
            self.assertEqual(first.shape, (64, 64, 3))
            self.assertFalse(np.array_equal(first, Generator(path).generate(1)))
            with np.load(path) as exported:
                self.assertEqual(set(exported.files), {"config", "mean", "root", "coordinate_scale"} |
                                 {f"weight_{i}" for i in range(6)})

    def test_single_neuron_outputs_follow_signed_teaching(self):
        config = ModelConfig(inputs=1, outputs=4, width=1, depth=1, gate="none",
                             readout="magnitude", output_bound=1, output_levels=256)
        backends = (NumpyModel, MojoModel) if os.environ.get("HEBBIAN_TEST_GPU") == "1" else (NumpyModel,)
        x = np.zeros((1, 1), np.float32)
        target = np.array([[1, 1, -1, 1]], np.float32)
        for backend in backends:
            with self.subTest(backend=backend.__name__):
                model = backend(config, 0)
                before = model.state()
                before["weight_0"][0] = 0
                before["weight_0"][1] = [.25, -.25, -.25, 1.25]
                model.load(before)
                np.testing.assert_allclose(model.predict_values(x), [[-127 / 255] * 3 + [1]], atol=1e-7)
                np.testing.assert_array_equal(model.step(x, target, .01, return_signs=True), [[1, 1, -1, 0]])
                change = np.abs(model.state()["weight_0"][1]) - np.abs(before["weight_0"][1])
                np.testing.assert_array_equal(np.sign(change), [1, 1, -1, 0])

    def test_vector_decoder_trains_and_exports(self):
        cases = ((0, 64, 1, False, 1), (16, 128, 2, False, 1), (16, 64, 1, True, 1),
                 (8, 64, 1, True, 2), (8, 64, 1, True, 3), (8, 64, 1, False, 2))
        for width, latent, output_width, skip, depth in cases:
            with self.subTest(hidden_width=width, code_width=latent, output_width=output_width, skip=skip, depth=depth):
                model = Autoencoder(3, images_per_batch=4, inputs=12, decoder_kind="vector",
                                    vector_hidden_width=width, vector_hidden_depth=depth, latent_width=latent,
                                    vector_output_class_width=output_width, vector_decoder_skip=skip,
                                    projected_decoder_votes=skip, decoder_rate_decay=.15 if skip else 1.,
                                    vector_gate_width=.38)
                self.assertEqual(model.decoder.config.gate_width, .38)
                initial = model.state()
                codes = model.encode(self.images)
                self.assertEqual(codes.shape, (4, latent))
                stages = [codes]
                components = model.components()
                for index in range(depth if width else 0):
                    name = "hidden" if index == 0 else f"hidden{index + 1}"
                    layer = components[name]
                    presynaptic = stages[-1]
                    self.assertEqual(layer.config.inputs, presynaptic.shape[1])
                    raw = np.column_stack((presynaptic, np.ones(len(codes), np.float32))) @ layer.weights[0]
                    stages.append(select(raw, layer.config)[0])
                expected_inputs = np.concatenate(stages, axis=1) if skip else stages[-1]
                expected_prediction = model.decoder.predict_values(expected_inputs).reshape(4, -1, 3)
                np.testing.assert_array_equal(model.decode(codes), expected_prediction)
                model.step(self.images, None, expected_prediction, .004)
                for key, array in model.state().items():
                    np.testing.assert_array_equal(array, initial[key], err_msg=key)
                targets = -np.sign(expected_prediction)
                model.step(self.images, None, targets, .004)
                for key, array in model.state().items():
                    self.assertFalse(np.array_equal(array, initial[key]), key)
                trained = model.state()
                model.load(initial)
                model.step(self.images, None, targets / 2, .004)
                for key, array in model.state().items():
                    np.testing.assert_array_equal(array, trained[key], err_msg=key)
                model.load(initial)
                model.load(trained)
                for key, restored in model.state().items():
                    np.testing.assert_array_equal(restored, trained[key])
                with TemporaryDirectory() as directory:
                    path = Path(directory) / "decoder.npz"
                    export_generator(path, model.decoder, model.encode(self.images), hidden=model.decoder_hidden,
                                     skip=model.decoder_skip)
                    generator = Generator(path)
                    noise = np.random.default_rng(0).standard_normal(latent)
                    code = (generator.mean + generator.root @ noise)[None].astype(np.float32)
                    expected = rgb_bytes(model.decode(code)).reshape(64, 64, 3)
                    np.testing.assert_array_equal(generator.generate(0), expected)
                    self.assertFalse(np.array_equal(generator.generate(0), generator.generate(1)))

    def test_balanced_votes_cancel_uniform_error_signs(self):
        projection = vote_assignment(96, 1, True, votes_per_pair=8)
        teaching = np.ones((2, 64 * 64 * 3), np.float32)
        accumulated, counts = projected_votes(teaching, projection, np.arange(96))
        np.testing.assert_array_equal(accumulated, 0)
        np.testing.assert_array_equal(counts, 8)
        self.assertEqual(len(np.unique(projection[0])), 96 * 8)

    def test_decoder_samples_mixture_without_encoder(self):
        model = Autoencoder(3, inputs=12, decoder_kind="vector")
        codes = model.encode(self.images)
        with TemporaryDirectory() as directory:
            base = Path(directory) / "base.npz"
            export_generator(base, model.decoder, codes)
            with np.load(base) as data:
                bundle = dict(data)
            mix = Path(directory) / "mixture.npz"
            np.savez(mix, **bundle, mixture_weights=[1., 0.], mixture_means=codes[:2],
                     mixture_roots=np.zeros((2, 64, 8)))
            generator = Generator(mix)
            expected = np.rint((model.decode(codes[:1])[0] + 1) * 127.5).astype(np.uint8).reshape(64, 64, 3)
            np.testing.assert_array_equal(generator.generate(9), expected)
            np.testing.assert_array_equal(generator.generate(3), expected)

    def test_bottleneck_pools_keep_both_classes_and_scale_locally(self):
        config = encoder_configs(12, "signsgd", (.5, .5))[1]
        layer = EncoderLayer(config, 1, 1, "numpy", True, pool_size=4)
        layer.model.weights[0].fill(0)
        layer.model.weights[0][-1, :8] = [1, -.9, .4, -.3, .2, -.18, .08, -.06]
        activity, _ = layer.forward(np.zeros((1, 192), np.float32))
        expected = np.array([1, -.9, .4, -.3]) * np.exp(-.5 * ((1 - np.array([1, .9, .4, .3])) / .5)**2)
        np.testing.assert_allclose(activity[0, :4], expected, atol=1e-7)
        np.testing.assert_allclose(activity[0, 4:8], expected * .2, atol=1e-7)
        np.testing.assert_array_equal(activity[0, 8:], 0)

    @unittest.skipUnless(os.environ.get("HEBBIAN_TEST_GPU") == "1", "GPU disabled")
    def test_gpu_encoder_decoder_step_agrees_with_numpy(self):
        for rule in ("signsgd", "lion", "lion-explicit", "lion-adaptive-agreement"):
            with self.subTest(rule=rule):
                cpu, gpu = self.model(rule=rule), self.model("mojo", rule)
                np.testing.assert_allclose(cpu.encode(self.images), gpu.encode(self.images), atol=5e-6)
                for _ in range(2):
                    for model in (cpu, gpu):
                        model.step(self.images, self.points, self.targets, .004)
                actual = gpu.state()
                for key, expected in cpu.state().items():
                    np.testing.assert_allclose(actual[key], expected, atol=3e-6, err_msg=key)

    @unittest.skipUnless(os.environ.get("HEBBIAN_TEST_GPU") == "1", "GPU disabled")
    def test_gpu_projected_bottleneck_signs_agree_with_numpy(self):
        settings = dict(images_per_batch=4, inputs=12, decoder_kind="vector",
                        projected_latent_votes=True)
        for rule in ("signsgd", "lion-adaptive-agreement"):
            with self.subTest(rule=rule):
                cpu = Autoencoder(3, rule=rule, **settings)
                gpu = Autoencoder(3, backend="mojo", rule=rule, **settings)
                targets = np.where(cpu.decode(cpu.encode(self.images)) > 0, -1., 1.).astype(np.float32)
                for _ in range(2):
                    for model in (cpu, gpu):
                        model.step(self.images, None, targets, .004)
                actual = gpu.state()
                for key, expected in cpu.state().items():
                    np.testing.assert_allclose(actual[key], expected, atol=3e-6, err_msg=key)

    @unittest.skipUnless(os.environ.get("HEBBIAN_TEST_GPU") == "1", "GPU disabled")
    def test_gpu_small_pools_and_balanced_encoder_feedback(self):
        for skip, depth, extra_widths in ((False, 1, ()), (True, 1, ()), (True, 2, ()), (False, 2, (48, 24))):
            with self.subTest(skip=skip, depth=depth, encoder_hidden_widths=extra_widths):
                settings = dict(images_per_batch=4, inputs=12, decoder_kind="vector",
                                projected_encoder_votes=True, bottleneck_pool_size=8, center_each_input=True,
                                encoder_hidden_widths=extra_widths,
                                vector_hidden_width=16, vector_hidden_depth=depth, vector_decoder_skip=skip, projected_decoder_votes=skip,
                                decoder_rate_decay=.15 if skip else 1.)
                cpu, gpu = Autoencoder(3, **settings), Autoencoder(3, backend="mojo", **settings)
                codes = cpu.encode(self.images)
                np.testing.assert_allclose(codes, gpu.encode(self.images), atol=5e-6)
                np.testing.assert_allclose(cpu.decode(codes), gpu.decode(codes), atol=2 / 255 + 1e-6)
                targets = np.where(cpu.decode(codes) > 0, -1., 1.).astype(np.float32)
                for model in (cpu, gpu):
                    model.step(self.images, None, targets, .004)
                actual = gpu.state()
                for key, expected in cpu.state().items():
                    if (skip or extra_widths) and key.startswith("decoder_"):
                        # Roundoff near zero coactivity can change a sign update by one step.
                        continue
                    np.testing.assert_allclose(actual[key], expected, atol=3e-6, err_msg=key)
                np.testing.assert_allclose(cpu.decode(codes), gpu.decode(codes), atol=2 / 255 + 1e-6)

    @unittest.skipUnless(os.environ.get("HEBBIAN_TEST_GPU") == "1", "GPU disabled")
    def test_gpu_top_k_matches_numpy_with_ties_and_a_short_final_pool(self):
        for width, latent in ((24, 26), (768, 770)):
            first, code = encoder_configs(12, "signsgd", encoder_width=width, latent_width=latent,
                                          top_k=(width // 12, latent // 6))
            for index, config in enumerate((first, code)):
                with self.subTest(width=config.units, bottleneck=bool(index)):
                    cpu = EncoderLayer(config, 1, 4, "numpy", bool(index))
                    gpu = EncoderLayer(config, 1, 4, "mojo", bool(index))
                    state = cpu.model.state()
                    weight = state["weight_0"]
                    weight.fill(0)
                    weight[-1] = np.resize([0, -3, 3, 1, -1, 0, -3], config.units)
                    cpu.model.load(state)
                    gpu.model.load(state)
                    samples = np.zeros((4, config.inputs), np.float32)
                    expected, _ = cpu.forward(samples)
                    actual, _ = gpu.forward(samples)
                    np.testing.assert_array_equal(actual, expected)

    @unittest.skipUnless(os.environ.get("HEBBIAN_TEST_GPU") == "1", "GPU disabled")
    def test_gpu_top_one_output_matches_numpy(self):
        settings = dict(images_per_batch=4, inputs=12, decoder_kind="vector",
                        encoder_width=24, latent_width=26, encoder_top_k=(5, 6), vector_top_k=1)
        cpu, gpu = Autoencoder(3, **settings), Autoencoder(3, backend="mojo", **settings)
        codes = cpu.encode(self.images)
        expected, cpu_masks = cpu.decoder.predict(codes)
        actual, gpu_masks = gpu.decoder.predict(codes)
        np.testing.assert_allclose(actual, expected, atol=2 / 255 + 1e-6)
        np.testing.assert_array_equal(gpu_masks[0], cpu_masks[0])

    @unittest.skipUnless(os.environ.get("HEBBIAN_TEST_GPU") == "1", "GPU disabled")
    def test_gpu_wider_layers_agree_with_numpy(self):
        for projected in (False, True):
            with self.subTest(projected=projected):
                settings = dict(images_per_batch=4, inputs=12, decoder_kind="vector",
                                encoder_width=384, latent_width=128, projected_encoder_votes=projected,
                                vector_output_class_width=2 if projected else 1)
                cpu, gpu = Autoencoder(3, **settings), Autoencoder(3, backend="mojo", **settings)
                codes = cpu.encode(self.images)
                np.testing.assert_allclose(codes, gpu.encode(self.images), atol=5e-6)
                targets = np.where(np.random.default_rng(51).random((4, 64 * 64, 3)) > .5,
                                   -1., 1.).astype(np.float32)
                for model in (cpu, gpu):
                    model.step(self.images, None, targets, .004)
                actual = gpu.state()
                for key, expected in cpu.state().items():
                    if key.startswith("encoder_"):
                        np.testing.assert_allclose(actual[key], expected, atol=3e-6, err_msg=key)
                np.testing.assert_allclose(cpu.decode(codes), gpu.decode(codes), atol=2 / 255 + 1e-6)
