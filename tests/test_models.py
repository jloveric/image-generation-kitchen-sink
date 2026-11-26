import os
import unittest
from dataclasses import replace
from types import SimpleNamespace

import numpy as np

from hebbian_fit.data import EDGES, scalar_data, step_metrics
from hebbian_fit.__main__ import training_epochs
from hebbian_fit.model import ModelConfig, NumpyModel, connection_inputs, inverse_participation, select, with_bias


class NumpyTests(unittest.TestCase):
    def test_checked_step_restores_weights_and_history_before_skipping_a_reversed_sign(self):
        cfg = ModelConfig(width=1, depth=1, gate="none")
        ordinary = NumpyModel(cfg, 5)
        checked = NumpyModel(replace(cfg, hidden_rule="lion-checked-step"), 5)
        weights = np.array([[0., 0.], [.2, .05]], dtype=np.float32)
        for model in (ordinary, checked):
            model.weights[0][:] = weights
        x = np.zeros((1, 1), np.float32)
        target = ordinary.predict_values(x) + .001
        saved = checked.state()
        ordinary.step(x, target, .05)
        self.assertLess(ordinary.teaching_signs(ordinary.predict_values(x), target).item(), 0)
        checked.step(x, target, .05)
        for key, value in checked.state().items():
            np.testing.assert_array_equal(value, saved[key])

    def test_centered_lion_matches_training_in_centered_feature_coordinates(self):
        cfg = ModelConfig(inputs=2, width=1, depth=1, gate="none")
        ordinary = NumpyModel(cfg, 7)
        centered_update = NumpyModel(replace(cfg, hidden_rule="lion-centered"), 7)
        x = np.array([[1., -.5], [3., .5], [2., 1.5]], np.float32)
        mean = x.mean(axis=0)
        ordinary.weights[0][-1] += mean @ ordinary.weights[0][:-1]
        target = centered_update.predict_values(x) + np.array([[1.], [-.5], [.25]], np.float32)
        ordinary.step(x - mean, target, .025)
        centered_update.step(x, target, .025)
        ordinary.weights[0][-1] -= mean @ ordinary.weights[0][:-1]
        np.testing.assert_allclose(centered_update.weights[0], ordinary.weights[0], atol=1e-7)
        np.testing.assert_allclose(centered_update.history[0], ordinary.history[0], atol=1e-7)

    def test_inverse_participation_counts_effectively_active_examples(self):
        activity = np.array([[0., -3., 9.], [0., 3., 0.], [0., -3., 0.]], dtype=np.float32)
        np.testing.assert_allclose(inverse_participation(activity), [0., 1/3, 1.])
        np.testing.assert_allclose(inverse_participation(activity * 1e-25), [0., 1/3, 1.])

    def test_lion_weight_power_scales_the_step_by_each_neurons_current_weights(self):
        cfg = ModelConfig(width=1, depth=1, gate="none", hidden_rule="lion-weight-power")
        model = NumpyModel(cfg, 5)
        weights = np.array([[1., .25], [1., .25]], dtype=np.float32)
        model.weights[0][:] = weights
        x = np.ones((1, 1), dtype=np.float32)
        rate = .05
        model.step(x, model.predict_values(x) + 1, rate)
        steps = rate * np.square(weights).sum(axis=0)
        direction = np.array([1., -1.]) / np.sqrt(2)
        expected = (weights + steps * direction) / (1 + steps * np.square(weights.sum(axis=0)))
        np.testing.assert_allclose(model.weights[0], expected, rtol=1e-6)

    def test_lion_coherence_damps_conflicting_observations_and_resumes(self):
        cfg = ModelConfig(width=1, depth=1, gate="none", hidden_rule="lion-coherence")
        model = NumpyModel(cfg, 5)
        model.weights[0][:] = [[1, 0], [1, 0]]
        x = np.ones((1, 1), dtype=np.float32)
        model.history[0][:, 0] = -.01
        model.magnitude_history[0][:, 0] = .04
        rate = .125
        target = model.predict_values(x) + 1
        restored = NumpyModel(cfg, 6)
        restored.load(model.state())
        for current in (model, restored):
            current.step(x, target, rate)
        signed = .99 * -.01 + .01 * 2
        magnitude = .99 * .04 + .01 * 2
        effective_rate = rate * signed / magnitude
        expected = (1 + effective_rate / np.sqrt(2)) / (1 + effective_rate * 4)
        np.testing.assert_allclose(model.weights[0][:, 0], expected)
        np.testing.assert_allclose(model.take_mean_step_multipliers(), [signed / magnitude])
        for name, value in model.state().items():
            np.testing.assert_array_equal(value, restored.state()[name])

    def test_adaptive_agreement_recovers_or_reduces_each_neurons_step(self):
        cfg = ModelConfig(width=1, depth=1, gate="none", hidden_rule="lion-adaptive-agreement")
        model = NumpyModel(cfg, 5)
        model.weights[0].fill(.5)
        model.history[0].fill(-.1)
        model.agreement_history[0].fill(.5)
        restored = NumpyModel(cfg, 6)
        restored.load(model.state())

        x = np.ones((1, 1), dtype=np.float32)
        target = model.predict_values(x) + 1
        for current in (model, restored):
            current.step(x, target, .05)

        gains = .5 * np.array([.99, 1 / .99])
        rates = .05 * gains
        expected = (.5 + rates * np.array([1, -1]) / np.sqrt(2)) / (1 + rates)
        np.testing.assert_allclose(model.agreement_history[0], gains)
        np.testing.assert_allclose(model.weights[0], np.broadcast_to(expected, (2, 2)))
        for name, value in model.state().items():
            np.testing.assert_array_equal(value, restored.state()[name])

    def test_interlayer_max_abs_uses_one_peak_per_example_across_classes(self):
        cfg = ModelConfig(width=1, depth=3, gate="none", normalize_hidden_inputs="max-abs")
        model = NumpyModel(cfg, 2)
        model.weights[0][:] = [[-4., 2.], [0., 0.]]
        x = np.array([[1.], [-2.], [0.]], dtype=np.float32)
        _, traces = model.forward(x)
        np.testing.assert_array_equal(traces[0][1], x * [[-4., 2.]])
        expected = np.array([[-1., .5], [1., -.5], [0., 0.]], dtype=np.float32)
        np.testing.assert_array_equal(traces[1][0], with_bias(expected))
        post = traces[1][1]
        expected = post / np.max(np.abs(post), axis=1, keepdims=True)
        np.testing.assert_allclose(traces[2][0], with_bias(expected))

    def test_hidden_rms_scales_connections_per_example_and_preserves_layer_outputs(self):
        cfg = ModelConfig(width=1, depth=2, gate="none", readout_layers="all", normalize_hidden_inputs=True)
        model = NumpyModel(cfg, 1)
        model.weights[0][:] = [[3, -4], [0, 0]]
        x = np.array([[0], [1], [-2]], np.float32)
        prediction, ((first_input, first_output, _), (second_input, second_output, _)) = model.forward(x)
        raw = x * np.array([[3, -4]], np.float32)
        normalized = raw / np.sqrt((raw**2).mean(axis=1, keepdims=True) + 1e-6)
        np.testing.assert_array_equal(first_input, with_bias(x))
        np.testing.assert_array_equal(first_output, raw)
        np.testing.assert_allclose(second_input, with_bias(normalized))
        np.testing.assert_allclose(prediction, (model.pool(raw) + model.pool(second_output)) / 2)
        np.testing.assert_allclose(prediction, np.vstack([model.predict_values(example[None]) for example in x]))

    def test_plain_cosine_starts_at_the_peak_and_decay_begins_immediately(self):
        data = SimpleNamespace(x=np.zeros((1, 1)), y=np.zeros((1, 1)))
        rates = []
        model = SimpleNamespace(step=lambda x, y, rate: rates.append(rate))
        settings = dict(seed=0, batch=1, rate=1., epochs=10)
        list(training_epochs(model, data, SimpleNamespace(**settings, schedule="cosine"),
                             np.array([0]), 10))
        np.testing.assert_allclose(rates, (1 + np.cos(np.pi * np.arange(10) / 10)) / 2)
        rates.clear()
        list(training_epochs(model, data, SimpleNamespace(**settings), np.array([0]), 10))
        self.assertEqual(rates[:3], [1., 1., 1.])
        self.assertLess(rates[3], 1.)

    def test_signsgd_uses_current_correlation_and_has_no_history(self):
        cfg = ModelConfig(width=1, depth=1, gate="none", hidden_rule="signsgd")
        model = NumpyModel(cfg, 1)
        model.weights[0][:] = [[1, 0], [1, 0]]
        x = np.ones((2, 1), dtype=np.float32)
        rate = .125
        model.step(x, model.predict_values(x) + 1, rate)
        first = (1 + rate / np.sqrt(2)) / (1 + rate * 4)
        np.testing.assert_allclose(model.weights[0][:, 0], first)
        # Nonempty, cancelling observations yield a zero signed direction but positive Q.
        pred = model.predict_values(x)
        model.step(x, pred + [[1], [-1]], rate)
        np.testing.assert_allclose(model.weights[0][:, 0], first / (1 + rate * (2 * first)**2))
        np.testing.assert_array_equal(model.weights[0][:, 1], 0)
        self.assertFalse(any(k.startswith("history_") for k in model.state()))
        restored = NumpyModel(cfg, 9)
        restored.load(model.state())
        for m in (model, restored):
            m.step(x, -np.ones_like(x), rate)
        for k, v in model.state().items():
            np.testing.assert_array_equal(restored.state()[k], v)

    def test_explicit_lion_uses_old_weight_for_decay_and_retains_coactivity_history(self):
        cfg = ModelConfig(width=1, depth=1, gate="none", hidden_rule="lion-explicit")
        model = NumpyModel(cfg, 1)
        model.weights[0][:] = [[1, 0], [1, 0]]
        x = np.ones((1, 1), np.float32)
        rate = .125
        model.step(x, model.predict_values(x) + 1, rate)
        expected = (1 - rate * 4) + rate / np.sqrt(2)
        np.testing.assert_allclose(model.weights[0][:, 0], expected)
        np.testing.assert_allclose(model.history[0][:, 0], .02)
        np.testing.assert_array_equal(model.weights[0][:, 1], 0)

    def test_lion_agreement_scales_the_whole_update_using_previous_history(self):
        cfg = ModelConfig(inputs=2, width=1, depth=1, gate="none", hidden_rule="lion-agreement")
        model = NumpyModel(cfg, 1)
        model.weights[0][:] = [[.2, 0], [.3, 0], [.5, 0]]
        model.history[0][:, 0] = [.5, -.5, 0.]
        restored = NumpyModel(cfg, 2)
        restored.load(model.state())
        x = np.ones((1, 2), np.float32)
        rate = .125
        rates = rate * np.array([1.5, .5, 1.])
        expected = (model.weights[0][:, 0] + rates / np.sqrt(3)) / (1 + rates)
        for current in (model, restored):
            current.step(x, current.predict_values(x) + 1, rate)
            np.testing.assert_allclose(current.weights[0][:, 0], expected)
            np.testing.assert_allclose(current.history[0][:, 0], [.505, -.485, .01])
            np.testing.assert_array_equal(current.weights[0][:, 1], 0.)
            self.assertEqual(current.take_mean_step_multipliers(), [1.])

    def test_lion_response_limits_each_neurons_change_before_competition(self):
        cfg = ModelConfig(width=1, depth=1, gate="none")
        ordinary = NumpyModel(cfg, 3)
        limited = NumpyModel(replace(cfg, hidden_rule="lion-response", response_limit_ratio=.02), 3)
        x = np.array([[-1], [.5], [1]], dtype=np.float32)
        rate = .1
        before = np.array([[.2, -.1], [.4, -.2]], dtype=np.float32)
        for model in (ordinary, limited):
            model.weights[0][:] = before
            model.step(x, model.predict_values(x) + 1, rate)
        incoming = with_bias(x)
        proposed = ordinary.weights[0] - before
        budget = (.02 * rate) * np.linalg.norm(incoming @ before, axis=0)
        factors = budget / np.linalg.norm(incoming @ proposed, axis=0)
        self.assertTrue((factors < 1).all())
        self.assertNotAlmostEqual(*factors)
        np.testing.assert_allclose(limited.weights[0], before + factors * proposed)
        np.testing.assert_array_equal(limited.history[0], ordinary.history[0])
        np.testing.assert_allclose(limited.take_mean_step_multipliers(), [factors.mean()], rtol=1e-6)

    def test_depth_scaled_step_matches_each_local_update(self):
        cfg = ModelConfig(depth=3, width=3, gate="gaussian", hidden_rule="signsgd")
        x = np.array([[.25], [-.5]], dtype=np.float32)
        y = np.ones_like(x)
        scaled = NumpyModel(replace(cfg, layer_rate_decay=.5), 3)
        scaled.step(x, y, .01)
        for layer, multiplier in enumerate((.25, .5, 1.)):
            uniform = NumpyModel(cfg, 3)
            uniform.step(x, y, .01 * multiplier)
            np.testing.assert_array_equal(scaled.weights[layer], uniform.weights[layer])

    def test_outputs_are_parameter_free_and_use_only_their_class_pair(self):
        post = np.array([[-3, 1, 2, -4, 5, -1, -3, 1]], dtype=np.float32)
        for pool, expected in (("fixed", [-1, 1]), ("class-sum", [-2, 2]), ("sum", [-4, 2]),
                               ("class-energy", [-5, 8])):
            config = ModelConfig(outputs=2, width=2, gate="none", readout=pool, hidden_rule="signsgd")
            model = NumpyModel(config, 1)
            np.testing.assert_array_equal(model.pool(post), [expected])
            self.assertEqual(set(model.state()), {"weight_0", "weight_1"})
            with self.assertRaises(ValueError):
                model.load({**model.state(), "output": np.ones((5, 2), dtype=np.float32)})

    def test_class_energy_sums_normalized_layer_scores(self):
        config = ModelConfig(width=2, depth=2, gate="none", readout_layers="all",
                             readout="class-energy", hidden_rule="signsgd")
        model = NumpyModel(config, 2)
        for weights in model.weights:
            weights.fill(0)
        model.weights[0][-1] = [-3, 1, 0, -2]  # (9 + 1 - 4) / 2 = 3
        model.weights[1][-1] = [0, 0, 1, 3]   # -(1 + 9) / 2 = -5
        np.testing.assert_array_equal(model.predict_values([[0.]]), [[-2.]])

    def test_output_bound_applies_to_combined_prediction_and_residual(self):
        config = ModelConfig(width=1, depth=2, gate="none", readout_layers="all",
                             readout="class-sum", hidden_rule="signsgd", output_bound=1.)
        model = NumpyModel(config, 1)
        x = np.zeros((1, 1), dtype=np.float32)
        for weights in model.weights:
            weights[:] = 0
        model.weights[0][-1] = [3, 0]
        model.weights[1][-1] = [0, 2]
        np.testing.assert_array_equal(model.predict_values(x), [[.5]])
        model.weights[0][-1] = [5, 0]
        np.testing.assert_array_equal(model.predict_values(x), [[1.]])
        before = model.state()
        model.step(x, np.ones_like(x), .01)
        for name, weights in before.items():
            np.testing.assert_array_equal(model.state()[name], weights)

    def test_matching_pixel_codes_skip_their_rgb_channels(self):
        cfg = ModelConfig(outputs=3, width=1, depth=1, gate="none", readout="class-sum",
                          hidden_rule="signsgd", output_bound=1., output_levels=256)
        x = np.ones((1, 1), dtype=np.float32)
        middle = np.float32(128) / np.float32(127.5) - 1
        y = np.array([[-1, middle, middle]], dtype=np.float32)
        for clip in (True, False):
            with self.subTest(clip=clip):
                model = NumpyModel(replace(cfg, clip_output=clip), 1)
                model.weights[0][:] = [[0] * 6, [0, 1.5, 1 + middle + .002, 1, 1 + middle + .01, 1]]
                before = model.weights[0].copy()
                prediction = model.predict_values(x)
                self.assertEqual(prediction[0, 1], y[0, 1])
                self.assertGreater(prediction[0, 2], y[0, 2])
                if clip:
                    self.assertEqual(prediction[0, 0], -1)
                else:
                    self.assertLess(prediction[0, 0], -1)
                model.step(x, y, .01)
                np.testing.assert_array_equal(model.weights[0][:, 2:4], before[:, 2:4])
                self.assertTrue(np.any(model.weights[0][:, 4:] != before[:, 4:]))
                if clip:
                    np.testing.assert_array_equal(model.weights[0][:, :2], before[:, :2])
                else:
                    self.assertGreater(model.predict_values(x)[0, 0], prediction[0, 0])

    def test_only_residual_sign_enters_training(self):
        x = np.array([[-.5], [.2], [.8]], dtype=np.float32)
        signs = np.array([[1, -1], [-1, 1], [1, 0]], dtype=np.float32)
        for rule in ("lion", "signsgd", "activity-sign"):
            for pool in ("fixed", "class-sum", "sum"):
                config = ModelConfig(outputs=2, depth=2, gate="gaussian", readout=pool,
                                     hidden_rule=rule, connection="discontinuous")
                a, b = NumpyModel(config, 11), NumpyModel(config, 11)
                initial = a.state()
                y = a.predict_values(x)
                a.step(x, y + signs, .001)
                b.step(x, y + 17 * signs, .001)
                for key, value in a.state().items():
                    np.testing.assert_array_equal(value, b.state()[key])
                for i, weights in enumerate(a.weights):
                    self.assertTrue(np.any(weights != initial[f"weight_{i}"]))

    def test_activity_scaling_respects_local_postsynaptic_power(self):
        cfg = ModelConfig(width=1, depth=1, gate="none", hidden_rule="activity-sign")
        model = NumpyModel(cfg, 1)
        model.weights[0][:] = [[0, 0], [.01, .5]]
        x = np.ones((1, 1), dtype=np.float32)
        rate = .1
        biases = model.weights[0][-1].copy()
        power = biases ** 2
        direction = np.array([1, -1]) * np.sqrt(power / (1 + power)) / np.sqrt(2)
        model.step(x, model.predict_values(x) + 1, rate)
        expected = (np.array([[0, 0], biases]) + rate * direction) / (1 + rate * power)
        np.testing.assert_allclose(model.weights[0], expected, atol=1e-8)
        self.assertEqual(set(model.state()), {"weight_0"})

    def test_two_piece_interpolation(self):
        x = np.array([[-2, -1, -.001, 0, .001, 1, 2]], dtype=np.float32).T
        for kind, endpoints, expected in (
            ("discontinuous", [-1, 1, -2, 0, 0], [-3, -1, .998, -2, -1.998, 0, 2]),
            ("continuous", [-1, 1, 0, 0], [-3, -1, .998, 1, .999, 0, -1]),
        ):
            actual = connection_inputs(x, kind) @ np.array(endpoints, np.float32)
            np.testing.assert_allclose(actual, expected, atol=1e-6)

    def test_affine_start_matches_hidden_functions(self):
        cfg = ModelConfig(inputs=2, outputs=2, width=3, depth=2, gate="none",
                          readout="fixed", initialization="affine-start")
        x = np.random.default_rng(31).normal(size=(17, 2)).astype(np.float32)
        expected, _ = NumpyModel(cfg, 4).forward(x)
        for connection in ("continuous", "discontinuous"):
            model = NumpyModel(replace(cfg, connection=connection), 4)
            actual, _ = model.forward(x)
            np.testing.assert_allclose(actual, expected, atol=1e-6)
            model.step(x, np.ones_like(expected), .001)
            for w in model.weights:
                endpoints = w[:-1].reshape(-1, model.config.basis_size, cfg.units)
                self.assertTrue(np.any(endpoints[:, 1] != 0))

    def test_two_piece_transition_matches_lines_and_bridges_the_jump(self):
        x = np.array([[-.3], [-.2], [-1e-6], [0], [1e-6], [.2], [.3]], dtype=np.float32)
        endpoints = np.array([1., 2., -3., 4., 0.], dtype=np.float32)
        sharp = connection_inputs(x, "discontinuous") @ endpoints
        smooth = connection_inputs(x, "discontinuous", .2) @ endpoints
        np.testing.assert_allclose(smooth[[0, 1, 5, 6]], sharp[[0, 1, 5, 6]], atol=1e-6)
        np.testing.assert_allclose(smooth[2:5], [-.5, -.5, -.5], atol=3e-5)
        np.testing.assert_allclose(connection_inputs(x, "discontinuous", .2)[:, :-1].sum(axis=1), 1, atol=2e-7)

    def test_piecewise_hebbian_updates_use_local_endpoint_coefficients(self):
        cfg = ModelConfig(width=1, depth=2, gate="none",
                          connection="discontinuous", readout_layers="all")
        model = NumpyModel(cfg, 12)
        x = np.array([[-.5], [0], [.25]], np.float32)
        y = np.array([[1], [-1], [.2]], np.float32)
        prediction, traces = model.forward(x)
        expected = []
        for incoming, post, _ in traces:
            expected.append(incoming.T @ (np.sign(y - prediction) * model.polarity * post) / len(x))
        model.step(x, y, .001)
        for product, history in zip(expected, model.history):
            np.testing.assert_allclose(history, .01 * product, atol=1e-8)

    def test_competition_ties_and_starvation(self):
        z = np.array([[-3, 3, 1, -2]], dtype=np.float32)
        signed, mask = select(z, ModelConfig(width=2, winners=1))
        np.testing.assert_array_equal(signed, [[-3, 0, 0, -2]])
        np.testing.assert_array_equal(mask, [[True, False, False, True]])
        signed, mask = select(z, ModelConfig(width=2, winners=1, competition="layer"))
        np.testing.assert_array_equal(signed, [[-3, 3, 0, 0]])
        np.testing.assert_array_equal(mask, [[True, True, False, False]])

    def test_scaled_gates_use_every_neuron(self):
        z = np.array([[-4, 2, 3, -1], [0, 0, 0, 0]], dtype=np.float32)
        config = ModelConfig(width=2, winners=1, competition="layer", gate_width=0.5)
        expected = {
            "gaussian": np.exp(-0.5 * np.array([0, 1, 0.5, 1.5]) ** 2),
            "tent": np.array([1, 0.01, 0.5, 0.01]),
        }
        for gate, scale in expected.items():
            with self.subTest(gate=gate):
                gated = replace(config, gate=gate)
                activity, mask = select(z, gated)
                np.testing.assert_allclose(activity, z * scale, rtol=1e-6)
                self.assertTrue(mask.all())
                np.testing.assert_array_equal(activity, select(z, replace(gated, winners=2))[0])
                by_class, _ = select(z, replace(gated, competition="class"))
                self.assertEqual(by_class[0, 2], z[0, 2])

    def test_output_competition_shares_peak_across_the_two_classes(self):
        config = ModelConfig(outputs=2, width=2, competition="output", gate="gaussian", gate_width=.5)
        preactivation = np.array([[4, 4, 2, 2, 200, 200, 100, 100]], dtype=np.float32)
        activity, mask = select(preactivation, config)
        scales = [1, 1, np.exp(-.5), np.exp(-.5)] * 2
        np.testing.assert_allclose(activity, preactivation * scales, rtol=1e-6)
        self.assertTrue(mask.all())

    def test_smooth_gaussian_uses_even_moment_center(self):
        config = ModelConfig(width=2, competition="layer", gate="gaussian-smooth", gate_width=.2)
        inputs = np.array([[1., -.96, .91, -.6], [0., 0., 0., 0.]], dtype=np.float32)
        activity, mask = select(inputs, config)
        values = inputs[0].astype(np.float64)
        center = np.sqrt(np.sum(values ** 16) / np.sum(values ** 14))
        expected = values * np.exp(-.5 * ((np.abs(values) - center) / (center * .2)) ** 2)
        np.testing.assert_allclose(activity[0], expected, rtol=1e-6)
        np.testing.assert_array_equal(activity[1], 0)
        self.assertTrue(mask.all())

    def test_two_gaussian_widths_split_each_class_evenly(self):
        config = ModelConfig(width=4, competition="layer", gate="gaussian-dual", gate_width=.2)
        z = np.array([[1., -.6, .6, -.6, -.6, .6, -.6, .6]], dtype=np.float32)
        activity, mask = select(z, config)
        widths = .2 * np.tile([.75, 1.25], 4)
        expected = z * np.exp(-.5 * ((1 - np.abs(z)) / widths) ** 2)
        np.testing.assert_allclose(activity, expected, rtol=1e-6)
        self.assertTrue(mask.all())
        with self.assertRaises(ValueError):
            replace(config, width=3)

    def test_sine_precedes_competition(self):
        z = np.array([[0.3, 1.5, -0.7, 1.2]], dtype=np.float32)
        for gate in ("none", "gaussian"):
            for pool in ("class", "layer"):
                cfg = ModelConfig(width=2, competition=pool, gate=gate, sine_omega=2)
                activity, mask = select(z, cfg)
                wave = np.sin(2 * z)
                expected = wave if gate == "none" else select(wave, replace(cfg, sine_omega=None))[0]
                np.testing.assert_allclose(activity, expected)
                self.assertTrue(mask.all())
        with self.assertRaises(ValueError):
            ModelConfig(sine_omega=float("nan"))

    def test_scaled_gates_are_continuous_at_winner_swaps(self):
        x = np.array([[-0.00001], [0], [0.00001]], dtype=np.float32)
        config = ModelConfig(width=2, winners=1, depth=1)
        for gate in ("hard", "gaussian", "tent"):
            with self.subTest(gate=gate):
                model = NumpyModel(replace(config, gate=gate), 1)
                model.weights[0][:] = [[1, -1, 0, 0], [1, 1, 0, 0]]
                _, traces = model.forward(x)
                jump = np.max(np.abs(np.diff(traces[0][1][:, 0])))
                if gate == "hard":
                    self.assertGreater(jump, 0.9)
                else:
                    self.assertLess(jump, 0.001)

    def test_layer_gate_widths_and_checkpoint(self):
        config = ModelConfig(width=3, gate="gaussian", layer_gate_widths=(0.2, 0.45))
        model = NumpyModel(config, 5)
        x = np.array([[-0.25], [0.65]], dtype=np.float32)
        y = np.array([[0.7], [-0.1]], dtype=np.float32)
        prediction, traces = model.forward(x)
        incoming = x
        for weights, width, (_, actual, _) in zip(model.weights, config.gate_widths, traces):
            expected, _ = select(with_bias(incoming) @ weights, replace(config, gate_width=width))
            np.testing.assert_array_equal(actual, expected)
            incoming = expected
        model.step(x, y, 0.0001)

        restored = NumpyModel(config, 8)
        restored.load(model.state())
        for instance in (restored, model):
            instance.step(x, y, 0.0001)
        for name, value in model.state().items():
            np.testing.assert_array_equal(restored.state()[name], value)

    def test_semi_implicit_update(self):
        model = NumpyModel(ModelConfig(width=1, winners=1, depth=1), 4)
        model.weights[0][:] = [[2, -1], [0, 0]]
        x = np.array([[1]], dtype=np.float32)
        y = np.array([[2]], dtype=np.float32)
        rate = 0.1
        model.step(x, y, rate)
        delta = rate / np.sqrt(2)
        expected = np.array([[2 + delta, -1 + delta], [delta, delta]]) / [1.4, 1.1]
        np.testing.assert_allclose(model.weights[0], expected, rtol=1e-6)
        np.testing.assert_allclose(model.history[0], [[0.02, 0.01], [0.02, 0.01]], rtol=1e-6)
        initial = model.state()
        model.step(x, model.predict(x)[0], rate)
        for key, value in model.state().items():
            np.testing.assert_array_equal(value, initial[key])

    def test_all_layers_share_one_residual(self):
        config = ModelConfig(width=4, depth=3, gate="gaussian",
                             readout_layers="all")
        x = np.array([[-0.7], [0.2], [0.9]], dtype=np.float32)
        y = np.array([[0.4], [-0.8], [0.3]], dtype=np.float32)
        model = NumpyModel(config, 9)
        before = model.state()
        prediction, traces = model.forward(x)
        heads = [np.abs(post[:, :4]).mean(1) - np.abs(post[:, 4:]).mean(1) for _, post, _ in traces]
        expected = np.mean(heads, axis=0)[:, None]
        np.testing.assert_allclose(prediction, expected)
        model.step(x, y, 0.001)
        for layer, (incoming, post, _) in enumerate(traces):
            signs = np.sign(y - expected) * model.polarity
            count = np.maximum(np.count_nonzero(signs * post, axis=0), 1)
            np.testing.assert_allclose(model.history[layer], .01 * (incoming.T @ (signs * post)) / count,
                                       atol=1e-7)
        for layer, weights in enumerate(model.weights):
            self.assertFalse(np.array_equal(weights, before[f"weight_{layer}"]))

    def test_unit_norm_updates_preserve_zero_residual_and_train_every_layer(self):
        config = ModelConfig(width=6, depth=2, gate="gaussian",
                             initialization="unit-affine", unit_norm_updates=True)
        model = NumpyModel(config, 3)
        x = np.array([[-0.7], [0.2]], dtype=np.float32)
        before = model.state()
        model.step(x, model.predict(x)[0], 0.01)
        for key, value in before.items():
            np.testing.assert_array_equal(value, model.state()[key])
        model.step(x, np.ones_like(x), 0.01)
        for i, weight in enumerate(model.weights):
            self.assertFalse(np.array_equal(weight, before[f"weight_{i}"]))
            np.testing.assert_allclose(np.linalg.norm(weight, axis=0), 1, atol=2e-7)

    def test_rgb_cross_links_and_conflicting_targets(self):
        config = ModelConfig(inputs=2, outputs=3, width=1, winners=1)
        model = NumpyModel(config, 4)
        model.weights[0][:-1] = 0
        model.weights[0][-1] = 1
        model.weights[1][:-1] = np.eye(config.units, dtype=np.float32)
        model.weights[1][-1] = 0
        x = np.array([[0.1, 0.1]], dtype=np.float32)
        before = model.predict(x)[0]
        model.weights[1][0, 4] += 0.5  # +Red source to +Blue receiver
        after = model.predict(x)[0]
        self.assertGreater(after[0, 2], before[0, 2])
        self.assertEqual(after[0, 0], before[0, 0])
        model.weights[0][-1, 0] += 0.1
        changed_source = model.predict(x)[0]
        self.assertGreater(changed_source[0, 2], after[0, 2])
        after = changed_source
        model.step(x, after + [[0.5, 0, -0.5]], 0.00075)
        corrected = model.predict(x)[0]
        self.assertGreater(corrected[0, 0], after[0, 0])
        self.assertLess(corrected[0, 2], after[0, 2])

    def test_staircase_rmse_for_large_finite_errors(self):
        data = scalar_data("steps")
        x = data.x[data.test]
        metrics = step_metrics(x, np.full_like(x, 1e21))
        np.testing.assert_allclose(metrics["smooth_rmse"], 1e21)

    def test_the_steps_test_grid_resolves_every_plateau(self):
        data = scalar_data("steps")
        for split in (data.train, data.validation, data.test):
            x = data.x[split, 0]
            for start, end in zip(EDGES[:-1], EDGES[1:]):
                quarter = (end - start) / 4
                self.assertGreaterEqual(np.count_nonzero(
                    (x >= start + quarter) & (x < end - quarter)), 3)


@unittest.skipUnless(os.getenv("HEBBIAN_TEST_GPU") == "1", "requires a Mojo GPU")
class GpuTests(unittest.TestCase):
    def test_dense_parity(self):
        from hebbian_fit.gpu import MojoModel

        for outputs in (1, 3):
            for competition in ("class", "output", "layer"):
                for readout in ("fixed", "class-sum", "sum"):
                    for gate in ("hard", "gaussian", "tent", "gaussian-smooth", "gaussian-dual"):
                        config = ModelConfig(2, outputs, 4, 2, 2, competition, readout, gate)
                        with self.subTest(config=config):
                            self.check_parity(NumpyModel(config, 17), MojoModel(config, 17))

    def test_dense_ties_and_zero_residual(self):
        from hebbian_fit.gpu import MojoModel

        config = ModelConfig(width=2, winners=1, depth=1, competition="layer")
        reference = NumpyModel(config, 17)
        reference.weights[0][:] = [[0, 0, 0, 0], [-3, 3, 1, -2]]
        gpu = MojoModel(config, 17)
        initial = reference.state()
        gpu.load(initial)
        x = np.array([[0]], dtype=np.float32)
        expected, masks = reference.predict(x)
        actual, gpu_masks = gpu.predict(x)
        np.testing.assert_array_equal(actual, expected)
        np.testing.assert_array_equal(gpu_masks, masks)
        gpu.step(x, expected, 0.1)
        for key, value in gpu.state().items():
            np.testing.assert_array_equal(value, initial[key])

    def test_layer_gate_widths_parity(self):
        from hebbian_fit.gpu import MojoModel

        for competition in ("class", "output", "layer"):
            config = ModelConfig(inputs=2, outputs=3, width=4, depth=3,
                                 competition=competition, gate="gaussian",
                                 layer_gate_widths=(0.35, 0.15, 0.4), layer_rate_decay=.7)
            self.check_parity(NumpyModel(config, 17), MojoModel(config, 17))

    def test_all_layer_readouts(self):
        from hebbian_fit.gpu import MojoModel
        for readout in ("fixed", "class-sum", "sum", "class-energy"):
            config = ModelConfig(inputs=2, outputs=3, width=4, depth=3, gate="gaussian",
                                 readout=readout, readout_layers="all")
            self.check_parity(NumpyModel(config, 17), MojoModel(config, 17))
        config = replace(config, initialization="unit-affine", unit_norm_updates=True)
        self.check_parity(NumpyModel(config, 17), MojoModel(config, 17))

    def test_wide_gaussian_pool_maxima(self):
        from hebbian_fit.gpu import MojoModel
        for pool in ("class", "layer"):
            config = ModelConfig(width=161, depth=1, competition=pool, gate="gaussian",
                                 initialization="unit-sphere")
            ref = NumpyModel(config, 2)
            gpu = MojoModel(config, 2)
            gpu.load(ref.state())
            x = np.linspace(-1, 1, 19, dtype=np.float32)[:, None]
            np.testing.assert_allclose(gpu.predict_values(x), ref.predict_values(x), atol=1e-6)

    def test_signsgd_gpu_parity_and_restore(self):
        from hebbian_fit.gpu import MojoModel
        cfg = ModelConfig(width=3, inputs=2, outputs=2, depth=2, gate="gaussian",
                          connection="discontinuous", hidden_rule="signsgd")
        ref, gpu = NumpyModel(cfg, 17), MojoModel(cfg, 17)
        self.check_parity(ref, gpu)
        self.assertEqual(set(gpu.state()), set(ref.state()))
        restored = MojoModel(cfg, 3)
        restored.load(gpu.state())
        x = np.array([[.2, -.8]], dtype=np.float32)
        for m in (gpu, restored):
            m.step(x, np.ones_like(x), .001)
        for key, value in gpu.state().items():
            np.testing.assert_array_equal(value, restored.state()[key])

    def test_bounded_output_and_training_match_numpy(self):
        from hebbian_fit.gpu import MojoModel
        cfg = ModelConfig(width=3, inputs=2, outputs=2, depth=2, gate="gaussian",
                          connection="discontinuous", hidden_rule="signsgd",
                          readout_layers="all", output_bound=.35)
        self.check_parity(NumpyModel(cfg, 17), MojoModel(cfg, 17))

    def test_discrete_output_and_training_match_numpy(self):
        from hebbian_fit.gpu import MojoModel
        cfg = ModelConfig(width=3, inputs=2, outputs=3, depth=2, gate="gaussian",
                          connection="discontinuous", hidden_rule="signsgd",
                          readout_layers="all", unit_norm_updates=True,
                          output_bound=1., output_levels=256)
        for clip in (True, False):
            with self.subTest(clip=clip):
                config = replace(cfg, clip_output=clip)
                self.check_parity(NumpyModel(config, 17), MojoModel(config, 17))

    def test_activity_scaled_updates_match_numpy(self):
        from hebbian_fit.gpu import MojoModel
        cfg = ModelConfig(width=3, inputs=2, outputs=2, depth=2, gate="gaussian",
                          connection="discontinuous", hidden_rule="activity-sign",
                          readout_layers="all")
        self.check_parity(NumpyModel(cfg, 17), MojoModel(cfg, 17))

    def test_soft_edge_transition_matches_numpy(self):
        from hebbian_fit.gpu import MojoModel
        cfg = ModelConfig(width=4, inputs=2, outputs=3, depth=2, gate="gaussian",
                          connection="discontinuous", hidden_rule="lion", competition="layer",
                          readout_layers="all", output_bound=1., output_levels=256,
                          edge_transitions=(.08, .1))
        self.check_parity(NumpyModel(cfg, 17), MojoModel(cfg, 17))

    def test_piecewise_parity(self):
        from hebbian_fit.gpu import MojoModel
        for kind in ("continuous", "discontinuous"):
            for readout in ("fixed", "class-sum", "sum"):
                cfg = ModelConfig(width=3, inputs=2, outputs=2, depth=2, gate="gaussian",
                                  readout=readout, readout_layers="all", connection=kind)
                self.check_parity(NumpyModel(cfg, 17), MojoModel(cfg, 17))

    def test_sine_training_parity(self):
        from hebbian_fit.gpu import MojoModel
        for gate in ("none", "gaussian"):
            for pool in ("class", "layer"):
                cfg = ModelConfig(width=5, inputs=2, depth=2, gate=gate, competition=pool,
                                  sine_omega=2, initialization="unit-sphere",
                                  unit_norm_updates=True)
                self.check_parity(NumpyModel(cfg, 17), MojoModel(cfg, 17))

    def check_parity(self, reference, gpu):
        config = reference.config
        rng = np.random.default_rng(123)
        x = rng.uniform(-1, 1, (17, config.inputs)).astype(np.float32)
        x[0] = -1
        x[1] = 1
        y = rng.normal(size=(len(x), config.outputs)).astype(np.float32)
        for _ in range(5):
            before, masks = reference.predict(x)
            gpu_before, gpu_masks = gpu.predict(x)
            # Unnormalized sums accumulate absolute rounding error across their terms.
            atol = 1e-6 if config.readout == "fixed" else config.width * 1e-6
            np.testing.assert_allclose(gpu_before, before, atol=atol)
            np.testing.assert_array_equal(gpu_masks, masks)
            reference.step(x, y, 0.00075)
            gpu.step(x, y, 0.00075)
            state = gpu.state()
            for key, value in reference.state().items():
                np.testing.assert_allclose(state[key], value, atol=5e-7, err_msg=key)


if __name__ == "__main__":
    unittest.main()
