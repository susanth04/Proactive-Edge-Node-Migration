"""Small runnable checks for forecast shapes and migration invariants."""
from __future__ import annotations

import copy
import sys
import unittest
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT))

from forecasting.models import make_model
from sim.environment import Environment
from sim.policies import _would_overload_without_move, run_policy


class BasicTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))

    def test_forecasters_share_prediction_shape(self):
        x = np.random.default_rng(1).random((8, 30, 6), dtype=np.float32)
        y = np.random.default_rng(2).random((8, 3, 4), dtype=np.float32)
        forecast_cfg = {**self.config["forecast"], "horizon": 3, "epochs": 1,
                        "patience": 1, "batch_size": 8, "hidden_size": 8, "layers": 1}
        for name in ("MovingAverage", "ARIMA", "GRU", "LSTM"):
            with self.subTest(model=name):
                model = make_model(name, forecast_cfg, 3)
                if name in {"GRU", "LSTM"}:
                    model.fit(x, y, (x, y))
                else:
                    model.fit(x, y)
                prediction = model.predict(x)
                self.assertEqual(prediction.shape, (8, 3, 4))
                if name == "ARIMA":
                    history = x[0, :, :4]
                    expected = history.mean(axis=0) + (
                        model.phi[None, :] ** np.arange(1, 4)[:, None]
                    ) * (history[-1] - history.mean(axis=0))
                    np.testing.assert_allclose(prediction[0], np.clip(expected, 0, 1), rtol=1e-5)

    def test_migration_moves_once_and_respects_memory(self):
        config = copy.deepcopy(self.config)
        config["data"].update(nodes=2, services=1)
        traces = np.zeros((3, 4, 4), dtype=np.float32)
        env = Environment(config, traces, 4, initial_hosts=[0])
        migration = env.start_migration(0, 1, 0)
        self.assertEqual(env.hosts, [0])
        self.assertEqual(env.complete_migrations(migration.complete_slot), [migration])
        self.assertEqual(env.hosts, [1])
        self.assertEqual(env.complete_migrations(migration.complete_slot + 1), [])
        self.assertTrue(np.all(env.used_memory + env.reserved_memory <=
                               np.array([node.memory_capacity_mb for node in env.nodes])))

    def test_rt_only_migrates_after_actual_overload(self):
        config = copy.deepcopy(self.config)
        config["data"].update(nodes=2, services=1)
        traces = np.zeros((3, 2, 4), dtype=np.float32)
        traces[:, 1, :] = 1.0
        forecasts = np.zeros((2, 3, 1, 4), dtype=np.float32)
        _, timeline = run_policy(config, "RT", traces, forecasts, 0, seed=2, keep_timeline=True)
        self.assertEqual(timeline["migration_slots"], [1])

    def test_fcm_waits_for_migration_lead_time(self):
        config = copy.deepcopy(self.config)
        config["data"].update(nodes=2, services=1)
        config["simulation"]["services"]["state_size_mb"] = [100, 100]
        config["simulation"]["network"]["bandwidth_mbps"] = [1, 1]
        traces = np.full((3, 2, 4), 0.1, dtype=np.float32)
        seed, horizon = 9, 17
        env = Environment(config, traces, seed)
        source = env.hosts[0]
        forecasts = np.full((1, 3, horizon, 4), 0.1, dtype=np.float32)
        forecasts[0, 2] = 0.3
        forecasts[0, source, :3] = 0.6
        _, early = run_policy(config, "FCM", traces, forecasts, 0, seed=seed, keep_timeline=True)
        self.assertEqual(early["migration_slots"], [])
        forecasts[0, source, :3] = 0.1
        forecasts[0, source, 13:16] = 0.6
        _, timely = run_policy(config, "FCM", traces, forecasts, 0, seed=seed, keep_timeline=True)
        self.assertEqual(timely["migration_slots"], [0])

    def test_rt_service_selection_uses_actual_load(self):
        config = copy.deepcopy(self.config)
        config["data"].update(nodes=2, services=2)
        seed = next(seed for seed in range(100)
                    if len(set(Environment(config, np.zeros((4, 2, 4)), seed).hosts)) == 1)
        env = Environment(config, np.zeros((4, 2, 4)), seed)
        source = env.hosts[0]
        traces = np.zeros((4, 2, 4), dtype=np.float32)
        traces[2, :, :] = 0.9
        traces[3, :, :] = 0.01
        traces[source, :, :] = 0.02
        predictions = np.zeros((1, 4, 1, 4), dtype=np.float32)
        predictions[0, 2, 0, :] = 0.01
        predictions[0, 3, 0, :] = 0.9
        _, timeline = run_policy(config, "RT", traces, predictions, 0, seed=seed, keep_timeline=True)
        self.assertEqual([migration.service_id for migration in timeline["migrations"]], [0])

    def test_unnecessary_oracle_excludes_current_slot(self):
        config = copy.deepcopy(self.config)
        config["data"].update(nodes=2, services=1)
        traces = np.zeros((3, 3, 4), dtype=np.float32)
        traces[:, 0, :] = 1.0
        env = Environment(config, traces, 2, initial_hosts=[0])
        weights = np.asarray(config["simulation"]["weights"])
        self.assertFalse(_would_overload_without_move(env, 0, 0, traces, 2, 0.85, weights))
        traces[:, 1, :] = 1.0
        self.assertTrue(_would_overload_without_move(env, 0, 0, traces, 2, 0.85, weights))

    def test_slow_links_have_multislot_migrations(self):
        config = copy.deepcopy(self.config)
        config["data"].update(nodes=4, services=1)
        config["simulation"]["network"]["bandwidth_mbps"] = [8, 40]
        config["simulation"]["services"]["state_size_mb"] = [64, 256]
        traces = np.zeros((5, 2, 4), dtype=np.float32)
        env = Environment(config, traces, 14)
        self.assertTrue(any(env.migration_slots(0, i, j) > 1
                            for i in range(len(env.nodes)) for j in range(len(env.nodes)) if i != j))

    def test_rt_cd_respects_recent_migration_cooldown(self):
        config = copy.deepcopy(self.config)
        config["data"].update(nodes=2, services=1)
        traces = np.full((3, 4, 4), 0.02, dtype=np.float32)
        traces[2, :, :] = 0.9
        predictions = np.zeros((4, 3, 1, 4), dtype=np.float32)
        _, timeline = run_policy(config, "RT-CD", traces, predictions, 0, seed=3, keep_timeline=True)
        self.assertEqual(len(timeline["migrations"]), 1)


if __name__ == "__main__":
    unittest.main()
