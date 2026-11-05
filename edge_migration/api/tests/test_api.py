from __future__ import annotations

import unittest

import pandas as pd
from fastapi.testclient import TestClient

from api.main import Overrides, _request_config, app, get_cached_case
from sim.policies import run_policy


class SimulatorApiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.client_context = TestClient(app)
        cls.client = cls.client_context.__enter__()

    @classmethod
    def tearDownClass(cls) -> None:
        cls.client_context.__exit__(None, None, None)

    def test_health(self) -> None:
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["status"], "ok")

    def test_performance_matches_saved_csv(self) -> None:
        response = self.client.get("/results/performance?scenario=default")
        expected = pd.read_csv("results/default/migration_performance.csv").to_dict(orient="records")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), expected)

    def test_simulation_rejects_zero_score_weights(self) -> None:
        response = self.client.post("/simulate", json={
            "scenario": "default", "seeds": [7], "policies": ["FCM"],
            "overrides": {"alpha": 0, "beta": 0, "gamma": 0},
        })
        self.assertEqual(response.status_code, 422)
        self.assertIn("At least one", response.json()["detail"])

    def test_simulation_adds_real_nodes_without_mutating_config(self) -> None:
        response = self.client.post("/simulate", json={
            "scenario": "default", "seeds": [7], "policies": ["RT"],
            "overrides": {"node_count": 11},
        })
        self.assertEqual(response.status_code, 200, response.text)
        payload = response.json()
        self.assertEqual(payload["timeline"]["node_count"], 11)
        self.assertIn("node_10", payload["timeline"]["policies"]["RT"]["points"][0])
        self.assertEqual(self.client.get("/config").json()["node_count"], 10)

    def test_simulation_rejects_node_count_outside_configured_range(self) -> None:
        response = self.client.post("/simulate", json={
            "scenario": "default", "seeds": [7], "policies": ["RT"],
            "overrides": {"node_count": 21},
        })
        self.assertEqual(response.status_code, 422)
        self.assertIn("between 2 and 20", response.json()["detail"])

    def test_simulation_matches_direct_policy_call(self) -> None:
        payload = {"scenario": "default", "seeds": [7], "policies": ["RT"]}
        response = self.client.post("/simulate", json=payload)
        self.assertEqual(response.status_code, 200, response.text)
        traces, forecasts, start = get_cached_case("default", 7)
        config, fcm_overrides = _request_config("default", Overrides())
        expected, _ = run_policy(config, "RT", traces, forecasts, start,
                                 fcm_overrides=fcm_overrides, seed=7)
        actual = response.json()["per_seed"][0]
        self.assertEqual(actual["Policy"], expected["Policy"])
        for key, value in expected.items():
            if key == "Decision time (ms/slot)":
                continue  # Wall-clock timing is inherently different between two calls.
            if isinstance(value, float):
                self.assertAlmostEqual(actual[key], value, places=7)
            else:
                self.assertEqual(actual[key], value)


if __name__ == "__main__":
    unittest.main()
