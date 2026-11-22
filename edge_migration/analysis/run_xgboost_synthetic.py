"""Run FLL and FCM using XGBoost as the policy forecaster on synthetic scenarios."""
from __future__ import annotations

import copy
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parents[1]))

import numpy as np
import pandas as pd
import yaml

from sim.run_experiments import build_forecasts, _performance_table
from sim.policies import run_policy
from forecasting.models import make_model
from forecasting.train import make_features, windows


def run_xgboost_synthetic(config_path: Path) -> dict[str, pd.DataFrame]:
    with open(config_path, encoding="utf-8") as f:
        config = yaml.safe_load(f)

    results_dir = config_path.parent / config["paths"]["results"]
    traces_dir = config_path.parent / config["paths"]["traces"]
    data_cfg, forecast_cfg = config["data"], config["forecast"]

    # Load seed 7 training trace to fit XGBoost
    train_trace_file = traces_dir / f"traces_seed_{int(config['seed'])}.npz"
    train_traces = np.load(train_trace_file)["traces"]
    total = train_traces.shape[1]
    train_end = int(total * float(data_cfg["train_fraction"]))
    val_end = int(total * (float(data_cfg["train_fraction"]) + float(data_cfg["validation_fraction"])))
    window, horizon = int(forecast_cfg["window"]), int(forecast_cfg["horizon"])
    stride = int(forecast_cfg["training_stride"])
    features = make_features(train_traces, int(data_cfg["slots_per_day"]))
    train_x, train_y = windows(features, train_traces, 0, train_end, window, horizon, stride)

    print("Fitting XGBoost policy model...")
    xgb_model = make_model("XGBoost", forecast_cfg, int(config["seed"]))
    xgb_model.fit(train_x, train_y)

    seeds = [int(s) for s in config["simulation"]["seeds"]]
    start = val_end
    stop = total - horizon

    # Precompute forecasts per seed
    forecasts_by_seed = {}
    traces_by_seed = {}
    for seed in seeds:
        t_file = traces_dir / f"traces_seed_{seed}.npz"
        tr = np.load(t_file)["traces"]
        traces_by_seed[seed] = tr
        forecasts_by_seed[seed] = build_forecasts(xgb_model, tr, config, start, stop)

    results = {}

    for scenario_name, scenario_vals in config.get("scenarios", {"default": {}}).items():
        scenario_cfg = copy.deepcopy(config)
        scenario_cfg["simulation"]["network"]["bandwidth_mbps"] = scenario_vals.get(
            "bandwidth_mbps", scenario_cfg["simulation"]["network"]["bandwidth_mbps"])
        scenario_cfg["simulation"]["services"]["state_size_mb"] = scenario_vals.get(
            "state_size_mb", scenario_cfg["simulation"]["services"]["state_size_mb"])

        rows = []
        for seed in seeds:
            for policy in ("RT", "RT-CD", "FLL", "FCM"):
                metrics, _ = run_policy(scenario_cfg, policy, traces_by_seed[seed],
                                        forecasts_by_seed[seed], start, seed=seed)
                rows.append({"Seed": seed, "Scenario": scenario_name, "PolicyModel": "XGBoost", **metrics})

        raw_df = pd.DataFrame(rows)
        perf_df = _performance_table(raw_df)
        results[scenario_name] = perf_df

        perf_df.to_csv(results_dir / f"synthetic_{scenario_name}_xgboost.csv", index=False)
        raw_df.to_csv(results_dir / f"synthetic_{scenario_name}_xgboost_by_seed.csv", index=False)

    return results


if __name__ == "__main__":
    cfg_p = Path(__file__).parents[1] / "config.yaml"
    res = run_xgboost_synthetic(cfg_p)
    for sc, df in res.items():
        print(f"\n=== Synthetic Scenario: {sc} (Policy Model: XGBoost) ===")
        cols = ["Policy", "Overload time (%)", "SLA violation (%)", "Migrations",
                "Total migration cost (s)", "Avg latency (ms)"]
        print(df[cols].to_string(index=False))
