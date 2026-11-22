"""Run 5-seed evaluation on real Google ClusterData trace across scenarios and policy models."""
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


def evaluate_real_trace_all(config_path: Path, policy_model_name: str = "LSTM") -> dict[str, pd.DataFrame]:
    with open(config_path, encoding="utf-8") as f:
        config = yaml.safe_load(f)

    data_dir = config_path.parent / config["paths"]["traces"]
    results_dir = config_path.parent / config["paths"]["results"]
    npz_path = data_dir / "traces_real_google.npz"

    traces = np.load(npz_path)["traces"]
    total_slots = traces.shape[1]
    train_end = int(total_slots * float(config["data"]["train_fraction"]))
    val_end = int(total_slots * (float(config["data"]["train_fraction"]) + float(config["data"]["validation_fraction"])))
    horizon = int(config["forecast"]["horizon"])
    window = int(config["forecast"]["window"])
    stride = int(config["forecast"]["training_stride"])
    start = val_end
    stop = total_slots - horizon

    features = make_features(traces, int(config["data"]["slots_per_day"]))
    train_x, train_y = windows(features, traces, 0, train_end, window, horizon, stride)
    val_x, val_y = windows(features, traces, train_end, val_end, window, horizon, stride)

    # Train model on real trace
    print(f"Training {policy_model_name} on real Google trace...")
    if policy_model_name in {"GRU", "LSTM"}:
        model = make_model(policy_model_name, config["forecast"], int(config["seed"]))
        model.fit(train_x, train_y, (val_x, val_y))
    else:
        model = make_model(policy_model_name, config["forecast"], int(config["seed"]))
        model.fit(train_x, train_y)

    forecasts = build_forecasts(model, traces, config, start, stop)
    seeds = [int(s) for s in config["simulation"]["seeds"]] # [7, 17, 27, 37, 47]

    results_by_scenario = {}

    for scenario_name, scenario_vals in config.get("scenarios", {"default": {}}).items():
        scenario_cfg = copy.deepcopy(config)
        scenario_cfg["simulation"]["network"]["bandwidth_mbps"] = scenario_vals.get(
            "bandwidth_mbps", scenario_cfg["simulation"]["network"]["bandwidth_mbps"])
        scenario_cfg["simulation"]["services"]["state_size_mb"] = scenario_vals.get(
            "state_size_mb", scenario_cfg["simulation"]["services"]["state_size_mb"])

        rows = []
        for seed in seeds:
            for policy in ("RT", "RT-CD", "FLL", "FCM"):
                metrics, _ = run_policy(scenario_cfg, policy, traces, forecasts, start, seed=seed)
                rows.append({"Seed": seed, "Scenario": scenario_name, "PolicyModel": policy_model_name, **metrics})

        raw_df = pd.DataFrame(rows)
        perf_df = _performance_table(raw_df)
        results_by_scenario[scenario_name] = perf_df

        out_name = f"real_trace_{scenario_name}_{policy_model_name.lower()}.csv"
        perf_df.to_csv(results_dir / out_name, index=False)
        raw_df.to_csv(results_dir / f"real_trace_{scenario_name}_{policy_model_name.lower()}_by_seed.csv", index=False)

    return results_by_scenario


if __name__ == "__main__":
    cfg_p = Path(__file__).parents[1] / "config.yaml"
    for m in ["LSTM", "XGBoost"]:
        res = evaluate_real_trace_all(cfg_p, policy_model_name=m)
        for sc, df in res.items():
            print(f"\n=== Real Google Trace: {sc} (Policy Model: {m}) ===")
            cols = ["Policy", "Overload time (%)", "SLA violation (%)", "Migrations",
                    "Total migration cost (s)", "Avg latency (ms)", "Unnecessary migrations (%)"]
            print(df[cols].to_string(index=False))
