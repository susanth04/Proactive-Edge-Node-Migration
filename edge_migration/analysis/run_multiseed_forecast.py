"""Train and evaluate all 5 forecasters across 5 distinct seeds for confidence intervals."""
from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parents[1]))

import numpy as np
import pandas as pd
import yaml
from scipy import stats

from data.generate_traces import generate_traces
from forecasting.models import make_model
from forecasting.train import make_features, windows, _composite
from sim.environment import Environment


def run_multiseed_forecast(config_path: Path, seeds: list[int] = [7, 17, 27, 37, 47]) -> pd.DataFrame:
    with open(config_path, encoding="utf-8") as f:
        config = yaml.safe_load(f)

    results_dir = config_path.parent / config["paths"]["results"]
    traces_dir = config_path.parent / config["paths"]["traces"]
    data_cfg, forecast_cfg = config["data"], config["forecast"]
    weights = np.asarray(config["simulation"]["weights"], dtype=np.float32)
    models = ["MovingAverage", "ARIMA", "XGBoost", "GRU", "LSTM"]

    results = []

    for seed in seeds:
        print(f"\n--- Training Seed {seed} ---")
        trace_file = traces_dir / f"traces_seed_{seed}.npz"
        if trace_file.exists():
            traces = np.load(trace_file)["traces"]
        else:
            traces, _ = generate_traces(config, seed)
            np.savez_compressed(trace_file, traces=traces)

        total = traces.shape[1]
        train_end = int(total * float(data_cfg["train_fraction"]))
        val_end = int(total * (float(data_cfg["train_fraction"]) + float(data_cfg["validation_fraction"])))
        window, horizon = int(forecast_cfg["window"]), int(forecast_cfg["horizon"])
        features = make_features(traces, int(data_cfg["slots_per_day"]))
        stride = int(forecast_cfg["training_stride"])

        train_x, train_y = windows(features, traces, 0, train_end, window, horizon, stride)
        val_x, val_y = windows(features, traces, train_end, val_end, window, horizon, stride)
        test_x, test_y = windows(features, traces, val_end, total, window, horizon, int(forecast_cfg["evaluation_step"]))

        env = Environment(config, traces, seed)
        n_series = traces.shape[0]
        samples = len(test_x) // n_series
        actual = _composite(test_y.reshape(n_series, samples, horizon, 4), len(env.nodes), env.hosts, weights)

        for name in models:
            m = make_model(name, forecast_cfg, seed)
            if name in {"GRU", "LSTM"}:
                m.fit(train_x, train_y, (val_x, val_y))
            else:
                m.fit(train_x, train_y)

            pred = m.predict(test_x)
            est = _composite(pred.reshape(n_series, samples, horizon, 4), len(env.nodes), env.hosts, weights)
            rmse = float(np.sqrt(np.mean((est - actual) ** 2)))
            mae = float(np.mean(np.abs(est - actual)))
            results.append({"Seed": seed, "Model": name, "RMSE": rmse, "MAE": mae})
            print(f"  {name:15s} Seed {seed}: RMSE={rmse:.5f}, MAE={mae:.5f}")

    df = pd.DataFrame(results)
    out_csv = results_dir / "forecast_accuracy_by_seed.csv"
    df.to_csv(out_csv, index=False)

    summary = []
    n_seeds = len(seeds)
    t_crit = stats.t.ppf(0.975, df=n_seeds - 1) # ~2.776 for N=5

    for name in models:
        sub = df[df["Model"] == name]
        rmse_m, rmse_s = sub["RMSE"].mean(), sub["RMSE"].std(ddof=1)
        mae_m, mae_s = sub["MAE"].mean(), sub["MAE"].std(ddof=1)
        rmse_ci = t_crit * (rmse_s / np.sqrt(n_seeds))
        mae_ci = t_crit * (mae_s / np.sqrt(n_seeds))
        summary.append({
            "Model": name,
            "RMSE Mean": rmse_m,
            "RMSE Std": rmse_s,
            "RMSE 95% CI": f"[{rmse_m - rmse_ci:.4f}, {rmse_m + rmse_ci:.4f}]",
            "MAE Mean": mae_m,
            "MAE Std": mae_s,
            "MAE 95% CI": f"[{mae_m - mae_ci:.4f}, {mae_m + mae_ci:.4f}]",
        })

    sum_df = pd.DataFrame(summary)
    sum_df.to_csv(results_dir / "forecast_accuracy_5seeds_summary.csv", index=False)
    print("\n=== 5-Seed Forecasting Performance Summary ===")
    print(sum_df.to_string(index=False))

    # Paired comparisons
    pivot_rmse = df.pivot(index="Seed", columns="Model", values="RMSE")
    print("\n=== Paired t-tests on RMSE across 5 seeds ===")
    for pair in [("XGBoost", "MovingAverage"), ("XGBoost", "LSTM"), ("LSTM", "MovingAverage")]:
        m1, m2 = pair
        diff = pivot_rmse[m1] - pivot_rmse[m2]
        t_stat, p_val = stats.ttest_rel(pivot_rmse[m1], pivot_rmse[m2])
        print(f"  {m1} vs {m2}: Diff={diff.mean():+.5f}, t={t_stat:.3f}, p={p_val:.4f}")

    return sum_df


if __name__ == "__main__":
    cfg_p = Path(__file__).parents[1] / "config.yaml"
    run_multiseed_forecast(cfg_p)
