"""Run policy comparisons and one-factor FCM sensitivity sweeps."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from data.generate_traces import generate_traces
from forecasting.train import make_features
from sim.fcm_sensitivity import run_sensitivity
from sim.policies import run_policy


def build_forecasts(model: Any, traces: np.ndarray, config: dict[str, Any],
                    start: int, stop: int) -> np.ndarray:
    """Forecast each independent background/service trace without aggregating early."""
    forecast_cfg = config["forecast"]
    window, horizon = int(forecast_cfg["window"]), int(forecast_cfg["horizon"])
    features = make_features(traces, int(config["data"]["slots_per_day"]))
    steps, series = max(0, stop - start), len(traces)
    forecast = np.empty((steps, series, horizon, 4), dtype=np.float32)
    chunk = 24
    for offset in range(0, steps, chunk):
        times = range(start + offset, min(stop, start + offset + chunk))
        windows = np.stack([features[:, t - window + 1:t + 1] for t in times], axis=1)
        count = windows.shape[1]
        predicted = model.predict(windows.reshape(series * count, window, features.shape[-1]))
        forecast[offset:offset + count] = predicted.reshape(series, count, horizon, 4).transpose(1, 0, 2, 3)
    return forecast


def _performance_table(raw: pd.DataFrame) -> pd.DataFrame:
    metric_columns = ["Overload time (%)", "Avg latency (ms)", "Migrations",
                      "Unnecessary migrations (%)", "Load std dev", "SLA violation (%)",
                      "Total migration cost (s)", "Decision time (ms/slot)",
                      "No feasible target events", "Max migration slots", "Mean migration slots"]
    grouped = raw.groupby("Policy", sort=False)
    mean = grouped[metric_columns].mean().reset_index()
    std = grouped[metric_columns].std(ddof=1).fillna(0).add_suffix(" std").reset_index(drop=True)
    return pd.concat((mean, std), axis=1)


def _run_ablation(config: dict[str, Any], forecasts: dict[int, np.ndarray],
                  trace_map: dict[int, np.ndarray], start: int) -> pd.DataFrame:
    variants = [
        ("FCM-full", {}),
        ("FCM-no-safe-filter", {"theta_safe": 1.0}),
        ("FCM-no-cooldown", {"cooldown_slots": 0}),
        ("FCM-persistence-1", {"persistence": 1}),
    ]
    metric_names = ["Overload time (%)", "Avg latency (ms)", "Migrations",
                    "Unnecessary migrations (%)", "Load std dev", "SLA violation (%)",
                    "Total migration cost (s)", "No feasible target events"]
    rows = []
    seeds = [int(seed) for seed in config["simulation"]["seeds"]]
    for variant, override in variants:
        runs = [run_policy(config, "FCM", trace_map[seed], forecasts[seed], start,
                           fcm_overrides=override, seed=seed)[0] for seed in seeds]
        row: dict[str, Any] = {"Variant": variant, "Runs": len(runs)}
        for metric in metric_names:
            values = np.asarray([float(run[metric]) for run in runs])
            row[metric] = float(values.mean())
            row[f"{metric} std"] = float(values.std(ddof=1)) if len(values) > 1 else 0.0
        rows.append(row)
    return pd.DataFrame(rows)


def run_experiments(config: dict[str, Any], model: Any, results_dir: Path,
                    start: int, stop: int) -> tuple[pd.DataFrame, dict[str, Any]]:
    seeds = [int(seed) for seed in config["simulation"]["seeds"]]
    forecast_seeds = list(dict.fromkeys(seeds + (
        [int(seed) for seed in config["sensitivity"]["seeds"]]
        if bool(config["sensitivity"]["enabled"]) else [])))
    traces_dir = Path(__file__).parents[1] / config["paths"]["traces"]
    traces_by_seed: dict[int, np.ndarray] = {}
    forecasts_by_seed: dict[int, np.ndarray] = {}
    for seed in forecast_seeds:
        path = traces_dir / f"traces_seed_{seed}.npz"
        if path.exists():
            traces = np.load(path)["traces"]
        else:
            traces, metadata = generate_traces(config, seed)
            path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(path, traces=traces)
            metadata.to_csv(path.parent / f"series_seed_{seed}.csv", index=False)
        traces_by_seed[seed] = traces
        forecasts_by_seed[seed] = build_forecasts(model, traces, config, start, stop)

    raw_rows: list[dict[str, Any]] = []
    sample_timeline: dict[str, Any] = {}
    gains_by_seed: dict[int, list[float]] = {}
    for seed in seeds:
        for policy in ("RT", "RT-CD", "FLL", "FCM"):
            gain_sink = [] if policy == "FCM" else None
            metrics, timeline = run_policy(
                config, policy, traces_by_seed[seed], forecasts_by_seed[seed], start,
                seed=seed, keep_timeline=(seed == seeds[0] and policy in {"RT", "FCM"}),
                gain_sink=gain_sink,
            )
            raw_rows.append({"Seed": seed, **metrics})
            if gain_sink is not None:
                gains_by_seed[seed] = gain_sink
            if timeline:
                sample_timeline[policy] = timeline

    raw = pd.DataFrame(raw_rows)
    raw.to_csv(results_dir / "migration_performance_by_seed.csv", index=False)
    performance = _performance_table(raw)
    performance.to_csv(results_dir / "migration_performance.csv", index=False)

    sensitivity = pd.DataFrame()
    diagnostics: dict[str, Any] = {}
    if bool(config["sensitivity"]["enabled"]):
        sensitivity, diagnostics = run_sensitivity(
            config, forecasts_by_seed, traces_by_seed, start, gains_by_seed)
        sensitivity.to_csv(results_dir / "fcm_sensitivity.csv", index=False)
    ablation = _run_ablation(config, forecasts_by_seed, traces_by_seed, start)
    ablation.to_csv(results_dir / "fcm_ablation.csv", index=False)
    sample_timeline["traces"] = traces_by_seed[seeds[0]]
    sample_timeline["forecasts"] = forecasts_by_seed[seeds[0]]
    sample_timeline["start_slot"] = start
    sample_timeline["seed"] = seeds[0]
    return performance, {"timeline": sample_timeline, "raw": raw, "sensitivity": sensitivity,
                         "ablation": ablation, "diagnostics": diagnostics}
