"""Run synthetic data, forecasting, simulations, summaries, and plots."""
from __future__ import annotations

import copy
import sys
from math import ceil
from pathlib import Path
from typing import Any

import yaml

from data.generate_traces import generate_traces
from forecasting.train import train_models
from sim.run_experiments import run_experiments
from analysis.make_tables_and_plots import make_forecast_plots, make_plots
from analysis.export_tables import export_tables
from sim.fcm_sensitivity import format_gain_diagnostics


def validate_config(config: dict[str, Any]) -> None:
    data, forecast, simulation = config["data"], config["forecast"], config["simulation"]
    weights = simulation["weights"]
    if len(weights) != 4 or any(float(w) < 0 for w in weights) or abs(sum(weights) - 1) > 1e-8:
        raise ValueError("simulation.weights must contain four non-negative values summing to one")
    if not 0 < float(data["train_fraction"]) < 1 or not 0 < float(data["validation_fraction"]) < 1:
        raise ValueError("Train and validation fractions must be between zero and one")
    if float(data["train_fraction"]) + float(data["validation_fraction"]) >= 1:
        raise ValueError("Train and validation fractions must leave a test interval")
    if int(forecast["horizon"]) < 1 or int(forecast["window"]) < 1:
        raise ValueError("Forecast window and horizon must be positive")
    if forecast["policy_model"] not in {"best_recurrent", "GRU", "LSTM"}:
        raise ValueError("forecast.policy_model must be best_recurrent, GRU, or LSTM")
    scenarios = config.get("scenarios", {"default": {}})
    for name, scenario in scenarios.items():
        bandwidth = scenario.get("bandwidth_mbps", simulation["network"]["bandwidth_mbps"])
        state_size = scenario.get("state_size_mb", simulation["services"]["state_size_mb"])
        minimum_bandwidth, maximum_state = float(bandwidth[0]), float(state_size[1])
        migration_slots = max(1, ceil((maximum_state * 8 / minimum_bandwidth) / 60))
        if int(forecast["horizon"]) < migration_slots:
            raise ValueError(
                f"forecast.horizon must be at least {migration_slots} migration slots for {name}")


def run(config_path: Path) -> None:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    validate_config(config)
    project = config_path.parent
    traces_dir = project / config["paths"]["traces"]
    results_dir = project / config["paths"]["results"]
    traces_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)
    (results_dir / "config_used.yaml").write_text(
        yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

    seed = int(config["seed"])
    traces, metadata = generate_traces(config, seed)
    import numpy as np
    np.savez_compressed(traces_dir / f"traces_seed_{seed}.npz", traces=traces)
    metadata.to_csv(traces_dir / f"series_seed_{seed}.csv", index=False)
    print(f"Generated {traces.shape[0]} traces x {traces.shape[1]} slots (seed {seed}).")

    models, train_meta = train_models(config, results_dir, traces_dir)
    chosen_model = models[train_meta["policy_model"]]
    forecast_figures = make_forecast_plots(results_dir, train_meta)

    print("\nForecast accuracy (composite node load):")
    print(train_meta["accuracy"].to_string(index=False, float_format=lambda x: f"{x:.5f}"))
    print(f"\nPolicy forecaster selected by validation RMSE: {train_meta['policy_model']} "
          f"({train_meta['validation_rmse'][train_meta['policy_model']]:.5f})")
    print("\nScenario results:")
    for scenario_name, scenario_values in config.get("scenarios", {"default": {}}).items():
        scenario_config = copy.deepcopy(config)
        scenario_config["simulation"]["network"]["bandwidth_mbps"] = scenario_values.get(
            "bandwidth_mbps", scenario_config["simulation"]["network"]["bandwidth_mbps"])
        scenario_config["simulation"]["services"]["state_size_mb"] = scenario_values.get(
            "state_size_mb", scenario_config["simulation"]["services"]["state_size_mb"])
        scenario_dir = results_dir / scenario_name
        scenario_dir.mkdir(parents=True, exist_ok=True)
        (scenario_dir / "config_used.yaml").write_text(
            yaml.safe_dump(scenario_config, sort_keys=False), encoding="utf-8")
        performance, experiment_meta = run_experiments(
            scenario_config, chosen_model, scenario_dir,
            int(train_meta["test_start"]), int(train_meta["test_end"]))
        figures = make_plots(scenario_dir, experiment_meta, performance)
        tex_path = export_tables(results_dir, scenario_name)
        raw = experiment_meta["raw"]
        migration_count = raw["Migrations"].sum()
        mean_slots = (raw["Mean migration slots"] * raw["Migrations"]).sum() / max(1, migration_count)
        max_slots = int(raw["Max migration slots"].max())
        print(f"\n{scenario_name}: migration_slots max={max_slots}, mean={mean_slots:.3f}")
        print(f"  FCM chosen-candidate gain distribution: "
              f"{format_gain_diagnostics(experiment_meta['diagnostics'].get('gain_by_seed', {}))}")
        outcomes = experiment_meta["diagnostics"].get("horizon_outcomes", {})
        print(f"  Horizon outcome changes vs H={outcomes.get('reference_horizon')}: "
              f"{outcomes.get('changed_vs_reference', {})}")
        print("  Migration performance (mean over seeds; std columns included):")
        print(performance.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
        print(f"  Tables: {tex_path.relative_to(project)}")
        print(f"  Scenario figures: {', '.join(path.name for path in figures)}")
    print("\nOutput files:")
    for path in sorted([*results_dir.rglob("*"), *traces_dir.glob("*")]):
        if path.is_file():
            print(f"  {path.relative_to(project)}")
    print("\nAssumptions/limitations: synthetic traces are normalized and generated independently by series;")
    print("the oracle unnecessary-migration metric uses actual future traces with placement frozen;")
    print("latency and migration are simplified edge-network models; random-onset bursts are not predictable;")
    print("models are trained on the single configured training seed only.")
    print(f"Created {len(forecast_figures)} shared forecast figures at 300 dpi.")


def main() -> None:
    package_dir = Path(__file__).resolve().parent
    sys.path.insert(0, str(package_dir))
    run(package_dir / "config.yaml")


if __name__ == "__main__":
    main()
