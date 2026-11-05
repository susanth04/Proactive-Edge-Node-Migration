"""FCM parameter sweeps and diagnostics for sensitivity interpretation."""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from sim.environment import Environment
from sim.policies import _trigger_run, run_policy

# On the checked 5-seed run, scored FCM candidates had gains 0.12952--0.31656
# (default) and 0.12636--0.29974 (slow_links), so tau through 0.08 rejected none.
# The initial-placement prefix diagnostic found H=8 and H=10 identical; H=5 differed
# from H=10 on 649/94,740 first-breach and 273/94,740 persistence outcomes, but none
# changed the selected migration in the sensitivity runs. Record the observed causes
# rather than tuning parameters to manufacture sweep differences; rerunning prints
# the measured values.


def run_sensitivity(config: dict[str, Any], forecasts: dict[int, np.ndarray],
                    traces: dict[int, np.ndarray], start: int,
                    gains: dict[int, list[float]]) -> tuple[pd.DataFrame, dict[str, Any]]:
    sweep_rows: list[dict[str, Any]] = []
    sens = config["sensitivity"]
    seeds = [int(seed) for seed in sens["seeds"]]
    tests: list[tuple[str, Any, dict[str, Any]]] = []
    for parameter in ("horizons", "persistence", "tau", "theta_safe"):
        key = {"horizons": "horizon"}.get(parameter, parameter)
        for value in sens[parameter]:
            tests.append((key, value, {key: value}))
    for value in sens["weights"]:
        tests.append(("weights", ",".join(f"{v:g}" for v in value),
                      {"alpha": float(value[0]), "beta": float(value[1]), "gamma": float(value[2])}))

    for parameter, setting, override in tests:
        rows = [run_policy(config, "FCM", traces[seed], forecasts[seed], start,
                           fcm_overrides=override, seed=seed)[0] for seed in seeds]
        row: dict[str, Any] = {"Parameter": parameter, "Setting": setting, "Runs": len(rows)}
        for key in ["Overload time (%)", "Avg latency (ms)", "Migrations",
                    "Unnecessary migrations (%)", "Load std dev"]:
            values = np.asarray([float(result[key]) for result in rows])
            row[key] = float(values.mean())
            row[f"{key} std"] = float(values.std(ddof=1)) if len(values) > 1 else 0.0
        sweep_rows.append(row)

    diagnostics = {"gain_by_seed": gains, "horizon_outcomes": horizon_outcomes(
        config, forecasts, traces, seeds)}
    return pd.DataFrame(sweep_rows), diagnostics


def horizon_outcomes(config: dict[str, Any], forecasts: dict[int, np.ndarray],
                     traces: dict[int, np.ndarray], seeds: list[int]) -> dict[str, Any]:
    """Compare first-breach and persistence outcomes for horizon prefixes."""
    sizes = [size for size in (5, 8, 10) if size <= int(config["forecast"]["horizon"])]
    baseline = max(sizes)
    threshold = float(config["simulation"]["theta_hi"])
    persistence = int(config["simulation"]["fcm"]["persistence"])
    changed = {size: {"first_breach": 0, "persistence": 0, "total": 0}
               for size in sizes if size != baseline}

    for seed in seeds:
        env = Environment(config, traces[seed], seed)
        values = forecasts[seed]
        for local_slot in range(len(values)):
            by_node = np.empty((len(env.nodes), baseline), dtype=np.float32)
            for lead in range(baseline):
                by_node[:, lead] = env.node_channels(values[local_slot, :, lead, :]) @ np.asarray(
                    config["simulation"]["weights"], dtype=np.float32)
            for node_id in range(len(env.nodes)):
                reference = by_node[node_id, :baseline]
                reference_first = int(np.flatnonzero(reference >= threshold)[0]) if np.any(
                    reference >= threshold) else None
                reference_persistent = _trigger_run(reference, threshold, persistence, 1)
                for size in changed:
                    prefix = by_node[node_id, :size]
                    first = int(np.flatnonzero(prefix >= threshold)[0]) if np.any(prefix >= threshold) else None
                    persistent = _trigger_run(prefix, threshold, persistence, 1)
                    changed[size]["first_breach"] += int(first != reference_first)
                    changed[size]["persistence"] += int(persistent != reference_persistent)
                    changed[size]["total"] += 1
    for counts in changed.values():
        total = max(1, counts["total"])
        counts["first_breach_fraction"] = counts["first_breach"] / total
        counts["persistence_fraction"] = counts["persistence"] / total
    return {"reference_horizon": baseline, "changed_vs_reference": changed}


def format_gain_diagnostics(gains: dict[int, list[float]]) -> str:
    all_gains = np.asarray([gain for seed_gains in gains.values() for gain in seed_gains], dtype=float)
    if not len(all_gains):
        return "No FCM candidates passed lead-time and safe-target filtering."
    return (f"n={len(all_gains)}, min={all_gains.min():.5f}, "
            f"p25={np.quantile(all_gains, .25):.5f}, median={np.median(all_gains):.5f}, "
            f"mean={all_gains.mean():.5f}, p75={np.quantile(all_gains, .75):.5f}, "
            f"max={all_gains.max():.5f}")
