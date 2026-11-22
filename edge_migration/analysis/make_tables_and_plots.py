"""Write paper-ready CSV summaries and 300-dpi figures."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def make_forecast_plots(results_dir: Path, train_meta: dict[str, Any]) -> list[Path]:
    results_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 10, "axes.titlesize": 12, "figure.dpi": 120})
    outputs: list[Path] = []
    actual, predicted = train_meta["preferred_actual"], train_meta["preferred_prediction"]
    step = int(train_meta["evaluation_step"])

    fig, ax = plt.subplots(figsize=(8, 3.8))
    count = min(96, len(actual))
    ax.plot(np.arange(count) * step, actual[:count, 0, 0], label="Actual", linewidth=1.5)
    ax.plot(np.arange(count) * step, predicted[:count, 0, 0], label="Forecast", linewidth=1.3)
    ax.set(title="One-step node-load forecast (node 0)", xlabel="Test interval (minutes)", ylabel="Composite load")
    ax.set_ylim(0, 1)
    ax.grid(alpha=0.25)
    ax.legend()
    path = results_dir / "load_forecast.png"
    fig.tight_layout(); fig.savefig(path, dpi=300); plt.close(fig); outputs.append(path)

    fig, ax = plt.subplots(figsize=(7, 3.8))
    ax.plot(np.arange(1, len(train_meta["horizon_rmse"]) + 1), train_meta["horizon_rmse"],
            marker="o", linewidth=1.5)
    ax.set(title="Forecast RMSE by lead time", xlabel="Forecast lead (slots)", ylabel="Composite load RMSE")
    ax.grid(alpha=0.3)
    path = results_dir / "forecast_rmse_horizon.png"
    fig.tight_layout(); fig.savefig(path, dpi=300); plt.close(fig); outputs.append(path)
    return outputs


def make_plots(results_dir: Path, experiment_meta: dict[str, Any],
               performance: pd.DataFrame) -> list[Path]:
    results_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 10, "axes.titlesize": 12, "figure.dpi": 120})
    outputs: list[Path] = []
    timeline = experiment_meta["timeline"]

    fig, ax = plt.subplots(figsize=(9, 3.8))
    for policy, style in (("RT", "-"), ("FCM", "-")):
        loads = timeline[policy]["node_loads"]
        ax.plot(loads[:, 0], style, linewidth=1.0, label=policy)
    events = [event for event, node in zip(timeline["FCM"]["migration_slots"],
                                            timeline["FCM"]["migration_nodes"]) if node == 0]
    for event in events:
        ax.axvline(event, color="tab:green", alpha=0.18, linewidth=0.8)
    ax.axhline(0.85, color="tab:red", linestyle="--", linewidth=1, label="Overload threshold")
    ax.set(title="Node 0 load under RT and FCM (green: FCM source migration)",
           xlabel="Test slot (minutes)", ylabel="Composite load", ylim=(0, 1))
    ax.grid(alpha=0.25); ax.legend()
    path = results_dir / "load_timeline_rt_fcm.png"
    fig.tight_layout(); fig.savefig(path, dpi=300); plt.close(fig); outputs.append(path)

    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    plot_metrics = [
        ("Overload time (%)", "Overload time (%)"),
        ("Migrations", "Migrations"),
        ("SLA violation (%)", "SLA violation (%)"),
        ("Total migration cost (s)", "Total migration cost (s)"),
    ]
    x = np.arange(len(performance))
    for ax, (metric, title) in zip(axes.flat, plot_metrics):
        error = performance.get(f"{metric} std", pd.Series(np.zeros(len(performance))))
        ax.bar(x, performance[metric], yerr=error, capsize=3)
        ax.set_title(title)
        ax.set_xticks(x, performance["Policy"], rotation=15)
        ax.grid(axis="y", alpha=0.25)
    path = results_dir / "policy_comparison.png"
    fig.tight_layout(); fig.savefig(path, dpi=300); plt.close(fig); outputs.append(path)

    # Dedicated Migration Cost & Latency Trade-off Figure
    fig, (ax_cost, ax_tradeoff) = plt.subplots(1, 2, figsize=(9.5, 4.0))
    colors = ["#2b5c8f", "#4682b4", "#e67e22", "#27ae60"]

    # (a) Total Migration Cost (seconds) on log scale
    costs = performance["Total migration cost (s)"]
    cost_err = performance.get("Total migration cost (s) std", pd.Series(np.zeros(len(performance))))
    bars = ax_cost.bar(x, costs, yerr=cost_err, capsize=4, color=colors, edgecolor="black", alpha=0.85)
    ax_cost.set_yscale("log")
    ax_cost.set_ylabel("Total Migration Cost (s, log scale)")
    ax_cost.set_xticks(x, performance["Policy"])
    ax_cost.set_title("(a) Migration Cost by Policy")
    ax_cost.grid(axis="y", alpha=0.3, which="both")
    for bar, val in zip(bars, costs):
        ax_cost.text(bar.get_x() + bar.get_width() / 2, max(val * 1.15, 1.2),
                     f"{val:.1f}s", ha="center", va="bottom", fontsize=8.5, weight="bold")

    # (b) Latency vs Migration Cost Trade-off
    latencies = performance["Avg latency (ms)"]
    lat_err = performance.get("Avg latency (ms) std", pd.Series(np.zeros(len(performance))))
    for i, pol in enumerate(performance["Policy"]):
        ax_tradeoff.errorbar(costs.iloc[i], latencies.iloc[i],
                             xerr=cost_err.iloc[i], yerr=lat_err.iloc[i],
                             marker="o", markersize=8, color=colors[i], capsize=3,
                             label=pol)
        offset_y = 0.3 if i % 2 == 0 else -0.5
        ax_tradeoff.annotate(pol, (costs.iloc[i], latencies.iloc[i]),
                             textcoords="offset points", xytext=(8, offset_y * 10),
                             weight="bold", fontsize=9, color=colors[i])
    ax_tradeoff.set_xscale("log")
    ax_tradeoff.set_xlabel("Total Migration Cost (s, log scale)")
    ax_tradeoff.set_ylabel("Average Latency (ms)")
    ax_tradeoff.set_title("(b) Latency vs Migration Cost Trade-off")
    ax_tradeoff.grid(alpha=0.3, which="both")
    ax_tradeoff.legend(loc="upper right", framealpha=0.8)

    fig.tight_layout()
    cost_lat_path = results_dir / "policy_cost_latency.png"
    fig.savefig(cost_lat_path, dpi=300)
    plt.close(fig)
    outputs.append(cost_lat_path)

    sensitivity: pd.DataFrame = experiment_meta["sensitivity"]
    if not sensitivity.empty:
        parameters = ("horizon", "persistence", "tau", "theta_safe", "weights")
        fig, axes = plt.subplots(2, 3, figsize=(12, 7))
        for ax, parameter in zip(axes.flat, parameters):
            subset = sensitivity[sensitivity["Parameter"] == parameter]
            xlabels = subset["Setting"].astype(str).tolist()
            xvals = np.arange(len(subset))
            ax.errorbar(xvals, subset["Overload time (%)"], yerr=subset["Overload time (%) std"],
                        marker="o", capsize=3, label="Overload")
            ax.set_xticks(xvals, xlabels)
            ax.set_title(f"FCM sensitivity: {parameter}")
            ax.set_ylabel("Overload time (%)")
            ax.grid(alpha=0.25)
            ax2 = ax.twinx()
            ax2.plot(xvals, subset["Migrations"], color="tab:orange", marker="s", label="Migrations")
            ax2.set_ylabel("Migrations")
        axes.flat[-1].set_visible(False)
        fig.tight_layout()
        path = results_dir / "fcm_sensitivity.png"
        fig.savefig(path, dpi=300); plt.close(fig); outputs.append(path)
    return outputs
