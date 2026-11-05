"""Export result CSVs as paper-ready LaTeX tabular environments."""
from __future__ import annotations

from pathlib import Path

import pandas as pd


def _number(value: float) -> str:
    return f"{value:.4g}"


def _table(title: str, headers: list[str], rows: list[list[str]]) -> str:
    spec = "l" + "r" * (len(headers) - 1)
    lines = [f"\\paragraph{{{title}}}", f"\\begin{{tabular}}{{{spec}}}", "\\hline",
             " & ".join(headers) + r" \\", "\\hline"]
    lines.extend(" & ".join(row) + r" \\" for row in rows)
    lines.extend(["\\hline", "\\end{tabular}", ""])
    return "\n".join(lines)


def export_tables(results_root: Path, scenario: str) -> Path:
    scenario_dir = results_root / scenario
    forecast = pd.read_csv(results_root / "forecast_accuracy.csv")
    policy = pd.read_csv(scenario_dir / "migration_performance.csv")
    sensitivity = pd.read_csv(scenario_dir / "fcm_sensitivity.csv")
    ablation = pd.read_csv(scenario_dir / "fcm_ablation.csv")

    output = [_table("Forecast accuracy", ["Model", "RMSE", "MAE", "Inference (ms)"], [
        [str(row["Model"]), *[_number(float(row[column])) for column in
                              ("RMSE", "MAE", "Inference (ms)")]]
        for _, row in forecast.iterrows()
    ])]
    policy_metrics = ["Overload time (%)", "Avg latency (ms)", "Migrations",
                      "Unnecessary migrations (%)", "Load std dev", "SLA violation (%)",
                      "Total migration cost (s)", "No feasible target events"]
    policy_headers = ["Policy", "Overload (\\%)", "Latency (ms)", "Migrations",
                      "Unnecessary (\\%)", "Load std", "SLA violation (\\%)",
                      "Migration cost (s)", "No feasible"]
    policy_rows = []
    for _, row in policy.iterrows():
        policy_rows.append([str(row["Policy"])] + [
            f"${_number(float(row[metric]))} \\pm {_number(float(row[f'{metric} std']))}$"
            for metric in policy_metrics
        ])
    output.append(_table(f"Policy comparison ({scenario})", policy_headers, policy_rows))

    sensitivity_rows = []
    for _, row in sensitivity.iterrows():
        sensitivity_rows.append([
            str(row["Parameter"]), str(row["Setting"]),
            f"${_number(float(row['Overload time (%)']))} \\pm "
            f"{_number(float(row['Overload time (%) std']))}$",
            f"${_number(float(row['Migrations']))} \\pm "
            f"{_number(float(row['Migrations std']))}$",
        ])
    output.append(_table(f"FCM sensitivity ({scenario})",
                         ["Parameter", "Setting", "Overload (\\%)", "Migrations"],
                         sensitivity_rows))

    ablation_metrics = ["Overload time (%)", "Migrations", "Total migration cost (s)",
                        "Unnecessary migrations (%)"]
    ablation_headers = ["Variant", "Overload (\\%)", "Migrations",
                        "Migration cost (s)", "Unnecessary (\\%)"]
    ablation_rows = []
    for _, row in ablation.iterrows():
        ablation_rows.append([str(row["Variant"])] + [
            f"${_number(float(row[metric]))} \\pm {_number(float(row[f'{metric} std']))}$"
            for metric in ablation_metrics
        ])
    output.append(_table(f"FCM ablation ({scenario})", ablation_headers, ablation_rows))

    path = scenario_dir / "tables.tex"
    path.write_text("\n\n".join(output), encoding="utf-8")
    print(f"\nLaTeX tables ({scenario}):\n" + "\n\n".join(output))
    return path
