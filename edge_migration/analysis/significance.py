"""Statistical significance analysis across experimental seeds."""
from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats


def evaluate_significance(scenario_dir: Path) -> tuple[pd.DataFrame, str]:
    seed_file = scenario_dir / "migration_performance_by_seed.csv"
    if not seed_file.exists():
        raise FileNotFoundError(f"Missing {seed_file}")

    df = pd.read_csv(seed_file)
    fcm_df = df[df["Policy"] == "FCM"].sort_values("Seed").reset_index(drop=True)

    metrics = [
        ("Migrations", "Migrations"),
        ("Total migration cost (s)", "Migration Cost (s)"),
        ("Overload time (%)", "Overload Time (%)"),
        ("SLA violation (%)", "SLA Violation (%)"),
    ]

    baselines = ["RT", "RT-CD", "FLL"]
    records = []

    for baseline in baselines:
        base_df = df[df["Policy"] == baseline].sort_values("Seed").reset_index(drop=True)
        if len(base_df) != len(fcm_df):
            continue

        for metric_col, metric_label in metrics:
            base_vals = base_df[metric_col].to_numpy(dtype=float)
            fcm_vals = fcm_df[metric_col].to_numpy(dtype=float)
            diff = fcm_vals - base_vals

            # Paired t-test
            t_stat, p_val = stats.ttest_rel(fcm_vals, base_vals)
            # Wilcoxon signed-rank test
            try:
                w_stat, w_pval = stats.wilcoxon(fcm_vals, base_vals)
            except Exception:
                w_pval = p_val

            sig_marker = "***" if p_val < 0.001 else ("**" if p_val < 0.01 else ("*" if p_val < 0.05 else "ns"))
            pct_change = ((fcm_vals.mean() - base_vals.mean()) / max(1e-8, base_vals.mean())) * 100

            records.append({
                "Comparison": f"FCM vs {baseline}",
                "Metric": metric_label,
                "Baseline Mean": float(base_vals.mean()),
                "Baseline Std": float(base_vals.std(ddof=1)),
                "FCM Mean": float(fcm_vals.mean()),
                "FCM Std": float(fcm_vals.std(ddof=1)),
                "Diff (FCM - Base)": float(diff.mean()),
                "Pct Change (%)": float(pct_change),
                "t-statistic": float(t_stat) if np.isfinite(t_stat) else 0.0,
                "p-value (t-test)": float(p_val) if np.isfinite(p_val) else 1.0,
                "p-value (Wilcoxon)": float(w_pval) if np.isfinite(w_pval) else 1.0,
                "Significance": sig_marker,
            })

    sig_df = pd.DataFrame(records)
    out_csv = scenario_dir / "statistical_significance.csv"
    sig_df.to_csv(out_csv, index=False)

    # Generate LaTeX table snippet
    lines = [
        r"\begin{table}[t]",
        r"\caption{Statistical significance of FCM improvements (paired $t$-test across 5 seeds).}",
        r"\label{tab:significance}",
        r"\centering",
        r"\scriptsize",
        r"\begin{tabular}{llcccc}",
        r"\toprule",
        r"Comparison & Metric & Baseline & FCM & Diff (\%) & $p$-value \\",
        r"\midrule",
    ]
    for rec in records:
        diff_str = f"{rec['Pct Change (%)']:+.1f}\\%"
        p_str = f"{rec['p-value (t-test)']:.4f} ({rec['Significance']})" if rec['p-value (t-test)'] >= 0.0001 else f"<0.0001 ({rec['Significance']})"
        base_str = f"{rec['Baseline Mean']:.2f} $\\pm$ {rec['Baseline Std']:.2f}"
        fcm_str = f"{rec['FCM Mean']:.2f} $\\pm$ {rec['FCM Std']:.2f}"
        lines.append(f"{rec['Comparison']} & {rec['Metric']} & {base_str} & {fcm_str} & {diff_str} & {p_str} \\\\")
    lines.extend([
        r"\bottomrule",
        r"\multicolumn{6}{l}{\scriptsize $^*p < 0.05$, $^{**}p < 0.01$, $^{***}p < 0.001$, ns: not significant.}",
        r"\end{tabular}",
        r"\end{table}",
    ])
    tex_str = "\n".join(lines)
    (scenario_dir / "significance_table.tex").write_text(tex_str, encoding="utf-8")
    return sig_df, tex_str


if __name__ == "__main__":
    scenario_path = Path(__file__).parents[1] / "results" / "default"
    if scenario_path.exists():
        df, tex = evaluate_significance(scenario_path)
        print(df[["Comparison", "Metric", "Pct Change (%)", "p-value (t-test)", "Significance"]])
