"use client";

import { useEffect, useState } from "react";
import { DataTable, ErrorMessage, LoadingMessage, PageHeading } from "@/components/shell";
import { api, responseSeconds, scenarioInfo, type Scenario } from "@/lib/api";

type ResultState = { sensitivity: Record<string, unknown>[]; ablation: Record<string, unknown>[] };
export default function SensitivityPage() {
  const [scenario, setScenario] = useState<Scenario>("default");
  const [data, setData] = useState<ResultState>({ sensitivity: [], ablation: [] });
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(true);
  const readableSensitivity = data.sensitivity.map((row) => {
    const labels: Record<string, string> = {
      horizon: "How far ahead the forecast looks",
      persistence: "How long a busy forecast must last",
      tau: "Minimum benefit required before a move",
      theta_safe: "How busy a destination may be",
      weights: "Which trade-off matters most",
    };
    const parameter = String(row.Parameter);
    const unit = ["horizon", "persistence"].includes(parameter) ? " minutes" : "";
    return {
      "What we changed": labels[parameter] ?? parameter,
      "Setting tested": `${String(row.Setting)}${unit}`,
      "Time computers were too busy": `${Number(row["Overload time (%)"]).toFixed(2)}%`,
      "Moves made": Number(row.Migrations).toFixed(1),
      "Typical user wait": responseSeconds(Number(row["Avg latency (ms)"])),
    };
  });
  const ablationNames: Record<string, string> = {
    "FCM-full": "Balanced rule with all checks",
    "FCM-no-safe-filter": "Allow moves to busier computers",
    "FCM-no-cooldown": "Remove the pause between repeat moves",
    "FCM-persistence-1": "React to one busy forecast",
  };
  const readableAblation = data.ablation.map((row) => ({
    "Rule version": ablationNames[String(row.Variant)] ?? row.Variant,
    "Time computers were too busy": `${Number(row["Overload time (%)"]).toFixed(2)}%`,
    "Moves made": Number(row.Migrations).toFixed(1),
    "Typical user wait": responseSeconds(Number(row["Avg latency (ms)"])),
  }));
  useEffect(() => {
    setLoading(true);
    setError("");
    Promise.all([
      api<Record<string, unknown>[]>(`/results/sensitivity?scenario=${scenario}`),
      api<Record<string, unknown>[]>(`/results/ablation?scenario=${scenario}`),
    ]).then(([sensitivity, ablation]) => setData({ sensitivity, ablation }))
      .catch((reason: Error) => setError(reason.message))
      .finally(() => setLoading(false));
  }, [scenario]);
  return (
    <>
      <PageHeading eyebrow="WHAT-IF CHECKS" title="What changes when the rules change?"
        detail="These saved tests adjust one setting at a time, then measure how often computers were too busy, how many apps moved, and how long users waited." />
      <div className="control-row">
        <label htmlFor="sensitivity-scenario">Connection conditions</label>
        <select className="select" id="sensitivity-scenario" value={scenario} onChange={(event) => setScenario(event.target.value as Scenario)}>
          {(Object.keys(scenarioInfo) as Scenario[]).map((value) => <option key={value} value={value}>{scenarioInfo[value].label}</option>)}
        </select>
      </div>
      {loading && <LoadingMessage />}{error && <ErrorMessage message={error} />}
      {!loading && !error && <>
        <section className="sheet">
          <div className="section-heading"><h2>Try different settings</h2><p>{scenarioInfo[scenario].label}</p></div>
          <DataTable rows={readableSensitivity} />
          <details className="advanced-details">
            <summary>Show the full engineering measurements</summary>
            <DataTable rows={data.sensitivity} />
          </details>
        </section>
        <section className="sheet">
          <div className="section-heading"><h2>What if one safety check is removed?</h2><p>Each row is a version of the same forecast-based rule.</p></div>
          <DataTable rows={readableAblation} />
          <details className="advanced-details">
            <summary>Show full policy names and engineering measurements</summary>
            <DataTable rows={data.ablation} />
          </details>
        </section>
      </>}
    </>
  );
}
