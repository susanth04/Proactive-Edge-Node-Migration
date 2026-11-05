"use client";

import { useEffect, useState } from "react";
import { ComparisonCharts, PolicyResultsTable } from "@/components/charts";
import { DataTable, ErrorMessage, LoadingMessage, PageHeading } from "@/components/shell";
import { api, policyInfo, scenarioInfo, type PerformanceRow, type Scenario } from "@/lib/api";

export default function DashboardPage() {
  const [scenario, setScenario] = useState<Scenario>("default");
  const [rows, setRows] = useState<PerformanceRow[]>([]);
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(true);
  const researchMetrics = [
    "Overload time (%)", "SLA violation (%)", "Avg latency (ms)", "Migrations",
    "Total migration cost (s)", "Unnecessary migrations (%)", "Load std dev",
    "No feasible target events",
  ];
  const summaryRows = rows.map((row) => ({
    Policy: row.Policy,
    busyPct: Number(row["Overload time (%)"]),
    responseMs: Number(row["Avg latency (ms)"]),
    migrations: Number(row.Migrations),
  }));
  const researchRows = rows.map((row) => ({
    Policy: row.Policy,
    "Plain-language description": policyInfo[row.Policy].description,
    ...Object.fromEntries(researchMetrics.map((key) => [
      key,
      `${Number(row[key]).toPrecision(4)} ± ${Number(row[`${key} std`]).toPrecision(4)}`,
    ])),
  }));

  useEffect(() => {
    setLoading(true);
    setError("");
    api<PerformanceRow[]>(`/results/performance?scenario=${scenario}`)
      .then(setRows).catch((reason: Error) => setError(reason.message))
      .finally(() => setLoading(false));
  }, [scenario]);

  return (
    <>
      <PageHeading eyebrow="START HERE" title="How well do different moving rules work?"
        detail="Apps run on nearby computers. When a computer gets too busy, the simulator tries moving an app to another one." />
      <div className="notice"><strong>A safe, made-up example.</strong> These results come from generated computer workloads, not live devices. Every number shown was measured by running the simulator.</div>
      <div className="control-row">
        <label htmlFor="scenario">Connection conditions</label>
        <select className="select" id="scenario" value={scenario} onChange={(event) => setScenario(event.target.value as Scenario)}>
          {(Object.keys(scenarioInfo) as Scenario[]).map((value) => <option key={value} value={value}>{scenarioInfo[value].label}</option>)}
        </select>
      </div>
      <p className="scenario-explanation">{scenarioInfo[scenario].description}</p>
      <section className="sheet">
        <div className="section-heading"><h2>Compare the four approaches</h2><p>Average from multiple simulator runs</p></div>
        {loading && <LoadingMessage />}
        {error && <ErrorMessage message={error} />}
        {!loading && !error && <>
          <PolicyResultsTable rows={summaryRows} />
          <div className="guide-grid">
            <p><strong>Busy time:</strong> how often computers were above the selected workload limit. Lower is better.</p>
            <p><strong>Average response:</strong> how long a user waits for an answer. Lower is better.</p>
            <p><strong>Moves made:</strong> moving an app can help, but every move takes time and network capacity.</p>
          </div>
          <div className="section-heading"><h2>What the results look like</h2><p>{scenarioInfo[scenario].label}</p></div>
          <ComparisonCharts rows={rows} />
          <details className="advanced-details">
            <summary>Show detailed research metrics</summary>
            <p className="chart-caption">Each value shows the average ± run-to-run variation. These are engineering measures used to compare policies in more detail.</p>
            <DataTable rows={researchRows} />
          </details>
        </>}
      </section>
    </>
  );
}
