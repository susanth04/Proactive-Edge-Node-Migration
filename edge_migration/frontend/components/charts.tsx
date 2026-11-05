"use client";

import {
  Bar, BarChart, CartesianGrid, ComposedChart, Legend, Line, ReferenceLine,
  ResponsiveContainer, Scatter, Tooltip, XAxis, YAxis,
} from "recharts";
import { policyInfo, responseSeconds, type PerformanceRow, type Policy, type SimulationResponse } from "@/lib/api";
import { useReducedMotion } from "@/lib/use-reduced-motion";

const seriesColors = ["var(--color-accent)", "var(--color-ink-2)", "var(--color-muted)", "var(--color-rule-2)"];
type ComparisonRow = Pick<PerformanceRow, "Policy" | "Overload time (%)" | "Migrations" | "Total migration cost (s)" | "Avg latency (ms)">;
const policyAxisLabels: Record<string, string> = {
  "React after overload": "React",
  "React, then pause": "React + pause",
  "Move before overload": "Forecast",
  "Balance the trade-offs": "Balanced",
};

export function PolicyResultsTable({ rows }: { rows: { Policy: PerformanceRow["Policy"]; busyPct: number; responseMs: number; migrations: number }[] }) {
  return (
    <div className="table-scroll">
      <table className="data-table policy-table">
        <thead><tr><th scope="col">Moving approach</th><th scope="col">Time computers were too busy</th><th scope="col">Average user wait</th><th scope="col">Moves made</th></tr></thead>
        <tbody>{rows.map((row) => (
          <tr key={row.Policy}>
            <th scope="row">
              <strong>{policyInfo[row.Policy].label}</strong>
              <small>{policyInfo[row.Policy].description}</small>
            </th>
            <td><span className="result-number">{row.busyPct.toFixed(2)}%</span><small>of computer time</small></td>
            <td><span className="result-number">{responseSeconds(row.responseMs)}</span><small>per response</small></td>
            <td><span className="result-number">{row.migrations.toLocaleString(undefined, { maximumFractionDigits: 1 })}</span><small>app moves on average</small></td>
          </tr>
        ))}</tbody>
      </table>
    </div>
  );
}

export function ComparisonCharts({ rows }: { rows: ComparisonRow[] }) {
  const reducedMotion = useReducedMotion();
  const metrics = [
    ["Overload time (%)", "Time computers were too busy", "%", "%"],
    ["Migrations", "Moves between computers", " moves", ""],
    ["Total migration cost (s)", "Transfer time and cost (seconds)", " s", " s"],
    ["Response time (s)", "Typical user wait (seconds)", " s", " s"],
  ] as const;
  const chartRows = rows.map((row) => ({
    ...row,
    Policy: policyInfo[row.Policy]?.label ?? row.Policy,
    "Response time (s)": Number(row["Avg latency (ms)"]) / 1000,
  }));
  return (
    <div className="chart-grid">
      {metrics.map(([key, title, unit, axisUnit]) => (
        <section className="chart-panel" key={key}>
          <h3>{title}</h3>
          <ResponsiveContainer width="100%" height={220}>
            <BarChart data={chartRows} margin={{ top: 8, right: 12, bottom: 4, left: 0 }}>
              <CartesianGrid vertical={false} stroke="var(--color-rule)" />
              <XAxis dataKey="Policy" interval={0} tick={{ fontSize: 11 }} tickFormatter={(value) => policyAxisLabels[String(value)] ?? String(value)} tickLine={false} axisLine={{ stroke: "var(--color-rule-2)" }} />
              <YAxis tickFormatter={(value) => `${Number(value).toLocaleString(undefined, { maximumFractionDigits: 1 })}${axisUnit}`} tickLine={false} axisLine={false} width={48} />
              <Tooltip formatter={(value) => [`${Number(value).toLocaleString(undefined, { maximumFractionDigits: 2 })}${unit}`, title]} />
              <Bar dataKey={key} fill="var(--color-accent)" maxBarSize={38}
                isAnimationActive={!reducedMotion} animationDuration={650} animationEasing="ease-out" />
            </BarChart>
          </ResponsiveContainer>
        </section>
      ))}
    </div>
  );
}

export function PolicyTimelineChart({
  result, policy, node, threshold,
}: {
  result: SimulationResponse;
  policy: Policy;
  node: number;
  threshold: number;
}) {
  const timeline = result.timeline.policies[policy];
  if (!timeline) return <p className="status-message">No timeline was returned.</p>;
  const key = `node_${node}`;
  const data = timeline.points.map((point) => ({ ...point, [key]: Number(point[key] ?? 0) * 100 }));
  const events = timeline.migrations
    .filter((event) => event.source === node)
    .map((event) => {
      const nearest = timeline.points.reduce((best, point) =>
        Math.abs(point.slot - event.slot) < Math.abs(best.slot - event.slot) ? point : best,
      timeline.points[0]);
      return { x: event.slot, y: Number(nearest[key] ?? 0) * 100 };
    });
  return (
    <>
      <div className="chart-caption">{timeline.total_points.toLocaleString()} simulated minutes, {timeline.migrations.length} moves, computer {node + 1}</div>
      <ResponsiveContainer width="100%" height={360}>
        <ComposedChart data={data} margin={{ top: 12, right: 18, bottom: 12, left: 6 }}>
          <CartesianGrid vertical={false} stroke="var(--color-rule)" />
          <XAxis dataKey="slot" type="number" domain={["dataMin", "dataMax"]} tickLine={false} axisLine={{ stroke: "var(--color-rule-2)" }} label={{ value: "Minutes into the simulation", position: "insideBottom", offset: -4 }} />
          <YAxis domain={[0, 100]} tickFormatter={(value) => `${value}%`} tickLine={false} axisLine={false} width={48} />
          <Tooltip formatter={(value) => [`${Number(value).toFixed(1)}%`, "Computer busy"]} />
          <ReferenceLine y={threshold * 100} stroke="var(--color-muted)" strokeDasharray="4 4" label={`Busy limit ${(threshold * 100).toFixed(0)}%`} />
          <Line dataKey={key} name={`Computer ${node + 1}`} stroke="var(--color-accent)" strokeWidth={2} dot={false} isAnimationActive={false} />
          <Scatter data={events} dataKey="y" name="Migration from this node" fill="var(--color-ink)" />
        </ComposedChart>
      </ResponsiveContainer>
      <p className="chart-caption">The line shows how busy this computer was. The dashed line is the move limit; markers show moves away from it.</p>
    </>
  );
}

export function SimulationComparison({ result }: { result: SimulationResponse }) {
  const summaryRows = result.summary.map(({ Policy, mean }) => ({
    Policy,
    busyPct: mean["Overload time (%)"],
    responseMs: mean["Avg latency (ms)"],
    migrations: mean.Migrations,
  }));
  const chartRows = result.summary.map((item) => ({
    Policy: item.Policy,
    "Overload time (%)": item.mean["Overload time (%)"],
    Migrations: item.mean.Migrations,
    "Total migration cost (s)": item.mean["Total migration cost (s)"],
    "Avg latency (ms)": item.mean["Avg latency (ms)"],
  }));
  return (
    <>
      <PolicyResultsTable rows={summaryRows} />
      <p className="chart-caption">These are averages across the selected test runs. Less busy time and shorter waits are better; moves also use transfer time.</p>
      <ComparisonCharts rows={chartRows} />
    </>
  );
}

export function ForecastComparison({ rows }: { rows: Record<string, unknown>[] }) {
  const reducedMotion = useReducedMotion();
  const modelLabels: Record<string, string> = {
    MovingAverage: "Recent average",
    ARIMA: "Statistical forecast",
    GRU: "Neural forecast A",
    LSTM: "Neural forecast B",
  };
  const data = rows.map((row, index) => ({
    Approach: modelLabels[String(row.Model)] ?? row.Model,
    "Typical error (points)": Number(row.RMSE) * 100,
    "Average error (points)": Number(row.MAE) * 100,
    fill: seriesColors[index % seriesColors.length],
  }));
  return (
    <ResponsiveContainer width="100%" height={260}>
      <BarChart data={data} margin={{ top: 8, right: 12, bottom: 4, left: 0 }}>
        <CartesianGrid vertical={false} stroke="var(--color-rule)" />
        <XAxis dataKey="Approach" interval={0} tick={{ fontSize: 11 }} tickLine={false} axisLine={{ stroke: "var(--color-rule-2)" }} />
        <YAxis tickLine={false} axisLine={false} width={48} />
        <Tooltip />
        <Legend />
        <Bar dataKey="Typical error (points)" fill="var(--color-accent)"
          isAnimationActive={!reducedMotion} animationDuration={650} animationEasing="ease-out" />
        <Bar dataKey="Average error (points)" fill="var(--color-ink-2)"
          isAnimationActive={!reducedMotion} animationDuration={650} animationEasing="ease-out" />
      </BarChart>
    </ResponsiveContainer>
  );
}
