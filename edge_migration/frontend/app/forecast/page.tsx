"use client";

import { useEffect, useState } from "react";
import { ForecastComparison } from "@/components/charts";
import { DataTable, ErrorMessage, LoadingMessage, PageHeading } from "@/components/shell";
import { api, figureUrl } from "@/lib/api";

export default function ForecastPage() {
  const [rows, setRows] = useState<Record<string, unknown>[]>([]);
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(true);
  const approachInfo: Record<string, string> = {
    MovingAverage: "Repeats the recent average conditions.",
    ARIMA: "Extends a statistical pattern from recent conditions.",
    GRU: "A neural-network forecast, type A.",
    LSTM: "A neural-network forecast, type B.",
  };
  const readableRows = rows.map((row) => ({
    "Forecast approach": approachInfo[String(row.Model)] ?? row.Model,
    "Typical difference from actual": `${(Number(row.RMSE) * 100).toFixed(2)} percentage points`,
    "Average difference from actual": `${(Number(row.MAE) * 100).toFixed(2)} percentage points`,
    "Time to make a forecast": `${Number(row["Inference (ms)"]).toFixed(3)} ms`,
  }));
  useEffect(() => {
    api<Record<string, unknown>[]>("/results/forecast-accuracy")
      .then(setRows).catch((reason: Error) => setError(reason.message))
      .finally(() => setLoading(false));
  }, []);
  return (
    <>
      <PageHeading eyebrow="HOW PREDICTIONS WORK" title="Can the simulator see a busy period coming?"
        detail="A forecast estimates how busy each computer will be soon. Better estimates can give the moving rules more time to act." />
      <section className="sheet">
        <div className="section-heading"><h2>How close were the estimates?</h2><p>Smaller differences mean a closer forecast.</p></div>
        {loading && <LoadingMessage />}{error && <ErrorMessage message={error} />}
        {!loading && !error && <>
          <DataTable rows={readableRows} />
          <p className="chart-caption">Load is shown from 0% to 100%. A difference of one percentage point is a small miss; lower values are better.</p>
          <ForecastComparison rows={rows} />
          <details className="advanced-details">
            <summary>Show technical model names and error measures</summary>
            <DataTable rows={rows} />
          </details>
        </>}
      </section>
      <section className="sheet">
        <div className="section-heading"><h2>Pictures from the same experiment</h2><p>These figures were generated from the saved simulation run.</p></div>
        <div className="image-grid">
          <figure className="figure"><img src={figureUrl("load_forecast.png")} alt="Actual and estimated computer workload over time" /><figcaption>Estimated workload compared with what actually happened.</figcaption></figure>
          <figure className="figure"><img src={figureUrl("forecast_rmse_horizon.png")} alt="Forecast difference at different look-ahead times" /><figcaption>How prediction error changes as the look-ahead gets longer.</figcaption></figure>
        </div>
      </section>
    </>
  );
}
