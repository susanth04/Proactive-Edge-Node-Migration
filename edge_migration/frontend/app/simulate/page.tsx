"use client";

import { useEffect, useState } from "react";
import { PolicyTimelineChart, SimulationComparison } from "@/components/charts";
import { MigrationFlow } from "@/components/migration-flow";
import { ErrorMessage, LoadingMessage, PageHeading } from "@/components/shell";
import { api, policyInfo, scenarioInfo, type Policy, type Scenario, type SimulationResponse } from "@/lib/api";

type Config = {
  node_count: number;
  node_count_range: [number, number];
  seeds: number[];
  policies: Policy[];
  simulation: { theta_hi: number; theta_safe: number; fcm: { persistence: number; cooldown_slots: number; tau: number; alpha: number; beta: number; gamma: number } };
  forecast: { horizon: number };
};
type FormValues = {
  node_count: number;
  theta_hi: number; theta_safe: number; persistence: number; cooldown_slots: number;
  tau: number; alpha: number; beta: number; gamma: number; horizon: number;
};

const initial: FormValues = {
  node_count: 10,
  theta_hi: 0.85, theta_safe: 0.7, persistence: 3, cooldown_slots: 10,
  tau: 0.03, alpha: 0.5, beta: 0.3, gamma: 0.2, horizon: 10,
};
const fields: { key: keyof FormValues; label: string; hint: string; min: number; max: number; step: number; scale?: number }[] = [
  { key: "theta_hi", label: "Move an app when a computer reaches (%)", hint: "This is the busy limit.", min: 50, max: 100, step: 1, scale: 100 },
  { key: "theta_safe", label: "Keep destination computers below (%)", hint: "A new home should stay comfortably below this level.", min: 50, max: 100, step: 1, scale: 100 },
  { key: "persistence", label: "Busy forecast must last (minutes)", hint: "A longer wait avoids reacting to a brief spike.", min: 1, max: 10, step: 1 },
  { key: "cooldown_slots", label: "Pause before moving the same app again (minutes)", hint: "Prevents an app from being moved back and forth.", min: 0, max: 60, step: 1 },
  { key: "tau", label: "Minimum benefit before a move", hint: "Advanced setting: reject moves with too little predicted gain.", min: 0, max: 0.2, step: 0.01 },
  { key: "alpha", label: "Workload importance (Alpha)", hint: "How much destination busyness matters to the proposed rule.", min: 0, max: 1, step: 0.05 },
  { key: "beta", label: "Response-time importance (Beta)", hint: "How much user response time matters to the proposed rule.", min: 0, max: 1, step: 0.05 },
  { key: "gamma", label: "Transfer-effort importance (Gamma)", hint: "How much data-transfer effort matters to the proposed rule.", min: 0, max: 1, step: 0.05 },
  { key: "horizon", label: "Look ahead (minutes)", hint: "How far ahead the simulator checks its forecast.", min: 1, max: 10, step: 1 },
];

export default function SimulatePage() {
  const [config, setConfig] = useState<Config | null>(null);
  const [scenario, setScenario] = useState<Scenario>("default");
  const [values, setValues] = useState<FormValues>(initial);
  const [seeds, setSeeds] = useState<number[]>([7]);
  const [policies, setPolicies] = useState<Policy[]>(["RT", "RT-CD", "FLL", "FCM"]);
  const [result, setResult] = useState<SimulationResponse | null>(null);
  const [policy, setPolicy] = useState<Policy>("FCM");
  const [node, setNode] = useState(0);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [resultThreshold, setResultThreshold] = useState(initial.theta_hi);
  const [minNodes, maxNodes] = config?.node_count_range ?? [2, 20];

  useEffect(() => {
    api<Config>("/config").then((loaded) => {
      setConfig(loaded);
      setSeeds([loaded.seeds[0]]);
      setPolicies(loaded.policies);
      setValues({
        ...initial,
        node_count: loaded.node_count,
        theta_hi: loaded.simulation.theta_hi,
        theta_safe: loaded.simulation.theta_safe,
        persistence: loaded.simulation.fcm.persistence,
        cooldown_slots: loaded.simulation.fcm.cooldown_slots,
        tau: loaded.simulation.fcm.tau,
        alpha: loaded.simulation.fcm.alpha,
        beta: loaded.simulation.fcm.beta,
        gamma: loaded.simulation.fcm.gamma,
        horizon: loaded.forecast.horizon,
      });
    }).catch((reason: Error) => setError(reason.message));
  }, []);

  function toggle<T>(items: T[], value: T, setter: (next: T[]) => void, limit?: number) {
    if (items.includes(value)) setter(items.filter((item) => item !== value));
    else if (!limit || items.length < limit) setter([...items, value]);
  }

  async function submit(event: React.FormEvent<HTMLFormElement>) {
    event.preventDefault();
    setError("");
    setLoading(true);
    try {
      const response = await api<SimulationResponse>("/simulate", {
        method: "POST",
        body: JSON.stringify({ scenario, seeds, policies, overrides: values }),
      });
      setResult(response);
      setResultThreshold(values.theta_hi);
      setNode(0);
      setPolicy(policies.includes("FCM") ? "FCM" : policies[0]);
    } catch (reason) {
      setError((reason as Error).message);
    } finally {
      setLoading(false);
    }
  }

  return (
    <>
      <PageHeading eyebrow="TRY IT YOURSELF" title="See what happens when apps move"
        detail="Choose how many computers to use, run the comparison, then click any computer to explore its workload and app moves." />
      <section className="sheet">
        <form onSubmit={submit}>
          <div className="simple-run">
            <label className="field">Connection conditions
              <select className="select" value={scenario} onChange={(event) => setScenario(event.target.value as Scenario)}>
                {(Object.keys(scenarioInfo) as Scenario[]).map((value) => <option key={value} value={value}>{scenarioInfo[value].label}</option>)}
              </select>
            </label>
            <p className="field-hint">{scenarioInfo[scenario].description}</p>
          </div>
          <p className="run-intro">The default compares all four approaches using one test run. You can start right away, or open the optional settings to change the details.</p>
          <div className="node-config">
            <div>
              <strong>Computers in the experiment</strong>
              <small>Each added computer gets its own generated workload. Run the comparison to see its effect.</small>
            </div>
            <div className="node-stepper" aria-label="Choose the number of simulated computers">
              <button className="button" type="button" aria-label="Remove one computer"
                disabled={loading || values.node_count <= minNodes}
                onClick={() => setValues({ ...values, node_count: values.node_count - 1 })}>Remove</button>
              <output aria-live="polite">{values.node_count} computers</output>
              <button className="button" type="button" aria-label="Add one computer"
                disabled={loading || values.node_count >= maxNodes}
                onClick={() => setValues({ ...values, node_count: values.node_count + 1 })}>Add</button>
            </div>
          </div>
          {result && result.timeline.node_count !== values.node_count &&
            <p className="status-message" role="status">Run the comparison again to see results for {values.node_count} computers.</p>}
          <div className="run-action">
            <button className="button button-primary" type="submit" disabled={loading || !seeds.length || !policies.length || values.alpha + values.beta + values.gamma <= 0 || values.theta_safe > values.theta_hi}>
              {loading ? "Comparing approaches…" : "Compare approaches"}
            </button>
            <span>Usually takes a few seconds on this computer.</span>
          </div>
          <details className="advanced-details">
            <summary>Optional settings for a more detailed experiment</summary>
            <p className="chart-caption">You can leave these as they are and still run the comparison.</p>
            <div className="section-heading"><h2>When should an app move?</h2></div>
            <div className="form-grid">
              {fields.map(({ key, label, hint, min, max, step, scale = 1 }) => (
                <label className="field" key={key}>
                  <span>{label}</span>
                  <input type="range" min={min} max={max} step={step} value={values[key] * scale}
                    onChange={(event) => setValues({ ...values, [key]: Number(event.target.value) / scale })} />
                  <input className="input" type="number" min={min} max={max} step={step} value={values[key] * scale}
                    onChange={(event) => setValues({ ...values, [key]: Number(event.target.value) / scale })} />
                  <small>{hint}</small>
                </label>
              ))}
            </div>
            {values.theta_safe > values.theta_hi && <ErrorMessage message="The destination limit should not be higher than the busy limit." />}
            {values.alpha + values.beta + values.gamma <= 0 && <ErrorMessage message="At least one of the three trade-off weights must be above zero." />}
            <div className="section-heading"><h2>Repeat the experiment</h2><p>Select up to five test runs.</p></div>
            <div className="choice-list">
              {(config?.seeds ?? [7, 17, 27, 37, 47]).map((seed) => (
                <label key={seed}><input type="checkbox" checked={seeds.includes(seed)} onChange={() => toggle(seeds, seed, setSeeds, 5)} /> Run {seed}</label>
              ))}
            </div>
            <div className="section-heading"><h2>Compare these approaches</h2></div>
            <div className="choice-list policy-choice-list">
              {(["RT", "RT-CD", "FLL", "FCM"] as Policy[]).map((name) => (
                <label className="policy-choice" key={name}>
                  <input type="checkbox" checked={policies.includes(name)} onChange={() => toggle(policies, name, setPolicies)} />
                  <span><strong>{policyInfo[name].label}</strong><small>{policyInfo[name].description}</small></span>
                </label>
              ))}
            </div>
          </details>

        </form>
        {error && <ErrorMessage message={error} />}
        {loading && <p className="status-message" role="status">Running the selected comparisons. This can take a little while…</p>}
      </section>
      {result && config && <section className="sheet">
        <div className="section-heading"><h2>What happened?</h2><p>{scenarioInfo[result.scenario].label} · {result.per_seed.length} comparisons</p></div>
        <SimulationComparison result={result} />
        <div className="section-heading"><h2>Watch apps move</h2><p>Choose a rule, play through the moves, and click a computer to see its details.</p></div>
        <div className="control-row">
          <label htmlFor="timeline-policy">Moving rule</label>
          <select className="select" id="timeline-policy" value={policy} onChange={(event) => setPolicy(event.target.value as Policy)}>
            {policies.map((name) => <option key={name} value={name}>{policyInfo[name].label}</option>)}
          </select>
        </div>
        <div className="section-heading"><h2>Explore computer workloads</h2><p>The timeline and move details follow the computer you select.</p></div>
        <MigrationFlow result={result} policy={policy} selectedNode={node} onSelectNode={setNode} />
        <div className="section-heading"><h2>How busy was this computer over time?</h2><p>The timeline shows the first selected simulator run.</p></div>
        <PolicyTimelineChart result={result} policy={policy} node={node} threshold={resultThreshold} />
      </section>}
    </>
  );
}
