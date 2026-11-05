export type Scenario = "default" | "slow_links";
export type Policy = "RT" | "RT-CD" | "FLL" | "FCM";

export const policyInfo: Record<Policy, { label: string; description: string }> = {
  RT: {
    label: "React after overload",
    description: "Moves an app only after its current computer gets too busy.",
  },
  "RT-CD": {
    label: "React, then pause",
    description: "Uses the same rule, but avoids moving the same app again too soon.",
  },
  FLL: {
    label: "Move before overload",
    description: "Uses a forecast and chooses the computer expected to be quietest.",
  },
  FCM: {
    label: "Balance the trade-offs",
    description: "Forecasts demand and considers response time, transfer effort, and move timing.",
  },
};

export const scenarioInfo: Record<Scenario, { label: string; description: string }> = {
  default: {
    label: "Typical connection speed",
    description: "Data moves between computers at the usual simulated network speed.",
  },
  slow_links: {
    label: "Slower connections",
    description: "Transfers take longer, so a move may need to start earlier.",
  },
};

export type PerformanceRow = {
  Policy: Policy;
  "Overload time (%)": number;
  "SLA violation (%)": number;
  "Avg latency (ms)": number;
  Migrations: number;
  "Total migration cost (s)": number;
  "Unnecessary migrations (%)": number;
  "Load std dev": number;
  "No feasible target events": number;
  [key: string]: string | number;
};

export type SimulationRequest = {
  scenario: Scenario;
  seeds: number[];
  policies: Policy[];
  overrides: {
    node_count: number;
    theta_hi: number;
    theta_safe: number;
    persistence: number;
    cooldown_slots: number;
    tau: number;
    alpha: number;
    beta: number;
    gamma: number;
    horizon: number;
  };
};

export type SimulationRun = {
  seed: number;
  Policy: Policy;
  [metric: string]: number | string;
};

export type TimelinePoint = { slot: number; [node: string]: number };
export type MigrationEvent = {
  slot: number;
  source: number;
  destination: number;
  service: number;
  complete_slot: number;
};
export type PolicyTimeline = {
  points: TimelinePoint[];
  migrations: MigrationEvent[];
  total_points: number;
};
export type SimulationResponse = {
  scenario: Scenario;
  per_seed: SimulationRun[];
  summary: {
    Policy: Policy;
    mean: Record<string, number>;
    std: Record<string, number>;
  }[];
  timeline: { seed: number; node_count: number; policies: Partial<Record<Policy, PolicyTimeline>> };
};

export function responseSeconds(milliseconds: number): string {
  return `${(milliseconds / 1000).toFixed(3)} s`;
}

const apiBase = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

export async function api<T>(path: string, init?: RequestInit): Promise<T> {
  let response: Response;
  try {
    response = await fetch(`${apiBase}${path}`, {
      ...init,
      headers: { "Content-Type": "application/json", ...init?.headers },
      cache: "no-store",
    });
  } catch {
    throw new Error("The API is unavailable. Start the FastAPI server on port 8000 and try again.");
  }
  if (!response.ok) {
    const body = await response.json().catch(() => null);
    const detail = body?.detail;
    throw new Error(Array.isArray(detail) ? detail.map((item) => item.msg).join("; ") : detail ?? response.statusText);
  }
  return response.json() as Promise<T>;
}

export function figureUrl(name: string): string {
  return `${apiBase}/figures/${name}`;
}
