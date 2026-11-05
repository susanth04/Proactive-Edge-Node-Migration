"""Reactive and forecast-driven migration policies."""
from __future__ import annotations

import time
from typing import Any

import numpy as np

from sim.environment import Environment


def _loads(env: Environment, predictions: np.ndarray, hosts: list[int],
           weights: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    channels = env.node_channels(predictions, hosts)
    return channels, channels @ weights


def _series_load(series: np.ndarray, weights: np.ndarray) -> float:
    return float(np.clip(series[:4], 0, 1) @ weights)


def _normalize(values: list[float]) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    span = values.max() - values.min()
    return np.zeros_like(values) if span < 1e-12 else (values - values.min()) / span


def _trigger_run(load: np.ndarray, threshold: float, persistence: int, migration_slots: int) -> bool:
    over = load >= threshold
    for start in range(max(0, len(over) - persistence + 1)):
        if start + 1 >= migration_slots and over[start:start + persistence].all():
            return True
    return False


def _select_service(env: Environment, node_id: int, values: np.ndarray, weights: np.ndarray,
                    excluded: set[int] | None = None) -> int | None:
    candidates = [s for s in env.services_on(node_id) if excluded is None or s not in excluded]
    if not candidates:
        return None
    offset = len(env.nodes)
    return max(candidates, key=lambda service: _series_load(values[offset + service], weights))


def _would_overload_without_move(env: Environment, service_id: int, slot: int, traces: np.ndarray,
                                 horizon: int, threshold: float, weights: np.ndarray) -> bool:
    """Oracle counterfactual: inspect actual future trace with current placement held fixed."""
    end = min(traces.shape[1], slot + horizon + 1)
    for future in range(slot + 1, end):
        load = env.node_channels(traces[:, future, :], env.hosts) @ weights
        if load[env.hosts[service_id]] >= threshold:
            return True
    return False


def run_policy(config: dict[str, Any], policy: str, traces: np.ndarray, predictions: np.ndarray,
               start_slot: int, *, fcm_overrides: dict[str, Any] | None = None,
               seed: int | None = None, keep_timeline: bool = False,
               gain_sink: list[float] | None = None) -> tuple[dict[str, float | str], dict[str, Any]]:
    """Simulate one policy over the prediction interval and return metrics and timeline."""
    if policy not in {"RT", "RT-CD", "FLL", "FCM"}:
        raise ValueError(f"Unknown policy: {policy}")
    sim = config["simulation"]
    weights = np.asarray(sim["weights"], dtype=np.float32)
    threshold, safe = float(sim["theta_hi"]), float(sim["theta_safe"])
    fcm = dict(sim["fcm"])
    if fcm_overrides:
        fcm.update(fcm_overrides)
        safe = float(fcm_overrides.get("theta_safe", safe))
    horizon = min(int(fcm.pop("horizon", predictions.shape[2])), predictions.shape[2])
    predictions = predictions[:, :, :horizon, :]
    env = Environment(config, traces, int(seed if seed is not None else config["seed"]))
    steps = min(len(predictions), traces.shape[1] - start_slot)
    horizon = predictions.shape[2]
    overload_slots, service_slots, sla_violations = 0, 0, 0
    latency_total, load_std_total, decision_time_total = 0.0, 0.0, 0.0
    unnecessary = 0
    node_timeline = np.zeros((steps, len(env.nodes)), dtype=np.float32)
    event_slots: list[int] = []
    event_nodes: list[int] = []
    unavailable = 0
    last_migration: dict[int, int] = {}
    migration_durations: list[int] = []

    for local_slot in range(steps):
        slot = start_slot + local_slot
        env.complete_migrations(slot)
        actual = env.node_channels(traces[:, slot, :], env.hosts)
        actual_loads = actual @ weights
        node_timeline[local_slot] = actual_loads
        overload_slots += int(np.count_nonzero(actual_loads >= threshold))
        load_std_total += float(np.std(actual_loads))

        timer = time.perf_counter()
        values = predictions[local_slot]
        planned_hosts = env.hosts.copy()
        predicted_by_node = np.empty((len(env.nodes), horizon), dtype=np.float32)
        for lead in range(horizon):
            _, predicted_by_node[:, lead] = _loads(env, values[:, lead, :], planned_hosts, weights)
        used_destinations: set[int] = set()

        for source in range(len(env.nodes)):
            if policy in {"RT", "RT-CD"}:
                triggered = actual_loads[source] >= threshold
            elif policy == "FLL":
                triggered = float(np.max(predicted_by_node[source])) >= threshold
            else:
                triggered = _trigger_run(predicted_by_node[source], threshold,
                                         int(fcm["persistence"]), 1)
            if not triggered:
                continue
            excluded = set()
            if policy in {"RT-CD", "FCM"}:
                cooldown = int(fcm["cooldown_slots"])
                excluded = {s for s, moved_at in last_migration.items() if slot - moved_at < cooldown}
            selection_values = traces[:, slot, :] if policy in {"RT", "RT-CD"} else values[:, 0, :]
            service_id = _select_service(env, source, selection_values, weights, excluded)
            if service_id is None:
                continue

            service = env.services[service_id]
            feasible = [node.node_id for node in env.nodes if node.node_id != source
                        and env.available_memory(node.node_id) >= service.memory_mb
                        and node.node_id not in used_destinations]
            target: int | None = None

            if policy in {"RT", "RT-CD"}:
                updated_loads = env.node_channels(traces[:, slot, :], planned_hosts) @ weights
                target = min(feasible, key=lambda n: updated_loads[n], default=None)
            elif policy == "FLL":
                destination_peaks = []
                for candidate in feasible:
                    trial = planned_hosts.copy()
                    trial[service_id] = candidate
                    peaks = []
                    for lead in range(horizon):
                        _, load = _loads(env, values[:, lead, :], trial, weights)
                        peaks.append(load[candidate])
                    destination_peaks.append(max(peaks))
                target = feasible[int(np.argmin(destination_peaks))] if feasible else None
            else:
                first_breach = np.flatnonzero(predicted_by_node[source] >= threshold)
                if not len(first_breach) or not _trigger_run(predicted_by_node[source], threshold,
                                                             int(fcm["persistence"]), 1):
                    continue
                run_start = next((i for i in range(max(0, horizon - int(fcm["persistence"]) + 1))
                                  if (predicted_by_node[source, i:i + int(fcm["persistence"])] >= threshold).all()), None)
                if run_start is None:
                    continue
                candidate_data = []
                src_before = float(np.max(predicted_by_node[source]))
                for candidate in feasible:
                    if first_breach[0] + 1 < env.migration_slots(service_id, source, candidate):
                        continue
                    trial = planned_hosts.copy()
                    trial[service_id] = candidate
                    peak_target, peak_source_after = [], []
                    for lead in range(horizon):
                        _, load = _loads(env, values[:, lead, :], trial, weights)
                        peak_target.append(float(load[candidate]))
                        peak_source_after.append(float(load[source]))
                    if max(peak_target) > safe:
                        continue
                    source_after = max(peak_source_after)
                    latency_src = env.latency_ms(service_id, source, src_before)
                    latency_dst = env.latency_ms(service_id, candidate, max(peak_target))
                    gain = src_before - source_after + float(fcm["latency_gain_weight"]) * max(
                        0.0, (latency_src - latency_dst) / float(sim["sla_latency_ms"]))
                    cost = env.migration_cost_seconds(service_id, source, candidate)
                    candidate_data.append((candidate, max(peak_target), latency_dst, cost, gain))
                if not candidate_data:
                    unavailable += 1
                    continue
                u_norm = _normalize([item[1] for item in candidate_data])
                l_norm = _normalize([item[2] for item in candidate_data])
                m_norm = _normalize([item[3] for item in candidate_data])
                scores = [float(fcm["alpha"]) * u + float(fcm["beta"]) * l + float(fcm["gamma"]) * m
                          for u, l, m in zip(u_norm, l_norm, m_norm)]
                best = candidate_data[int(np.argmin(scores))]
                if gain_sink is not None:
                    gain_sink.append(best[4])
                if best[4] < float(fcm["tau"]):
                    continue
                target = best[0]

            if target is None:
                if not feasible:
                    if policy != "FCM":
                        unavailable += 1
                continue
            if not _would_overload_without_move(env, service_id, slot, traces, horizon, threshold, weights):
                unnecessary += 1
            migration = env.start_migration(service_id, target, slot)
            migration_durations.append(migration.complete_slot - slot)
            planned_hosts[service_id] = target
            used_destinations.add(target)
            event_slots.append(local_slot)
            event_nodes.append(source)
            last_migration[service_id] = slot
            # Recompute after each accepted decision so later sources see the reserved placement.
            for lead in range(horizon):
                _, predicted_by_node[:, lead] = _loads(env, values[:, lead, :], planned_hosts, weights)

        decision_time_total += (time.perf_counter() - timer) * 1000

        for service_id, node_id in enumerate(env.hosts):
            latency = env.latency_ms(service_id, node_id, actual_loads[node_id],
                                     service_id in env.pending)
            latency_total += latency
            service_slots += 1
            sla_violations += int(actual_loads[node_id] >= threshold or latency > float(sim["sla_latency_ms"]))

    n_nodes, n_services = len(env.nodes), len(env.services)
    metrics: dict[str, float | str] = {
        "Policy": policy,
        "Overload time (%)": 100 * overload_slots / max(1, steps * n_nodes),
        "SLA violation (%)": 100 * sla_violations / max(1, service_slots),
        "Avg latency (ms)": latency_total / max(1, service_slots),
        "Migrations": float(len(env.migrations)),
        "Total migration cost (s)": float(sum(m.cost_seconds for m in env.migrations)),
        "Unnecessary migrations (%)": 100 * unnecessary / max(1, len(env.migrations)),
        "Load std dev": load_std_total / max(1, steps),
        "Decision time (ms/slot)": decision_time_total / max(1, steps),
        "No feasible target events": float(unavailable),
        "Max migration slots": float(max(migration_durations, default=0)),
        "Mean migration slots": float(np.mean(migration_durations)) if migration_durations else 0.0,
    }
    timeline = {"node_loads": node_timeline, "migration_slots": event_slots, "migration_nodes": event_nodes,
                "hosts": env.hosts.copy(), "migrations": env.migrations}
    if not keep_timeline:
        timeline = {}
    return metrics, timeline
