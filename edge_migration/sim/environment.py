"""Plain-Python node/service state, resource accounting, latency, and migration."""
from __future__ import annotations

from dataclasses import dataclass
from math import ceil, hypot
from typing import Any

import numpy as np


@dataclass(frozen=True)
class Node:
    node_id: int
    x_km: float
    y_km: float
    memory_capacity_mb: float


@dataclass(frozen=True)
class Service:
    service_id: int
    state_size_mb: float
    memory_mb: float
    user_x_km: float
    user_y_km: float


@dataclass
class Migration:
    service_id: int
    source: int
    destination: int
    complete_slot: int
    cost_seconds: float


class Environment:
    """Owns the single-host invariant and temporary destination reservations."""

    def __init__(self, config: dict[str, Any], traces: np.ndarray, seed: int,
                 initial_hosts: list[int] | None = None):
        self.config, self.traces = config, traces
        self.rng = np.random.default_rng(seed)
        sim = config["simulation"]
        node_cfg, service_cfg = sim["nodes"], sim["services"]
        n_nodes, n_services = int(config["data"]["nodes"]), int(config["data"]["services"])
        area = float(node_cfg["area_km"])
        cap_lo, cap_hi = map(float, node_cfg["memory_capacity_mb"])
        self.nodes = [Node(i, float(self.rng.uniform(0, area)), float(self.rng.uniform(0, area)),
                           float(self.rng.uniform(cap_lo, cap_hi))) for i in range(n_nodes)]
        state_lo, state_hi = map(float, service_cfg["state_size_mb"])
        mem_lo, mem_hi = map(float, service_cfg["memory_requirement_mb"])
        self.services = [Service(i, float(self.rng.uniform(state_lo, state_hi)),
                                 float(self.rng.uniform(mem_lo, mem_hi)),
                                 float(self.rng.uniform(0, area)), float(self.rng.uniform(0, area)))
                         for i in range(n_services)]
        self.hosts = list(initial_hosts) if initial_hosts is not None else self._initial_assignment()
        if len(self.hosts) != n_services or any(not 0 <= node < n_nodes for node in self.hosts):
            raise ValueError("initial_hosts must map every service to a valid node")
        self.used_memory = np.zeros(n_nodes, dtype=float)
        for service_id, node in enumerate(self.hosts):
            self.used_memory[node] += self.services[service_id].memory_mb
        for node_id, node in enumerate(self.nodes):
            if self.used_memory[node_id] > node.memory_capacity_mb:
                raise ValueError(f"Initial placement exceeds node {node_id} memory")
        self.reserved_memory = np.zeros(n_nodes, dtype=float)
        self.pending: dict[int, Migration] = {}
        self.migrations: list[Migration] = []
        self.bandwidth_mbps = self.rng.uniform(*map(float, sim["network"]["bandwidth_mbps"]),
                                               (n_nodes, n_nodes))
        self.bandwidth_mbps = (self.bandwidth_mbps + self.bandwidth_mbps.T) / 2
        np.fill_diagonal(self.bandwidth_mbps, np.inf)

    def _initial_assignment(self) -> list[int]:
        assignments = self.rng.integers(len(self.nodes), size=len(self.services)).tolist()
        # Capacities are intentionally generous; this still makes placement valid for custom configs.
        used = np.zeros(len(self.nodes))
        for service_id, node_id in enumerate(assignments):
            required = self.services[service_id].memory_mb
            if used[node_id] + required > self.nodes[node_id].memory_capacity_mb:
                available = [i for i, node in enumerate(self.nodes)
                             if used[i] + required <= node.memory_capacity_mb]
                if not available:
                    raise ValueError("No memory-feasible initial placement for all services")
                node_id = int(available[0])
                assignments[service_id] = node_id
            used[node_id] += required
        return assignments

    def services_on(self, node_id: int, include_migrating: bool = False) -> list[int]:
        return [s for s, host in enumerate(self.hosts)
                if host == node_id and (include_migrating or s not in self.pending)]

    def available_memory(self, node_id: int) -> float:
        return self.nodes[node_id].memory_capacity_mb - self.used_memory[node_id] - self.reserved_memory[node_id]

    def distance_km(self, x: float, y: float, node_id: int) -> float:
        node = self.nodes[node_id]
        return hypot(x - node.x_km, y - node.y_km)

    def migration_cost_seconds(self, service_id: int, source: int, destination: int) -> float:
        service = self.services[service_id]
        transfer_seconds = service.state_size_mb * 8 / self.bandwidth_mbps[source, destination]
        return float(transfer_seconds + float(self.config["simulation"]["network"]["downtime_seconds"]))

    def migration_slots(self, service_id: int, source: int, destination: int) -> int:
        seconds = self.services[service_id].state_size_mb * 8 / self.bandwidth_mbps[source, destination]
        return max(1, ceil(seconds / 60))

    def start_migration(self, service_id: int, destination: int, slot: int) -> Migration:
        if service_id in self.pending:
            raise ValueError(f"Service {service_id} is already migrating")
        source = self.hosts[service_id]
        if source == destination or self.available_memory(destination) < self.services[service_id].memory_mb:
            raise ValueError("Migration destination is invalid or lacks memory")
        duration = self.migration_slots(service_id, source, destination)
        migration = Migration(service_id, source, destination, slot + duration,
                              self.migration_cost_seconds(service_id, source, destination))
        self.reserved_memory[destination] += self.services[service_id].memory_mb
        self.pending[service_id] = migration
        self.migrations.append(migration)
        return migration

    def complete_migrations(self, slot: int) -> list[Migration]:
        completed = []
        for service_id, migration in list(self.pending.items()):
            if migration.complete_slot <= slot:
                memory = self.services[service_id].memory_mb
                self.used_memory[migration.source] -= memory
                self.reserved_memory[migration.destination] -= memory
                self.used_memory[migration.destination] += memory
                self.hosts[service_id] = migration.destination
                del self.pending[service_id]
                completed.append(migration)
        return completed

    def node_channels(self, slot_values: np.ndarray, hosts: list[int] | None = None) -> np.ndarray:
        """Sum background and hosted service channels; clip per-channel node utilization."""
        placement = self.hosts if hosts is None else hosts
        n_nodes = len(self.nodes)
        channels = np.zeros((n_nodes, 4), dtype=np.float32)
        for node_id in range(n_nodes):
            channels[node_id] = slot_values[node_id, :4]
        offset = len(self.nodes)
        for service_id, node_id in enumerate(placement):
            channels[node_id] += slot_values[offset + service_id, :4]
        return np.clip(channels, 0, 1)

    def latency_ms(self, service_id: int, node_id: int, load: float, degraded: bool = False) -> float:
        service = self.services[service_id]
        node_cfg = self.config["simulation"]["nodes"]
        distance = self.distance_km(service.user_x_km, service.user_y_km, node_id)
        network = float(node_cfg["base_latency_ms"]) + float(node_cfg["latency_ms_per_km"]) * distance
        processing = float(node_cfg["processing_p0_ms"]) / (1 - min(float(load), 0.95))
        migration_penalty = float(self.config["simulation"]["services"]["degraded_latency_ms"]) if degraded else 0
        return network + processing + migration_penalty
