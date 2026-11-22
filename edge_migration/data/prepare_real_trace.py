"""Process real Google ClusterData 2011 task_usage tables into 4-channel edge traces."""
from __future__ import annotations

import gzip
from pathlib import Path
import numpy as np
import pandas as pd


def prepare_google_trace(real_dir: Path, out_dir: Path, n_nodes: int = 10, n_services: int = 30) -> Path:
    files = sorted(real_dir.glob("part-*.csv.gz"))
    if not files:
        raise FileNotFoundError(f"No part-*.csv.gz files found in {real_dir}")

    print(f"Processing {len(files)} Google ClusterData partition files...")

    # First pass: find top 10 machine IDs and top 30 job IDs by activity
    machine_counts: dict[str, int] = {}
    job_counts: dict[str, int] = {}

    with gzip.open(files[0], "rt") as f:
        for idx, line in enumerate(f):
            parts = line.strip().split(",")
            m = parts[4]
            j = parts[2]
            machine_counts[m] = machine_counts.get(m, 0) + 1
            job_counts[j] = job_counts.get(j, 0) + 1
            if idx >= 500000:
                break

    top_machines = [m for m, _ in sorted(machine_counts.items(), key=lambda x: x[1], reverse=True)[:n_nodes]]
    top_jobs = [j for j, _ in sorted(job_counts.items(), key=lambda x: x[1], reverse=True)[:n_services]]

    machine_to_idx = {m: i for i, m in enumerate(top_machines)}
    job_to_idx = {j: i for i, j in enumerate(top_jobs)}

    print(f"Selected {len(top_machines)} machines (nodes) and {len(top_jobs)} jobs (services).")

    # Time slot aggregation: 1-minute bins = 60,000,000 microseconds
    SLOT_US = 60_000_000

    # Determine global time range
    with gzip.open(files[0], "rt") as f:
        first_line = f.readline().strip().split(",")
        base_time = float(first_line[0])

    slot_accum: dict[tuple[int, int], list[float]] = {}
    slot_counts: dict[tuple[int, int], int] = {}
    max_slot = 0

    for file_path in files:
        print(f"Reading {file_path.name}...")
        with gzip.open(file_path, "rt") as f:
            for line in f:
                parts = line.strip().split(",")
                st = float(parts[0])
                m = parts[4]
                j = parts[2]

                series_idx = -1
                if m in machine_to_idx:
                    series_idx = machine_to_idx[m] # 0 .. 9 (node backgrounds)
                elif j in job_to_idx:
                    series_idx = n_nodes + job_to_idx[j] # 10 .. 39 (services)

                if series_idx == -1:
                    continue

                slot = int((st - base_time) // SLOT_US)
                if slot < 0:
                    continue
                max_slot = max(max_slot, slot)

                # Column mappings from Google ClusterData 2011 task_usage schema:
                # Field 5: mean CPU usage rate -> CPU channel
                # Field 6: canonical memory usage -> Memory channel
                # Field 8: unmapped page cache memory / sample count -> Request rate proxy
                # Field 11: mean disk I/O time -> Network I/O transfer proxy
                try:
                    cpu = float(parts[5]) if parts[5] else 0.0
                    mem = float(parts[6]) if parts[6] else 0.0
                    req = float(parts[8]) if len(parts) > 8 and parts[8] else 0.0001
                    net = float(parts[11]) if len(parts) > 11 and parts[11] else 0.0
                except (ValueError, IndexError):
                    continue

                key = (series_idx, slot)
                if key not in slot_accum:
                    slot_accum[key] = [cpu, mem, req, net]
                    slot_counts[key] = 1
                else:
                    slot_accum[key][0] += cpu
                    slot_accum[key][1] += mem
                    slot_accum[key][2] += req
                    slot_accum[key][3] += net
                    slot_counts[key] += 1

    total_slots = max_slot + 1
    print(f"Total aggregated slots: {total_slots} (covering {total_slots} minutes = {total_slots/60:.2f} hours)")

    # Construct dense array: (n_series, total_slots, 4)
    n_series = n_nodes + n_services
    traces = np.zeros((n_series, total_slots, 4), dtype=np.float32)

    for (s_idx, slot), vals in slot_accum.items():
        cnt = slot_counts[(s_idx, slot)]
        traces[s_idx, slot] = [v / cnt for v in vals]

    # Fill any empty slots using forward fill + interpolation
    for s_idx in range(n_series):
        for c in range(4):
            series = traces[s_idx, :, c]
            zero_mask = series == 0
            if zero_mask.all():
                traces[s_idx, :, c] = 0.05
            elif zero_mask.any():
                # Forward fill
                last_val = series[~zero_mask][0]
                for t in range(total_slots):
                    if series[t] == 0:
                        series[t] = last_val
                    else:
                        last_val = series[t]

    # Normalize each channel to [0, 1] using min-max scaling across entities
    for c in range(4):
        c_min = traces[:, :, c].min()
        c_max = traces[:, :, c].max()
        span = c_max - c_min
        if span > 1e-8:
            traces[:, :, c] = (traces[:, :, c] - c_min) / span
        traces[:, :, c] = np.clip(traces[:, :, c], 0, 1)

    # Scale node backgrounds to sensible edge range [0.10, 0.35]
    # and service demands to [0.05, 0.25] so sum produces realistic overloads
    for i in range(n_nodes):
        traces[i] = traces[i] * 0.35 + 0.10
    for i in range(n_nodes, n_series):
        traces[i] = traces[i] * 0.25 + 0.05

    out_dir.mkdir(parents=True, exist_ok=True)
    out_npz = out_dir / "traces_real_google.npz"
    np.savez_compressed(out_npz, traces=traces)

    # Save node 0 representative CSV
    df0 = pd.DataFrame(traces[0], columns=["cpu", "memory", "request_rate", "network"])
    df0.to_csv(out_dir / "google_cluster_trace.csv", index=False)

    print(f"Successfully generated real Google cluster trace: {traces.shape} saved to {out_npz}")
    return out_npz


if __name__ == "__main__":
    base = Path(__file__).parents[1]
    prepare_google_trace(base / "data" / "real", base / "data")
