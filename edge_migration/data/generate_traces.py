"""Generate deterministic, correlated four-channel edge workload traces."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open(encoding="utf-8") as stream:
        return yaml.safe_load(stream)


def generate_traces(config: dict[str, Any], seed: int) -> tuple[np.ndarray, pd.DataFrame]:
    """Return [backgrounds + services, slots, 4 channels] traces in [0, 1]."""
    rng = np.random.default_rng(seed)
    data_cfg, sim_cfg = config["data"], config["simulation"]
    n_nodes, n_services = int(data_cfg["nodes"]), int(data_cfg["services"])
    slots_day, slots = int(data_cfg["slots_per_day"]), int(data_cfg["days"]) * int(data_cfg["slots_per_day"])
    n_series = n_nodes + n_services
    channels = len(data_cfg["channels"])
    weights = np.asarray(sim_cfg["weights"], dtype=np.float32)
    shared = rng.normal(0, 1, slots)
    tod = np.arange(slots) % slots_day
    weekly = np.sin(2 * np.pi * np.arange(slots) / (7 * slots_day))
    values = np.empty((n_series, slots, channels), dtype=np.float32)
    labels = []

    for series in range(n_series):
        is_service = series >= n_nodes
        scale = np.asarray(sim_cfg["services"]["channel_scale"] if is_service
                           else data_cfg["background_channel_scale"])
        heterogeneity = rng.uniform(*map(float, data_cfg["entity_scale_range"]), channels)
        phase = rng.uniform(*map(float, data_cfg["phase_range"]), channels)
        correlated = rng.uniform(*map(float, data_cfg["channel_correlation_range"]), channels)
        daily_amp = rng.uniform(*map(float, data_cfg["daily_amplitude_range"]), channels)
        weekly_amp = rng.uniform(*map(float, data_cfg["weekly_amplitude_range"]), channels)
        noise_cfg = data_cfg["shared_noise_std"]["service" if is_service else "background"]
        latent = rng.normal(0, float(noise_cfg), (slots, channels))
        noise = rng.normal(0, float(data_cfg["channel_noise_std"]), (slots, channels))
        common = shared[:, None] * correlated[None, :] * float(data_cfg["shared_signal_scale"])
        base = scale[None, :] * heterogeneity[None, :] * (
            1 + daily_amp[None, :] * np.sin(2 * np.pi * tod[:, None] / slots_day + phase[None, :])
            + weekly_amp[None, :] * weekly[:, None]
        )
        trace = base + latent + noise + common

        lo, hi = map(int, data_cfg["burst_count_range"])
        dur_lo, dur_hi = map(int, data_cfg["burst_duration_range"])
        for _ in range(int(rng.integers(lo, hi + 1))):
            start = int(rng.integers(0, max(1, slots - dur_hi)))
            duration = int(rng.integers(dur_lo, dur_hi + 1))
            intensity = float(rng.uniform(*map(float, data_cfg["burst_intensity_range"])))
            affected = rng.uniform(*map(float, data_cfg["burst_channel_effect_range"]), channels)
            trace[start:min(slots, start + duration)] += intensity * affected
        values[series] = np.clip(trace, 0, 1)
        labels.append({"series_id": series, "kind": "service" if is_service else "background",
                       "entity_id": series - n_nodes if is_service else series,
                       "load_weight": float(weights @ np.mean(values[series], axis=0))})

    return values, pd.DataFrame(labels)


def load_real_trace(path: str | Path) -> np.ndarray:
    """Load normalized CSV columns cpu,memory,request_rate,network (values in [0,1])."""
    frame = pd.read_csv(path)
    columns = ["cpu", "memory", "request_rate", "network"]
    missing = set(columns) - set(frame.columns)
    if missing:
        raise ValueError(f"Trace CSV is missing columns: {', '.join(sorted(missing))}")
    values = frame[columns].to_numpy(dtype=np.float32)
    if not len(values) or not np.isfinite(values).all() or (values < 0).any() or (values > 1).any():
        raise ValueError("Trace channels must be non-empty, finite, and normalized to [0, 1]")
    return values


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(Path(__file__).parents[1] / "config.yaml"))
    parser.add_argument("--seed", type=int)
    args = parser.parse_args()
    config = load_config(args.config)
    seed = int(args.seed if args.seed is not None else config["seed"])
    out = Path(config["paths"]["traces"])
    out.mkdir(parents=True, exist_ok=True)
    traces, metadata = generate_traces(config, seed)
    np.savez_compressed(out / f"traces_seed_{seed}.npz", traces=traces)
    metadata.to_csv(out / f"series_seed_{seed}.csv", index=False)
    print(f"Saved {traces.shape[0]} series x {traces.shape[1]} slots to {out}")


if __name__ == "__main__":
    main()
