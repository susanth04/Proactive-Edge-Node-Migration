"""Train, evaluate, and persist the configured shared forecasting models."""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from data.generate_traces import generate_traces
from forecasting.models import make_model
from sim.environment import Environment


def make_features(traces: np.ndarray, slots_per_day: int) -> np.ndarray:
    tod = np.arange(traces.shape[1]) % slots_per_day
    angle = 2 * np.pi * tod / slots_per_day
    clock = np.stack((np.sin(angle), np.cos(angle)), axis=-1).astype(np.float32)
    return np.concatenate((traces, np.broadcast_to(clock, (*traces.shape[:2], 2))), axis=-1)


def windows(features: np.ndarray, traces: np.ndarray, start: int, stop: int,
            window: int, horizon: int, stride: int) -> tuple[np.ndarray, np.ndarray]:
    indices = np.arange(max(start + window - 1, window - 1), min(stop - horizon, traces.shape[1] - horizon), stride)
    if not len(indices):
        raise ValueError("Insufficient trace length for configured window and horizon")
    x = np.stack([features[:, t - window + 1:t + 1] for t in indices], axis=1)
    y = np.stack([traces[:, t + 1:t + horizon + 1] for t in indices], axis=1)
    return x.reshape(-1, window, features.shape[-1]), y.reshape(-1, horizon, traces.shape[-1])


def _composite(channels: np.ndarray, n_nodes: int, hosts: list[int], weights: np.ndarray) -> np.ndarray:
    """Aggregate [series, samples, horizon, channels] into [samples, nodes, horizon]."""
    out = np.zeros((channels.shape[1], n_nodes, channels.shape[2], 4), dtype=np.float32)
    for series in range(n_nodes):
        out[:, series] += channels[series]
    for service_id, node_id in enumerate(hosts):
        out[:, node_id] += channels[n_nodes + service_id]
    return (np.clip(out, 0, 1) * weights).sum(axis=-1)


def train_models(config: dict[str, Any], results_dir: Path, traces_dir: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    data_cfg, forecast_cfg = config["data"], config["forecast"]
    seed = int(config["seed"])
    trace_file = traces_dir / f"traces_seed_{seed}.npz"
    if trace_file.exists():
        traces = np.load(trace_file)["traces"]
    else:
        traces, metadata = generate_traces(config, seed)
        traces_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(trace_file, traces=traces)
        metadata.to_csv(traces_dir / f"series_seed_{seed}.csv", index=False)

    total = traces.shape[1]
    train_end = int(total * float(data_cfg["train_fraction"]))
    val_end = int(total * (float(data_cfg["train_fraction"]) + float(data_cfg["validation_fraction"])))
    window, horizon = int(forecast_cfg["window"]), int(forecast_cfg["horizon"])
    features = make_features(traces, int(data_cfg["slots_per_day"]))
    stride = int(forecast_cfg["training_stride"])
    train_x, train_y = windows(features, traces, 0, train_end, window, horizon, stride)
    val_x, val_y = windows(features, traces, train_end, val_end, window, horizon, stride)
    test_x, test_y = windows(features, traces, val_end, total, window, horizon, int(forecast_cfg["evaluation_step"]))
    results_dir.mkdir(parents=True, exist_ok=True)
    models: dict[str, Any] = {}
    metrics: list[dict[str, float | str]] = []
    environment = Environment(config, traces, seed)
    test_composites: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    validation_scores: dict[str, float] = {}
    model_names = ["MovingAverage", "ARIMA", "GRU", "LSTM"]
    for name in model_names:
        model = make_model(name, forecast_cfg, seed)
        if name in {"GRU", "LSTM"}:
            model.fit(train_x, train_y, (val_x, val_y))
            model.save(str(results_dir / f"{name.lower()}_forecaster.pt"))
            val_pred = model.predict(val_x)
            n_series = traces.shape[0]
            val_samples = len(val_x) // n_series
            val_actual = _composite(val_y.reshape(n_series, val_samples, horizon, 4), len(environment.nodes),
                                    environment.hosts, np.asarray(config["simulation"]["weights"], dtype=np.float32))
            val_estimated = _composite(val_pred.reshape(n_series, val_samples, horizon, 4),
                                       len(environment.nodes), environment.hosts,
                                       np.asarray(config["simulation"]["weights"], dtype=np.float32))
            validation_scores[name] = float(np.sqrt(np.mean((val_estimated - val_actual) ** 2)))
        else:
            model.fit(train_x, train_y)
        start = time.perf_counter()
        pred = model.predict(test_x)
        elapsed_ms = (time.perf_counter() - start) * 1000 / len(test_x)
        weights = np.asarray(config["simulation"]["weights"], dtype=np.float32)
        n_series = traces.shape[0]
        samples = len(test_x) // n_series
        actual = _composite(test_y.reshape(n_series, samples, horizon, 4), len(environment.nodes),
                            environment.hosts, weights)
        estimated = _composite(pred.reshape(n_series, samples, horizon, 4), len(environment.nodes),
                               environment.hosts, weights)
        metrics.append({"Model": name, "RMSE": float(np.sqrt(np.mean((estimated - actual) ** 2))),
                        "MAE": float(np.mean(np.abs(estimated - actual))), "Inference (ms)": elapsed_ms})
        test_composites[name] = estimated, actual
        models[name] = model

    requested_model = config["forecast"]["policy_model"]
    if requested_model == "best_recurrent":
        selected_model = min(validation_scores, key=validation_scores.get)
    elif requested_model in {"GRU", "LSTM"}:
        selected_model = requested_model
    else:
        raise ValueError("forecast.policy_model must be best_recurrent, GRU, or LSTM")
    (results_dir / "model_meta.json").write_text(
        json.dumps({"selected_model": selected_model, "validation_rmse": validation_scores}, indent=2) + "\n",
        encoding="utf-8",
    )
    preferred_prediction, preferred_actual = test_composites[selected_model]
    horizon_rmse = np.sqrt(np.mean((preferred_prediction - preferred_actual) ** 2, axis=(0, 1)))
    accuracy = pd.DataFrame(metrics)
    accuracy.to_csv(results_dir / "forecast_accuracy.csv", index=False)
    test_meta = {"traces": traces, "features": features, "train_end": train_end,
                 "validation_end": val_end, "test_start": val_end, "test_end": total - horizon,
                 "test_x": test_x, "test_y": test_y, "accuracy": accuracy,
                 "preferred_prediction": preferred_prediction, "preferred_actual": preferred_actual,
                 "horizon_rmse": horizon_rmse, "evaluation_step": int(forecast_cfg["evaluation_step"]),
                 "policy_model": selected_model, "validation_rmse": validation_scores}
    return models, test_meta


if __name__ == "__main__":
    import yaml
    cfg_path = Path(__file__).parents[1] / "config.yaml"
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    models, _ = train_models(cfg, Path(cfg["paths"]["results"]), Path(cfg["paths"]["traces"]))
    print(pd.read_csv(Path(cfg["paths"]["results"]) / "forecast_accuracy.csv").to_string(index=False))
