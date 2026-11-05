"""Forecasting baselines and small shared PyTorch recurrent forecasters."""
from __future__ import annotations

import copy
import time
from typing import Any

import numpy as np
import torch
from torch import nn
from statsmodels.tsa.arima.model import ARIMA


class MovingAverage:
    """Repeat the trailing-window mean for all requested horizons."""

    def fit(self, x: np.ndarray, y: np.ndarray | None = None, validation_data: Any = None) -> "MovingAverage":
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        mean = np.asarray(x)[..., :4].mean(axis=-2, keepdims=True)
        return np.repeat(mean, self.horizon, axis=-2)

    def set_horizon(self, horizon: int) -> None:
        self.horizon = int(horizon)


class ARIMAForecaster:
    """AR(1) around the window mean, with pooled phi estimated by statsmodels."""

    def __init__(self, horizon: int, order: tuple[int, int, int] = (1, 0, 0)):
        self.horizon = int(horizon)
        self.order = order
        self.phi = np.zeros(4, dtype=np.float32)

    def fit(self, x: np.ndarray, y: np.ndarray | None = None, validation_data: Any = None) -> "ARIMAForecaster":
        # Estimate one autoregressive coefficient per channel from pooled training windows.
        histories = np.asarray(x)[::max(1, len(x) // 400), :, :4]
        pooled = histories.reshape(-1, 4)
        for channel in range(4):
            fitted = ARIMA(pooled[:, channel], order=self.order, trend="c").fit()
            self.phi[channel] = float(np.clip(fitted.arparams[0], -0.98, 0.98))
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x)
        history = x[..., :4].astype(np.float32, copy=False)
        window_mean = history.mean(axis=-2)
        deviation = history[..., -1, :] - window_mean
        steps = np.arange(1, self.horizon + 1, dtype=np.int32)
        decay = self.phi[None, None, :] ** steps[None, :, None]
        result = window_mean[..., None, :] + decay * deviation[..., None, :]
        result = np.clip(result, 0, 1).astype(np.float32)
        return result


class _RecurrentNetwork(nn.Module):
    def __init__(self, kind: str, input_size: int, hidden_size: int, layers: int, horizon: int):
        super().__init__()
        recurrent = nn.GRU if kind == "GRU" else nn.LSTM
        self.encoder = recurrent(input_size, hidden_size, num_layers=layers, batch_first=True)
        self.head = nn.Linear(hidden_size, horizon * 4)
        self.horizon = horizon

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        encoded, _ = self.encoder(x)
        return self.head(encoded[:, -1]).view(-1, self.horizon, 4)


class RecurrentForecaster:
    """Early-stopped shared GRU/LSTM trained on pooled series windows."""

    def __init__(self, kind: str, config: dict[str, Any], seed: int):
        if kind not in {"GRU", "LSTM"}:
            raise ValueError(f"Unsupported recurrent model: {kind}")
        self.kind, self.config, self.seed = kind, config, int(seed)
        self.horizon = int(config["horizon"])
        self.network: _RecurrentNetwork | None = None

    def fit(self, x: np.ndarray, y: np.ndarray, validation_data: tuple[np.ndarray, np.ndarray]) -> "RecurrentForecaster":
        torch.manual_seed(self.seed)
        torch.set_num_threads(int(self.config.get("torch_threads", 1)))
        self.network = _RecurrentNetwork(
            self.kind, x.shape[-1], int(self.config["hidden_size"]),
            int(self.config["layers"]), self.horizon,
        )
        optimizer = torch.optim.Adam(self.network.parameters(), lr=float(self.config["learning_rate"]))
        loss_fn = nn.MSELoss()
        train_x, train_y = torch.from_numpy(x), torch.from_numpy(y)
        val_x, val_y = map(torch.from_numpy, validation_data)
        generator = torch.Generator().manual_seed(self.seed)
        best_loss, best_state, stale = float("inf"), None, 0
        batch_size = int(self.config["batch_size"])

        for _ in range(int(self.config["epochs"])):
            self.network.train()
            order = torch.randperm(len(train_x), generator=generator)
            for start in range(0, len(order), batch_size):
                indices = order[start:start + batch_size]
                optimizer.zero_grad(set_to_none=True)
                loss = loss_fn(self.network(train_x[indices]), train_y[indices])
                loss.backward()
                optimizer.step()
            self.network.eval()
            with torch.no_grad():
                val_loss = float(loss_fn(self.network(val_x), val_y))
            if val_loss < best_loss:
                best_loss, best_state, stale = val_loss, copy.deepcopy(self.network.state_dict()), 0
            else:
                stale += 1
                if stale >= int(self.config["patience"]):
                    break
        if best_state is None:
            raise RuntimeError(f"{self.kind} training produced no validation checkpoint")
        self.network.load_state_dict(best_state)
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        if self.network is None:
            raise RuntimeError(f"{self.kind} must be fitted before prediction")
        self.network.eval()
        outputs = []
        batch_size = int(self.config.get("inference_batch_size", 256))
        with torch.no_grad():
            for start in range(0, len(x), batch_size):
                tensor = torch.from_numpy(np.asarray(x[start:start + batch_size], dtype=np.float32))
                outputs.append(self.network(tensor).numpy())
        return np.clip(np.concatenate(outputs, axis=0), 0, 1)

    def save(self, path: str) -> None:
        if self.network is None:
            raise RuntimeError(f"{self.kind} must be fitted before saving")
        torch.save(self.network.state_dict(), path)

    def load(self, path: str) -> "RecurrentForecaster":
        """Load a saved CPU state dictionary without fitting the model."""
        self.network = _RecurrentNetwork(
            self.kind, 6, int(self.config["hidden_size"]),
            int(self.config["layers"]), self.horizon,
        )
        self.network.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
        self.network.eval()
        return self


def make_model(name: str, config: dict[str, Any], seed: int):
    """Construct one of the four supported forecasters."""
    horizon = int(config["horizon"])
    if name == "MovingAverage":
        model = MovingAverage()
        model.set_horizon(horizon)
        return model
    if name == "ARIMA":
        return ARIMAForecaster(horizon, tuple(config["arima_order"]))
    if name in {"GRU", "LSTM"}:
        return RecurrentForecaster(name, config, seed)
    raise ValueError(f"Unknown forecasting model: {name}")


def timed_predict(model: Any, x: np.ndarray) -> tuple[np.ndarray, float]:
    start = time.perf_counter()
    prediction = model.predict(x)
    return prediction, (time.perf_counter() - start) * 1000 / max(1, len(x))
