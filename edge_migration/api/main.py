"""FastAPI endpoints over the existing trace, forecast, and simulation code."""
from __future__ import annotations

import copy
import json
from contextlib import asynccontextmanager
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd
import yaml
from fastapi import FastAPI, HTTPException
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, ConfigDict, Field, model_validator

from data.generate_traces import generate_traces
from forecasting.models import RecurrentForecaster
from sim.environment import Environment
from sim.policies import run_policy
from sim.run_experiments import build_forecasts

PROJECT_DIR = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT_DIR / "results"
_CONFIG: dict[str, Any] | None = None
_MODEL: RecurrentForecaster | None = None
_MODEL_NAME: str | None = None
SCENARIO = Literal["default", "slow_links"]
POLICY = Literal["RT", "RT-CD", "FLL", "FCM"]


@asynccontextmanager
async def lifespan(_: FastAPI):
    global _CONFIG, _MODEL, _MODEL_NAME
    with (PROJECT_DIR / "config.yaml").open(encoding="utf-8") as file:
        _CONFIG = yaml.safe_load(file)
    meta_path = RESULTS_DIR / "model_meta.json"
    if not meta_path.is_file():
        raise FileNotFoundError(f"Trained model metadata not found: {meta_path}; run forecasting/train.py first.")
    metadata = json.loads(meta_path.read_text(encoding="utf-8"))
    _MODEL_NAME = str(metadata["selected_model"]).upper()
    if _MODEL_NAME not in {"GRU", "LSTM"}:
        raise ValueError(f"Unsupported selected model in {meta_path}: {_MODEL_NAME}")
    model_path = RESULTS_DIR / f"{_MODEL_NAME.lower()}_forecaster.pt"
    if not model_path.is_file():
        raise FileNotFoundError(f"Trained model weights not found: {model_path}")
    _MODEL = RecurrentForecaster(_MODEL_NAME, _CONFIG["forecast"], int(_CONFIG["seed"])).load(str(model_path))
    yield
    clear_case_cache()
    _CONFIG = _MODEL = None
    _MODEL_NAME = None


app = FastAPI(title="Edge Migration Simulator API", version="1.0.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


@lru_cache(maxsize=4)
def get_cached_case(scenario: str, seed: int, node_count: int | None = None) -> tuple[np.ndarray, np.ndarray, int]:
    """Cache generated or saved traces and forecasts for a scenario/seed pair."""
    if _CONFIG is None or _MODEL is None:
        raise RuntimeError("API model is not initialized")
    config = copy.deepcopy(_CONFIG)
    base_node_count = int(config["data"]["nodes"])
    selected_node_count = base_node_count if node_count is None else int(node_count)
    config["data"]["nodes"] = selected_node_count
    scenario_config = config["scenarios"][scenario]
    config["simulation"]["network"]["bandwidth_mbps"] = scenario_config["bandwidth_mbps"]
    config["simulation"]["services"]["state_size_mb"] = scenario_config["state_size_mb"]
    traces_path = PROJECT_DIR / config["paths"]["traces"] / f"traces_seed_{seed}.npz"
    if selected_node_count == base_node_count and traces_path.is_file():
        traces = np.load(traces_path)["traces"]
    else:
        traces, _ = generate_traces(config, seed)
    environment = Environment(config, traces, seed)
    expected_series = len(environment.nodes) + len(environment.services)
    if traces.shape[0] != expected_series:
        raise ValueError(f"Trace contains {traces.shape[0]} series; configuration requires {expected_series}")
    total = traces.shape[1]
    start = int(total * (float(config["data"]["train_fraction"]) +
                         float(config["data"]["validation_fraction"])))
    stop = total - int(config["forecast"]["horizon"])
    forecasts = build_forecasts(_MODEL, traces, config, start, stop)
    return traces, forecasts, start


def clear_case_cache() -> None:
    get_cached_case.cache_clear()


class Overrides(BaseModel):
    model_config = ConfigDict(extra="forbid")

    theta_hi: float | None = Field(default=None, gt=0, le=1)
    theta_safe: float | None = Field(default=None, gt=0, le=1)
    persistence: int | None = Field(default=None, ge=1, le=10)
    cooldown_slots: int | None = Field(default=None, ge=0)
    tau: float | None = None
    alpha: float | None = Field(default=None, ge=0)
    beta: float | None = Field(default=None, ge=0)
    gamma: float | None = Field(default=None, ge=0)
    horizon: int | None = Field(default=None, ge=1, le=10)
    node_count: int | None = Field(default=None, ge=1)

    @model_validator(mode="after")
    def validate_thresholds(self) -> "Overrides":
        if self.theta_safe is not None and self.theta_hi is not None and self.theta_safe > self.theta_hi:
            raise ValueError("theta_safe must be less than or equal to theta_hi")
        return self


class SimulationRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    scenario: SCENARIO
    seeds: list[int] = Field(min_length=1, max_length=5)
    policies: list[POLICY] = Field(min_length=1)
    overrides: Overrides = Field(default_factory=Overrides)

    @model_validator(mode="after")
    def validate_seed_and_policy_lists(self) -> "SimulationRequest":
        if len(set(self.seeds)) != len(self.seeds):
            raise ValueError("seeds must not contain duplicates")
        if len(set(self.policies)) != len(self.policies):
            raise ValueError("policies must not contain duplicates")
        return self


def _request_config(scenario: str, overrides: Overrides) -> tuple[dict[str, Any], dict[str, Any]]:
    if _CONFIG is None:
        raise RuntimeError("API config is not initialized")
    config = copy.deepcopy(_CONFIG)
    selected_scenario = config["scenarios"][scenario]
    config["simulation"]["network"]["bandwidth_mbps"] = selected_scenario["bandwidth_mbps"]
    config["simulation"]["services"]["state_size_mb"] = selected_scenario["state_size_mb"]
    values = overrides.model_dump(exclude_none=True)
    node_count = int(values.pop("node_count", config["data"]["nodes"]))
    node_min, node_max = map(int, config["data"]["node_count_range"])
    if not node_min <= node_count <= node_max:
        raise HTTPException(status_code=422, detail=f"node_count must be between {node_min} and {node_max}")
    config["data"]["nodes"] = node_count
    if "horizon" in values and values["horizon"] > int(config["forecast"]["horizon"]):
        raise HTTPException(status_code=422, detail=f"horizon must be at most {config['forecast']['horizon']}")
    if values.get("theta_safe", config["simulation"]["theta_safe"]) > values.get(
            "theta_hi", config["simulation"]["theta_hi"]):
        raise HTTPException(status_code=422, detail="theta_safe must be less than or equal to theta_hi")
    config["simulation"]["theta_hi"] = values.pop("theta_hi", config["simulation"]["theta_hi"])
    config["simulation"]["theta_safe"] = values.pop("theta_safe", config["simulation"]["theta_safe"])
    horizon = values.pop("horizon", int(config["forecast"]["horizon"]))
    config["forecast"]["horizon"] = horizon
    for key in ("persistence", "cooldown_slots", "tau", "alpha", "beta", "gamma"):
        if key in values:
            config["simulation"]["fcm"][key] = values[key]
    score_weights = config["simulation"]["fcm"]
    if sum(float(score_weights[key]) for key in ("alpha", "beta", "gamma")) <= 0:
        raise HTTPException(status_code=422, detail="At least one of alpha, beta, or gamma must be greater than 0")
    return config, {"horizon": horizon}


def _timeline_payload(timeline: dict[str, Any]) -> dict[str, Any]:
    loads = np.asarray(timeline["node_loads"])
    indices = np.unique(np.linspace(0, len(loads) - 1, min(600, len(loads)), dtype=int))
    points = [{"slot": int(index), **{f"node_{node}": float(value)
                                     for node, value in enumerate(loads[index])}}
              for index in indices]
    migrations = [
        {"slot": int(slot), "source": int(item.source), "destination": int(item.destination),
         "service": int(item.service_id), "complete_slot": int(item.complete_slot)}
        for slot, item in zip(timeline["migration_slots"], timeline["migrations"])
    ]
    return {"points": points, "migrations": migrations, "total_points": len(loads)}


def _simulate(request: SimulationRequest) -> dict[str, Any]:
    config, fcm_overrides = _request_config(request.scenario, request.overrides)
    overrides = request.overrides.model_dump(exclude_none=True)
    runs: list[dict[str, Any]] = []
    timelines: dict[str, Any] = {}
    for seed in request.seeds:
        traces, forecasts, start = get_cached_case(request.scenario, seed, int(config["data"]["nodes"]))
        for policy in request.policies:
            metrics, timeline = run_policy(
                config, policy, traces, forecasts, start, fcm_overrides=fcm_overrides,
                seed=seed, keep_timeline=(seed == request.seeds[0]),
            )
            runs.append({"seed": seed, **metrics})
            if seed == request.seeds[0]:
                timelines[policy] = _timeline_payload(timeline)
    metric_names = [key for key, value in runs[0].items() if key not in {"seed", "Policy"}
                    and isinstance(value, (int, float))]
    summary = []
    for policy in request.policies:
        selected = [run for run in runs if run["Policy"] == policy]
        summary.append({
            "Policy": policy,
            "mean": {name: float(np.mean([run[name] for run in selected])) for name in metric_names},
            "std": {name: float(np.std([run[name] for run in selected], ddof=1)) if len(selected) > 1 else 0.0
                    for name in metric_names},
        })
    return {"scenario": request.scenario, "per_seed": runs, "summary": summary,
            "timeline": {"seed": request.seeds[0], "node_count": int(config["data"]["nodes"]),
                         "policies": timelines}}


def _csv_records(name: str, scenario: str | None = None) -> list[dict[str, Any]]:
    path = RESULTS_DIR / scenario / name if scenario else RESULTS_DIR / name
    if not path.is_file():
        raise HTTPException(status_code=404, detail=f"Result file not found: {path.name}")
    frame = pd.read_csv(path)
    return frame.astype(object).where(pd.notna(frame), None).to_dict(orient="records")


@app.get("/health")
def health() -> dict[str, str]:
    if _CONFIG is None or _MODEL is None:
        raise HTTPException(status_code=503, detail="Model is not loaded")
    return {"status": "ok", "model": _MODEL_NAME or ""}


@app.get("/config")
def get_config() -> dict[str, Any]:
    if _CONFIG is None:
        raise HTTPException(status_code=503, detail="Config is not loaded")
    return {
        "scenarios": list(_CONFIG["scenarios"]),
        "seeds": _CONFIG["simulation"]["seeds"],
        "policies": ["RT", "RT-CD", "FLL", "FCM"],
        "node_count": int(_CONFIG["data"]["nodes"]),
        "node_count_range": _CONFIG["data"]["node_count_range"],
        "simulation": _CONFIG["simulation"],
        "forecast": {"horizon": _CONFIG["forecast"]["horizon"]},
    }


@app.get("/results/forecast-accuracy")
def forecast_accuracy() -> list[dict[str, Any]]:
    return _csv_records("forecast_accuracy.csv")


@app.get("/results/performance")
def performance(scenario: SCENARIO = "default") -> list[dict[str, Any]]:
    return _csv_records("migration_performance.csv", scenario)


@app.get("/results/by-seed")
def by_seed(scenario: SCENARIO = "default") -> list[dict[str, Any]]:
    return _csv_records("migration_performance_by_seed.csv", scenario)


@app.get("/results/sensitivity")
def sensitivity(scenario: SCENARIO = "default") -> list[dict[str, Any]]:
    return _csv_records("fcm_sensitivity.csv", scenario)


@app.get("/results/ablation")
def ablation(scenario: SCENARIO = "default") -> list[dict[str, Any]]:
    return _csv_records("fcm_ablation.csv", scenario)


@app.get("/figures/{name:path}")
def figure(name: str) -> Any:
    relative = Path(name)
    if relative.is_absolute() or ".." in relative.parts or relative.suffix.lower() != ".png":
        raise HTTPException(status_code=404, detail="Figure not found")
    path = (RESULTS_DIR / relative).resolve()
    if not path.is_relative_to(RESULTS_DIR.resolve()) or not path.is_file():
        raise HTTPException(status_code=404, detail="Figure not found")
    from fastapi.responses import FileResponse
    return FileResponse(path, media_type="image/png")


@app.post("/simulate")
async def simulate(request: SimulationRequest) -> dict[str, Any]:
    return await run_in_threadpool(_simulate, request)
