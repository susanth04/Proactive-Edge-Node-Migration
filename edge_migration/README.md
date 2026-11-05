# Proactive Edge Node Migration Using Resource Overload Forecasting

A deterministic, CPU-friendly Python simulation comparing reactive threshold migration (RT), forecast-plus-least-loaded (FLL), and forecast/cost-aware migration (FCM).

## Run

From the workspace root:

```powershell
python -m pip install -r edge_migration\requirements.txt
python run_all.py
```

`run_all.py` generates traces, trains the four forecasters once, then evaluates both `default` and `slow_links` scenarios across the configured five seeds. Scenario tables, plots, sensitivity sweeps, ablations, and `tables.tex` are written under `edge_migration\results\<scenario>\`; shared forecast metrics and forecast plots remain at the top level. Python 3.10+ is supported. If your `python` command points to another interpreter, run the same commands with that interpreter (for example `py -3.11 run_all.py`).

The default workload is 11 days at one-minute resolution (60/20/20 chronological split), 10 nodes, 30 services, a 30-slot input window, and a 10-slot horizon. Training uses pooled series and early stopping. The default policy forecaster is selected between GRU and LSTM by validation composite-load RMSE; set `forecast.policy_model` to `GRU` or `LSTM` to override. To keep runs laptop-friendly, neural models are small single-layer 64-hidden-unit GRU/LSTM models, and forecasts for each test trace are precomputed once for simulation and sensitivity runs.

## Outputs

- `forecast_accuracy.csv`: composite-node-load RMSE/MAE and inference milliseconds per series-window for MovingAverage, ARIMA(1,0,0), GRU, and LSTM.
- `results/<scenario>/migration_performance.csv`: policy means and standard deviations for each numeric metric; `migration_performance_by_seed.csv` contains individual runs.
- `results/<scenario>/fcm_sensitivity.csv` and `fcm_ablation.csv`: sensitivity and ablation summaries; `tables.tex` exports the result CSVs as LaTeX tables.
- `results/<scenario>/load_timeline_rt_fcm.png`, `policy_comparison.png`, and `fcm_sensitivity.png`, plus top-level `load_forecast.png` and `forecast_rmse_horizon.png`: paper-ready 300-dpi figures.
- `config_used.yaml`, trained recurrent state dictionaries, generated trace archives, and per-seed series metadata provide the run inputs and reproducibility record.

## Model and interpretation

Each node has a background trace; every service has its own independent four-channel demand trace. Per-channel utilization is the clipped sum of background and hosted service demand, and node load is the configured weighted sum. Nodes and user locations are sampled within the configured square region. Latency combines distance-based network delay, load-dependent processing delay, and a fixed migration degradation penalty. Migration cost is transfer time in seconds (MB converted to megabits / Mbps) plus configured downtime; transfer duration is rounded up to at least one one-minute slot. Destination memory is reserved immediately, while service load remains on the source until migration completion.

RT uses current actual service load to select a service; FLL and FCM use the forecast. RT-CD uses RT's trigger/target rule with the same post-migration cooldown applied by FCM. FCM's gain combines the predicted source peak-load reduction with a small normalized latency benefit (`latency_gain_weight`). Its `tau` therefore thresholds a dimensionless heuristic, not a physical unit. Its feasibility check computes the target forecast after adding the service demand; FLL instead picks the least predicted peak without latency/cost/safety filtering. Same-slot destinations are unique and projected placement is updated after each decision.

“Unnecessary migration” is an oracle counterfactual: using only slots strictly after the decision through the configured horizon, hold the pre-decision placement fixed and count whether the source would have crossed the overload threshold. This is retrospective evaluation, not an online policy input. “No feasible target events” counts triggered decisions for which no candidate survives lead-time and/or safe-load filtering. Migration-slot summaries count transfer duration rounded up to whole minute slots.

The `default` scenario uses the configured current bandwidth and state-size ranges. `slow_links` changes only bandwidth to 8–40 Mbps and state size to 64–256 MB, so transfer duration can span multiple slots. Both scenarios retain the same seeds and all other settings. The FCM ablation compares the full policy with no safe-load filter, no cooldown, and persistence set to one.

## Real trace input

`data.generate_traces.load_real_trace(path)` accepts a CSV with numeric `cpu`, `memory`, `request_rate`, and `network` columns in `[0,1]`; additional columns such as timestamps are ignored. It returns a NumPy array in that channel order. Replace a generated series with this array only after aligning its slot duration, length, and scale with the configured simulation.

## Tests and extension

Run the focused checks with `python -m unittest discover -s edge_migration\tests`. Change values in `config.yaml` to vary sizes, thresholds, model settings, migration behavior, seeds, or sweep points.

## Run the demo

Install the Python dependencies with `python -m pip install -r requirements.txt`, then install the web dependencies with `cd frontend; npm install`. From the `edge_migration` directory, start the services in separate PowerShell windows:

```powershell
uvicorn api.main:app --reload --port 8000
cd frontend
npm run dev
```

Or launch both from the project directory with `.\scripts\dev.ps1`. Open `http://localhost:3000`; the UI reads saved CSVs and figures and calls the API for simulations. The API loads the selected pre-trained GRU/LSTM weights and never trains at startup. On the simulation page, add or remove computers (2–20) and rerun to generate a matching experiment. During replay, select a computer in the topology to see its measured workload, incoming/outgoing moves, and event list beside the animation.

The interface has separate Dashboard (`/`), Simulation (`/simulate`), Forecast (`/forecast`), and Sensitivity (`/sensitivity`) pages, available from the top navigation.

Endpoints: `GET /health`, `/config`, `/results/forecast-accuracy`, `/results/performance?scenario=default|slow_links`, `/results/by-seed`, `/results/sensitivity`, `/results/ablation`, and `/figures/{name}`; `POST /simulate` runs selected policies and seeds. `POST /simulate` accepts up to five seeds and the documented threshold, horizon, persistence, cooldown, hysteresis, score-weight, and `node_count` overrides. The node count is limited by `data.node_count_range` in `config.yaml`.

For a demo recording, capture the scenario toggle and comparison table, the overload/migration/cost/latency charts, a simulation with the threshold and migration markers, forecast scores, and the sensitivity/ablation results. State that traces are synthetic and values are measured.

Run API checks with `python -m unittest discover -s api\tests` from this directory.

## Assumptions and limitations

Trace channels are synthetic, normalized utilization proxies; service traces are exogenous. Bursts have random onset and are not predictable from the trace window, so principally the daily component is predictable. Forecast models are trained once on the single configured training seed and reused across simulation seeds and scenarios. Network propagation, bandwidth, processing delay, memory placement, and migration downtime are simplified and do not model packet-level contention or workload feedback. The simulation makes decisions once per slot.
