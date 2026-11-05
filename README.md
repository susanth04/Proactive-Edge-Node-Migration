# Proactive Edge Node Migration

A deterministic simulation and web dashboard for comparing ways to move workloads between edge computers when demand rises. Explore reactive and forecast-based policies using generated workloads; the dashboard is a research and learning tool, not live device monitoring.

## Run the dashboard

You need Python 3.10 or later and Node.js. From the repository root, open two terminals.

Start the API in the first terminal:

```powershell
Set-Location edge_migration
python -m pip install -r requirements.txt
uvicorn api.main:app --reload --port 8000
```

Start the frontend in the second:

```powershell
Set-Location edge_migration\frontend
npm ci
npm run dev
```

Open [http://localhost:3000](http://localhost:3000). The API runs at [http://localhost:8000](http://localhost:8000).

## Pages

- **Dashboard** (`/`) — compare migration policies and connection scenarios.
- **Simulation** (`/simulate`) — configure and replay a simulation.
- **Forecast** (`/forecast`) — compare forecasting approaches.
- **Sensitivity** (`/sensitivity`) — explore parameter sweeps and policy ablations.

## Run the experiments

From the repository root:

```powershell
python -m pip install -r edge_migration\requirements.txt
python run_all.py
```

This generates workloads, trains the forecasters, and writes experiment results under `edge_migration\results\`.

See the [full project guide](edge_migration/README.md) for configuration, output details, API endpoints, tests, and model assumptions.
