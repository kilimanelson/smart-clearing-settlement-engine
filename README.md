# Smart Clearing & Settlement Engine
When multiple participants owe money to one another, settling every individual obligation creates unnecessary transactions, increases processing costs, and complicates reconciliation.

Consider a simple example:
- Mike owes John $100.
- John owes Rachel $200.
- Mike owes Rachel $400.

Instead of processing three separate payments, we can calculate each participant's net position and consolidate the obligations into fewer transactions.

This same challenge exists at a much larger scale in financial systems, where banks and payment institutions must clear and settle thousands or millions of transactions while managing liquidity requirements and settlement risk.

## The Solution

Smart Clearing & Settlement Engine is a Python-based simulation of a payment clearing and settlement system that combines multilateral netting, machine learning, and liquidity optimization to:
- Calculate net obligations between financial institutions.
- Reduce the number of settlement transfers through debt consolidation.
- Predict potential liquidity shortfalls using XGBoost.
- Optimize funding allocation while preserving settlement obligations.
- Visualize clearing cycles, settlement positions, and liquidity risks.

The project explores how financial institutions can make settlement processes more efficient, transparent, and risk-aware.

## What it does

1. Accepts simplified card-clearing records between an issuing bank and an acquiring bank.
2. Builds bilateral obligations and multilateral net positions for one currency and one cycle.
3. Generates a reproducible synthetic history with a prediction time and a later cutoff.
4. Compares a rules baseline, logistic regression, and XGBoost on the question: will available liquidity be short at cutoff?
5. Recommends additional funding under explicit caps. If the caps cannot cover the requirement, the result is an unresolved risk.
6. Serves the simulation over FastAPI and a Streamlit operations view.

Negative net position means the participant owes money. Positive net position means the participant is due to receive money. Every accepted clearing report satisfies `sum(net positions) = 0`. Money is stored as integer minor units.

## Architecture

```text
simulator  ->  transactions + pre-cutoff snapshots
clearing   ->  bilateral obligations and net positions
features   ->  model matrix and chronological cycle splits
model      ->  baseline, logistic regression, XGBoost
optimizer  ->  funding recommendations (does not edit DNS positions)
api        ->  FastAPI + PostgreSQL
app        ->  Streamlit dashboard
```

See [docs/architecture.md](docs/architecture.md), [docs/data_dictionary.md](docs/data_dictionary.md), and [docs/model_card.md](docs/model_card.md).

## Install

Python 3.11 or newer. On macOS, XGBoost needs the OpenMP runtime: `brew install libomp`. From this directory:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
cp .env.example .env
```

`.env.example` contains a local placeholder password (`change-me`). Change it before any shared deployment. The application reads `SCSE_DATABASE_URL` and does not embed a live credential.

## Tests and lint

```bash
pytest
ruff check src tests scripts
ruff format --check src tests scripts
```

Clearing, features, the simulator, and the optimizer can be imported without starting the API or the dashboard. API tests use SQLite only when `SCSE_ALLOW_SQLITE=1`, which the test fixture sets itself. That is not the application database.

PostgreSQL coverage runs when `SCSE_POSTGRES_TEST_URL` is set. GitHub Actions starts PostgreSQL and sets that variable. Locally:

```bash
docker compose up -d postgres
export SCSE_POSTGRES_TEST_URL=postgresql+psycopg://scse:change-me@localhost:5432/scse
pytest tests/test_api.py -k postgres
```

Use the password you put in `.env`, not necessarily the example value.

## Generate data and reproduce the evaluation

```bash
python scripts/generate_data.py --seed 42 --cycles 18 --banks 4
python scripts/run_evaluation.py --seed 42
```

`generate_data.py` writes `data/transactions.csv` and `data/snapshots.csv`. Those files are synthetic and are gitignored.

`run_evaluation.py` overwrites [docs/experimental_results.md](docs/experimental_results.md) with metrics from that run and saves `artifacts/shortfall_model.joblib`. Do not treat those metrics as measured card-network performance.

## API

Start PostgreSQL, then the API, from this directory with `.env` present:

```bash
docker compose up -d postgres
uvicorn smart_clearing.api:app --reload
```

| Method | Path | Purpose |
| --- | --- | --- |
| GET | `/health` | Process and database check |
| POST | `/transactions` | Idempotent ingestion |
| GET | `/cycles` | List cycles |
| POST | `/cycles/{cycle_id}/clear` | Run DNS and store positions |
| GET | `/cycles/{cycle_id}/positions` | Stored net positions |
| POST | `/cycles/{cycle_id}/predict-risk` | Score supplied pre-cutoff snapshots |
| POST | `/cycles/{cycle_id}/optimize-liquidity` | Store funding recommendations |
| GET | `/cycles/{cycle_id}/report` | Clearing, scores, recommendations, reconciliation |

State moves `open -> cleared -> predicted -> optimized`. A cleared cycle rejects new transaction ids. Replaying the same transaction payload is a no-op. A different payload for an existing id returns 409. The service has no authentication; do not expose it beyond a local demo.

If `artifacts/shortfall_model.joblib` is missing, prediction uses a fixed-scale liquidity-gap rule and records that version. It does not invent a trained XGBoost score.

## Dashboard

```bash
streamlit run streamlit_app.py
```

`streamlit_app.py` is the file Streamlit Community Cloud should run. It loads the same dashboard as `src/smart_clearing/app.py`.

After this branch is on GitHub, open [share.streamlit.io](https://share.streamlit.io), choose the repository, and set the main file to `streamlit_app.py` on branch `main`. The app installs `requirements.txt` there. It still only simulates settlement. It does not move money.

The page simulates weekday cycles in memory, clears the selected cycle, shows model probabilities separately from the simulated outcome, and lists funding recommendations. The recommendation table is explicit that nothing was executed. The first load fits the three models on the demo sample; that can take a short time.

The dashboard sample is the in-memory demo: seed 42 and 18 weekday cycles unless you change the sidebar. Those scores are a different sample from [docs/experimental_results.md](docs/experimental_results.md), which is the 36-cycle experiment. A captured view of cycle `CY-20240125` is in [docs/images/dashboard-cycle.png](docs/images/dashboard-cycle.png).

Docker Compose also defines `api` and `dashboard` services:

```bash
docker compose up --build
```

The dashboard listens on port 8501 and the API on port 8000.

## Limitations

- One settlement currency per cycle. A second currency is quarantined, not converted.
- No interchange, scheme fees, partial settlement windows, or nostro/vostro accounts.
- Reversals are explicit records. They are not independent purchases.
- Liquidity, shocks, and transaction sizes are simulation assumptions. See `synthetic_assumptions()` in `src/smart_clearing/simulator.py`.
- A predicted probability is not a guarantee of a future shortfall rate.
- The optimizer never changes DNS net positions. Unresolved risk stays visible when a cap binds.
- The API does not authenticate callers and must not be pointed at real payment data.

## Project layout

```text
streamlit_app.py       Community Cloud entry point
src/smart_clearing/    library, API, and dashboard
tests/                 pytest modules
scripts/               data generation and evaluation
docs/                  architecture, data dictionary, model card, results
data/                  generated CSV (gitignored)
artifacts/             saved model (gitignored)
```
