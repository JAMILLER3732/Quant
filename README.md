# Quant Analytics Platform

A quantitative-finance portfolio analytics platform: upload Excel/CSV data, pick a
quant methodology, and get a chart + statistics calculated by a real, tested Python
engine — not fabricated numbers.

> **Status: Phase 1 + Phase 2 + Phase 3 complete.** Upload → validate → map
> columns → select method → calculate → chart → stats → export, for 15
> methods spanning descriptive stats, technical/backtesting, risk,
> simulation, portfolio optimization, factor models, and econometrics.
> See [Roadmap](#roadmap) for what's next.

## Architecture

Two independently deployable services:

```
/web     Next.js 16 (App Router) + TypeScript + Tailwind — frontend, deploys to Vercel
/engine  FastAPI (Python) — quant calculation engine, deploys to Render/Railway/Fly
```

They talk over plain HTTP/JSON; `web` never runs quant math itself — it calls
`engine`, which does everything with pandas/NumPy/SciPy/Plotly and returns a
Plotly figure spec + statistics + tabular results as JSON.

### `engine/app` layout

```
data/       ingestion (CSV/Excel) -> column-role detection -> reshape (wide/long
            panels) -> data-quality validation
quant/      the QuantMethod framework (base.py), shared math (calc.py), chart
            theming, and methods/ (one file per methodology)
api/        FastAPI routes: upload, mapping, methods, requirements, calculate
store.py    in-memory dataset session store (see note below)
tests/      quantitative correctness tests (hand-calculated values) + API tests
```

Every methodology is a `QuantMethod` subclass declaring its required inputs,
parameters, assumptions, and limitations, and implementing one `calculate()`
method. The API and frontend discover everything generically — adding a new
method means writing one file and registering it in `quant/registry.py`, no
routing or UI changes required.

**Note on state:** dataset sessions live in an in-memory dict keyed by a
`dataset_id` UUID, since the engine runs as a single persistent process. This
is intentionally simple for this stage — it means state is lost on redeploy
and won't survive horizontal scaling. Revisit with Redis/a database if/when
that matters (see Roadmap).

### `web/src` layout

```
app/lab/page.tsx           the Quant Lab flow: upload -> map -> select method -> results
components/lab/            FileUpload, DataPreviewTable, QualityReportPanel,
                            ColumnMapper, MethodPicker, MethodInfoPanel,
                            ParamsForm, ResultsView, PlotlyChart
lib/api.ts                 typed API client
lib/roles.ts                column-role list (mirrors engine's ROLES)
lib/csv.ts                  client-side CSV export helper
```

## Implemented methods

| Method | Category | What it does |
|---|---|---|
| Returns & Descriptive Statistics | Returns & Descriptive Statistics | Simple/log returns, cumulative return, rolling volatility, distribution/skew/kurtosis |
| Rolling Z-Score & Std-Dev Bands | Returns & Descriptive Statistics | Rolling mean/stdev price bands + rolling Z-score of returns |
| EWMA Crossover Strategy | Technical / Time-Series | Fast/slow EWMA crossover signal, backtested vs. buy-and-hold, with costs |
| Performance & Risk Dashboard | Risk Analytics | Sharpe/Sortino/Calmar/MaxDD/VaR/CVaR ranking across all mapped securities |
| Monte Carlo Simulation (GBM) | Simulation | Vectorized geometric Brownian motion, percentile bands, terminal distribution |
| Efficient Frontier & Optimization | Portfolio Optimization | Random-portfolio cloud + SciPy-optimized min-vol/max-Sharpe portfolios and frontier |
| Correlation & Covariance Analysis | Correlation & Dependence | Pearson/Spearman/Kendall matrix heatmap + rolling pairwise correlation |
| Value at Risk & Expected Shortfall | Risk Analytics | Historical, parametric, and Monte Carlo VaR/CVaR compared side by side |
| Stress Testing & Scenario Analysis | Tail Risk & Stress Testing | Custom or historical-benchmark (beta-scaled) shocks, portfolio P&L breakdown |
| Mean-Reversion Z-Score Backtest | Backtesting | Rule-based long/short backtest with stop-loss/take-profit, no look-ahead |
| Factor Analysis (CAPM) | Factor Analysis | OLS market-model regression: alpha, beta, R², rolling beta/alpha |
| Pairs Trading & Cointegration | Pairs Trading / Mean Reversion | Engle-Granger test, hedge ratio, spread Z-score, half-life, backtest |
| GARCH Volatility Forecasting | Statistical / Econometric | GARCH(p,q) conditional volatility + forward forecast |
| Volatility & Trend Regime Analysis | Regime Analysis | Rolling-volatility tercile regimes + MA-based bull/bear regimes, per-regime stats |
| Hierarchical Risk Parity (HRP) | Portfolio Optimization | Correlation-clustering + recursive bisection allocation (no matrix inversion) |

All formulas live in `engine/app/quant/calc.py` (core stats/risk),
`engine/app/quant/portfolio.py` (portfolio optimization + HRP), and
`engine/app/quant/simulation.py` (Monte Carlo), unit-tested against
hand-calculated/analytical values in `engine/app/tests/`.

## Local development

### Engine (Python)

```bash
cd engine
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
# API docs: http://localhost:8000/docs
```

Run tests:

```bash
source .venv/bin/activate && python -m pytest app/tests/ -v
```

### Web (Next.js)

```bash
cd web
npm install
cp .env.local.example .env.local   # NEXT_PUBLIC_API_BASE_URL=http://localhost:8000
npm run dev
# http://localhost:3000
```

## Deployment

This project is meant to be published at your existing Vercel project
(`project-v0ezh.vercel.app`) for the frontend, plus a separate always-on
Python host for the engine (Python quant workloads don't fit Vercel's
serverless function limits well — see below).

### 1. Push this repo to GitHub

```bash
git remote add origin <your-new-github-repo-url>
git branch -M main
git push -u origin main
```

### 2. Deploy the engine (Render, or Railway/Fly similarly)

1. In Render: **New → Web Service → connect this GitHub repo**.
2. Root directory: `engine`. Render will pick up `engine/Dockerfile` (or use
   the included `engine/render.yaml` blueprint).
3. Set the environment variable `ALLOWED_ORIGINS` to your Vercel URL(s), e.g.
   `https://project-v0ezh.vercel.app` (comma-separate multiple origins).
4. Deploy. Confirm `https://<your-engine-url>/api/health` returns `{"status":"ok"}`.

### 3. Connect the frontend in Vercel

1. In the Vercel dashboard, open the `project-v0ezh` project → **Settings →
   Git** → connect this GitHub repo.
2. Set **Root Directory** to `web`.
3. Add environment variable `NEXT_PUBLIC_API_BASE_URL` = your Render engine
   URL from step 2 (e.g. `https://quant-engine.onrender.com`).
4. Deploy. Vercel will build/redeploy automatically on every push to `main`.

### Why two services instead of one Vercel deployment?

Vercel's Python serverless functions have tight execution-time and bundle-size
limits that don't fit the heavier quant workloads on the roadmap (Monte
Carlo, portfolio optimization, GARCH). Splitting frontend (Vercel) from the
Python engine (a normal long-running container host) avoids hitting that
wall later. Phase 1's methods would run fine on Vercel Python functions, but
this way nothing has to be re-architected in Phase 2.

## Roadmap

- **Phase 1 (done):** upload → validate → map → select method → calculate →
  chart → stats → export. Returns/stats, EWMA crossover, rolling Z-score,
  performance/risk dashboard.
- **Phase 2 (done):** Monte Carlo (GBM), efficient frontier / portfolio
  optimization, correlation analysis, standalone VaR/CVaR, stress testing,
  rule-based backtesting engine.
- **Phase 3 (done):** factor analysis (CAPM), pairs trading/cointegration,
  GARCH volatility forecasting, volatility/trend regime analysis, HRP.
- **Phase 3, not yet done:** multi-factor (Fama-French) models, Brinson
  attribution (needs portfolio+benchmark holdings/weights data this platform
  doesn't yet collect), ARIMA/SARIMA/VAR forecasting, Black-Litterman.
- **Phase 4:** research library, blog/CMS, admin panel, PDF report generation,
  user accounts.

See the original product spec for full detail on each phase.
