# DS Salary Prediction — Week 1

End-to-end ML pipeline that predicts data science salaries, with a FastAPI serving the model, an orchestrator script that runs batch predictions and LLM analysis, and a Streamlit dashboard to visualise results.

## What's been built

| Component                    | Status | Notes                                        |
|------------------------------|--------|----------------------------------------------|
| EDA notebooks                | Done   | `data/w1_data_science_jobs_eda.ipynb`        |
| ML model training            | Done   | `data/w1_data_science_jobs_ml.ipynb`         |
| CatBoost model               | Done   | `models/catboost_model.joblib`               |
| FastAPI prediction endpoint  | Done   | `POST /predict` with validated inputs        |
| Batch analysis orchestrator  | Done   | `analyse.py` — calls API, runs LLM analysis  |
| Groq LLM integration         | Done   | `services/groq_client.py`                    |
| Streamlit dashboard          | Done   | `dashboard.py` — charts + AI insights panel  |

## Architecture

```
analyse.py → FastAPI (/predict) → CatBoost model
                ↓
         Groq LLM (analysis)
                ↓
         data/dashboard_data.csv
                ↓
         streamlit dashboard.py
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r api/requirements.txt
```

Copy `.env.example` to `.env` and fill in your API keys.

## Key commands

**Run the FastAPI server:**
```bash
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
```
Docs available at `http://localhost:8000/docs`.

**Run batch predictions + LLM analysis (requires API server running):**
```bash
python analyse.py --save
```
This reads `data/llm_test_data.csv`, calls `/predict` for each row, runs Groq analysis, and writes `data/dashboard_data.csv`.

**Run the dashboard:**
```bash
streamlit run dashboard.py
```

## Dataset

[Data Science Job Salaries](https://kaggle.com/datasets/ruchi798/data-science-job-salaries) — Kaggle.
