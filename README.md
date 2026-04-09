# Salary Prediction Application for Data Science Jobs

End-to-end ML pipeline that predicts salaries for data science jobs.

## Architecture

```
Python script → FastAPI (Decision Tree) → Ollama LLM → Supabase → Streamlit Dashboard
```

## Stack

| Component | Purpose |
|---|---|
| Decision Tree (scikit-learn) | Trained on Kaggle dataset, predicts salaries |
| FastAPI | Serves the model via REST API |
| Python script | Orchestrator — calls FastAPI, passes results to Ollama, saves to Supabase |
| Ollama (llama3.2) | Local LLM, generates narrative analysis |
| Supabase | Cloud storage for predictions and narratives |
| Streamlit | Dashboard reading from Supabase |

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r api/requirements.txt
```

## Running the API

```bash
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
```

API will be available at `http://localhost:8000`. Interactive docs at `http://localhost:8000/docs`.

## Dataset

[Data Science Job Salaries](https://kaggle.com/datasets/ruchi798/data-science-job-salaries) from Kaggle.
