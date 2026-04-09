"""
Orchestrator script: calls FastAPI for salary predictions, then asks Ollama
to analyse the batch of predicted vs. actual salaries.

Usage:
    python analyse.py
"""

import httpx
from services import ollama_client as ollama

FASTAPI_URL = "http://localhost:8000"

# Sample rows — swap in real test data later.
# actual_salary_usd is the ground truth from the dataset.
SAMPLE_ROWS = [
    {
        "work_year": 2022,
        "experience_level": "SE",
        "job_title": "Data Scientist",
        "remote_ratio": 100,
        "company_location": "US",
        "company_size": "M",
        "is_abroad": False,
        "actual_salary_usd": 140000,
    },
    {
        "work_year": 2022,
        "experience_level": "EN",
        "job_title": "Data Analyst",
        "remote_ratio": 0,
        "company_location": "GB",
        "company_size": "S",
        "is_abroad": False,
        "actual_salary_usd": 45000,
    },
    {
        "work_year": 2023,
        "experience_level": "EX",
        "job_title": "Machine Learning Engineer",
        "remote_ratio": 50,
        "company_location": "US",
        "company_size": "L",
        "is_abroad": True,
        "actual_salary_usd": 210000,
    },
]

PREDICT_FIELDS = {
    "work_year", "experience_level", "job_title",
    "remote_ratio", "company_location", "company_size", "is_abroad",
}


def get_predictions(rows: list[dict]) -> list[dict]:
    """Call /predict on each row and return results with predicted salary attached."""
    results = []
    with httpx.Client(base_url=FASTAPI_URL, timeout=30.0) as client:
        for row in rows:
            payload = {k: v for k, v in row.items() if k in PREDICT_FIELDS}
            response = client.post("/predict", json=payload)
            response.raise_for_status()
            predicted = response.json()["predicted_salary_usd"]
            results.append({**row, "predicted_salary_usd": predicted})
    return results


def main():
    print("Fetching predictions from FastAPI...")
    try:
        results = get_predictions(SAMPLE_ROWS)
    except httpx.ConnectError:
        print(f"Error: could not connect to FastAPI at {FASTAPI_URL}. Is it running?")
        return
    except httpx.HTTPStatusError as e:
        print(f"Error: FastAPI returned {e.response.status_code}: {e.response.text}")
        return
    print(f"Got {len(results)} predictions.\n")

    for r in results:
        print(
            f"  {r['job_title']:35s} predicted: ${r['predicted_salary_usd']:>9,}  "
            f"actual: ${r['actual_salary_usd']:>9,}  "
            f"error: ${r['predicted_salary_usd'] - r['actual_salary_usd']:>+9,}"
        )

    print("\nAsking Ollama for analysis...")
    try:
        analysis = ollama.analyse_predictions(results)
    except httpx.ConnectError:
        print(f"Error: could not connect to Ollama at {ollama.OLLAMA_URL}. Is it running?")
        return
    except httpx.HTTPStatusError as e:
        print(f"Error: Ollama returned {e.response.status_code}: {e.response.text}")
        return
    print(f"\n--- Ollama Analysis ---\n{analysis}\n")


if __name__ == "__main__":
    main()
