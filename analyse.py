"""
Orchestrator script: reads test data from a CSV, calls FastAPI for salary
predictions, then asks an LLM to analyse the batch of predicted vs. actual salaries.

Expected CSV columns:
    work_year, experience_level, job_title, remote_ratio,
    company_location, company_size, is_abroad, salary_in_usd

Usage:
    python analyse.py [path/to/test_data.csv]

If no path is given it defaults to test_data.csv in the project directory.
"""

import os
import sys
import csv
from pathlib import Path

from dotenv import load_dotenv
import httpx
from services import groq_client as llm_client

load_dotenv()

FASTAPI_URL = "http://localhost:8000"

PREDICT_FIELDS = {
    "work_year", "experience_level", "job_title",
    "remote_ratio", "company_location", "company_size", "is_abroad",
}


def load_csv(path: Path) -> list[dict]:
    """Read test data rows from a CSV file. Returns a list of dicts."""
    rows = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader, start=1):
            # Coerce numeric and boolean fields from their string representations
            try:
                row["work_year"] = int(row["work_year"])
                row["remote_ratio"] = int(row["remote_ratio"])
                row["salary_in_usd"] = int(row["salary_in_usd"])
                row["is_abroad"] = row["is_abroad"].strip().lower() in ("true", "1", "yes")
            except (KeyError, ValueError) as e:
                print(f"Warning: skipping row {i} due to parse error: {e}")
                continue
            rows.append(row)
    return rows


def get_predictions(rows: list[dict]) -> list[dict]:
    """Call /predict on each row and return results with predicted salary and error attached."""
    results = []
    with httpx.Client(base_url=FASTAPI_URL, timeout=30.0) as client:
        for i, row in enumerate(rows, start=1):
            payload = {k: v for k, v in row.items() if k in PREDICT_FIELDS}
            response = client.post("/predict", json=payload)
            response.raise_for_status()
            predicted = response.json()["predicted_salary_usd"]
            error = predicted - row["salary_in_usd"]
            results.append({
                **row,
                "predicted_salary_usd": predicted,
                "error_usd": error,
            })
            print(f"  [{i:2d}/{len(rows)}] {row['job_title']:35s}  "
                  f"predicted: ${predicted:>9,}  "
                  f"actual: ${row['salary_in_usd']:>9,}  "
                  f"error: ${error:>+9,}")
    return results


def main():
    default_path = Path(__file__).parent / os.getenv("TEST_DATA_FILE", "data/llm_test_data.csv")
    csv_path = Path(sys.argv[1]) if len(sys.argv) > 1 else default_path

    if not csv_path.exists():
        print(f"Error: CSV file not found at {csv_path}")
        print("Provide a path as an argument: python analyse.py path/to/file.csv")
        sys.exit(1)

    print(f"Loading test data from {csv_path}...")
    rows = load_csv(csv_path)
    if not rows:
        print("Error: no valid rows found in CSV.")
        sys.exit(1)
    print(f"Loaded {len(rows)} rows.\n")

    print("Fetching predictions from FastAPI...")
    try:
        results = get_predictions(rows)
    except KeyboardInterrupt:
        print("\nCancelled.")
        sys.exit(0)
    except httpx.ConnectError:
        print(f"Error: could not connect to FastAPI at {FASTAPI_URL}. Is it running?")
        sys.exit(1)
    except httpx.HTTPStatusError as e:
        print(f"Error: FastAPI returned {e.response.status_code}: {e.response.text}")
        sys.exit(1)
    print(f"\nGot {len(results)} predictions.\n")

    print("Asking LLM for analysis...")
    try:
        analysis = llm_client.analyse_predictions(results)
    except KeyboardInterrupt:
        print("\nCancelled.")
        sys.exit(0)
    except ValueError as e:
        print(f"Error: {e}")
        sys.exit(1)

    print(f"\n--- LLM Analysis ---\n{analysis}\n")


if __name__ == "__main__":
    main()
