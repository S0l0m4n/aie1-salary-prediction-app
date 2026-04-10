"""
Ollama client helpers for salary analysis.
"""

import json
import httpx
from pathlib import Path

OLLAMA_URL = "http://localhost:11434"
OLLAMA_MODEL = "llama3.2"

_FEATURE_GUIDE_PATH = Path(__file__).parent.parent / "data" / "feature_guide.json"


def _load_feature_guide() -> dict:
    with open(_FEATURE_GUIDE_PATH, encoding="utf-8") as f:
        return json.load(f)


def analyse_predictions(results: list[dict]) -> str:
    """Send a batch of predicted-vs-actual salary results to Ollama and return its analysis."""
    feature_guide = _load_feature_guide()

    system_message = (
        "You are a data analyst reviewing the output of a machine learning salary prediction model "
        "trained on data science job data. "
        "Use the feature guide below to interpret any coded values (e.g. experience level, company size) "
        "that appear in the prediction results."
    )

    user_message = (
        "## Feature Guide\n\n"
        f"{json.dumps(feature_guide, indent=2)}\n\n"
        f"## Prediction Results ({len(results)} test cases)\n\n"
        f"{json.dumps(results, indent=2)}\n\n"
        "## Task\n\n"
        "Analyse the model's performance. Cover:\n"
        "- Overall accuracy and the distribution of prediction errors\n"
        "- Patterns in over- or underprediction (e.g. by role, experience level, location, company size)\n"
        "- Likely explanations for those patterns based on the feature guide and the data"
    )

    payload = {
        "model": OLLAMA_MODEL,
        "messages": [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_message},
        ],
        "stream": False,
    }

    response = httpx.post(f"{OLLAMA_URL}/api/chat", json=payload, timeout=60.0)
    response.raise_for_status()
    return response.json()["message"]["content"]
