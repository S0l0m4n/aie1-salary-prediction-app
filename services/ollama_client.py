"""
Ollama client helpers for salary analysis.
"""

import json
import httpx

OLLAMA_URL = "http://localhost:11434"
OLLAMA_MODEL = "llama3.2"


def analyse_predictions(results: list[dict]) -> str:
    """Send a batch of predicted-vs-actual salary results to Ollama and return its analysis."""
    prompt = (
        "You are a data analyst reviewing the output of a salary prediction model "
        "trained on data science job data. Here are the predicted vs. actual salaries "
        f"for {len(results)} test cases:\n\n"
        f"{json.dumps(results, indent=2)}\n\n"
        "Comment on the model's accuracy, any patterns in the errors "
        "(e.g. over/underpredicting certain roles or experience levels), and what might "
        "explain those patterns."
    )

    payload = {
        "model": OLLAMA_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
    }

    response = httpx.post(f"{OLLAMA_URL}/api/chat", json=payload, timeout=60.0)
    response.raise_for_status()
    return response.json()["message"]["content"]
