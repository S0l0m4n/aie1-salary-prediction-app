import httpx
from fastapi import HTTPException

from .config import OLLAMA_BASE_URL, OLLAMA_MODEL

_available: bool = False


def check_ollama() -> bool:
    """Ping the Ollama server. Returns True if reachable, False otherwise."""
    global _available
    try:
        response = httpx.get(f"{OLLAMA_BASE_URL.rstrip('/')}/api/tags", timeout=5.0)
        _available = response.status_code == 200
    except httpx.RequestError:
        _available = False
    return _available


def explain_prediction(features: dict, salary: int, actual_salary: int | None = None) -> str:
    profile = (
        f"- Work year: {features['work_year']}\n"
        f"- Experience level: {features['experience_level']}\n"
        f"- Job title: {features['job_title']}\n"
        f"- Remote ratio: {features['remote_ratio']}%\n"
        f"- Company location: {features['company_location']}\n"
        f"- Company size: {features['company_size']}\n"
        f"- Working abroad: {features['is_abroad']}"
    )

    if actual_salary is not None:
        prompt = (
            f"A data science salary prediction model estimated an annual salary of ${salary:,} USD "
            f"for the following profile:\n{profile}\n\n"
            f"The actual salary was ${actual_salary:,} USD. "
            f"In 2-3 sentences, comment on the difference between the predicted and actual values "
            f"and why the gap might exist given these factors."
        )
    else:
        prompt = (
            f"A data science salary prediction model estimated an annual salary of ${salary:,} USD "
            f"for the following profile:\n{profile}\n\n"
            f"In 2-3 sentences, explain why this salary estimate is reasonable given these factors."
        )

    payload = {
        "model": OLLAMA_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
    }

    url = f"{OLLAMA_BASE_URL.rstrip('/')}/api/chat"

    try:
        response = httpx.post(url, json=payload, timeout=60.0)
    except httpx.TimeoutException as exc:
        raise HTTPException(status_code=504, detail=f"Ollama request timed out: {exc}") from exc
    except httpx.RequestError as exc:
        raise HTTPException(status_code=502, detail=f"Could not reach Ollama at {OLLAMA_BASE_URL}: {exc}") from exc

    if response.status_code != 200:
        raise HTTPException(status_code=502, detail=f"Ollama returned {response.status_code}: {response.text}")

    data = response.json()
    content = (data.get("message") or {}).get("content")
    if not content:
        raise HTTPException(status_code=502, detail=f"Ollama response missing message content: {data}")

    return content
