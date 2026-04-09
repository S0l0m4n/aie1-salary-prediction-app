from fastapi import FastAPI, HTTPException
from .schema import PredictRequest, PredictResponse
from . import model, ollama

app = FastAPI(title="Salary Prediction API")


@app.on_event("startup")
def startup():
    try:
        model.load_model()
    except FileNotFoundError as e:
        # Allow the server to start without the model — /predict will return 503
        print(f"WARNING: {e}")

    if not ollama.check_ollama():
        print(f"WARNING: Ollama not reachable at {ollama.OLLAMA_BASE_URL} — /predict will return 503")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest):
    if model._model is None:
        raise HTTPException(status_code=503, detail="Model not loaded.")
    if not ollama._available:
        raise HTTPException(status_code=503, detail="Ollama not available.")
    features = {
        "work_year": request.work_year,
        "experience_level": request.experience_level.value,
        "job_title": request.job_title,
        "remote_ratio": request.remote_ratio.value,
        "company_location": request.company_location.value,
        "company_size": request.company_size.value,
        "is_abroad": request.is_abroad,
    }
    salary = model.predict(features)
    rounded_salary = round(salary / 1000) * 1000
    explanation = ollama.explain_prediction(features, rounded_salary, request.actual_salary_usd)
    return PredictResponse(predicted_salary_usd=rounded_salary, explanation=explanation)
