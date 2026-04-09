"""
Salary predictor router. Hit the /predict endpoint with the data science job
parameters to get a predicted value for the salary.

There is also a dummy /health endpoint to check the server is running.
""" 

from fastapi import APIRouter, HTTPException

from .. import ml_model as model
from ..schemas.schema import PredictRequest, PredictResponse

router = APIRouter(prefix="/predict")

@router.on_event("startup")
def startup():
    try:
        model.load_model()
    except FileNotFoundError as e:
        # Allow the server to start without the model — /predict will return 503
        print(f"WARNING: {e}")


@router.post("", response_model=PredictResponse)
def predict(request: PredictRequest):
    if model._model is None:
        raise HTTPException(status_code=503, detail="Model not loaded.")
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
    return PredictResponse(predicted_salary_usd=round(salary / 1000) * 1000)
