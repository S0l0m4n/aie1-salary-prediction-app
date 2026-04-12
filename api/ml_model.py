import os

import joblib
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

MODEL_PATH = os.getenv("MODEL_PATH", "models/catboost_model.joblib")

_model = None


def load_model():
    global _model
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Model file not found at '{MODEL_PATH}'. "
            "Drop the joblib file there and restart the server."
        )
    _model = joblib.load(MODEL_PATH)


def predict(features: dict) -> float:
    if _model is None:
        raise RuntimeError("Model is not loaded.")
    df = pd.DataFrame([features])
    result = _model.predict(df)
    return float(result[0])
