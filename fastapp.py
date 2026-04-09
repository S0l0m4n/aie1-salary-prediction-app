"""
Main entry point for Salary Prediction API.

After the app instance is created, the predict router /predict is registered.
"""

from fastapi import FastAPI

from app.routers import predict

app = FastAPI(title="Salary Prediction API")

# The predict.router syntax means to look for the `router` variable inside the
# predict.py router file
app.include_router(predict.router)

# A sanity health check
@app.get("/health")
def health():
    return {"status": "ok"}
