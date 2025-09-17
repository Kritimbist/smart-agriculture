from fastapi import FastAPI, Request, HTTPException
import os
from model import predict

app = FastAPI()

# API keys from environment variable
API_KEYS = os.getenv("API_KEYS", "demo123").split(",")

@app.middleware("http")
async def check_api_key(request: Request, call_next):
    api_key = request.headers.get("x-api-key")
    if api_key not in API_KEYS:
        raise HTTPException(status_code=401, detail="Invalid API Key")
    response = await call_next(request)
    return response

@app.post("/predict")
async def get_prediction(data: dict):
    result = predict(data)
    return {"prediction": result}
