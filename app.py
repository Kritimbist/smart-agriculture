from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import os
from model import predict

app = FastAPI(
    title="ML Model API",
    description="API for model predictions",
    version="1.0.0"
)

# Add CORS middleware to allow requests from your website
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Change to your website domain in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# API keys from environment variable
API_KEYS = os.getenv("API_KEYS", "demo123").split(",")

@app.middleware("http")
async def check_api_key(request: Request, call_next):
    # Skip API key check for docs and health endpoints
    if request.url.path in ["/", "/docs", "/openapi.json", "/health"]:
        response = await call_next(request)
        return response
    
    api_key = request.headers.get("x-api-key")
    if api_key not in API_KEYS:
        raise HTTPException(status_code=401, detail="Invalid API Key")
    response = await call_next(request)
    return response

@app.get("/")
async def root():
    return {
        "message": "ML Model API is running",
        "version": "1.0.0",
        "endpoints": {
            "predict": "/predict",
            "health": "/health",
            "docs": "/docs"
        }
    }

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

@app.post("/predict")
async def get_prediction(data: dict):
    try:
        result = predict(data)
        return {"prediction": result}  # Fixed: Changed // to #
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
