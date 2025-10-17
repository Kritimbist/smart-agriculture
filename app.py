from fastapi import FastAPI, File, UploadFile, Request, Form
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import uvicorn
import shutil
import os

# Import your model functions
from model import load_model, get_prediction_from_path

# -------------------------------
# FastAPI app setup
# -------------------------------
app = FastAPI(title="🌿 Plant Disease Detection API")

# Static files and templates
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# Global model variable
model = None


# -------------------------------
# Startup event
# -------------------------------
@app.on_event("startup")
async def load_model_on_startup():
    """Load the ML model from Hugging Face on startup."""
    global model
    hf_token = os.getenv("HF_TOKEN")  # optional: for private repos
    model = load_model(token=hf_token)
    print("✅ Model loaded successfully on startup")


# -------------------------------
# Routes
# -------------------------------
@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    """Serve the upload web interface."""
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/predict", response_class=HTMLResponse)
async def predict_image(request: Request, file: UploadFile = File(...)):
    """Handle image upload and show prediction on web UI."""
    try:
        # Save uploaded file temporarily
        upload_dir = "uploads"
        os.makedirs(upload_dir, exist_ok=True)
        file_path = os.path.join(upload_dir, file.filename)

        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Predict (returns label and confidence)
        label, confidence = get_prediction_from_path(file_path, model)

        # Delete temporary file
        os.remove(file_path)

        # Render result on same HTML page
        return templates.TemplateResponse(
            "index.html",
            {
                "request": request,
                "prediction": label,
                "confidence": f"{confidence * 100:.2f}%",
            },
        )

    except Exception as e:
        return templates.TemplateResponse(
            "index.html",
            {"request": request, "error": str(e)},
            status_code=500,
        )


@app.post("/api/predict", response_class=JSONResponse)
async def api_predict_image(file: UploadFile = File(...)):
    """JSON API endpoint for prediction."""
    try:
        upload_dir = "uploads"
        os.makedirs(upload_dir, exist_ok=True)
        file_path = os.path.join(upload_dir, file.filename)

        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        label, confidence = get_prediction_from_path(file_path, model)

        os.remove(file_path)
        return JSONResponse(
            content={"prediction": label, "confidence": round(confidence * 100, 2)}
        )

    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)


# -------------------------------
# Run the app (local testing)
# -------------------------------
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
