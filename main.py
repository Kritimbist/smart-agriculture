from fastapi import FastAPI, File, UploadFile, Request, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from model import load_model, predict  # Changed: import predict instead of get_prediction_from_path
import shutil
import os
import logging

# -------------------------------
# App Setup
# -------------------------------
app = FastAPI(title="Plant Disease Detection API")

# Mount static and templates directories
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# Upload directory
UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Allowed file extensions
 ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png"}

# -------------------------------
# Helper function to validate files
# -------------------------------
def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

# -------------------------------
# Load the model once at startup
# -------------------------------
logger = logging.getLogger("uvicorn")
try:
    hf_token = os.getenv("HF_TOKEN")  # optional for private HF repo
    model = load_model(token=hf_token)
    logger.info("✅ Model loaded successfully at startup")
except Exception as e:
    logger.error(f"❌ Failed to load model: {e}")
    model = None

# -------------------------------
# Routes
# -------------------------------
@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/predict", response_class=HTMLResponse)
async def predict_disease(request: Request, file: UploadFile = File(...)):
    if not allowed_file(file.filename):
        raise HTTPException(status_code=400, detail="Invalid file type. Only JPG, JPEG, PNG allowed.")
    
    if model is None:
        raise HTTPException(status_code=500, detail="Model is not loaded. Try again later.")

    file_path = os.path.join(UPLOAD_DIR, file.filename)

    try:
        # Save uploaded file temporarily
        with open(file_path, "wb") as buffer:       
            shutil.copyfileobj(file.file, buffer)

        # Get prediction using the predict function from model.py
        with open(file_path, "rb") as img_file:
            result = predict(img_file.read(), model)
        
        # Extract label and confidence from result
        label = result["label"]
        confidence = result["confidence"]

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing file: {e}")

    finally:
        # Clean up uploaded file
        if os.path.exists(file_path):
            os.remove(file_path)

    # Return template with prediction and confidence
    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "prediction": label,
            "confidence": f"{confidence*100:.2f}%",  # formatted as percentage
            "description": result.get("description", ""),
            "remedy": result.get("remedy", "")
        }
    )
