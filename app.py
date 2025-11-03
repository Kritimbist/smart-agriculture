from fastapi import FastAPI, File, UploadFile, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import shutil
import os

from model import load_model, predict

# -------------------------------
# FastAPI app setup
# -------------------------------
app = FastAPI(title="Plant Disease Detection")

# Mount static directory
app.mount("/static", StaticFiles(directory="static"), name="static")

# Templates
templates = Jinja2Templates(directory="templates")

# Load model
model = load_model()


# -------------------------------
# Routes
# -------------------------------
@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {
        "request": request,
        "result": None,
        "error": None
    })


@app.post("/predict", response_class=HTMLResponse)
async def predict_disease(request: Request, file: UploadFile = File(...)):
    try:
        # Save uploaded file temporarily
        upload_dir = "static/uploads"
        os.makedirs(upload_dir, exist_ok=True)
        file_path = os.path.join(upload_dir, file.filename)
        
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Predict
        with open(file_path, "rb") as img_file:
            result = predict(img_file.read(), model)

        # Prepare context for HTML
        context = {
            "request": request,
            "result": result,
            "image_path": "/" + file_path.replace("\\", "/"),
            "error": None
        }
        return templates.TemplateResponse("index.html", context)

    except Exception as e:
        # Log the error
        print(f"Error during prediction: {str(e)}")
        import traceback
        traceback.print_exc()
        
        return templates.TemplateResponse("index.html", {
            "request": request,
            "result": None,
            "error": str(e),
            "image_path": None
        })


@app.get("/health")
async def health_check():
    """Health check endpoint for deployment"""
    return {"status": "healthy", "model_loaded": model is not None}


@app.get("/about")
async def about():
    return {"message": "Plant Disease Detection API with FastAPI"}
