from fastapi import FastAPI, UploadFile, File
from model import load_model, predict
import torch
import io

app = FastAPI(title="Plant Disease Detection API")

# Load model once at startup
model = load_model(torch.device("cpu"))

@app.get("/")
def root():
    return {"message": "Welcome to Plant Disease Detection API!"}

@app.post("/predict")
async def predict_image(file: UploadFile = File(...)):
    contents = await file.read()
    image_bytes = io.BytesIO(contents)
    result = predict(image_bytes, model)
    return {"prediction": result}
