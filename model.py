import logging
import os
import io
from typing import Union, BinaryIO, Optional, Tuple

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
from huggingface_hub import hf_hub_download

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Model configuration (can be overridden with env vars if needed)
IMAGE_SIZE = int(os.getenv("IMAGE_SIZE", 256))
IN_CHANNELS = 3
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
REPO_ID = os.getenv("HF_REPO_ID", "kritimbista/my-model-weights")
MODEL_FILENAME = os.getenv("HF_MODEL_FILENAME", "model_weights.pth")


# -----------------------------
# Model Definition (ResNet9)
# -----------------------------
class ResNet9(nn.Module):
    """ResNet9 architecture for plant disease classification"""

    def __init__(self, in_channels: int, num_classes: int):
        super().__init__()

        def conv_block(in_channels: int, out_channels: int, pool: bool = False) -> nn.Sequential:
            layers = [
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
            ]
            if pool:
                layers.append(nn.MaxPool2d(2))
            return nn.Sequential(*layers)

        self.conv1 = conv_block(in_channels, 64)
        self.conv2 = conv_block(64, 128, pool=True)
        self.res1 = nn.Sequential(conv_block(128, 128), conv_block(128, 128))
        self.conv3 = conv_block(128, 256, pool=True)
        self.conv4 = conv_block(256, 512, pool=True)
        self.res2 = nn.Sequential(conv_block(512, 512), conv_block(512, 512))
        
        # Classifier with AdaptiveAvgPool2d to handle any input size
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),  # This outputs (batch, 512, 1, 1)
            nn.Flatten(),              # This flattens to (batch, 512)
            nn.Linear(512, num_classes),
        )

    def forward(self, xb: torch.Tensor) -> torch.Tensor:
        out = self.conv1(xb)
        out = self.conv2(out)
        out = self.res1(out) + out
        out = self.conv3(out)
        out = self.conv4(out)
        out = self.res2(out) + out
        out = self.classifier(out)
        return out


# -----------------------------
# Class Names
# -----------------------------
class_names = [
    "Apple___Apple_scab",
    "Apple___Black_rot",
    "Apple___Cedar_apple_rust",
    "Apple___healthy",
    "Blueberry___healthy",
    "Cherry_(including_sour)___Powdery_mildew",
    "Cherry_(including_sour)___healthy",
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot",
    "Corn_(maize)___Common_rust_",
    "Corn_(maize)___Northern_Leaf_Blight",
    "Corn_(maize)___healthy",
    "Grape___Black_rot",
    "Grape___Esca_(Black_Measles)",
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)",
    "Grape___healthy",
    "Orange___Haunglongbing_(Citrus_greening)",
    "Peach___Bacterial_spot",
    "Peach___healthy",
    "Pepper,_bell___Bacterial_spot",
    "Pepper,_bell___healthy",
    "Potato___Early_blight",
    "Potato___Late_blight",
    "Potato___healthy",
    "Raspberry___healthy",
    "Soybean___healthy",
    "Squash___Powdery_mildew",
    "Strawberry___Leaf_scorch",
    "Strawberry___healthy",
    "Tomato___Bacterial_spot",
    "Tomato___Early_blight",
    "Tomato___Late_blight",
    "Tomato___Leaf_Mold",
    "Tomato___Septoria_leaf_spot",
    "Tomato___Spider_mites Two-spotted_spider_mite",
    "Tomato___Target_Spot",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
    "Tomato___Tomato_mosaic_virus",
    "Tomato___healthy",
]


# -----------------------------
# Image Transformation
# -----------------------------
transform = transforms.Compose(
    [
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)


# -----------------------------
# Load Model Function
# -----------------------------
def load_model(device: torch.device = DEVICE, token: Optional[str] = None) -> nn.Module:
    """
    Load the ResNet9 model with pretrained weights from Hugging Face Hub.

    Args:
        device: torch.device to place model on.
        token: optional HF token for private repos (or None for public repos / cached auth).

    Returns:
        nn.Module (model ready for inference)
    """
    try:
        # instantiate architecture
        model = ResNet9(in_channels=IN_CHANNELS, num_classes=len(class_names))

        # download from HF (hf_hub_download will cache locally)
        logger.info(f"Downloading weights from HF repo='{REPO_ID}' filename='{MODEL_FILENAME}'")
        model_path = hf_hub_download(repo_id=REPO_ID, filename=MODEL_FILENAME, token=token)

        # load checkpoint (robust to different save styles)
        checkpoint = torch.load(model_path, map_location=device)

        # Try several common checkpoint formats
        loaded_model = None
        if isinstance(checkpoint, nn.Module):
            # saved entire model object
            loaded_model = checkpoint
            logger.info("Loaded a full model object from checkpoint.")
        elif isinstance(checkpoint, dict):
            # common cases:
            # 1) saved state_dict directly
            # 2) saved dict with 'model_state_dict' or 'state_dict'
            if "model_state_dict" in checkpoint and isinstance(checkpoint["model_state_dict"], dict):
                # Load with strict=False to handle architecture mismatches
                missing, unexpected = model.load_state_dict(checkpoint["model_state_dict"], strict=False)
                if missing:
                    logger.warning(f"Missing keys when loading: {missing}")
                if unexpected:
                    logger.warning(f"Unexpected keys when loading: {unexpected}")
                logger.info("Loaded 'model_state_dict' from checkpoint dict.")
            elif "state_dict" in checkpoint and isinstance(checkpoint["state_dict"], dict):
                missing, unexpected = model.load_state_dict(checkpoint["state_dict"], strict=False)
                if missing:
                    logger.warning(f"Missing keys when loading: {missing}")
                if unexpected:
                    logger.warning(f"Unexpected keys when loading: {unexpected}")
                logger.info("Loaded 'state_dict' from checkpoint dict.")
            else:
                # maybe checkpoint is the state_dict itself
                try:
                    missing, unexpected = model.load_state_dict(checkpoint, strict=False)
                    if missing:
                        logger.warning(f"Missing keys when loading: {missing}")
                    if unexpected:
                        logger.warning(f"Unexpected keys when loading: {unexpected}")
                    logger.info("Loaded checkpoint as state_dict.")
                except Exception as e:
                    # fallback: try to detect if this dict includes keys matching model.state_dict()
                    ck_keys = set(k.split(".")[0] for k in checkpoint.keys())
                    model_keys = set(k.split(".")[0] for k in model.state_dict().keys())
                    if ck_keys & model_keys:
                        missing, unexpected = model.load_state_dict(checkpoint, strict=False)
                        if missing:
                            logger.warning(f"Missing keys when loading: {missing}")
                        if unexpected:
                            logger.warning(f"Unexpected keys when loading: {unexpected}")
                        logger.info("Loaded checkpoint as state_dict (fallback detection).")
                    else:
                        raise RuntimeError(
                            "Checkpoint dict doesn't look like a state_dict and no 'model_state_dict' field found."
                        ) from e
        else:
            raise RuntimeError("Unrecognized checkpoint format.")

        # if checkpoint was a full model object, use it
        if loaded_model is not None:
            model = loaded_model

        model.to(device)
        model.eval()
        logger.info(f"Model loaded successfully on {device} (from {model_path})")
        return model

    except Exception as e:
        logger.exception("Failed to load model")
        raise RuntimeError(f"Model loading failed: {e}") from e


# -----------------------------
# Predict Function (for FastAPI)
# -----------------------------
def predict(
    image_bytes: Union[bytes, BinaryIO], model: nn.Module, device: torch.device = DEVICE
) -> Tuple[str, float]:
    """
    Predict plant disease from image bytes.

    Returns:
        (label, confidence) tuple.
    """
    try:
        # Ensure bytes -> file-like for PIL
        if isinstance(image_bytes, (bytes, bytearray)):
            image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        else:
            # BinaryIO / file-like
            image = Image.open(image_bytes).convert("RGB")

        image_tensor = transform(image).unsqueeze(0).to(device)

        with torch.no_grad():
            outputs = model(image_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            confidence, predicted_idx = torch.max(probabilities, dim=1)
            confidence_value = float(confidence[0].cpu().item())
            predicted_idx_value = int(predicted_idx[0].cpu().item())

        label = class_names[predicted_idx_value]
        logger.info(f"Prediction: {label} (confidence: {confidence_value:.4f})")
        return label, confidence_value

    except Exception as e:
        logger.exception("Prediction failed")
        raise RuntimeError(f"Prediction failed: {e}") from e


# Backwards-compatible helper returning only the label (string)
def predict_label(image_bytes: Union[bytes, BinaryIO], model: nn.Module, device: torch.device = DEVICE) -> str:
    """Compatibility wrapper: returns only the predicted label string."""
    label, _ = predict(image_bytes, model, device=device)
    return label


# -----------------------------
# Helper for local file path (for FastAPI use)
# -----------------------------
def get_prediction_from_path(image_path: str, model: nn.Module) -> Tuple[str, float]:
    """Predict directly from image file path (used in FastAPI upload)."""
    with open(image_path, "rb") as f:
        return predict(f, model)


# -----------------------------
# Test Run (only when executed directly)
# -----------------------------
if __name__ == "__main__":
    try:
        hf_token = os.getenv("HF_TOKEN")  # set if private repo
        m = load_model(token=hf_token)
        logger.info("✅ Model test successful")
        # quick smoke test (no image file provided here)
    except Exception as e:
        logger.error(f"❌ Model test failed: {e}")
