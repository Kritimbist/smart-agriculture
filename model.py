import logging
import os
import io
from typing import Union, BinaryIO, Optional

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
from huggingface_hub import hf_hub_download

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# -----------------------------
# Model Configuration
# -----------------------------
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

        def conv_block(in_c: int, out_c: int, pool: bool = False) -> nn.Sequential:
            layers = [
                nn.Conv2d(in_c, out_c, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_c),
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
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
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
# Class Names (ALPHABETICAL ORDER - matches ImageFolder)
# -----------------------------
class_names = [
    'Apple___Apple_scab',
    'Apple___Black_rot',
    'Apple___Cedar_apple_rust',
    'Apple___healthy',
    'Blueberry___healthy',
    'Cherry_(including_sour)___Powdery_mildew',
    'Cherry_(including_sour)___healthy',
    'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot',
    'Corn_(maize)___Common_rust_',
    'Corn_(maize)___Northern_Leaf_Blight',
    'Corn_(maize)___healthy',
    'Grape___Black_rot',
    'Grape___Esca_(Black_Measles)',
    'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
    'Grape___healthy',
    'Orange___Haunglongbing_(Citrus_greening)',
    'Peach___Bacterial_spot',
    'Peach___healthy',
    'Pepper,_bell___Bacterial_spot',
    'Pepper,_bell___healthy',
    'Potato___Early_blight',
    'Potato___Late_blight',
    'Potato___healthy',
    'Raspberry___healthy',
    'Soybean___healthy',
    'Squash___Powdery_mildew',
    'Strawberry___Leaf_scorch',
    'Strawberry___healthy',
    'Tomato___Bacterial_spot',
    'Tomato___Early_blight',
    'Tomato___Late_blight',
    'Tomato___Leaf_Mold',
    'Tomato___Septoria_leaf_spot',
    'Tomato___Spider_mites Two-spotted_spider_mite',
    'Tomato___Target_Spot',
    'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
    'Tomato___Tomato_mosaic_virus',
    'Tomato___healthy'
]


# -----------------------------
# Disease Info (Descriptions + Remedies)
# -----------------------------
disease_info = {
    # ---------------- Apple ----------------
    "Apple___Apple_scab": {
        "description": "Apple scab is a fungal disease caused by Venturia inaequalis. It produces dark, scabby lesions on leaves, fruit, and young twigs.",
        "remedy": "Use resistant apple varieties, apply fungicides like captan or mancozeb, and remove fallen leaves to reduce fungal spores."
    },
    "Apple___Black_rot": {
        "description": "Black rot is caused by the fungus Botryosphaeria obtusa. It affects leaves, fruit, and bark, forming circular lesions and rotting fruit.",
        "remedy": "Prune infected branches, burn fallen debris, and use fungicides such as thiophanate-methyl or copper-based sprays."
    },
    "Apple___Cedar_apple_rust": {
        "description": "Cedar apple rust is caused by Gymnosporangium juniperi-virginianae. It forms orange, gelatinous spots on leaves and fruits.",
        "remedy": "Remove nearby cedar trees if possible, apply fungicides during early spring, and grow resistant apple varieties."
    },
    "Apple___healthy": {
        "description": "This apple leaf is healthy and free from any disease or fungal infection.",
        "remedy": "Continue proper irrigation, pruning, and nutrient management for optimal growth."
    },

    # ---------------- Blueberry ----------------
    "Blueberry___healthy": {
        "description": "The blueberry plant shows no disease symptoms and appears vigorous and healthy.",
        "remedy": "Maintain soil acidity, water regularly, and apply mulch to prevent weeds."
    },

    # ---------------- Cherry ----------------
    "Cherry_(including_sour)___Powdery_mildew": {
        "description": "Powdery mildew is a fungal disease causing white powder-like patches on cherry leaves and shoots.",
        "remedy": "Prune overcrowded branches, improve air circulation, and apply sulfur or neem oil sprays."
    },
    "Cherry_(including_sour)___healthy": {
        "description": "Cherry leaves are healthy, showing no fungal or bacterial infections.",
        "remedy": "Ensure proper watering and fertilization; maintain clean surroundings to prevent pests."
    },

    # ---------------- Corn ----------------
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot": {
        "description": "Gray leaf spot is caused by Cercospora species, forming rectangular gray lesions on leaves.",
        "remedy": "Rotate crops, use resistant hybrids, and apply fungicides like strobilurins when necessary."
    },
    "Corn_(maize)___Common_rust_": {
        "description": "Common rust is caused by Puccinia sorghi. It shows reddish-brown pustules on leaves.",
        "remedy": "Use resistant varieties and apply fungicides if rust severity increases."
    },
    "Corn_(maize)___Northern_Leaf_Blight": {
        "description": "Northern leaf blight is caused by Exserohilum turcicum, producing long gray lesions on leaves.",
        "remedy": "Use resistant hybrids, rotate crops, and spray fungicides at early infection stages."
    },
    "Corn_(maize)___healthy": {
        "description": "Corn leaves are green and healthy with no visible disease.",
        "remedy": "Maintain proper nitrogen levels and irrigation for optimal yield."
    },

    # ---------------- Grape ----------------
    "Grape___Black_rot": {
        "description": "Black rot is a fungal disease caused by Guignardia bidwellii, producing black spots on leaves and shriveled fruit.",
        "remedy": "Prune infected parts, improve air circulation, and apply fungicides like myclobutanil."
    },
    "Grape___Esca_(Black_Measles)": {
        "description": "Esca (Black Measles) causes black stripes and spots on leaves and fruit, eventually killing the vine.",
        "remedy": "Remove and destroy infected vines, avoid wounding, and improve drainage."
    },
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)": {
        "description": "Leaf blight caused by Pseudocercospora vitis results in angular brown leaf spots.",
        "remedy": "Prune affected leaves and apply protective fungicides."
    },
    "Grape___healthy": {
        "description": "The grapevine is healthy with vibrant green leaves.",
        "remedy": "Maintain balanced fertilization and proper pruning to enhance airflow."
    },

    # ---------------- Orange ----------------
    "Orange___Haunglongbing_(Citrus_greening)": {
        "description": "Citrus greening (HLB) is caused by Candidatus Liberibacter bacteria, leading to yellow shoots and misshapen fruit.",
        "remedy": "Remove infected trees, control psyllid vectors, and plant disease-free seedlings."
    },

    # ---------------- Peach ----------------
    "Peach___Bacterial_spot": {
        "description": "Bacterial spot is caused by Xanthomonas campestris, leading to black lesions on leaves and fruit.",
        "remedy": "Apply copper-based bactericides and avoid overhead irrigation."
    },
    "Peach___healthy": {
        "description": "Peach leaves are healthy and disease-free.",
        "remedy": "Maintain good soil drainage and use resistant cultivars."
    },

    # ---------------- Pepper ----------------
    "Pepper,_bell___Bacterial_spot": {
        "description": "Bacterial spot in bell pepper is caused by Xanthomonas species, forming dark water-soaked spots.",
        "remedy": "Use certified seeds, rotate crops, and apply copper fungicides as preventive measures."
    },
    "Pepper,_bell___healthy": {
        "description": "The bell pepper plant is healthy and free of bacterial or fungal infection.",
        "remedy": "Maintain proper watering and pest management."
    },

    # ---------------- Potato ----------------
    "Potato___Early_blight": {
        "description": "Early blight is caused by Alternaria solani, producing concentric brown rings on leaves.",
        "remedy": "Use disease-free seeds, rotate crops, and apply fungicides such as chlorothalonil."
    },
    "Potato___Late_blight": {
        "description": "Late blight is caused by Phytophthora infestans, leading to dark lesions on leaves and tubers.",
        "remedy": "Avoid wet conditions, use resistant varieties, and apply fungicides like metalaxyl."
    },
    "Potato___healthy": {
        "description": "The potato plant is healthy with no visible blight or disease symptoms.",
        "remedy": "Maintain proper hilling and watering practices."
    },

    # ---------------- Raspberry ----------------
    "Raspberry___healthy": {
        "description": "Raspberry plant appears healthy with no visible diseases.",
        "remedy": "Ensure proper pruning and air circulation to prevent fungal infections."
    },

    # ---------------- Soybean ----------------
    "Soybean___healthy": {
        "description": "Soybean plants show no signs of disease and appear vigorous.",
        "remedy": "Rotate crops and control weeds to maintain plant health."
    },

    # ---------------- Squash ----------------
    "Squash___Powdery_mildew": {
        "description": "Powdery mildew appears as white powdery spots on leaves and stems of squash.",
        "remedy": "Remove infected leaves, improve ventilation, and apply sulfur-based fungicides."
    },

    # ---------------- Strawberry ----------------
    "Strawberry___Leaf_scorch": {
        "description": "Leaf scorch causes reddish-brown spots on strawberry leaves, eventually leading to withering.",
        "remedy": "Remove infected leaves, avoid overhead watering, and use resistant cultivars."
    },
    "Strawberry___healthy": {
        "description": "The strawberry plant is healthy with lush green leaves.",
        "remedy": "Ensure proper sunlight and spacing to prevent fungal growth."
    },

    # ---------------- Tomato ----------------
    "Tomato___Bacterial_spot": {
        "description": "Bacterial spot is caused by Xanthomonas species, creating small dark lesions on leaves and fruit.",
        "remedy": "Use disease-free seeds, copper sprays, and rotate crops regularly."
    },
    "Tomato___Early_blight": {
        "description": "Early blight is caused by Alternaria solani, leading to concentric dark rings on older leaves.",
        "remedy": "Prune infected leaves, improve air circulation, and use chlorothalonil-based fungicides."
    },
    "Tomato___Late_blight": {
        "description": "Late blight caused by Phytophthora infestans forms large water-soaked spots on leaves and fruit.",
        "remedy": "Avoid overhead watering, destroy infected plants, and apply systemic fungicides."
    },
    "Tomato___Leaf_Mold": {
        "description": "Leaf mold is caused by Passalora fulva, forming yellow patches on the upper leaf surface.",
        "remedy": "Improve ventilation, reduce humidity, and apply copper-based fungicides."
    },
    "Tomato___Septoria_leaf_spot": {
        "description": "Septoria leaf spot causes numerous small circular spots with gray centers.",
        "remedy": "Remove infected leaves, avoid wet foliage, and use fungicides such as mancozeb."
    },
    "Tomato___Spider_mites Two-spotted_spider_mite": {
        "description": "Spider mites cause yellow stippling and webbing on leaves, leading to leaf drop.",
        "remedy": "Spray neem oil or insecticidal soap and maintain humidity to deter mites."
    },
    "Tomato___Target_Spot": {
        "description": "Target spot, caused by Corynespora cassiicola, produces target-like concentric lesions.",
        "remedy": "Remove infected leaves, rotate crops, and apply preventive fungicides."
    },
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus": {
        "description": "This viral disease causes upward curling of leaves and stunted growth.",
        "remedy": "Control whitefly vectors, remove infected plants, and use resistant hybrids."
    },
    "Tomato___Tomato_mosaic_virus": {
        "description": "TMV causes mosaic-like mottling and deformation of tomato leaves.",
        "remedy": "Avoid handling plants after tobacco use, disinfect tools, and use resistant varieties."
    },
    "Tomato___healthy": {
        "description": "The tomato plant is healthy, showing no signs of disease.",
        "remedy": "Maintain consistent watering and nutrient balance to prevent stress."
    },
}


# -----------------------------
# Image Transformation (MUST MATCH TRAINING)
# -----------------------------
transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
])


# -----------------------------
# Load Model Function
# -----------------------------
def load_model(device: torch.device = DEVICE, token: Optional[str] = None) -> nn.Module:
    try:
        logger.info(f"Downloading weights from HF repo='{REPO_ID}' filename='{MODEL_FILENAME}'")
        model_path = hf_hub_download(repo_id=REPO_ID, filename=MODEL_FILENAME, token=token)
        checkpoint = torch.load(model_path, map_location=device)

        # Determine number of classes
        num_classes = len(class_names)
        if isinstance(checkpoint, dict) and "num_classes" in checkpoint:
            num_classes = checkpoint["num_classes"]
        
        model = ResNet9(in_channels=IN_CHANNELS, num_classes=num_classes)

        if isinstance(checkpoint, dict):
            if "model_state_dict" in checkpoint:
                model.load_state_dict(checkpoint["model_state_dict"], strict=False)
            else:
                model.load_state_dict(checkpoint, strict=False)
        else:
            model.load_state_dict(checkpoint, strict=False)

        model.to(device)
        model.eval()
        logger.info(f"Model loaded successfully on {device} (from {model_path})")
        return model
    except Exception as e:
        logger.exception("Failed to load model")
        raise RuntimeError(f"Model loading failed: {e}") from e


# -----------------------------
# Predict Function
# -----------------------------
def predict(image_bytes: Union[bytes, BinaryIO], model: nn.Module, device: torch.device = DEVICE) -> dict:
    try:
        if isinstance(image_bytes, (bytes, bytearray)):
            image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        else:
            image = Image.open(image_bytes).convert("RGB")

        image_tensor = transform(image).unsqueeze(0).to(device)

        with torch.no_grad():
            outputs = model(image_tensor)
            probs = torch.nn.functional.softmax(outputs, dim=1)
            
            # Get top 5 for debugging
            top5_probs, top5_indices = torch.topk(probs, min(5, len(class_names)), dim=1)
            
            logger.info("Top 5 predictions:")
            for i in range(min(5, len(class_names))):
                idx = int(top5_indices[0][i].item())
                prob = float(top5_probs[0][i].item())
                logger.info(f"  {i+1}. {class_names[idx]}: {prob:.4f}")
            
            confidence, predicted_idx = torch.max(probs, dim=1)
            label = class_names[int(predicted_idx.item())]
            confidence_value = float(confidence.item())

        info = disease_info.get(label, {
            "description": "No detailed info available for this class.",
            "remedy": "Please consult an agricultural expert."
        })

        return {
            "label": label,
            "confidence": confidence_value,
            "description": info["description"],
            "remedy": info["remedy"]
        }

    except Exception as e:
        logger.exception("Prediction failed")
        raise RuntimeError(f"Prediction failed: {e}") from e