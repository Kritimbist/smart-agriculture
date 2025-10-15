import logging
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
from huggingface_hub import hf_hub_download
from typing import Union, BinaryIO

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Model configuration
IMAGE_SIZE = 256
IN_CHANNELS = 3
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
REPO_ID = "kritimbista/my-model-weights"
MODEL_FILENAME = "model_weights.pth"

class ResNet9(nn.Module):
    """ResNet9 architecture for plant disease classification"""
    
    def __init__(self, in_channels: int, num_classes: int):
        super().__init__()
        
        def conv_block(in_channels: int, out_channels: int, pool: bool = False) -> nn.Sequential:
            layers = [
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
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
            nn.MaxPool2d(4),
            nn.Flatten(),
            nn.Dropout(0.2),  # Added dropout for regularization
            nn.Linear(512, num_classes)
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

# Class names (unchanged)
class_names = [
   'Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
    'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew', 'Cherry_(including_sour)___healthy',
    'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 'Corn_(maize)___Common_rust_', 'Corn_(maize)___Northern_Leaf_Blight',
    'Corn_(maize)___healthy', 'Grape___Black_rot', 'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
    'Grape___healthy', 'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot', 'Peach___healthy',
    'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy', 'Potato___Early_blight', 'Potato___Late_blight',
    'Potato___healthy', 'Raspberry___healthy', 'Soybean___healthy', 'Squash___Powdery_mildew',
    'Strawberry___Leaf_scorch', 'Strawberry___healthy', 'Tomato___Bacterial_spot', 'Tomato___Early_blight',
    'Tomato___Late_blight', 'Tomato___Leaf_Mold', 'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite',
    'Tomato___Target_Spot', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus', 'Tomato___healthy'
]


# Image transformation pipeline
transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

def load_model(device: torch.device = DEVICE) -> ResNet9:
    """
    Load the ResNet9 model with pretrained weights.
    
    Args:
        device: torch.device to load the model on
        
    Returns:
        ResNet9: Loaded and configured model
        
    Raises:
        RuntimeError: If model loading fails
    """
    try:
        model = ResNet9(in_channels=IN_CHANNELS, num_classes=len(class_names))
        model_path = hf_hub_download(
            repo_id=REPO_ID,
            filename=MODEL_FILENAME
        )
        state_dict = torch.load(model_path, map_location=device)
        model.load_state_dict(state_dict)
        model.to(device)
        model.eval()
        logger.info(f"Model loaded successfully on {device}")
        return model
    except Exception as e:
        logger.error(f"Failed to load model: {str(e)}")
        raise RuntimeError(f"Model loading failed: {str(e)}")

def predict(image_bytes: Union[bytes, BinaryIO], model: ResNet9, device: torch.device = DEVICE) -> str:
    """
    Predict plant disease from image.
    
    Args:
        image_bytes: Image file in bytes or file-like object
        model: Loaded ResNet9 model
        device: torch.device to run inference on
        
    Returns:
        str: Predicted class name
        
    Raises:
        RuntimeError: If prediction fails
    """
    try:
        if not isinstance(model, ResNet9):
            raise ValueError("Invalid model type")
            
        image = Image.open(image_bytes).convert("RGB")
        image_tensor = transform(image).unsqueeze(0).to(device)
        
        with torch.no_grad():
            outputs = model(image_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            _, predicted = torch.max(probabilities, 1)
            confidence = probabilities[0][predicted].item()
            
        prediction = class_names[predicted.item()]
        logger.info(f"Prediction: {prediction} (confidence: {confidence:.2f})")
        return prediction
        
    except Exception as e:
        logger.error(f"Prediction failed: {str(e)}")
        raise RuntimeError(f"Prediction failed: {str(e)}")

if __name__ == "__main__":
    # Simple test to verify model loading
    try:
        model = load_model()
        logger.info("Model test successful")
    except Exception as e:
        logger.error(f"Model test failed: {str(e)}")
