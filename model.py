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
# -*- coding: utf-8 -*-

# disease_info.py (Full Nepali version, formal language)

disease_info = {
    # ---------------- स्याउ (Apple) ----------------
    "Apple___Apple_scab": {
        "description": "स्याउ स्क्याब Venturia inaequalis फफूंदीले हुने रोग हो। यसले पात, फल, र साना टहनीमा कालो र खुरदुरा दागहरू बनाउँछ।",
        "remedy": "रोग प्रतिरोधी स्याउ प्रजाति रोप्नुहोस्, कैप्टान वा म्यान्कोसेब जस्ता फफूंदी नाशक प्रयोग गर्नुहोस्, र झरेका पातहरू हटाएर फफूंदीको बीउ कम गर्नुहोस्।"
    },
    "Apple___Black_rot": {
        "description": "कालो गलन Botryosphaeria obtusa फफूंदीले हुने रोग हो। यसले पात, फल र छालामा वृत्ताकार दागहरू बनाउँछ र फल सडाउँछ।",
        "remedy": "संक्रमित टहनी काटेर हटाउनुहोस्, झरेका पात जलाउनुहोस्, र थायोफेनेट-मेथिल वा तामा आधारित छिड़काव गर्नुहोस्।"
    },
    "Apple___Cedar_apple_rust": {
        "description": "सिडर स्याउ रस्ट Gymnosporangium juniperi-virginianae फफूंदीले हुने रोग हो। यसले पात र फलमा सुन्तला रंगका जेली जस्ता दागहरू बनाउँछ।",
        "remedy": "सम्भव भए नजिकका सिडर रूख हटाउनुहोस्, वसन्तमा प्रारम्भिक फफूंदी नाशक छिड़काव गर्नुहोस्, र रोग प्रतिरोधी स्याउ प्रजाति रोप्नुहोस्।"
    },
    "Apple___healthy": {
        "description": "यो स्याउको पात स्वस्थ छ र कुनै रोग छैन।",
        "remedy": "नियमित सिंचाइ, छाँटकाँट, र पोषण व्यवस्थापन जारी राख्नुहोस्।"
    },

    # ---------------- ब्लूबेरी (Blueberry) ----------------
    "Blueberry___healthy": {
        "description": "ब्लूबेरी बिरुवा स्वस्थ छ र रोगको कुनै संकेत देखिँदैन।",
        "remedy": "माटोको अम्लीयता कायम राख्नुहोस्, नियमित पानी दिनुहोस्, र मल्च लगाएर झार–झार रोक्नुहोस्।"
    },

    # ---------------- चेरी (Cherry) ----------------
    "Cherry_(including_sour)___Powdery_mildew": {
        "description": "पाउडरी मिल्ड्यू फफूंदी रोग हो जसले चेरीको पात र कांडमा सेतो पाउडर जस्ता दागहरू बनाउँछ।",
        "remedy": "भीडभाड भएका टहनी काट्नुहोस्, हावा परिसंचरण सुधार गर्नुहोस्, र गन्धक वा निम तेलको छिड़काव गर्नुहोस्।"
    },
    "Cherry_(including_sour)___healthy": {
        "description": "चेरीका पात स्वस्थ छन्, कुनै फफूंदी वा ब्याक्टेरियल संक्रमण छैन।",
        "remedy": "उचित पानी र मल दिनुहोस्; सरसफाइ कायम राख्नुहोस्।"
    },

    # ---------------- मकै (Corn) ----------------
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot": {
        "description": "ग्रे लीफ स्पट Cercospora जातिको फफूंदीले हुने रोग हो, जसले पातमा आयताकार खरानी दाग बनाउँछ।",
        "remedy": "बाली घुमाउने गर्नुहोस्, रोग प्रतिरोधी हाइब्रिड प्रयोग गर्नुहोस्, र आवश्यक भए स्ट्रोबिलुरिन जस्ता फफूंदी नाशक छिड़काव गर्नुहोस्।"
    },
    "Corn_(maize)___Common_rust_": {
        "description": "कमन रस्ट Puccinia sorghi फफूंदीले हुने रोग हो। यसले पातमा रातो-खैरो पोथ्री जस्ता दागहरू बनाउँछ।",
        "remedy": "प्रतिरोधी प्रजाति प्रयोग गर्नुहोस् र रोग बढे फफूंदी नाशक छिड़काव गर्नुहोस्।"
    },
    "Corn_(maize)___Northern_Leaf_Blight": {
        "description": "नर्दन लीफ ब्लाइट Exserohilum turcicum फफूंदीले हुने रोग हो, जसले पातमा लामो खरानी दाग बनाउँछ।",
        "remedy": "प्रतिरोधी हाइब्रिड प्रयोग गर्नुहोस्, बाली घुमाउनुहोस्, र प्रारम्भिक चरणमा फफूंदी नाशक छिड़काव गर्नुहोस्।"
    },
    "Corn_(maize)___healthy": {
        "description": "मकैका पात हरियो र स्वस्थ छन्।",
        "remedy": "उचित नाइट्रोजन स्तर र सिंचाइ कायम राख्नुहोस्।"
    },

    # ---------------- अंगुर (Grape) ----------------
    "Grape___Black_rot": {
        "description": "कालो गलन Guignardia bidwellii फफूंदीले हुने रोग हो, जसले पातमा कालो दाग र फल सुकाउँछ।",
        "remedy": "संक्रमित भाग काट्नुहोस्, हावा परिसंचरण सुधार गर्नुहोस्, र माइक्लोब्युटानिल जस्ता फफूंदी नाशक छिड़काव गर्नुहोस्।"
    },
    "Grape___Esca_(Black_Measles)": {
        "description": "एस्का (ब्ल्याक मिजल्स) रोगले पात र फलमा कालो रेखा र दागहरू बनाउँछ र बाइनलाई मर्न पुर्‍याउँछ।",
        "remedy": "संक्रमित बाइन हटाउनुहोस् र जलाउनुहोस्, घाउ नहोस्, र पानी निकास सुधार गर्नुहोस्।"
    },
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)": {
        "description": "पात ब्लाइट Pseudocercospora vitis फफूंदीले हुने रोग हो, जसले पातमा खैरो कोणीय दाग बनाउँछ।",
        "remedy": "संक्रमित पात काट्नुहोस् र सुरक्षात्मक फफूंदी नाशक प्रयोग गर्नुहोस्।"
    },
    "Grape___healthy": {
        "description": "अंगुरको बाइन स्वस्थ छ, पात हरियो र जीवन्त छन्।",
        "remedy": "सन्तुलित मल र उचित छाँटकाँट गरेर हावा परिसंचरण सुधार गर्नुहोस्।"
    },

    # ---------------- सुन्तला (Orange) ----------------
    "Orange___Haunglongbing_(Citrus_greening)": {
        "description": "सिट्रस ग्रिनिङ (HLB) Candidatus Liberibacter ब्याक्टेरियाले हुने रोग हो, जसले हरा-पीला सिसाहरू र विकृत फल बनाउँछ।",
        "remedy": "संक्रमित रूख हटाउनुहोस्, सायलिड कीरा नियन्त्रण गर्नुहोस्, र रोगरहित बीउ प्रयोग गर्नुहोस्।"
    },

    # ---------------- पीच (Peach) ----------------
    "Peach___Bacterial_spot": {
        "description": "ब्याक्टेरियल स्पट Xanthomonas campestris द्वारा हुने रोग हो, जसले पात र फलमा कालो दाग बनाउँछ।",
        "remedy": "तामा आधारित ब्याक्टेरिसाइड प्रयोग गर्नुहोस् र माथि पानी दिने सिंचाइबाट बच्नुहोस्।"
    },
    "Peach___healthy": {
        "description": "पीचका पात स्वस्थ छन्, रोगरहित छन्।",
        "remedy": "राम्रो माटोको निकास सुनिश्चित गर्नुहोस् र रोग प्रतिरोधी प्रजाति प्रयोग गर्नुहोस्।"
    },

    # ---------------- खुर्सानी (Pepper, bell) ----------------
    "Pepper,_bell___Bacterial_spot": {
        "description": "बेल खुर्सानीमा ब्याक्टेरियल स्पट Xanthomonas जातिको कारणले हुने रोग हो, जसले गाढा दाग बनाउँछ।",
        "remedy": "प्रमाणित बीउ प्रयोग गर्नुहोस्, बाली घुमाउनुहोस्, र रोकथामका लागि तामा आधारित फफूंदी नाशक छिड़काव गर्नुहोस्।"
    },
    "Pepper,_bell___healthy": {
        "description": "बेल खुर्सानी स्वस्थ छ, कुनै रोग छैन।",
        "remedy": "उचित पानी र कीरा नियन्त्रण कायम राख्नुहोस्।"
    },

    # ---------------- आलु (Potato) ----------------
    "Potato___Early_blight": {
        "description": "अर्ली ब्लाइट Alternaria solani फफूंदीले हुने रोग हो, जसले पातमा वृत्ताकार खैरो घेराहरू बनाउँछ।",
        "remedy": "रोगरहित बीउ प्रयोग गर्नुहोस्, बाली घुमाउनुहोस्, र क्लोरोथालोनिल जस्ता फफूंदी नाशक प्रयोग गर्नुहोस्।"
    },
    "Potato___Late_blight": {
        "description": "लेट ब्लाइट Phytophthora infestans फफूंदीले हुने रोग हो, जसले पात र आलुको भागमा कालो दाग बनाउँछ।",
        "remedy": "भिजेको अवस्था टार्नुहोस्, प्रतिरोधी प्रजाति प्रयोग गर्नुहोस्, र मेटालाक्सिल जस्ता फफूंदी नाशक छिड़काव गर्नुहोस्।"
    },
    "Potato___healthy": {
        "description": "आलु स्वस्थ छ, कुनै रोग वा ब्लाइट छैन।",
        "remedy": "उचित हिलिङ र पानी व्यवस्थापन कायम राख्नुहोस्।"
    },

    # ---------------- रास्पबेरी (Raspberry) ----------------
    "Raspberry___healthy": {
        "description": "रास्पबेरी स्वस्थ छ, कुनै रोग छैन।",
        "remedy": "उचित छाँटकाँट गर्नुहोस् र हावा परिसंचरण सुनिश्चित गर्नुहोस्।"
    },

    # ---------------- सोयाबिन (Soybean) ----------------
    "Soybean___healthy": {
        "description": "सोयाबिन स्वस्थ छ, कुनै रोगको संकेत छैन।",
        "remedy": "बाली घुमाउनुहोस् र झार–झार नियन्त्रण गर्नुहोस्।"
    },

    # ---------------- स्क्वाश (Squash) ----------------
    "Squash___Powdery_mildew": {
        "description": "पाउडरी मिल्ड्यू पात र कांडमा सेतो पाउडर जस्ता दाग बनाउँछ।",
        "remedy": "संक्रमित पात हटाउनुहोस्, हावा परिसंचरण सुधार गर्नुहोस्, र गन्धक आधारित फफूंदी नाशक छिड़काव गर्नुहोस्।"
    },

    # ---------------- स्ट्रबेरी (Strawberry) ----------------
    "Strawberry___Leaf_scorch": {
        "description": "लीफ स्कोर्चले पातमा रातो–खैरो दाग बनाउँछ र अन्ततः पात मर्न जान्छ।",
        "remedy": "संक्रमित पात हटाउनुहोस्, माथि पानीबाट बच्नुहोस्, र रोग प्रतिरोधी प्रजाति प्रयोग गर्नुहोस्।"
    },
    "Strawberry___healthy": {
        "description": "स्ट्रबेरी स्वस्थ छ, पात हरियो र तन्दुरुस्त छन्।",
        "remedy": "पर्याप्त सूर्यप्रकाश र दूरी कायम गर्नुहोस्।"
    },

    # ---------------- टमाटर (Tomato) ----------------
    "Tomato___Bacterial_spot": {
        "description": "ब्याक्टेरियल स्पट Xanthomonas जातिले पात र फलमा साना कालो दाग बनाउँछ।",
        "remedy": "रोगरहित बीउ प्रयोग गर्नुहोस्, तामा स्प्रे गर्नुहोस्, र बाली घुमाउनुहोस्।"
    },
    "Tomato___Early_blight": {
        "description": "अर्ली ब्लाइट Alternaria solani फफूंदीले पुराना पातमा घेराबद्ध खैरो दाग बनाउँछ।",
        "remedy": "संक्रमित पात काट्नुहोस्, हावा परिसंचरण सुधार गर्नुहोस्, र क्लोरोथालोनिल फफूंदी नाशक प्रयोग गर्नुहोस्।"
    },
    "Tomato___Late_blight": {
        "description": "लेट ब्लाइट Phytophthora infestans फफूंदीले पात र फलमा ठूला पानी–सिक्त दाग बनाउँछ।",
        "remedy": "माथि पानीबाट बच्नुहोस्, संक्रमित बिरुवा नष्ट गर्नुहोस्, र प्रणालीगत फफूंदी नाशक प्रयोग गर्नुहोस्।"
    },
    "Tomato___Leaf_Mold": {
        "description": "पात मोल्ड Passalora fulva फफूंदीले पातको माथिल्लो भागमा पहेँलो दाग बनाउँछ।",
        "remedy": "हावा परिसंचरण सुधार्नुहोस्, आर्द्रता कम गर्नुहोस्, र तामा आधारित फफूंदी नाशक छिड़काव गर्नुहोस्।"
    },
    "Tomato___Septoria_leaf_spot": {
        "description": "सेप्टोरिया लीफ स्पटले साना वृत्ताकार दाग बनाउँछ, केन्द्र खरानी हुन्छ।",
        "remedy": "संक्रमित पात हटाउनुहोस्, पात सुक्खा राख्नुहोस्, र म्यान्कोसेब जस्ता फफूंदी नाशक प्रयोग गर्नुहोस्।"
    },
    "Tomato___Spider_mites Two-spotted_spider_mite": {
        "description": "स्पाइडर माइटले पातमा पहेँलो–खैरो धब्बा र जाल बनाउँछ, जसले पात झर्न सक्छ।",
        "remedy": "निम तेल वा कीट नाशक साबुन छिड़काव गर्नुहोस् र आर्द्रता कायम गर्नुहोस्।"
    },
    "Tomato___Target_Spot": {
        "description": "टार्गेट स्पट Corynespora cassiicola फफूंदीले लक्ष्य–जस्तो घेराबद्ध दाग बनाउँछ।",
        "remedy": "संक्रमित पात हटाउनुहोस्, बाली घुमाउनुहोस्, र रोकथामका लागि फफूंदी नाशक छिड़काव गर्नुहोस्।"
    },
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus": {
        "description": "यस भाइरल रोगले पात माथि ताने जस्तो मोडिन्छ र बिरुवा झन् स–छोटो हुन्छ।",
        "remedy": "ह्वाइटफ्लाई नियन्त्रण गर्नुहोस्, संक्रमित बिरुवा हटाउनुहोस्, र रोग प्रतिरोधी प्रजाति प्रयोग गर्नुहोस्।"
    },
    "Tomato___Tomato_mosaic_virus": {
        "description": "TMV ले पातमा मोजाइक जस्तो धब्बा र विकृति ल्याउँछ।",
        "remedy": "तम्बाकू प्रयोग पछि बिरुवा छोएन, उपकरण सफा गर्नुहोस्, र प्रतिरोधी प्रजाति प्रयोग गर्नुहोस्।"
    },
    "Tomato___healthy": {
        "description": "टमाटर स्वस्थ छ, कुनै रोग छैन।",
        "remedy": "नियमित पानी र पोषणको सन्तुलन कायम राख्नुहोस्।"
    }
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
