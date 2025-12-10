# backend/model_inference.py

import torch
import torch.nn as nn
from PIL import Image
import io
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
from torchvision import transforms, datasets
import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # Suppress TensorFlow logs entirely
os.environ["TOKENIZERS_PARALLELISM"] = "false"
from torchvision.models import mobilenet_v3_large, MobileNet_V3_Large_Weights
import timm
from torchvision.models import densenet121, DenseNet121_Weights
from torchvision.models import resnet18, ResNet18_Weights
import numpy as np
import cv2


device = "cuda" if torch.cuda.is_available() else "cpu"

# Get current folder (backend/)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")

# ============================================================
# MRI VALIDATION FUNCTION (must be ABOVE predict_mri)
# ============================================================
def is_mri_image(img: Image.Image) -> bool:
    """
    Lightweight validation to avoid non-MRI images.
    Uses color variance, edge density, and circle detection.
    """

    img_np = np.array(img)
    img_gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)

    # 3. Detect circular brain-like region
    circles = cv2.HoughCircles(
        img_gray, cv2.HOUGH_GRADIENT,
        dp=1.2, minDist=80,
        param1=50, param2=30,
        minRadius=40, maxRadius=120
    )

    if circles is None:
        return False

    # 1. Too colorful → not MRI
    if np.std(img_np) > 60:
        return False

    # 2. Too many edges → likely a real-life photo
    edges = cv2.Canny(img_gray, 50, 150)
    if (edges > 0).sum() / edges.size > 0.10:
        return False

    return True


# Custom class definition (must match training)
# ----------------------------
class EfficientNetB0_Regularized(nn.Module):
    def __init__(self, dropout_p=0.5):
        super().__init__()
        base = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
        
        # 🔹 Freeze feature extractor initially (same as training start)
        for param in base.features.parameters():
            param.requires_grad = False

        num_ftrs = base.classifier[1].in_features
        base.classifier = nn.Sequential(
            nn.Dropout(p=dropout_p),
            nn.Linear(num_ftrs, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(p=dropout_p),
            nn.Linear(512, 2)
        )
        self.model = base

    def forward(self, x):
        return self.model(x)
    
class MobileNetmobilV3_Regularized(nn.Module):
    def __init__(self, dropout_p=0.5):
        super().__init__()
        base = mobilenet_v3_large(weights=MobileNet_V3_Large_Weights.IMAGENET1K_V1)
        
        # 🔹 Freeze feature extractor initially
        for param in base.features.parameters():
            param.requires_grad = False

        # Extract number of features from the last layer before classifier
        num_ftrs = base.classifier[0].in_features  # 960 for MobileNetV3-Large

        # 🔧 Replace classifier head with regularized one
        base.classifier = nn.Sequential(
            nn.Dropout(p=dropout_p),
            nn.Linear(num_ftrs, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(p=dropout_p),
            nn.Linear(512, 2)
        )
        self.model = base

    def forward(self, x):
        return self.model(x)
    
class DenseNet121_Regularized(nn.Module):
    def __init__(self, dropout_p=0.5):
        super().__init__()
        base = densenet121(weights=DenseNet121_Weights.IMAGENET1K_V1)

        # 🔹 Freeze feature extractor initially
        for param in base.features.parameters():
            param.requires_grad = False

        # 🔹 Replace classifier with regularized head
        num_ftrs = base.classifier.in_features
        base.classifier = nn.Sequential(
            nn.Dropout(p=dropout_p),
            nn.Linear(num_ftrs, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(p=dropout_p),
            nn.Linear(512, 2)
        )

        self.model = base

    def forward(self, x):
        return self.model(x)
    
class ResNet18_Regularized(nn.Module):
    def __init__(self, dropout_p=0.5):
        super().__init__()
        base = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)

        # 🔹 Freeze convolutional backbone initially
        for param in base.layer1.parameters():
            param.requires_grad = False
        for param in base.layer2.parameters():
            param.requires_grad = False
        for param in base.layer3.parameters():
            param.requires_grad = False
        for param in base.layer4.parameters():
            param.requires_grad = False

        # 🔹 Replace classification head with a regularized one
        num_ftrs = base.fc.in_features
        base.fc = nn.Sequential(
            nn.Dropout(p=dropout_p),
            nn.Linear(num_ftrs, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(p=dropout_p),
            nn.Linear(512, 2)
        )

        self.model = base

    def forward(self, x):
        return self.model(x)

    
loaded_models = {}
# ============================================================
# 🧩 Model Loader
# ============================================================
def load_model(name: str):

    if name in loaded_models:
        return loaded_models[name]

    MODEL_REGISTRY = {
        "mobilenet": (
            "mobilenet_mss_bests.pth",
            MobileNetmobilV3_Regularized(dropout_p=0.4)
        ),
        "efficientnet": (
            "efficientnet_b0_mss_best_weights.pth",
            EfficientNetB0_Regularized(dropout_p=0.5)
        ),
        "densenet": (
            "densenet_121_new_mss_best.pth",
            DenseNet121_Regularized(dropout_p=0.5)
        ),
        "resnet": (
            "resnet_18_mss_best.pth",
            ResNet18_Regularized(dropout_p=0.5)
        ),
    }

    if name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model name: {name}")

    filename, model = MODEL_REGISTRY[name]
    path = os.path.join(MODEL_DIR, filename)

    state = torch.load(path, map_location=device)
    model.load_state_dict(state, strict=False)
    model.to(device).eval()

    loaded_models[name] = model 

    return model

# ----------------------------
# Define preprocessing
# ----------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
])

# ----------------------------
# Prediction function
# ----------------------------
def predict_mri(file_bytes, model_name: str):
    img = Image.open(io.BytesIO(file_bytes)).convert("RGB")

    # 🔍 MRI VALIDATION STEP
    if not is_mri_image(img):
        return {
            "error": True,
            "message": "❌ The uploaded image does not appear to be an MRI scan. Please upload a valid brain MRI."
        }

    model = load_model(model_name)
    tensor = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(tensor)
        probs = torch.softmax(logits, dim=1)
        conf, pred = torch.max(probs, 1)

    diagnosis = "Multiple Sclerosis Detected" if pred.item() == 1 else "Healthy Brain"

    return {
        "error": False,
        "model": model_name,
        "diagnosis": diagnosis,
        "confidence": round(conf.item() * 100, 2)
    }
