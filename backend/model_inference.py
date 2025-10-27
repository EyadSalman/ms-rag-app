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
import matplotlib.pyplot as plt
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights
import timm
from torchvision.models import densenet121, DenseNet121_Weights
from torchvision.models import resnet18, ResNet18_Weights


device = "cuda" if torch.cuda.is_available() else "cpu"

# ===== Dataset 1: Mendeley 60 Patients =====
MENDELEY_DIR = r"D:/ms/Brain MRI Dataset of MS"
MS_NIFTI_DIR = os.path.join(MENDELEY_DIR, "MS")
NORMAL_DIR   = os.path.join(MENDELEY_DIR, "Normal")

# ===== Dataset 2: Cropped Dataset =====
CROPPED_DIR = r"D:/ms/Control and MS"
CROPPED_MS_FOLDERS = ["MS_Axial_crop", "MS_Sagittal_crop"]
CROPPED_CONTROL_FOLDERS = ["Control_Axial_crop", "Control_Sagittal_crop"]

# ===== Output Folders =====
CONVERTED_DIR = r"D:/mss/MS_Normal"
CONVERTED_MS = os.path.join(CONVERTED_DIR, "MS")
CONVERTED_HEALTHY = os.path.join(CONVERTED_DIR, "Healthy")

MERGED_DIR = r"D:/mss/MS_Merged_Datasets"
FINAL_SPLIT_DIR = r"D:/mss/MS_Final_Split_Datasets"


# ----------------------------
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
    
class MobileNetV2_Dropout(nn.Module):
    def __init__(self, dropout_p=0.4):
        super().__init__()
        self.model = mobilenet_v2(weights=None)   # no pretrained weights for CPU stability
        num_ftrs = self.model.classifier[1].in_features
        # replace the classifier head with dropout + linear
        self.model.classifier = nn.Sequential(
            nn.Dropout(p=dropout_p),
            nn.Linear(num_ftrs, 2)
        )

    def forward(self, x):
        return self.model(x)
    
class CNN_Regularized(nn.Module):
    def __init__(self, dropout_p=0.5):
        super(CNN_Regularized, self).__init__()

        # Feature extractor
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),            # 112×112

            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),            # 56×56

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),            # 28×28

            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2),            # 14×14
        )

        # Adaptive pooling → independent of input size
        self.gap = nn.AdaptiveAvgPool2d((1, 1))

        # Classification head
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(p=dropout_p),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(p=dropout_p),
            nn.Linear(128, 2)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.gap(x)
        x = self.classifier(x)
        return x
    
class ViT_Tiny_Regularized(nn.Module):
    def __init__(self, dropout_p=0.5):
        super().__init__()
        base = timm.create_model("vit_tiny_patch16_224", pretrained=True)

        # 🔹 Freeze the transformer encoder initially
        for param in base.blocks.parameters():
            param.requires_grad = False

        num_ftrs = base.head.in_features
        base.head = nn.Sequential(
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

    

# ============================================================
# 🧩 Model Loader
# ============================================================
def load_model(name: str):
    if name == "mobilenet":
        model = MobileNetV2_Dropout(dropout_p=0.4)
        path = "D:/mss/models/mobilenetv2_ms_classifier.pth"

    elif name == "efficientnet":
        model = EfficientNetB0_Regularized(dropout_p=0.5)
        path = "D:/mss/models/efficientnet_b0_mss_bests.pth"
    elif name == "cnn":
        model = CNN_Regularized(dropout_p=0.5)
        path = "D:/mss/models/cnn_mss_best.pth"
    elif name == "vit":
        model = ViT_Tiny_Regularized(dropout_p=0.5)
        path = "D:/mss/models/vit_mss_best.pth"
    elif name == "densenet":
        model = DenseNet121_Regularized(dropout_p=0.5)
        path = "D:/mss/models/densenet121_mss_best.pth"
    elif name == "resnet":
        model = ResNet18_Regularized(dropout_p=0.5)
        path = "D:/mss/models/resnet_18_mss_best.pth"

    else:
        raise ValueError("Unknown model name")

    # Load trained weights
    state = torch.load(path, map_location=device)
    model.load_state_dict(state, strict=False)
    model.to(device).eval()
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
    model = load_model(model_name)
    img = Image.open(io.BytesIO(file_bytes)).convert("RGB")
    tensor = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(tensor)
        probs = torch.softmax(logits, dim=1)
        conf, pred = torch.max(probs, 1)

    diagnosis = "Multiple Sclerosis Detected" if pred.item() == 1 else "Healthy Brain"
    return {
        "model": model_name,
        "diagnosis": diagnosis,
        "confidence": round(conf.item() * 100, 2)
    }