import torch
from torch import nn
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights

class FruitClassifier(nn.Module):
    def __init__(self, num_classes):
        super(FruitClassifier, self).__init__()
        self.mobilenet = mobilenet_v2(weights=MobileNet_V2_Weights.DEFAULT)
        
        # Lấy phần feature extractor của MobileNetV2
        self.features = self.mobilenet.features
        
        # Tạo classifier thay thế phần cuối của MobileNetV2
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(1280, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x
