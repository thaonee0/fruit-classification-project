import torch
from torch import nn
from torchvision.models import resnet50, ResNet50_Weights

class FruitClassifier(nn.Module):
    def __init__(self, num_classes):
        super(FruitClassifier, self).__init__()
        self.resnet = resnet50(weights=ResNet50_Weights.DEFAULT)
        self.features = nn.Sequential(*list(self.resnet.children())[:-1])
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(2048, 512),
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